from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reid.config import ExperimentConfig, str2bool
from reid.data import build_dataloaders
from reid.diagnostics import (
    apply_checkpoint_model_config,
    batch_branch_statistics,
    collect_model_state_diagnostics,
    evaluate_branch_mode,
    gradient_branch_statistics,
    load_residual_scale_history,
)
from reid.losses import ReIDCriterion
from reid.model import build_model
from reid.performance import configure_torch_runtime, model_to_device
from reid.utils import load_checkpoint, resolve_device


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), Path(path.strip())
    path = Path(spec.strip())
    return path.parent.name, path


def format_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def run_diagnostics(args: argparse.Namespace, label: str, checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    device = resolve_device(args.device)
    configure_torch_runtime(device, args.cudnn_benchmark, args.allow_tf32)
    checkpoint = load_checkpoint(str(checkpoint_path), device)

    config = ExperimentConfig(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        use_amp=args.use_amp,
        channels_last=args.channels_last,
        eval_gpu_distance=args.eval_gpu_distance,
        eval_gpu_distance_max_elements=args.eval_gpu_distance_max_elements,
        distance_metric=args.distance_metric,
        max_rank=args.max_rank,
    )
    apply_checkpoint_model_config(config, checkpoint)
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.prefetch_factor = args.prefetch_factor
    config.persistent_workers = args.persistent_workers
    config.pin_memory = args.pin_memory
    config.use_amp = args.use_amp
    config.channels_last = args.channels_last
    config.eval_gpu_distance = args.eval_gpu_distance
    config.eval_gpu_distance_max_elements = args.eval_gpu_distance_max_elements
    config.distance_metric = args.distance_metric
    config.max_rank = args.max_rank

    train_loader, query_loader, gallery_loader, bundle = build_dataloaders(config)
    model = model_to_device(build_model(config, num_classes=bundle.num_train_ids), device, args.channels_last)
    model.load_state_dict(checkpoint["model"])

    criterion = ReIDCriterion(
        ce_weight=config.ce_weight,
        triplet_weight=config.triplet_weight,
        label_smoothing=config.label_smoothing,
        triplet_margin=config.triplet_margin,
        local_loss_weight=config.local_loss_weight,
    )
    state_stats = collect_model_state_diagnostics(model)
    feature_stats = batch_branch_statistics(
        model=model,
        loader=train_loader,
        device=device,
        channels_last=args.channels_last,
        use_amp=args.use_amp,
        max_batches=args.max_train_batches,
    )
    grad_stats = gradient_branch_statistics(
        model=model,
        loader=train_loader,
        criterion=criterion,
        device=device,
        channels_last=args.channels_last,
        max_batches=args.max_train_batches,
        use_amp=args.grad_use_amp,
    )

    retrieval: dict[str, dict[str, float] | None] = {}
    if not args.skip_retrieval_modes:
        for mode in args.feature_modes:
            retrieval[mode] = evaluate_branch_mode(
                model=model,
                query_loader=query_loader,
                gallery_loader=gallery_loader,
                device=device,
                mode=mode,
                distance_metric=args.distance_metric,
                max_rank=args.max_rank,
                channels_last=args.channels_last,
                use_amp=args.use_amp,
                gpu_distance=args.eval_gpu_distance,
                gpu_distance_max_elements=args.eval_gpu_distance_max_elements,
            )

    return {
        "label": label,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset_name": config.dataset_name,
        "experiment_name": config.experiment_name,
        "fusion_mode": config.fusion_mode,
        "local_residual_weight": config.local_residual_weight,
        "local_residual_learnable": config.local_residual_learnable,
        "local_loss_weight": config.local_loss_weight,
        "use_transformer": config.use_transformer,
        "use_fusion_gate": config.use_fusion_gate,
        "num_parts": config.num_parts,
        "state": state_stats,
        "feature_stats": feature_stats,
        "gradient_stats": grad_stats,
        "retrieval": retrieval,
        "scale_history": load_residual_scale_history(checkpoint_path.with_name("train_log.jsonl")),
    }


def build_markdown(results: list[dict[str, Any]]) -> str:
    scale_rows = []
    feature_rows = []
    grad_rows = []
    retrieval_rows = []
    notes = []

    for result in results:
        state = result["state"]
        feature = result["feature_stats"]
        grad = result["gradient_stats"]
        label = result["label"]
        scale_rows.append(
            [
                label,
                result["fusion_mode"],
                str(result["local_residual_learnable"]),
                format_float(state.get("local_residual_scale"), 8),
                format_float(state.get("local_residual_scale_abs"), 8),
                str(len(result["scale_history"])),
            ]
        )
        feature_rows.append(
            [
                label,
                format_float(feature.get("global_feature_norm")),
                format_float(feature.get("local_feature_norm")),
                format_float(feature.get("local_to_global_feature_norm_ratio")),
                format_float(feature.get("fused_delta_to_global_feature_norm_ratio"), 6),
                format_float(feature.get("full_minus_global_logit_delta_norm")),
                format_float(feature.get("target_logit_delta_full_minus_global")),
                format_float(feature.get("local_aux_logit_confidence")),
            ]
        )
        grad_rows.append(
            [
                label,
                format_float(grad.get("backbone")),
                format_float(grad.get("global_projection")),
                format_float(grad.get("local_branch")),
                format_float(grad.get("fusion")),
                format_float(grad.get("global_head")),
                format_float(grad.get("local_aux_head")),
            ]
        )
        for mode, metrics in result["retrieval"].items():
            retrieval_rows.append(
                [
                    label,
                    mode,
                    format_percent(None if metrics is None else metrics.get("rank1")),
                    format_percent(None if metrics is None else metrics.get("mAP")),
                    format_percent(None if metrics is None else metrics.get("rank5")),
                    format_percent(None if metrics is None else metrics.get("rank10")),
                ]
            )
        if not result["scale_history"] and result["state"].get("use_local_branch"):
            notes.append(
                f"- `{label}` 的旧训练日志没有逐 epoch `local_residual_scale` 字段；本次只能报告 checkpoint 端点值。"
            )

    lines = [
        "# Branch Marginalization Diagnostics",
        "",
        "This report checks whether the local branch is actually used by the trained ReID model.",
        "",
        "## Residual Scale",
        "",
        markdown_table(
            ["Run", "Fusion", "Learnable", "Scale", "Abs Scale", "Logged Epochs"],
            scale_rows,
        ),
        "",
        "## Feature And Logit Contribution",
        "",
        markdown_table(
            [
                "Run",
                "Global Norm",
                "Local Norm",
                "Local/Global",
                "Fused Delta/Global",
                "Logit Delta Norm",
                "Target Logit Delta",
                "Local Aux Conf",
            ],
            feature_rows,
        ),
        "",
        "## Gradient Norms",
        "",
        markdown_table(
            ["Run", "Backbone", "Global Proj", "Local Branch", "Fusion", "Global Head", "Local Aux Head"],
            grad_rows,
        ),
    ]
    if retrieval_rows:
        lines.extend(
            [
                "",
                "## Retrieval Discriminability By Feature Mode",
                "",
                markdown_table(["Run", "Feature", "Rank-1", "mAP", "Rank-5", "Rank-10"], retrieval_rows),
            ]
        )
    if notes:
        lines.extend(["", "## Notes", "", *notes])
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- If `Scale` is close to zero and `Fused Delta/Global` is tiny, the local branch is structurally present but contributes little to retrieval.",
            "- If `Local Branch` gradient norm is much smaller than the global path and there is no `Local Aux Head`, the local branch is weakly supervised.",
            "- If local-only retrieval is poor while fixed residual plus auxiliary supervision improves fused retrieval, the evidence supports the conclusion that local supervision and controlled fusion are solving the same marginalization problem.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze local-branch marginalization from trained ReID checkpoints.")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--run", action="append", required=True, help="Run spec in the form label=checkpoint_path")
    parser.add_argument("--output-dir", default="outputs/analysis/branch_marginalization")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", type=str2bool, default=True)
    parser.add_argument("--pin-memory", type=str2bool, default=True)
    parser.add_argument("--channels-last", type=str2bool, default=True)
    parser.add_argument("--use-amp", type=str2bool, default=True)
    parser.add_argument("--grad-use-amp", type=str2bool, default=False)
    parser.add_argument("--cudnn-benchmark", type=str2bool, default=True)
    parser.add_argument("--allow-tf32", type=str2bool, default=True)
    parser.add_argument("--distance-metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--max-rank", type=int, default=20)
    parser.add_argument("--eval-gpu-distance", type=str2bool, default=True)
    parser.add_argument("--eval-gpu-distance-max-elements", type=int, default=200_000_000)
    parser.add_argument("--feature-modes", nargs="+", default=["fused", "global", "local"], choices=["fused", "global", "local"])
    parser.add_argument("--max-train-batches", type=int, default=4)
    parser.add_argument("--skip-retrieval-modes", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [run_diagnostics(args, *parse_run_spec(run_spec)) for run_spec in args.run]

    (output_dir / "branch_diagnostics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    feature_rows = [{"run": item["label"], **item["feature_stats"]} for item in results]
    grad_rows = [{"run": item["label"], **item["gradient_stats"]} for item in results]
    state_rows = [{"run": item["label"], **item["state"]} for item in results]
    retrieval_rows = []
    scale_rows = []
    for item in results:
        for mode, metrics in item["retrieval"].items():
            if metrics is None:
                retrieval_rows.append({"run": item["label"], "feature_mode": mode})
            else:
                retrieval_rows.append({"run": item["label"], "feature_mode": mode, **metrics})
        for record in item["scale_history"]:
            scale_rows.append({"run": item["label"], **record})

    write_csv(output_dir / "feature_stats.csv", feature_rows)
    write_csv(output_dir / "gradient_stats.csv", grad_rows)
    write_csv(output_dir / "model_state.csv", state_rows)
    write_csv(output_dir / "retrieval_by_feature.csv", retrieval_rows)
    write_csv(output_dir / "residual_scale_history.csv", scale_rows)
    (output_dir / "summary.md").write_text(build_markdown(results), encoding="utf-8")
    print(f"Saved branch diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
