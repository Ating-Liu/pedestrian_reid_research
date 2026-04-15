from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reid.config import ExperimentConfig, str2bool
from reid.data import build_dataloaders
from reid.evaluation import extract_features
from reid.metrics import compute_distance_matrix
from reid.model import build_model
from reid.performance import configure_torch_runtime, model_to_device
from reid.utils import load_checkpoint, resolve_device


MODEL_CONFIG_KEYS = (
    "backbone",
    "pretrained",
    "last_stride",
    "embedding_dim",
    "use_local_branch",
    "use_transformer",
    "use_fusion_gate",
    "num_parts",
    "transformer_dim",
    "transformer_heads",
    "transformer_layers",
    "transformer_dropout",
    "fusion_mode",
    "local_residual_weight",
    "local_residual_learnable",
    "local_loss_weight",
)


@dataclass
class QueryCase:
    query_index: int
    query_path: str
    query_id: int
    query_camera: int
    baseline_first_correct_rank: int
    target_first_correct_rank: int
    baseline_rank1_correct: bool
    target_rank1_correct: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate retrieval success and failure case visualizations.")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--target-checkpoint", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", type=str2bool, default=False)
    parser.add_argument("--distance-metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--use-amp", type=str2bool, default=True)
    parser.add_argument("--channels-last", type=str2bool, default=True)
    parser.add_argument("--eval-gpu-distance", type=str2bool, default=True)
    parser.add_argument("--eval-gpu-distance-max-elements", type=int, default=200_000_000)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--num-cases", type=int, default=4)
    return parser.parse_args()


def apply_checkpoint_model_config(config: ExperimentConfig, checkpoint: dict) -> None:
    saved = checkpoint.get("config", {})
    for key in MODEL_CONFIG_KEYS:
        if key in saved:
            setattr(config, key, saved[key])


def load_model_from_checkpoint(
    checkpoint_path: str,
    base_config: ExperimentConfig,
    num_classes: int,
    device: torch.device,
):
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_config = copy.deepcopy(base_config)
    apply_checkpoint_model_config(model_config, checkpoint)
    model = model_to_device(build_model(model_config, num_classes=num_classes), device, base_config.channels_last)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def filtered_ranking(
    distmat: np.ndarray,
    q_idx: int,
    query_ids: np.ndarray,
    query_camids: np.ndarray,
    gallery_ids: np.ndarray,
    gallery_camids: np.ndarray,
) -> list[int]:
    order = np.argsort(distmat[q_idx])
    q_pid = query_ids[q_idx]
    q_camid = query_camids[q_idx]
    ranking: list[int] = []
    for g_idx in order:
        same_camera = gallery_ids[g_idx] == q_pid and gallery_camids[g_idx] == q_camid
        if same_camera:
            continue
        ranking.append(int(g_idx))
    return ranking


def first_correct_rank(ranking: list[int], query_id: int, gallery_ids: np.ndarray) -> int:
    for rank, g_idx in enumerate(ranking, start=1):
        if gallery_ids[g_idx] == query_id:
            return rank
    return len(ranking) + 1


def build_query_cases(
    baseline_distmat: np.ndarray,
    target_distmat: np.ndarray,
    query_ids: np.ndarray,
    query_camids: np.ndarray,
    query_paths: list[str],
    gallery_ids: np.ndarray,
    gallery_camids: np.ndarray,
) -> tuple[list[QueryCase], list[list[int]], list[list[int]]]:
    cases: list[QueryCase] = []
    baseline_rankings: list[list[int]] = []
    target_rankings: list[list[int]] = []

    for q_idx, query_path in enumerate(query_paths):
        baseline_ranking = filtered_ranking(baseline_distmat, q_idx, query_ids, query_camids, gallery_ids, gallery_camids)
        target_ranking = filtered_ranking(target_distmat, q_idx, query_ids, query_camids, gallery_ids, gallery_camids)
        baseline_rankings.append(baseline_ranking)
        target_rankings.append(target_ranking)

        baseline_rank = first_correct_rank(baseline_ranking, int(query_ids[q_idx]), gallery_ids)
        target_rank = first_correct_rank(target_ranking, int(query_ids[q_idx]), gallery_ids)
        cases.append(
            QueryCase(
                query_index=q_idx,
                query_path=query_path,
                query_id=int(query_ids[q_idx]),
                query_camera=int(query_camids[q_idx]),
                baseline_first_correct_rank=baseline_rank,
                target_first_correct_rank=target_rank,
                baseline_rank1_correct=baseline_rank == 1,
                target_rank1_correct=target_rank == 1,
            )
        )
    return cases, baseline_rankings, target_rankings


def pick_cases(cases: list[QueryCase], num_cases: int) -> dict[str, list[QueryCase]]:
    recovered = [
        case
        for case in cases
        if not case.baseline_rank1_correct and case.target_rank1_correct
    ]
    recovered.sort(key=lambda c: (-c.baseline_first_correct_rank, c.query_index))

    improved = [
        case
        for case in cases
        if case.target_first_correct_rank < case.baseline_first_correct_rank
        and not (not case.baseline_rank1_correct and case.target_rank1_correct)
    ]
    improved.sort(key=lambda c: (c.target_first_correct_rank, -c.baseline_first_correct_rank, c.query_index))

    failures = [case for case in cases if not case.target_rank1_correct]
    failures.sort(key=lambda c: (-c.target_first_correct_rank, c.baseline_rank1_correct, c.query_index))

    return {
        "recovered": recovered[:num_cases],
        "improved": improved[:num_cases],
        "target_failures": failures[:num_cases],
    }


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_text_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    draw.text(xy, text, fill=(30, 30, 30), font=font)


def image_cell(path: str, width: int, height: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((width, height))


def draw_case_grid(
    output_path: Path,
    case: QueryCase,
    baseline_ranking: list[int],
    target_ranking: list[int],
    gallery_paths: list[str],
    gallery_ids: np.ndarray,
    topk: int,
    target_name: str,
) -> None:
    cell_width, cell_height = 128, 256
    margin = 10
    label_height = 32
    title_height = 54
    row_gap = 20
    columns = topk + 1
    canvas_width = margin + columns * (cell_width + margin)
    canvas_height = title_height + 2 * (cell_height + label_height) + row_gap + margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(248, 248, 244))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(18)
    label_font = load_font(14)

    title = (
        f"query #{case.query_index} | pid {case.query_id} | "
        f"baseline first correct rank {case.baseline_first_correct_rank} | "
        f"{target_name} first correct rank {case.target_first_correct_rank}"
    )
    draw_text_box(draw, (margin, 12), title, title_font)

    def draw_row(row_name: str, ranking: list[int], y: int) -> None:
        paths = [case.query_path] + [gallery_paths[g_idx] for g_idx in ranking[:topk]]
        for idx, path in enumerate(paths):
            x = margin + idx * (cell_width + margin)
            canvas.paste(image_cell(path, cell_width, cell_height), (x, y))
            if idx == 0:
                border = (40, 80, 220)
                label = f"{row_name}: query"
            else:
                g_idx = ranking[idx - 1]
                matched = gallery_ids[g_idx] == case.query_id
                border = (20, 160, 70) if matched else (210, 45, 45)
                label = f"{row_name}: top{idx}"
            draw.rectangle([x, y, x + cell_width, y + cell_height], outline=border, width=4)
            draw_text_box(draw, (x, y + cell_height + 4), label, label_font)

    baseline_y = title_height
    target_y = title_height + cell_height + label_height + row_gap
    draw_row("baseline", baseline_ranking, baseline_y)
    draw_row(target_name, target_ranking, target_y)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    configure_torch_runtime(device, cudnn_benchmark=True, allow_tf32=True)

    config = ExperimentConfig(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        use_amp=args.use_amp,
        distance_metric=args.distance_metric,
        channels_last=args.channels_last,
        eval_gpu_distance=args.eval_gpu_distance,
        eval_gpu_distance_max_elements=args.eval_gpu_distance_max_elements,
    )
    _, query_loader, gallery_loader, bundle = build_dataloaders(config)

    baseline_model = load_model_from_checkpoint(args.baseline_checkpoint, config, bundle.num_train_ids, device)
    target_model = load_model_from_checkpoint(args.target_checkpoint, config, bundle.num_train_ids, device)

    baseline_query_features, query_ids, query_camids, query_paths = extract_features(
        baseline_model,
        query_loader,
        device,
        use_amp=args.use_amp,
        channels_last=args.channels_last,
        keep_on_device=args.eval_gpu_distance and device.type == "cuda",
    )
    baseline_gallery_features, gallery_ids, gallery_camids, gallery_paths = extract_features(
        baseline_model,
        gallery_loader,
        device,
        use_amp=args.use_amp,
        channels_last=args.channels_last,
        keep_on_device=args.eval_gpu_distance and device.type == "cuda",
    )
    target_query_features, target_query_ids, target_query_camids, target_query_paths = extract_features(
        target_model,
        query_loader,
        device,
        use_amp=args.use_amp,
        channels_last=args.channels_last,
        keep_on_device=args.eval_gpu_distance and device.type == "cuda",
    )
    target_gallery_features, target_gallery_ids, target_gallery_camids, target_gallery_paths = extract_features(
        target_model,
        gallery_loader,
        device,
        use_amp=args.use_amp,
        channels_last=args.channels_last,
        keep_on_device=args.eval_gpu_distance and device.type == "cuda",
    )

    if not (
        np.array_equal(query_ids, target_query_ids)
        and np.array_equal(query_camids, target_query_camids)
        and query_paths == target_query_paths
        and np.array_equal(gallery_ids, target_gallery_ids)
        and np.array_equal(gallery_camids, target_gallery_camids)
        and gallery_paths == target_gallery_paths
    ):
        raise RuntimeError("Baseline and target dataloaders produced different query/gallery order.")

    distance_elements = len(query_ids) * len(gallery_ids)
    use_gpu_distance = (
        args.eval_gpu_distance
        and device.type == "cuda"
        and distance_elements <= args.eval_gpu_distance_max_elements
    )
    if use_gpu_distance:
        baseline_distmat = compute_distance_matrix(
            baseline_query_features, baseline_gallery_features, metric=args.distance_metric
        ).cpu().numpy()
        target_distmat = compute_distance_matrix(
            target_query_features, target_gallery_features, metric=args.distance_metric
        ).cpu().numpy()
    else:
        baseline_distmat = compute_distance_matrix(
            baseline_query_features.cpu(), baseline_gallery_features.cpu(), metric=args.distance_metric
        ).numpy()
        target_distmat = compute_distance_matrix(
            target_query_features.cpu(), target_gallery_features.cpu(), metric=args.distance_metric
        ).numpy()

    cases, baseline_rankings, target_rankings = build_query_cases(
        baseline_distmat,
        target_distmat,
        query_ids,
        query_camids,
        query_paths,
        gallery_ids,
        gallery_camids,
    )
    selected = pick_cases(cases, args.num_cases)

    output_root = Path(args.output_dir)
    manifest = {
        "dataset_name": args.dataset_name,
        "baseline_checkpoint": args.baseline_checkpoint,
        "target_checkpoint": args.target_checkpoint,
        "target_name": args.target_name,
        "topk": args.topk,
        "num_queries": len(cases),
        "counts": {
            "baseline_rank1_correct": sum(case.baseline_rank1_correct for case in cases),
            "target_rank1_correct": sum(case.target_rank1_correct for case in cases),
            "recovered_rank1": len([case for case in cases if not case.baseline_rank1_correct and case.target_rank1_correct]),
            "regressed_rank1": len([case for case in cases if case.baseline_rank1_correct and not case.target_rank1_correct]),
            "both_rank1_correct": len([case for case in cases if case.baseline_rank1_correct and case.target_rank1_correct]),
            "both_rank1_wrong": len([case for case in cases if not case.baseline_rank1_correct and not case.target_rank1_correct]),
        },
        "cases": {},
    }

    for category, category_cases in selected.items():
        manifest["cases"][category] = []
        for offset, case in enumerate(category_cases):
            filename = f"{category}_{offset:02d}_query_{case.query_index:04d}.jpg"
            draw_case_grid(
                output_root / filename,
                case,
                baseline_rankings[case.query_index],
                target_rankings[case.query_index],
                gallery_paths,
                gallery_ids,
                args.topk,
                args.target_name,
            )
            payload = asdict(case)
            payload["visualization"] = filename
            manifest["cases"][category].append(payload)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "case_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Saved case analysis to {output_root}")


if __name__ == "__main__":
    main()
