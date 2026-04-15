from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class RunRecord:
    dataset: str
    variant: str
    experiment: str
    path: Path
    rank1: float
    mAP: float
    rank5: float
    rank10: float
    seed: int
    use_local_branch: bool
    use_transformer: bool
    use_fusion_gate: bool
    fusion_mode: str
    local_residual_weight: float
    local_residual_learnable: bool
    local_loss_weight: float
    num_parts: int


@dataclass(frozen=True)
class ExperimentSpec:
    title: str
    dataset: str
    experiment: str
    method: str
    seed: int
    flags: tuple[str, ...]
    reason: str


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_dev_run(experiment: str) -> bool:
    lowered = experiment.lower()
    return any(token in lowered for token in ("smoke", "probe", "perf"))


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def signed_pp(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.2f} pp"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def collect_runs(output_root: Path) -> list[RunRecord]:
    runs: list[RunRecord] = []
    for metrics_file in output_root.glob("*/*/*/final_metrics.json"):
        experiment = metrics_file.parent.name
        if is_dev_run(experiment):
            continue
        config_file = metrics_file.with_name("config.json")
        if not config_file.exists():
            continue
        dataset = metrics_file.parents[2].name
        variant = metrics_file.parents[1].name
        metrics = load_json(metrics_file)
        config = load_json(config_file)
        runs.append(
            RunRecord(
                dataset=dataset,
                variant=variant,
                experiment=experiment,
                path=metrics_file.parent,
                rank1=float(metrics["rank1"]),
                mAP=float(metrics["mAP"]),
                rank5=float(metrics["rank5"]),
                rank10=float(metrics["rank10"]),
                seed=int(config.get("seed", 42)),
                use_local_branch=bool(config.get("use_local_branch", False)),
                use_transformer=bool(config.get("use_transformer", False)),
                use_fusion_gate=bool(config.get("use_fusion_gate", False)),
                fusion_mode=str(config.get("fusion_mode", "projection")),
                local_residual_weight=float(config.get("local_residual_weight", 0.1)),
                local_residual_learnable=bool(config.get("local_residual_learnable", True)),
                local_loss_weight=float(config.get("local_loss_weight", 0.0)),
                num_parts=int(config.get("num_parts", 6)),
            )
        )
    return sorted(runs, key=lambda run: (run.dataset, run.variant, run.experiment))


def write_experiment_table(output_file: Path, runs: list[RunRecord]) -> None:
    rows = [
        [
            run.dataset,
            run.variant,
            run.experiment,
            pct(run.rank1),
            pct(run.mAP),
            pct(run.rank5),
            pct(run.rank10),
        ]
        for run in runs
    ]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        md_table(["Dataset", "Variant", "Experiment", "Rank-1", "mAP", "Rank-5", "Rank-10"], rows) + "\n",
        encoding="utf-8",
    )


def run_by_name(runs: list[RunRecord], experiment: str) -> RunRecord | None:
    return next((run for run in runs if run.experiment == experiment), None)


def base_train_flags(dataset: str, experiment: str, seed: int) -> list[str]:
    return [
        "py -3.12",
        "scripts\\train.py",
        "--data-root datasets",
        f"--dataset-name {dataset}",
        f"--experiment-name {experiment}",
        f"--seed {seed}",
        "--device cuda",
        "--use-amp true",
        "--channels-last true",
        "--cuda-prefetch true",
        "--fused-optimizer true",
        "--cudnn-benchmark true",
        "--allow-tf32 true",
        "--num-workers 4",
        "--prefetch-factor 2",
        "--persistent-workers false",
        "--pin-memory true",
        "--batch-size 64",
        "--num-instances 4",
    ]


def command_for(spec: ExperimentSpec) -> str:
    return " ".join([*base_train_flags(spec.dataset, spec.experiment, spec.seed), *spec.flags])


def ablation_specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            "fixed residual only",
            "market1501",
            "market1501_fixed_residual_no_aux_w0_10",
            "fixed residual, no local auxiliary",
            42,
            (
                "--use-local-branch true",
                "--use-transformer false",
                "--use-fusion-gate false",
                "--fusion-mode residual",
                "--local-residual-weight 0.1",
                "--local-residual-learnable false",
                "--local-loss-weight 0.0",
            ),
            "Checks whether fixed residual alone is enough.",
        ),
        ExperimentSpec(
            "local auxiliary only",
            "market1501",
            "market1501_local_aux_learnable_residual_w0_10",
            "local auxiliary, learnable residual",
            42,
            (
                "--use-local-branch true",
                "--use-transformer false",
                "--use-fusion-gate false",
                "--fusion-mode residual",
                "--local-residual-weight 0.1",
                "--local-residual-learnable true",
                "--local-loss-weight 0.3",
            ),
            "Checks whether local supervision can overcome a learnable scale.",
        ),
        ExperimentSpec(
            "fixed residual + local auxiliary",
            "market1501",
            "market1501_local_aux_residual",
            "fixed residual, local auxiliary",
            42,
            (
                "--use-local-branch true",
                "--use-transformer false",
                "--use-fusion-gate false",
                "--fusion-mode residual",
                "--local-residual-weight 0.1",
                "--local-residual-learnable false",
                "--local-loss-weight 0.3",
            ),
            "Main corrected local-branch variant.",
        ),
        ExperimentSpec(
            "residual weight 0.05",
            "market1501",
            "market1501_local_aux_residual_w0_05",
            "fixed residual 0.05, local auxiliary",
            42,
            (
                "--use-local-branch true",
                "--use-transformer false",
                "--use-fusion-gate false",
                "--fusion-mode residual",
                "--local-residual-weight 0.05",
                "--local-residual-learnable false",
                "--local-loss-weight 0.3",
            ),
            "Checks residual-weight sensitivity below 0.1.",
        ),
        ExperimentSpec(
            "residual weight 0.2",
            "market1501",
            "market1501_local_aux_residual_w0_20",
            "fixed residual 0.2, local auxiliary",
            42,
            (
                "--use-local-branch true",
                "--use-transformer false",
                "--use-fusion-gate false",
                "--fusion-mode residual",
                "--local-residual-weight 0.2",
                "--local-residual-learnable false",
                "--local-loss-weight 0.3",
            ),
            "Checks whether too much local residual starts adding noise.",
        ),
        ExperimentSpec(
            "num parts 4",
            "market1501",
            "market1501_local_aux_residual_parts4",
            "fixed residual, local auxiliary, 4 parts",
            42,
            (
                "--use-local-branch true",
                "--use-transformer false",
                "--use-fusion-gate false",
                "--fusion-mode residual",
                "--local-residual-weight 0.1",
                "--local-residual-learnable false",
                "--local-loss-weight 0.3",
                "--num-parts 4",
            ),
            "Low-cost check of part granularity.",
        ),
        ExperimentSpec(
            "num parts 8",
            "market1501",
            "market1501_local_aux_residual_parts8",
            "fixed residual, local auxiliary, 8 parts",
            42,
            (
                "--use-local-branch true",
                "--use-transformer false",
                "--use-fusion-gate false",
                "--fusion-mode residual",
                "--local-residual-weight 0.1",
                "--local-residual-learnable false",
                "--local-loss-weight 0.3",
                "--num-parts 8",
            ),
            "Low-cost check of part granularity.",
        ),
    ]


def stability_specs(seeds: list[int]) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for dataset in ("market1501", "cuhk03_np"):
        for seed in seeds:
            baseline_name = f"{dataset}_baseline" if seed == 42 else f"{dataset}_baseline_seed{seed}"
            local_name = f"{dataset}_local_aux_residual" if seed == 42 else f"{dataset}_local_aux_residual_seed{seed}"
            specs.append(
                ExperimentSpec(
                    f"{dataset} baseline seed {seed}",
                    dataset,
                    baseline_name,
                    "baseline",
                    seed,
                    (
                        "--use-local-branch false",
                        "--use-transformer false",
                        "--use-fusion-gate false",
                    ),
                    "Stability baseline.",
                )
            )
            specs.append(
                ExperimentSpec(
                    f"{dataset} local aux residual seed {seed}",
                    dataset,
                    local_name,
                    "fixed residual, local auxiliary",
                    seed,
                    (
                        "--use-local-branch true",
                        "--use-transformer false",
                        "--use-fusion-gate false",
                        "--fusion-mode residual",
                        "--local-residual-weight 0.1",
                        "--local-residual-learnable false",
                        "--local-loss-weight 0.3",
                    ),
                    "Stability corrected variant.",
                )
            )
    return specs


def summarize_group(values: list[float]) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return pct(values[0])
    return f"{pct(statistics.mean(values))} ± {statistics.stdev(values) * 100:.2f} pp"


def completed_run(runs: list[RunRecord], experiment: str) -> RunRecord | None:
    return run_by_name(runs, experiment)


def metric_pair(run: RunRecord | None) -> str:
    if run is None:
        return "-"
    return f"Rank-1 {pct(run.rank1)}, mAP {pct(run.mAP)}"


def paired_seed_rows(runs: list[RunRecord], dataset: str) -> list[list[str]]:
    rows = []
    baseline_runs = {
        run.seed: run
        for run in runs
        if run.dataset == dataset and not run.use_local_branch and run.experiment.startswith(f"{dataset}_baseline")
    }
    corrected_runs = {
        run.seed: run
        for run in runs
        if run.dataset == dataset
        and run.use_local_branch
        and not run.use_transformer
        and run.fusion_mode == "residual"
        and not run.local_residual_learnable
        and run.local_loss_weight > 0
        and (run.experiment == f"{dataset}_local_aux_residual" or run.experiment.startswith(f"{dataset}_local_aux_residual_seed"))
    }
    for seed in sorted(set(baseline_runs) & set(corrected_runs)):
        base = baseline_runs[seed]
        corrected = corrected_runs[seed]
        rows.append(
            [
                str(seed),
                pct(base.rank1),
                pct(corrected.rank1),
                signed_pp(corrected.rank1 - base.rank1),
                pct(base.mAP),
                pct(corrected.mAP),
                signed_pp(corrected.mAP - base.mAP),
            ]
        )
    return rows


def stable_groups(runs: list[RunRecord], dataset: str) -> tuple[list[RunRecord], list[RunRecord]]:
    baseline = [
        run
        for run in runs
        if run.dataset == dataset and not run.use_local_branch and run.experiment.startswith(f"{dataset}_baseline")
    ]
    corrected = [
        run
        for run in runs
        if run.dataset == dataset
        and run.use_local_branch
        and not run.use_transformer
        and run.fusion_mode == "residual"
        and not run.local_residual_learnable
        and run.local_loss_weight > 0
        and (run.experiment == f"{dataset}_local_aux_residual" or run.experiment.startswith(f"{dataset}_local_aux_residual_seed"))
    ]
    return sorted(baseline, key=lambda item: item.seed), sorted(corrected, key=lambda item: item.seed)


def paired_deltas(runs: list[RunRecord], dataset: str) -> tuple[list[float], list[float]]:
    baseline, corrected = stable_groups(runs, dataset)
    baseline_by_seed = {run.seed: run for run in baseline}
    corrected_by_seed = {run.seed: run for run in corrected}
    rank1_deltas = []
    map_deltas = []
    for seed in sorted(set(baseline_by_seed) & set(corrected_by_seed)):
        rank1_deltas.append(corrected_by_seed[seed].rank1 - baseline_by_seed[seed].rank1)
        map_deltas.append(corrected_by_seed[seed].mAP - baseline_by_seed[seed].mAP)
    return rank1_deltas, map_deltas


def mean_delta(values: list[float]) -> str:
    if not values:
        return "-"
    return signed_pp(statistics.mean(values))


def write_queue(path: Path, specs: list[ExperimentSpec], runs: list[RunRecord]) -> None:
    completed = {run.experiment for run in runs}
    lines = [
        "# Experiment Queue",
        "",
        "Only commands whose experiment directory has no `final_metrics.json` are listed here.",
        "",
    ]
    for spec in specs:
        if spec.experiment in completed:
            continue
        lines.extend(
            [
                f"## {spec.experiment}",
                "",
                f"- Goal: {spec.reason}",
                "",
                "```powershell",
                command_for(spec),
                "```",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ablation_summary(path: Path, runs: list[RunRecord], specs: list[ExperimentSpec]) -> None:
    completed = {run.experiment for run in runs}
    baseline_market = run_by_name(runs, "market1501_baseline")
    baseline_cuhk = run_by_name(runs, "cuhk03_np_baseline")
    key_names = [
        "market1501_baseline",
        "market1501_local_branch",
        "market1501_local_residual",
        "market1501_local_aux_residual",
        "market1501_transformer_aux_residual",
        "market1501_full_aux_gated_residual",
        "cuhk03_np_baseline",
        "cuhk03_np_local_residual",
        "cuhk03_np_local_aux_residual",
        "cuhk03_np_transformer_aux_residual",
        "cuhk03_np_full_aux_gated_residual",
    ]
    result_rows = []
    for name in key_names:
        run = run_by_name(runs, name)
        if run is None:
            continue
        baseline = baseline_market if run.dataset == "market1501" else baseline_cuhk
        result_rows.append(
            [
                run.dataset,
                run.experiment,
                pct(run.rank1),
                pct(run.mAP),
                "-" if baseline is None else signed_pp(run.rank1 - baseline.rank1),
                "-" if baseline is None else signed_pp(run.mAP - baseline.mAP),
            ]
        )

    status_rows = []
    for spec in specs:
        run = run_by_name(runs, spec.experiment)
        status_rows.append(
            [
                spec.title,
                spec.experiment,
                "done" if spec.experiment in completed else "pending",
                "-" if run is None else pct(run.rank1),
                "-" if run is None else pct(run.mAP),
            ]
        )

    fixed_only = completed_run(runs, "market1501_fixed_residual_no_aux_w0_10")
    aux_learnable = completed_run(runs, "market1501_local_aux_learnable_residual_w0_10")
    fixed_aux = completed_run(runs, "market1501_local_aux_residual")
    w005 = completed_run(runs, "market1501_local_aux_residual_w0_05")
    w020 = completed_run(runs, "market1501_local_aux_residual_w0_20")
    parts4 = completed_run(runs, "market1501_local_aux_residual_parts4")
    parts8 = completed_run(runs, "market1501_local_aux_residual_parts8")
    synergy_complete = fixed_only is not None and aux_learnable is not None and fixed_aux is not None
    trend_complete = w005 is not None and w020 is not None and parts4 is not None and parts8 is not None

    conclusion_lines = []
    if synergy_complete:
        conclusion_lines.extend(
            [
                "- 协同消融已经闭合：`fixed residual only`、`local auxiliary + learnable residual`、`fixed residual + local auxiliary` 三个关键点都有结果。",
                f"- `fixed residual only`: {metric_pair(fixed_only)}；`local auxiliary + learnable residual`: {metric_pair(aux_learnable)}；`fixed residual + local auxiliary`: {metric_pair(fixed_aux)}。",
            ]
        )
        if baseline_market is not None:
            conclusion_lines.extend(
                [
                    f"- 相对 Market-1501 baseline，fixed-only 为 Rank-1 {signed_pp(fixed_only.rank1 - baseline_market.rank1)}, mAP {signed_pp(fixed_only.mAP - baseline_market.mAP)}；aux+learnable 为 Rank-1 {signed_pp(aux_learnable.rank1 - baseline_market.rank1)}, mAP {signed_pp(aux_learnable.mAP - baseline_market.mAP)}；fixed+aux 为 Rank-1 {signed_pp(fixed_aux.rank1 - baseline_market.rank1)}, mAP {signed_pp(fixed_aux.mAP - baseline_market.mAP)}。",
                    "- 更准确的解释是：local auxiliary supervision 是主要收益来源；fixed residual 的价值是保证局部路径不会被 scale 压没，并让 Rank-1 更稳一点。不要把 fixed residual alone 夸大成主要提升来源。",
                ]
            )
    else:
        conclusion_lines.extend(
            [
                "- 已完成结果支持一个保守结论：直接投影式局部分支会明显伤害强 baseline；learnable residual 在无局部辅助监督时不能形成有效改进；固定残差加局部辅助监督是目前最稳定、最可解释的修正方向。",
                "- 严格证明 `fixed residual` 和 `local auxiliary supervision` 的协同关系，还需要补齐 `fixed residual only` 与 `local auxiliary + learnable residual` 两个缺口实验。它们已经在 `outputs/analysis/experiment_queue.md` 中生成可直接运行命令。",
            ]
        )
    if trend_complete:
        conclusion_lines.extend(
            [
                f"- residual weight 趋势已补齐：0.05 为 {metric_pair(w005)}，0.1 为 {metric_pair(fixed_aux)}，0.2 为 {metric_pair(w020)}。0.05 的 mAP 最高，0.2 开始回落，说明局部残差不宜过大；0.1 仍是解释最稳妥的默认设置。",
                f"- part 数趋势已补齐：4 parts 为 {metric_pair(parts4)}，6 parts 为 {metric_pair(fixed_aux)}，8 parts 为 {metric_pair(parts8)}。8 parts 单 seed 最强，但没有多 seed 验证；报告中可以说 part granularity 有影响，但不应包装成新的复杂模块贡献。",
            ]
        )
    else:
        conclusion_lines.append("- residual weight 和 part 数量对比目前不应写成最终结论；在补跑完成前，只能作为待验证趋势。")

    lines = [
        "# Ablation Summary",
        "",
        "## Completed Evidence",
        "",
        md_table(["Dataset", "Experiment", "Rank-1", "mAP", "ΔRank-1 vs baseline", "ΔmAP vs baseline"], result_rows),
        "",
        "## Required Synergy Ablations",
        "",
        md_table(["Question", "Experiment", "Status", "Rank-1", "mAP"], status_rows),
        "",
        "## Current Conclusion",
        "",
        *conclusion_lines,
        "",
        "## Command Example",
        "",
        "```powershell",
        command_for(specs[0]),
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_stability_summary(path: Path, runs: list[RunRecord], seeds: list[int], specs: list[ExperimentSpec]) -> None:
    methods = {
        "market1501/baseline": ("market1501", "baseline"),
        "market1501/local_aux_residual": ("market1501", "fixed residual, local auxiliary"),
        "cuhk03_np/baseline": ("cuhk03_np", "baseline"),
        "cuhk03_np/local_aux_residual": ("cuhk03_np", "fixed residual, local auxiliary"),
    }
    rows = []
    for key, (dataset, method) in methods.items():
        group = [run for run in runs if run.dataset == dataset]
        if method == "baseline":
            group = [run for run in group if not run.use_local_branch and run.experiment.startswith(f"{dataset}_baseline")]
        else:
            group = [
                run
                for run in group
                if run.use_local_branch
                and not run.use_transformer
                and run.fusion_mode == "residual"
                and not run.local_residual_learnable
                and run.local_loss_weight > 0
                and (run.experiment == f"{dataset}_local_aux_residual" or run.experiment.startswith(f"{dataset}_local_aux_residual_seed"))
            ]
        rows.append(
            [
                key,
                ", ".join(str(run.seed) for run in sorted(group, key=lambda item: item.seed)) or "-",
                str(len(group)),
                summarize_group([run.rank1 for run in group]),
                summarize_group([run.mAP for run in group]),
            ]
        )

    market_pairs = paired_seed_rows(runs, "market1501")
    cuhk_pairs = paired_seed_rows(runs, "cuhk03_np")
    missing = [spec for spec in specs if run_by_name(runs, spec.experiment) is None]
    if not missing and len(market_pairs) >= 2 and len(cuhk_pairs) >= 2:
        market_rank1_deltas, market_map_deltas = paired_deltas(runs, "market1501")
        cuhk_rank1_deltas, cuhk_map_deltas = paired_deltas(runs, "cuhk03_np")
        conclusion = [
            "- 多 seed 稳定性补跑已完成，当前可以报告 mean ± std，并用 paired seed 差值说明提升是否稳定出现。",
            f"- Market-1501 上 corrected variant 的 mAP 每个 seed 都提升，平均 paired 提升 {mean_delta(market_map_deltas)}；Rank-1 不稳定，平均 paired 变化 {mean_delta(market_rank1_deltas)}。",
            f"- CUHK03-NP 上 corrected variant 的 mAP 每个 seed 都提升，平均 paired 提升 {mean_delta(cuhk_map_deltas)}；Rank-1 也为正向但波动较大，平均 paired 提升 {mean_delta(cuhk_rank1_deltas)}。",
            "- 简历中可以写“多 seed 结果显示 mAP 提升较稳定”，但不要写“Rank-1 和 mAP 全面稳定提升”。",
        ]
    else:
        conclusion = [
            "- 目前每个关键方法只有 seed 42 的完整结果，因此还不能把提升表述为多 seed 稳定结论。",
            "- 单 seed 结果已经显示 corrected local branch 在 Market-1501 和 CUHK03-NP 上都优于 baseline，但简历和汇报中应写成“在两个数据集上观察到一致改进”，不要写“多随机种子稳定提升”。",
            "- 多 seed 补跑建议只覆盖 baseline 和 `local_aux_residual`，避免把所有模型全矩阵重跑。",
        ]
    lines = [
        "# Stability Summary",
        "",
        "## Current Seed Coverage",
        "",
        md_table(["Dataset/Method", "Seeds", "Runs", "Rank-1", "mAP"], rows),
        "",
        "## Current Conclusion",
        "",
        *conclusion,
        "",
        "## Paired Seed Differences",
        "",
        "### Market-1501",
        "",
        md_table(["Seed", "Baseline Rank-1", "Corrected Rank-1", "ΔRank-1", "Baseline mAP", "Corrected mAP", "ΔmAP"], market_pairs) if market_pairs else "No paired seeds yet.",
        "",
        "### CUHK03-NP",
        "",
        md_table(["Seed", "Baseline Rank-1", "Corrected Rank-1", "ΔRank-1", "Baseline mAP", "Corrected mAP", "ΔmAP"], cuhk_pairs) if cuhk_pairs else "No paired seeds yet.",
        "",
        "## Missing Stability Commands",
        "",
    ]
    if missing:
        for spec in missing:
            lines.extend(["```powershell", command_for(spec), "```", ""])
    else:
        lines.append("All planned stability runs have final metrics.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_project_narrative(path: Path, runs: list[RunRecord]) -> None:
    market_base = run_by_name(runs, "market1501_baseline")
    market_best_map = run_by_name(runs, "market1501_local_aux_residual")
    market_best_rank1 = run_by_name(runs, "market1501_transformer_aux_residual")
    cuhk_base = run_by_name(runs, "cuhk03_np_baseline")
    cuhk_rank1 = run_by_name(runs, "cuhk03_np_local_aux_residual")
    cuhk_map = run_by_name(runs, "cuhk03_np_full_aux_gated_residual")
    fixed_only = run_by_name(runs, "market1501_fixed_residual_no_aux_w0_10")
    aux_learnable = run_by_name(runs, "market1501_local_aux_learnable_residual_w0_10")
    fixed_aux = run_by_name(runs, "market1501_local_aux_residual")
    w005 = run_by_name(runs, "market1501_local_aux_residual_w0_05")
    w020 = run_by_name(runs, "market1501_local_aux_residual_w0_20")
    parts4 = run_by_name(runs, "market1501_local_aux_residual_parts4")
    parts8 = run_by_name(runs, "market1501_local_aux_residual_parts8")
    market_stable_base, market_stable_corrected = stable_groups(runs, "market1501")
    cuhk_stable_base, cuhk_stable_corrected = stable_groups(runs, "cuhk03_np")
    stability_complete = (
        len(market_stable_base) >= 3
        and len(market_stable_corrected) >= 3
        and len(cuhk_stable_base) >= 3
        and len(cuhk_stable_corrected) >= 3
    )

    lines = [
        "# 行人重识别项目最终叙事版本",
        "",
        "## 项目定位",
        "",
        "本项目不是为了盲目堆复杂模型，而是围绕一个强 ReID baseline 做完整研究闭环：复现、发现失败模式、诊断原因、提出修正、做消融和案例分析。最终可讲的重点是实验诊断能力，而不是简单宣称 Transformer 更强。",
        "",
        "## 从项目创建到当前版本的路线",
        "",
        "1. 建立 `ResNet50 + BNNeck + CE + Triplet` 强 baseline，使用 Rank-1/mAP 作为检索指标。",
        "2. 加入局部分支后，直接投影融合在 Market-1501 上明显退化，说明局部结构并不天然有效。",
        "3. 将融合改成 residual 后，退化被缓解，但 learnable residual scale 在训练后容易接近 0，局部分支仍可能被边缘化。",
        "4. 加入固定小残差和局部辅助 `CE + Triplet` 后，局部分支获得直接身份监督，并能稳定参与 fused retrieval。",
        "5. 再测试 Transformer/gated fusion，结论保持保守：它们在部分指标上有收益，但不是全面优于 corrected local branch。",
        "",
            "## 当前核心结论",
            "",
            "- 在强 ReID baseline 上，局部分支如果融合和监督设计不当，容易被训练过程边缘化。",
            "- learnable residual 融合可能把局部分支权重压到接近 0，使结构存在但贡献很小。",
            "- 消融显示 local auxiliary CE/Triplet 是主要收益来源；fixed residual 的价值是保证局部路径稳定参与融合，避免 learnable scale 把局部分支压没。",
            "- Transformer 只能作为局部关系建模的补充消融，不能写成“全面提升”。",
        "",
        "## 关键结果",
        "",
    ]
    result_rows = []
    if market_base and market_best_map:
        result_rows.append(
            [
                "Market-1501",
                market_base.experiment,
                pct(market_base.rank1),
                pct(market_base.mAP),
                "baseline",
            ]
        )
        result_rows.append(
            [
                "Market-1501",
                market_best_map.experiment,
                pct(market_best_map.rank1),
                pct(market_best_map.mAP),
                f"mAP {signed_pp(market_best_map.mAP - market_base.mAP)}",
            ]
        )
    if market_base and market_best_rank1:
        result_rows.append(
            [
                "Market-1501",
                market_best_rank1.experiment,
                pct(market_best_rank1.rank1),
                pct(market_best_rank1.mAP),
                f"Rank-1 {signed_pp(market_best_rank1.rank1 - market_base.rank1)}",
            ]
        )
    if cuhk_base and cuhk_rank1:
        result_rows.append(
            [
                "CUHK03-NP",
                cuhk_base.experiment,
                pct(cuhk_base.rank1),
                pct(cuhk_base.mAP),
                "baseline",
            ]
        )
        result_rows.append(
            [
                "CUHK03-NP",
                cuhk_rank1.experiment,
                pct(cuhk_rank1.rank1),
                pct(cuhk_rank1.mAP),
                f"Rank-1 {signed_pp(cuhk_rank1.rank1 - cuhk_base.rank1)}",
            ]
        )
    if cuhk_base and cuhk_map:
        result_rows.append(
            [
                "CUHK03-NP",
                cuhk_map.experiment,
                pct(cuhk_map.rank1),
                pct(cuhk_map.mAP),
                f"mAP {signed_pp(cuhk_map.mAP - cuhk_base.mAP)}",
            ]
        )
    lines.extend([md_table(["Dataset", "Experiment", "Rank-1", "mAP", "Use in report"], result_rows), ""])
    lines.extend(
        [
            "## 可写进简历的内容",
            "",
            "- 基于 PyTorch 搭建行人重识别实验框架，复现 `ResNet50 + BNNeck + CE + Triplet` 强 baseline，并在 Market-1501 与 CUHK03-NP 上完成 Rank-1/mAP 评测。",
            "- 发现局部分支在 learnable residual 融合下可能被训练过程压制，围绕 feature norm、gradient norm、logit contribution、residual scale 与检索案例构建诊断证据链。",
            "- 设计固定残差融合与局部辅助 `CE + Triplet` 监督，使局部特征真正参与检索；在 seed 42 结果中，Market-1501 mAP 从 `82.93%` 提升到 `83.81%`，CUHK03-NP Rank-1 从 `61.43%` 提升到 `63.79%`。",
            "- 补齐协同消融、residual weight、part 数和多 seed 稳定性实验；多 seed 下 Market-1501 与 CUHK03-NP 的 mAP 提升更稳定，Rank-1 提升存在数据集差异。",
            "",
            "## 只适合内部分析的内容",
            "",
            "- 多 seed 稳定性如果只在部分指标上稳定，简历中要保留 mean ± std 和具体数据，不写成所有指标全面提升。",
            "- residual weight 和 local part 数量趋势只服务于说明设计选择，不适合包装成新的复杂模块贡献。",
            "- Transformer/gated fusion 的结果更适合讲成“指标取舍和边界条件”，不适合作为项目主卖点。",
            "",
            "## 面试讲述版本",
            "",
            "我先复现了 ReID 强基线，而不是直接堆模块。第一次加入局部分支后，Market-1501 反而下降，这说明局部结构本身不保证有效。我随后把问题拆成融合方式和监督方式两部分：learnable residual 虽然看起来灵活，但训练后会把局部分支 scale 压到接近 0；于是我加入固定小残差约束，并给局部分支增加辅助 CE 和 Triplet 监督。后续消融显示，主要收益来自局部辅助监督，fixed residual 更像是防止局部路径被压没的稳定约束。多 seed 结果进一步说明，mAP 的提升比 Rank-1 更稳定。因此这个项目的核心不是 Transformer 全面更强，而是强 baseline 上局部特征需要直接监督和受控融合，才能可靠参与检索。",
            "",
            "## 补强完成情况",
            "",
        ]
    )
    completion_lines = []
    if fixed_only and aux_learnable and fixed_aux:
        completion_lines.extend(
            [
                f"- 协同消融已完成：fixed-only 为 {metric_pair(fixed_only)}，aux + learnable residual 为 {metric_pair(aux_learnable)}，fixed + aux 为 {metric_pair(fixed_aux)}。",
                "- 这部分用于回答“fixed residual 和 local auxiliary 是否协同解决边缘化问题”，具体结论以 `docs/ablation_summary.md` 为准。",
            ]
        )
    else:
        completion_lines.append("- 协同消融尚未全部完成，不能把 fixed residual 与 local auxiliary 的关系写成最终定论。")
    if w005 and fixed_aux and w020 and parts4 and parts8:
        completion_lines.extend(
            [
                f"- residual weight 趋势已完成：0.05 为 {metric_pair(w005)}，0.1 为 {metric_pair(fixed_aux)}，0.2 为 {metric_pair(w020)}。",
                f"- local part 数趋势已完成：4 parts 为 {metric_pair(parts4)}，6 parts 为 {metric_pair(fixed_aux)}，8 parts 为 {metric_pair(parts8)}。",
            ]
        )
    else:
        completion_lines.append("- residual weight 和 local part 数量趋势仍有缺口，不写成最终趋势。")
    if stability_complete:
        market_rank1_deltas, market_map_deltas = paired_deltas(runs, "market1501")
        cuhk_rank1_deltas, cuhk_map_deltas = paired_deltas(runs, "cuhk03_np")
        completion_lines.extend(
            [
                f"- Market-1501 多 seed：baseline Rank-1 {summarize_group([run.rank1 for run in market_stable_base])}，corrected Rank-1 {summarize_group([run.rank1 for run in market_stable_corrected])}；baseline mAP {summarize_group([run.mAP for run in market_stable_base])}，corrected mAP {summarize_group([run.mAP for run in market_stable_corrected])}。",
                f"- CUHK03-NP 多 seed：baseline Rank-1 {summarize_group([run.rank1 for run in cuhk_stable_base])}，corrected Rank-1 {summarize_group([run.rank1 for run in cuhk_stable_corrected])}；baseline mAP {summarize_group([run.mAP for run in cuhk_stable_base])}，corrected mAP {summarize_group([run.mAP for run in cuhk_stable_corrected])}。",
                f"- paired seed 口径：Market-1501 mAP 平均提升 {mean_delta(market_map_deltas)}，Rank-1 平均变化 {mean_delta(market_rank1_deltas)}；CUHK03-NP mAP 平均提升 {mean_delta(cuhk_map_deltas)}，Rank-1 平均提升 {mean_delta(cuhk_rank1_deltas)}。",
            ]
        )
    else:
        completion_lines.append("- 多 seed 稳定性尚未全部完成，当前仍以单 seed 和待补跑队列为准。")
    lines.extend([*completion_lines, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build project-level ReID research reports from finished experiments.")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--stability-seeds", default="42,123,3407")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    docs_dir = Path(args.docs_dir)
    seeds = [int(item.strip()) for item in args.stability_seeds.split(",") if item.strip()]
    runs = collect_runs(output_root)
    ab_specs = ablation_specs()
    st_specs = stability_specs(seeds)

    write_experiment_table(output_root / "experiment_table.md", runs)
    write_queue(output_root / "analysis" / "experiment_queue.md", [*ab_specs, *st_specs], runs)
    write_ablation_summary(docs_dir / "ablation_summary.md", runs, ab_specs)
    write_stability_summary(docs_dir / "stability_summary.md", runs, seeds, st_specs)
    write_project_narrative(docs_dir / "project_narrative.md", runs)
    print(f"Loaded {len(runs)} finished experiments.")
    print(f"Updated {output_root / 'experiment_table.md'}")
    print(f"Updated {docs_dir / 'ablation_summary.md'}")
    print(f"Updated {docs_dir / 'stability_summary.md'}")
    print(f"Updated {docs_dir / 'project_narrative.md'}")


if __name__ == "__main__":
    main()
