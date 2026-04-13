from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def is_development_run(experiment_name: str) -> bool:
    lowered = experiment_name.lower()
    return any(token in lowered for token in ("smoke", "probe", "perf"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize re-ID experiment metrics into a Markdown table")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--datasets", default="market1501,cuhk03_np,msmt17")
    parser.add_argument("--variants", default="baseline,local_branch,transformer_branch,full_model")
    parser.add_argument("--output-file", default="outputs/experiment_table.md")
    parser.add_argument("--include-dev-runs", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]

    lines = [
        "| Dataset | Variant | Experiment | Rank-1 | mAP | Rank-5 | Rank-10 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset in datasets:
        for variant in variants:
            variant_dir = output_root / dataset / variant
            if not variant_dir.exists():
                continue
            for metrics_file in sorted(variant_dir.glob("*/final_metrics.json")):
                experiment = metrics_file.parent.name
                if is_development_run(experiment) and not args.include_dev_runs:
                    continue
                metrics = load_metrics(metrics_file)
                lines.append(
                    "| {dataset} | {variant} | {experiment} | {rank1:.2%} | {mAP:.2%} | {rank5:.2%} | {rank10:.2%} |".format(
                        dataset=dataset,
                        variant=variant,
                        experiment=experiment,
                        rank1=metrics["rank1"],
                        mAP=metrics["mAP"],
                        rank5=metrics["rank5"],
                        rank10=metrics["rank10"],
                    )
                )

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved summary table to {output_file}")


if __name__ == "__main__":
    main()
