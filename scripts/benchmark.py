from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


VARIANTS = {
    "baseline": {
        "use_local_branch": "false",
        "use_transformer": "false",
        "use_fusion_gate": "false",
        "fusion_mode": "projection",
        "local_residual_learnable": "true",
        "local_loss_weight": "0.0",
    },
    "local_branch": {
        "use_local_branch": "true",
        "use_transformer": "false",
        "use_fusion_gate": "false",
        "fusion_mode": "residual",
        "local_residual_learnable": "false",
        "local_loss_weight": "0.3",
    },
    "transformer_branch": {
        "use_local_branch": "true",
        "use_transformer": "true",
        "use_fusion_gate": "false",
        "fusion_mode": "residual",
        "local_residual_learnable": "false",
        "local_loss_weight": "0.3",
    },
    "full_model": {
        "use_local_branch": "true",
        "use_transformer": "true",
        "use_fusion_gate": "true",
        "fusion_mode": "gated_residual",
        "local_residual_learnable": "false",
        "local_loss_weight": "0.3",
    },
}


def build_command(python_executable: str, dataset: str, variant: str, data_root: str, output_dir: str, epochs: int) -> list[str]:
    flags = VARIANTS[variant]
    experiment_name = f"{dataset}_{variant}"
    return [
        python_executable,
        "scripts/train.py",
        "--dataset-name",
        dataset,
        "--data-root",
        data_root,
        "--output-dir",
        output_dir,
        "--experiment-name",
        experiment_name,
        "--epochs",
        str(epochs),
        "--use-local-branch",
        flags["use_local_branch"],
        "--use-transformer",
        flags["use_transformer"],
        "--use-fusion-gate",
        flags["use_fusion_gate"],
        "--fusion-mode",
        flags["fusion_mode"],
        "--local-residual-learnable",
        flags["local_residual_learnable"],
        "--local-loss-weight",
        flags["local_loss_weight"],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the recommended re-ID benchmark matrix")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--datasets", default="market1501,cuhk03_np,msmt17")
    parser.add_argument("--variants", default="baseline,local_branch,transformer_branch,full_model")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]

    for dataset in datasets:
        for variant in variants:
            command = build_command(args.python_executable, dataset, variant, args.data_root, args.output_dir, args.epochs)
            printable = " ".join(command)
            print(printable)
            if args.execute:
                subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
