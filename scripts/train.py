from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reid.config import parse_args
from reid.engine import run_training


def main() -> None:
    config = parse_args(description="Train a pedestrian re-identification model")
    metrics = run_training(config)
    print("Training finished.")
    print(metrics)


if __name__ == "__main__":
    main()
