from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reid.data import dataset_summary, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a pedestrian re-ID dataset")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--dataset-name", default="market1501")
    args = parser.parse_args()

    bundle = load_dataset(args.dataset_name, args.data_root)
    print(json.dumps(dataset_summary(bundle), indent=2))


if __name__ == "__main__":
    main()
