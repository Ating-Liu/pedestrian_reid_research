from __future__ import annotations

import sys
from pathlib import Path


def mark(message: str) -> None:
    print(message, flush=True)


mark("start")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
mark("path_inserted")

import reid.config  # noqa: E402

mark("reid.config_imported")

import torch  # noqa: E402

mark("torch_imported")

import torchvision  # noqa: E402

mark("torchvision_imported")

from reid.engine import run_training  # noqa: E402

mark(f"run_training_imported:{callable(run_training)}")
