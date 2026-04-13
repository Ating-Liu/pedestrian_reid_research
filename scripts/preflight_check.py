from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _status(label: str, ok: bool, detail: str) -> bool:
    tag = "OK" if ok else "FAIL"
    print(f"[{tag}] {label}: {detail}")
    return ok


def _warn(label: str, detail: str) -> None:
    print(f"[WARN] {label}: {detail}")


def check_runtime() -> bool:
    print("== Runtime ==")
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    python_ok = _status("Python executable", True, sys.executable)
    version_ok = _status("Python version", sys.version_info[:2] >= (3, 12), version)
    if sys.version_info[:2] != (3, 12):
        _warn("Python version", "Project was verified on Python 3.12; use `py -3.12` on Windows.")
    _status("Platform", True, platform.platform())
    return python_ok and version_ok


def check_torch() -> bool:
    print("\n== PyTorch ==")
    try:
        import torch
        import torchvision
    except Exception as exc:
        return _status("PyTorch import", False, str(exc))

    ok = True
    ok &= _status("torch", True, torch.__version__)
    ok &= _status("torchvision", True, torchvision.__version__)
    cuda_available = torch.cuda.is_available()
    ok &= _status("CUDA available", cuda_available, str(cuda_available))
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        ok &= _status("CUDA device", True, device_name)
    else:
        _warn("CUDA device", "Training will fall back to CPU.")
    return bool(ok)


def check_tests(run_tests: bool) -> bool:
    print("\n== Tests ==")
    if not run_tests:
        _warn("Unit tests", "Skipped. Re-run with `--run-tests` before the first full training job.")
        return True

    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    return _status("Unit tests", result.returncode == 0, f"exit_code={result.returncode}")


def check_dataset(data_root: str, dataset_name: str) -> bool:
    print("\n== Dataset ==")
    try:
        from reid.data import dataset_summary, load_dataset
    except Exception as exc:
        return _status("Dataset imports", False, str(exc))

    dataset_path = Path(data_root) / dataset_name
    if not dataset_path.exists():
        _warn("Dataset root", f"Missing {dataset_path}. Download and organize the dataset first.")
        return False

    try:
        bundle = load_dataset(dataset_name, data_root)
    except Exception as exc:
        return _status("Dataset load", False, str(exc))

    print(json.dumps(dataset_summary(bundle), indent=2))
    return _status("Dataset load", True, f"{dataset_name} is ready")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the re-ID environment before training")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--dataset-name", default="market1501")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()

    checks = [
        check_runtime(),
        check_torch(),
        check_tests(args.run_tests),
        check_dataset(args.data_root, args.dataset_name),
    ]

    if not all(checks):
        raise SystemExit(1)

    print("\nAll required checks passed. You can start the first training run.")


if __name__ == "__main__":
    main()
