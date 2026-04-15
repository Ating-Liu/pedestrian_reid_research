from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_research_reports import (
    ExperimentSpec,
    ablation_specs,
    collect_runs,
    command_for,
    stability_specs,
)


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def selected_specs(args: argparse.Namespace) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    if args.suite in {"all", "ablation"}:
        specs.extend(ablation_specs())
    if args.suite in {"all", "stability"}:
        specs.extend(stability_specs(parse_seeds(args.stability_seeds)))
    return specs


def expected_variant(spec: ExperimentSpec) -> str:
    joined = " ".join(spec.flags)
    if "--use-local-branch false" in joined:
        return "baseline"
    if "--use-transformer false" in joined:
        return "local_branch"
    if "--use-fusion-gate false" in joined:
        return "transformer_branch"
    return "full_model"


def command_with_resume(spec: ExperimentSpec, output_root: Path) -> str:
    command = command_for(spec)
    run_dir = output_root / spec.dataset / expected_variant(spec) / spec.experiment
    final_metrics = run_dir / "final_metrics.json"
    last_checkpoint = run_dir / "last_model.pth"
    if not final_metrics.exists() and last_checkpoint.exists() and "--checkpoint" not in command:
        command = f"{command} --checkpoint {last_checkpoint}"
    return command


def refresh_reports(output_root: Path, docs_dir: Path, stability_seeds: str, repo_root: Path, log_dir: Path) -> None:
    report_log = log_dir / "report_refresh.log"
    with report_log.open("a", encoding="utf-8") as handle:
        subprocess.run(
            [
                sys.executable,
                "scripts/build_research_reports.py",
                "--output-root",
                str(output_root),
                "--docs-dir",
                str(docs_dir),
                "--stability-seeds",
                stability_seeds,
            ],
            cwd=repo_root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pending narrative-strengthening ReID experiments sequentially.")
    parser.add_argument("--suite", choices=["all", "ablation", "stability"], default="all")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--stability-seeds", default="42,123,3407")
    parser.add_argument("--log-dir", default="outputs/analysis/run_logs")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    docs_dir = Path(args.docs_dir)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    specs = selected_specs(args)
    status_path = log_dir / "queue_status.json"
    queue_log_path = log_dir / "queue.log"

    with queue_log_path.open("a", encoding="utf-8") as queue_log:
        queue_log.write(f"\n=== Queue started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        queue_log.flush()

        for spec in specs:
            runs = collect_runs(output_root)
            completed = {run.experiment for run in runs}
            if spec.experiment in completed:
                queue_log.write(f"[skip] {spec.experiment} already has final_metrics.json\n")
                queue_log.flush()
                continue

            command = command_with_resume(spec, output_root)
            train_log_path = log_dir / f"{spec.experiment}.log"
            status = {
                "state": "running",
                "experiment": spec.experiment,
                "dataset": spec.dataset,
                "suite": args.suite,
                "command": command,
                "log": str(train_log_path),
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            queue_log.write(f"[run] {spec.experiment}\n{command}\n")
            queue_log.flush()

            start = time.perf_counter()
            with train_log_path.open("w", encoding="utf-8") as train_log:
                proc = subprocess.run(
                    command,
                    cwd=repo_root,
                    shell=True,
                    stdout=train_log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            seconds = time.perf_counter() - start
            queue_log.write(f"[done] {spec.experiment} exit={proc.returncode} seconds={seconds:.1f}\n")
            queue_log.flush()

            refresh_reports(output_root, docs_dir, args.stability_seeds, repo_root, log_dir)
            if proc.returncode != 0 and not args.continue_on_error:
                status_path.write_text(
                    json.dumps(
                        {
                            "state": "failed",
                            "experiment": spec.experiment,
                            "returncode": proc.returncode,
                            "seconds": seconds,
                            "log": str(train_log_path),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return proc.returncode

        refresh_reports(output_root, docs_dir, args.stability_seeds, repo_root, log_dir)
        status_path.write_text(
            json.dumps({"state": "finished", "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
            encoding="utf-8",
        )
        queue_log.write(f"=== Queue finished {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
