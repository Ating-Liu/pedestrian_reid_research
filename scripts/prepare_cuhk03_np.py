from __future__ import annotations

import argparse
import csv
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import scipy.io as sio


SPLIT_TO_FOLDER = {
    "train": "bounding_box_train",
    "query": "query",
    "gallery": "bounding_box_test",
}


@dataclass(frozen=True)
class ExtractTask:
    split: str
    source_name: str
    source_index: int
    person_id: int
    camera_id: int
    zip_entry: str
    source_size: int
    target_path: Path


_thread_local = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CUHK03-NP in Market-style folder layout.")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--archive", default=None, help="Path to archive.zip. Defaults to <data-root>/archive.zip.")
    parser.add_argument("--split-file", default=None, help="New-protocol .mat split file.")
    parser.add_argument("--output-name", default="cuhk03_np")
    parser.add_argument("--variant", choices=["detected", "labeled"], default="detected")
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--force", action="store_true", help="Delete and recreate the output directory.")
    return parser.parse_args()


def _mat_string(value) -> str:
    while hasattr(value, "shape") and getattr(value, "size", None) == 1:
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _load_split(split_file: Path) -> dict:
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    data = sio.loadmat(split_file)
    required = ["filelist", "labels", "camId", "train_idx", "query_idx", "gallery_idx"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Split file is missing keys: {missing}")
    return data


def _build_zip_index(archive: Path, image_subdir: str) -> dict[str, tuple[str, int]]:
    if not archive.exists():
        raise FileNotFoundError(f"Missing archive: {archive}")
    suffix = f"/{image_subdir}/"
    entries: dict[str, tuple[str, int]] = {}
    with ZipFile(archive) as zip_file:
        for info in zip_file.infolist():
            normalized = info.filename.replace("\\", "/")
            if info.is_dir() or suffix not in f"/{normalized}":
                continue
            entries[Path(normalized).name] = (info.filename, info.file_size)
    if not entries:
        raise FileNotFoundError(f"No images found under {image_subdir} in {archive}")
    return entries


def _make_tasks(data: dict, zip_index: dict[str, tuple[str, int]], output_dir: Path) -> list[ExtractTask]:
    filelist = data["filelist"].reshape(-1)
    labels = data["labels"].reshape(-1)
    cam_ids = data["camId"].reshape(-1)
    tasks: list[ExtractTask] = []
    missing: list[str] = []

    for split, folder in SPLIT_TO_FOLDER.items():
        indices = data[f"{split}_idx"].reshape(-1)
        for raw_index in indices:
            source_index = int(raw_index)
            zero_index = source_index - 1
            source_name = _mat_string(filelist[zero_index])
            if source_name not in zip_index:
                missing.append(source_name)
                continue
            person_id = int(labels[zero_index])
            camera_id = int(cam_ids[zero_index])
            zip_entry, source_size = zip_index[source_name]
            target_name = f"{person_id:04d}_c{camera_id}s1_{source_index:06d}.png"
            tasks.append(
                ExtractTask(
                    split=split,
                    source_name=source_name,
                    source_index=source_index,
                    person_id=person_id,
                    camera_id=camera_id,
                    zip_entry=zip_entry,
                    source_size=source_size,
                    target_path=output_dir / folder / target_name,
                )
            )

    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} split images were not found in the archive. First missing: {preview}")
    return tasks


def _get_zip_file(archive: Path) -> ZipFile:
    archive_key = str(archive.resolve())
    cache = getattr(_thread_local, "zip_cache", None)
    if cache is None:
        cache = {}
        _thread_local.zip_cache = cache
    if archive_key not in cache:
        cache[archive_key] = ZipFile(archive)
    return cache[archive_key]


def _extract_one(archive: Path, task: ExtractTask) -> str:
    if task.target_path.exists() and task.target_path.stat().st_size == task.source_size:
        return "skipped"
    task.target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = task.target_path.with_suffix(task.target_path.suffix + ".tmp")
    zip_file = _get_zip_file(archive)
    with zip_file.open(task.zip_entry) as source, temp_path.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    temp_path.replace(task.target_path)
    return "written"


def _write_manifest(output_dir: Path, tasks: list[ExtractTask], variant: str, split_file: Path, archive: Path) -> None:
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "target_path", "source_name", "source_index", "person_id", "camera_id"],
        )
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "split": task.split,
                    "target_path": str(task.target_path.relative_to(output_dir)),
                    "source_name": task.source_name,
                    "source_index": task.source_index,
                    "person_id": task.person_id,
                    "camera_id": task.camera_id,
                }
            )

    summary = {
        "dataset": "CUHK03-NP",
        "variant": variant,
        "archive": str(archive),
        "split_file": str(split_file),
        "train_images": sum(1 for task in tasks if task.split == "train"),
        "query_images": sum(1 for task in tasks if task.split == "query"),
        "gallery_images": sum(1 for task in tasks if task.split == "gallery"),
        "train_ids": len({task.person_id for task in tasks if task.split == "train"}),
        "query_ids": len({task.person_id for task in tasks if task.split == "query"}),
        "gallery_ids": len({task.person_id for task in tasks if task.split == "gallery"}),
    }
    with (output_dir / "prepare_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    archive = Path(args.archive) if args.archive else data_root / "archive.zip"
    split_file = (
        Path(args.split_file)
        if args.split_file
        else data_root / f"cuhk03_new_protocol_config_{args.variant}.mat"
    )
    output_dir = data_root / args.output_name
    image_subdir = f"images_{args.variant}"

    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for folder in SPLIT_TO_FOLDER.values():
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    print(f"Loading split: {split_file}")
    data = _load_split(split_file)
    print(f"Indexing archive entries under {image_subdir}: {archive}")
    zip_index = _build_zip_index(archive, image_subdir)
    tasks = _make_tasks(data, zip_index, output_dir)
    print(f"Prepared {len(tasks)} extraction tasks with jobs={args.jobs}")

    written = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [executor.submit(_extract_one, archive, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result == "written":
                written += 1
            else:
                skipped += 1
            if done % 500 == 0 or done == len(futures):
                print(f"Processed {done}/{len(futures)} images; written={written}, skipped={skipped}")

    _write_manifest(output_dir, tasks, args.variant, split_file, archive)
    print(f"Saved CUHK03-NP {args.variant} dataset to {output_dir}")
    print(f"Train/query/gallery: {sum(1 for t in tasks if t.split == 'train')}/"
          f"{sum(1 for t in tasks if t.split == 'query')}/"
          f"{sum(1 for t in tasks if t.split == 'gallery')}")


if __name__ == "__main__":
    main()
