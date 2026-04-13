from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class ImageRecord:
    path: str
    person_id: int
    camera_id: int
    dataset_id: int = 0


@dataclass
class DatasetBundle:
    name: str
    train: list[ImageRecord]
    query: list[ImageRecord]
    gallery: list[ImageRecord]
    num_train_ids: int


class ReIDImageDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform: Callable | None = None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "images": image,
            "person_ids": torch.tensor(record.person_id, dtype=torch.long),
            "camera_ids": torch.tensor(record.camera_id, dtype=torch.long),
            "paths": record.path,
        }


class RandomIdentitySampler(Sampler[int]):
    def __init__(self, records: list[ImageRecord], batch_size: int, num_instances: int):
        if batch_size % num_instances != 0:
            raise ValueError("batch_size must be divisible by num_instances")
        self.records = records
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.index_dic: dict[int, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            self.index_dic[record.person_id].append(index)
        self.person_ids = list(self.index_dic.keys())
        self.length = self._estimate_length()

    def _estimate_length(self) -> int:
        length = 0
        for pid in self.person_ids:
            num = len(self.index_dic[pid])
            if num < self.num_instances:
                num = self.num_instances
            length += num - num % self.num_instances
        return length

    def __iter__(self):
        batch_indices: list[int] = []
        pid_to_chunks: dict[int, list[list[int]]] = {}
        for pid, indices in self.index_dic.items():
            pid_indices = indices.copy()
            if len(pid_indices) < self.num_instances:
                sampled = torch.randint(len(pid_indices), (self.num_instances,), dtype=torch.long).tolist()
                pid_indices = [pid_indices[i] for i in sampled]
            else:
                pid_indices = [pid_indices[i] for i in torch.randperm(len(pid_indices)).tolist()]
            chunks = []
            for start in range(0, len(pid_indices), self.num_instances):
                chunk = pid_indices[start : start + self.num_instances]
                if len(chunk) < self.num_instances:
                    chunk = chunk + chunk[: self.num_instances - len(chunk)]
                chunks.append(chunk)
            pid_to_chunks[pid] = chunks

        available_pids = self.person_ids.copy()
        while len(available_pids) >= self.num_pids_per_batch:
            selected_indices = torch.randperm(len(available_pids))[: self.num_pids_per_batch].tolist()
            selected_pids = [available_pids[i] for i in selected_indices]
            for pid in selected_pids:
                batch_indices.extend(pid_to_chunks[pid].pop(0))
                if not pid_to_chunks[pid]:
                    available_pids.remove(pid)
        return iter(batch_indices)

    def __len__(self) -> int:
        return self.length


def build_transforms(height: int, width: int, is_train: bool) -> Callable:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((height, width)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Pad(10),
                transforms.RandomCrop((height, width)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
            ]
        )
    return transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor(), normalize])


def _iter_image_paths(directory: Path) -> list[Path]:
    return sorted([path for path in directory.glob("*") if path.suffix.lower() in IMAGE_EXTENSIONS])


def _parse_folder(directory: Path, relabel: bool, regex: re.Pattern[str] = re.compile(r"([-\d]+)_c(\d+)")) -> tuple[list[ImageRecord], int]:
    if not directory.exists():
        raise FileNotFoundError(f"Missing dataset directory: {directory}")
    records: list[ImageRecord] = []
    person_ids = []
    for path in _iter_image_paths(directory):
        match = regex.search(path.name)
        if match is None:
            continue
        pid, camid = int(match.group(1)), int(match.group(2))
        if pid == -1:
            continue
        person_ids.append(pid)
        records.append(ImageRecord(path=str(path), person_id=pid, camera_id=camid - 1))
    pid_mapping = {pid: index for index, pid in enumerate(sorted(set(person_ids)))} if relabel else {}
    if relabel:
        records = [ImageRecord(path=r.path, person_id=pid_mapping[r.person_id], camera_id=r.camera_id, dataset_id=r.dataset_id) for r in records]
    return records, len(set(person_ids))


def _parse_msmt_list(list_path: Path, image_root: Path, relabel: bool) -> tuple[list[ImageRecord], int]:
    if not list_path.exists():
        raise FileNotFoundError(f"Missing MSMT17 list file: {list_path}")
    records: list[ImageRecord] = []
    person_ids = []
    with list_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            rel_path = parts[0]
            pid = int(parts[1]) if len(parts) > 1 else int(re.findall(r"\d+", Path(rel_path).stem)[0])
            stem_numbers = [int(item) for item in re.findall(r"\d+", Path(rel_path).stem)]
            camid = stem_numbers[1] if len(stem_numbers) > 1 else 0
            records.append(ImageRecord(path=str(image_root / rel_path), person_id=pid, camera_id=camid))
            person_ids.append(pid)
    pid_mapping = {pid: index for index, pid in enumerate(sorted(set(person_ids)))} if relabel else {}
    if relabel:
        records = [ImageRecord(path=r.path, person_id=pid_mapping[r.person_id], camera_id=r.camera_id, dataset_id=r.dataset_id) for r in records]
    return records, len(set(person_ids))


def load_dataset(name: str, root: str) -> DatasetBundle:
    dataset_root = Path(root) / name
    normalized = name.lower()
    if normalized == "market1501":
        train, num_train_ids = _parse_folder(dataset_root / "bounding_box_train", relabel=True)
        query, _ = _parse_folder(dataset_root / "query", relabel=False)
        gallery, _ = _parse_folder(dataset_root / "bounding_box_test", relabel=False)
    elif normalized == "cuhk03_np":
        train, num_train_ids = _parse_folder(dataset_root / "bounding_box_train", relabel=True)
        query, _ = _parse_folder(dataset_root / "query", relabel=False)
        gallery, _ = _parse_folder(dataset_root / "bounding_box_test", relabel=False)
    elif normalized == "msmt17":
        train, num_train_ids = _parse_msmt_list(dataset_root / "list_train.txt", dataset_root / "train", relabel=True)
        query, _ = _parse_msmt_list(dataset_root / "list_query.txt", dataset_root / "test", relabel=False)
        gallery, _ = _parse_msmt_list(dataset_root / "list_gallery.txt", dataset_root / "test", relabel=False)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return DatasetBundle(name=name, train=train, query=query, gallery=gallery, num_train_ids=num_train_ids)


def build_dataloaders(config) -> tuple[DataLoader, DataLoader, DataLoader, DatasetBundle]:
    bundle = load_dataset(config.dataset_name, config.data_root)
    train_transform = build_transforms(config.image_height, config.image_width, is_train=True)
    eval_transform = build_transforms(config.image_height, config.image_width, is_train=False)

    train_dataset = ReIDImageDataset(bundle.train, transform=train_transform)
    query_dataset = ReIDImageDataset(bundle.query, transform=eval_transform)
    gallery_dataset = ReIDImageDataset(bundle.gallery, transform=eval_transform)

    loader_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": True,
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = config.persistent_workers
        loader_kwargs["prefetch_factor"] = config.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=RandomIdentitySampler(bundle.train, config.batch_size, config.num_instances),
        drop_last=True,
        **loader_kwargs,
    )
    query_loader = DataLoader(query_dataset, batch_size=config.batch_size, shuffle=False, **loader_kwargs)
    gallery_loader = DataLoader(gallery_dataset, batch_size=config.batch_size, shuffle=False, **loader_kwargs)
    return train_loader, query_loader, gallery_loader, bundle


def dataset_summary(bundle: DatasetBundle) -> dict[str, int]:
    return {
        "train_images": len(bundle.train),
        "query_images": len(bundle.query),
        "gallery_images": len(bundle.gallery),
        "train_ids": bundle.num_train_ids,
    }
