from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import amp

from .metrics import compute_distance_matrix, evaluate_rankings
from .performance import images_to_device


@torch.no_grad()
def extract_features(
    model,
    dataloader,
    device: torch.device,
    use_amp: bool = True,
    channels_last: bool = False,
    keep_on_device: bool = False,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    features = []
    person_ids = []
    camera_ids = []
    paths: list[str] = []

    for batch in dataloader:
        images = images_to_device(batch["images"], device, channels_last)
        with amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            outputs = model(images)
        embeddings = outputs["bn_embeddings"].detach().float()
        features.append(embeddings if keep_on_device else embeddings.cpu())
        person_ids.append(batch["person_ids"].numpy())
        camera_ids.append(batch["camera_ids"].numpy())
        paths.extend(batch["paths"])

    return torch.cat(features, dim=0), np.concatenate(person_ids), np.concatenate(camera_ids), paths


@torch.no_grad()
def evaluate_model(
    model,
    query_loader,
    gallery_loader,
    device: torch.device,
    distance_metric: str = "cosine",
    max_rank: int = 20,
    use_amp: bool = True,
    channels_last: bool = False,
    gpu_distance: bool = True,
    gpu_distance_max_elements: int = 200_000_000,
) -> dict[str, float]:
    keep_features_on_device = device.type == "cuda" and gpu_distance
    query_features, query_ids, query_camids, _ = extract_features(
        model,
        query_loader,
        device,
        use_amp=use_amp,
        channels_last=channels_last,
        keep_on_device=keep_features_on_device,
    )
    gallery_features, gallery_ids, gallery_camids, _ = extract_features(
        model,
        gallery_loader,
        device,
        use_amp=use_amp,
        channels_last=channels_last,
        keep_on_device=keep_features_on_device,
    )
    distance_elements = len(query_ids) * len(gallery_ids)
    if keep_features_on_device and distance_elements <= gpu_distance_max_elements:
        distmat = compute_distance_matrix(query_features, gallery_features, metric=distance_metric).cpu().numpy()
    else:
        distmat = compute_distance_matrix(query_features.cpu(), gallery_features.cpu(), metric=distance_metric).numpy()
    return evaluate_rankings(
        distmat,
        query_ids=query_ids,
        gallery_ids=gallery_ids,
        query_camids=query_camids,
        gallery_camids=gallery_camids,
        max_rank=max_rank,
    )


def save_ranked_results(
    model,
    query_loader,
    gallery_loader,
    device: torch.device,
    output_dir: str | Path,
    topk: int = 10,
    num_queries: int = 10,
    distance_metric: str = "cosine",
    use_amp: bool = True,
    channels_last: bool = False,
    gpu_distance: bool = True,
    gpu_distance_max_elements: int = 200_000_000,
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    keep_features_on_device = device.type == "cuda" and gpu_distance
    query_features, query_ids, query_camids, query_paths = extract_features(
        model,
        query_loader,
        device,
        use_amp=use_amp,
        channels_last=channels_last,
        keep_on_device=keep_features_on_device,
    )
    gallery_features, gallery_ids, gallery_camids, gallery_paths = extract_features(
        model,
        gallery_loader,
        device,
        use_amp=use_amp,
        channels_last=channels_last,
        keep_on_device=keep_features_on_device,
    )
    distance_elements = len(query_ids) * len(gallery_ids)
    if keep_features_on_device and distance_elements <= gpu_distance_max_elements:
        distmat = compute_distance_matrix(query_features, gallery_features, metric=distance_metric).cpu().numpy()
    else:
        distmat = compute_distance_matrix(query_features.cpu(), gallery_features.cpu(), metric=distance_metric).numpy()
    indices = np.argsort(distmat, axis=1)

    selected_queries = min(num_queries, len(query_paths))
    for q_idx in range(selected_queries):
        ranking = []
        for g_idx in indices[q_idx]:
            same_camera = query_ids[q_idx] == gallery_ids[g_idx] and query_camids[q_idx] == gallery_camids[g_idx]
            if same_camera:
                continue
            ranking.append(g_idx)
            if len(ranking) >= topk:
                break
        grid = _build_ranking_grid(query_paths[q_idx], ranking, gallery_paths, query_ids[q_idx], gallery_ids)
        grid.save(output_root / f"query_{q_idx:03d}.jpg")


def _build_ranking_grid(query_path: str, ranking: list[int], gallery_paths: list[str], query_id: int, gallery_ids: np.ndarray) -> Image.Image:
    cell_width, cell_height = 128, 256
    margin = 8
    canvas_width = (len(ranking) + 1) * (cell_width + margin) + margin
    canvas_height = cell_height + 2 * margin + 20
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    images = [Image.open(query_path).convert("RGB").resize((cell_width, cell_height))]
    for index in ranking:
        images.append(Image.open(gallery_paths[index]).convert("RGB").resize((cell_width, cell_height)))

    for idx, image in enumerate(images):
        x = margin + idx * (cell_width + margin)
        y = margin
        canvas.paste(image, (x, y))
        if idx == 0:
            border = (0, 0, 255)
            label = "query"
        else:
            matched = gallery_ids[ranking[idx - 1]] == query_id
            border = (0, 180, 0) if matched else (220, 0, 0)
            label = f"top{idx}"
        draw.rectangle([x, y, x + cell_width, y + cell_height], outline=border, width=4)
        draw.text((x, y + cell_height + 2), label, fill=(20, 20, 20))
    return canvas
