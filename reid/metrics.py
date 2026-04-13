from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def compute_distance_matrix(query_features: torch.Tensor, gallery_features: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
    if metric == "cosine":
        query = F.normalize(query_features, dim=1)
        gallery = F.normalize(gallery_features, dim=1)
        return 1.0 - torch.mm(query, gallery.t())
    if metric == "euclidean":
        return torch.cdist(query_features, gallery_features, p=2)
    raise ValueError(f"Unsupported distance metric: {metric}")


def evaluate_rankings(
    distmat: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    query_camids: np.ndarray,
    gallery_camids: np.ndarray,
    max_rank: int = 20,
) -> dict[str, float]:
    num_query, num_gallery = distmat.shape
    if num_query == 0 or num_gallery == 0:
        raise ValueError("Distance matrix must be non-empty")
    max_rank = min(max_rank, num_gallery)
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_ap = []
    valid_queries = 0

    for q_idx in range(num_query):
        q_pid = query_ids[q_idx]
        q_camid = query_camids[q_idx]

        order = indices[q_idx]
        remove = (gallery_ids[order] == q_pid) & (gallery_camids[order] == q_camid)
        keep = np.invert(remove)
        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        cmc = cmc[:max_rank]
        if len(cmc) < max_rank:
            cmc = np.pad(cmc, (0, max_rank - len(cmc)), mode="edge")
        all_cmc.append(cmc)
        valid_queries += 1

        num_rel = raw_cmc.sum()
        precision = raw_cmc.cumsum() / (np.arange(len(raw_cmc)) + 1.0)
        ap = (precision * raw_cmc).sum() / num_rel
        all_ap.append(ap)

    if valid_queries == 0:
        raise RuntimeError("No valid queries remained after camera filtering")

    cmc_curve = np.mean(np.asarray(all_cmc), axis=0)
    return {
        "rank1": float(cmc_curve[0]),
        "rank5": float(cmc_curve[min(4, len(cmc_curve) - 1)]),
        "rank10": float(cmc_curve[min(9, len(cmc_curve) - 1)]),
        "mAP": float(np.mean(all_ap)),
    }
