from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import amp
from torch import nn

from .metrics import compute_distance_matrix, evaluate_rankings
from .performance import images_to_device


MODEL_CONFIG_KEYS = (
    "backbone",
    "pretrained",
    "last_stride",
    "embedding_dim",
    "use_local_branch",
    "use_transformer",
    "use_fusion_gate",
    "num_parts",
    "transformer_dim",
    "transformer_heads",
    "transformer_layers",
    "transformer_dropout",
    "fusion_mode",
    "local_residual_weight",
    "local_residual_learnable",
    "local_loss_weight",
)


def apply_checkpoint_model_config(config: Any, checkpoint: dict[str, Any]) -> None:
    saved = checkpoint.get("config", {})
    for key in MODEL_CONFIG_KEYS:
        if key in saved:
            setattr(config, key, saved[key])


def collect_model_state_diagnostics(model: nn.Module) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    raw_model = model.module if hasattr(model, "module") else model
    scale = getattr(raw_model, "local_residual_scale", None)
    if scale is not None:
        scale_tensor = scale.detach().float()
        diagnostics["local_residual_scale"] = float(scale_tensor.mean().item())
        diagnostics["local_residual_scale_abs"] = float(scale_tensor.abs().mean().item())
        diagnostics["local_residual_scale_learnable"] = isinstance(scale, nn.Parameter)
    diagnostics["use_local_branch"] = bool(getattr(raw_model, "use_local_branch", False))
    diagnostics["use_transformer"] = bool(getattr(raw_model, "use_transformer", False))
    diagnostics["use_fusion_gate"] = bool(getattr(raw_model, "use_fusion_gate", False))
    diagnostics["use_local_auxiliary"] = bool(getattr(raw_model, "use_local_auxiliary", False))
    diagnostics["fusion_mode"] = str(getattr(raw_model, "fusion_mode", "none"))
    return diagnostics


def parameter_group_name(name: str) -> str:
    if name.startswith("backbone."):
        return "backbone"
    if name.startswith("global_projection."):
        return "global_projection"
    if name.startswith(("part_projection.", "local_projection.", "transformer.", "positional_embedding")):
        return "local_branch"
    if name.startswith(("fusion_projection.", "fusion_gate.", "local_residual_scale")):
        return "fusion"
    if name.startswith(("bnneck.", "classifier.")):
        return "global_head"
    if name.startswith(("local_bnneck.", "local_classifier.")):
        return "local_aux_head"
    return "other"


def grouped_gradient_norms(model: nn.Module) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = parameter_group_name(name)
        norm = float(parameter.grad.detach().float().norm(2).item())
        sums[group] += norm * norm
    return {group: math.sqrt(value) for group, value in sorted(sums.items())}


def mean_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    return {
        key: float(np.mean([item[key] for item in items if key in item]))
        for key in keys
        if any(key in item for item in items)
    }


def _mean_tensor_norm(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().norm(dim=1).mean().item())


def _mean_softmax_confidence(logits: torch.Tensor) -> float:
    return float(torch.softmax(logits.detach().float(), dim=1).max(dim=1).values.mean().item())


@torch.no_grad()
def batch_branch_statistics(
    model: nn.Module,
    loader,
    device: torch.device,
    channels_last: bool,
    use_amp: bool,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    stats: list[dict[str, float]] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        images = images_to_device(batch["images"], device, channels_last)
        targets = batch["person_ids"].to(device, non_blocking=True)
        with amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            outputs = model(images)

        batch_stats: dict[str, float] = {
            "num_samples": float(images.size(0)),
            "global_feature_norm": _mean_tensor_norm(outputs["global_embeddings"]),
            "fused_feature_norm": _mean_tensor_norm(outputs["embeddings"]),
            "full_logit_confidence": _mean_softmax_confidence(outputs["logits"]),
        }

        global_embeddings = outputs["global_embeddings"].detach().float()
        global_bn = model.bnneck(global_embeddings)
        global_logits = model.classifier(global_bn)
        logit_delta = outputs["logits"].detach().float() - global_logits.detach().float()
        batch_stats["global_only_logit_confidence"] = _mean_softmax_confidence(global_logits)
        batch_stats["full_minus_global_logit_delta_norm"] = _mean_tensor_norm(logit_delta)
        batch_stats["target_logit_delta_full_minus_global"] = float(
            logit_delta.gather(1, targets.view(-1, 1)).mean().item()
        )

        local_embeddings = outputs.get("local_embeddings")
        if local_embeddings is not None:
            global_norm = global_embeddings.norm(dim=1).clamp_min(1e-12)
            local_norm = local_embeddings.detach().float().norm(dim=1)
            fused_delta_norm = (outputs["embeddings"].detach().float() - global_embeddings).norm(dim=1)
            batch_stats["local_feature_norm"] = float(local_norm.mean().item())
            batch_stats["local_to_global_feature_norm_ratio"] = float((local_norm / global_norm).mean().item())
            batch_stats["fused_delta_to_global_feature_norm_ratio"] = float((fused_delta_norm / global_norm).mean().item())

            local_logits = outputs.get("local_logits")
            if local_logits is not None:
                local_logits = local_logits.detach().float()
                batch_stats["local_aux_logit_confidence"] = _mean_softmax_confidence(local_logits)
                batch_stats["local_aux_target_logit"] = float(
                    local_logits.gather(1, targets.view(-1, 1)).mean().item()
                )

            if getattr(model, "fusion_gate", None) is not None:
                fused_inputs = torch.cat([global_embeddings, local_embeddings.detach().float()], dim=1)
                gate = model.fusion_gate(fused_inputs).detach().float()
                batch_stats["fusion_gate_mean"] = float(gate.mean().item())
                batch_stats["fusion_gate_std"] = float(gate.std(unbiased=False).item())

        stats.append(batch_stats)
    return mean_dicts(stats)


def gradient_branch_statistics(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    channels_last: bool,
    max_batches: int,
    use_amp: bool = False,
) -> dict[str, float]:
    model.eval()
    per_batch: list[dict[str, float]] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        images = images_to_device(batch["images"], device, channels_last)
        targets = batch["person_ids"].to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        with amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            outputs = model(images)
            loss, _ = criterion(outputs, targets)
        loss.backward()
        per_batch.append(grouped_gradient_norms(model))
    model.zero_grad(set_to_none=True)
    return mean_dicts(per_batch)


@torch.no_grad()
def extract_branch_features(
    model: nn.Module,
    loader,
    device: torch.device,
    mode: str,
    channels_last: bool,
    use_amp: bool,
    keep_on_device: bool,
) -> tuple[torch.Tensor | None, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    features: list[torch.Tensor] = []
    person_ids = []
    camera_ids = []
    paths: list[str] = []

    for batch in loader:
        images = images_to_device(batch["images"], device, channels_last)
        with amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            outputs = model(images)
            if mode == "fused":
                embeddings = outputs["bn_embeddings"]
            elif mode == "global":
                embeddings = model.bnneck(outputs["global_embeddings"])
            elif mode == "local":
                local_embeddings = outputs.get("local_bn_embeddings")
                if local_embeddings is None:
                    local_embeddings = outputs.get("local_embeddings")
                if local_embeddings is None:
                    return None, np.array([], dtype=np.int64), np.array([], dtype=np.int64), []
                embeddings = local_embeddings
            else:
                raise ValueError(f"Unsupported feature mode: {mode}")

        embeddings = embeddings.detach().float()
        features.append(embeddings if keep_on_device else embeddings.cpu())
        person_ids.append(batch["person_ids"].numpy())
        camera_ids.append(batch["camera_ids"].numpy())
        paths.extend(batch["paths"])

    return torch.cat(features, dim=0), np.concatenate(person_ids), np.concatenate(camera_ids), paths


def evaluate_branch_mode(
    model: nn.Module,
    query_loader,
    gallery_loader,
    device: torch.device,
    mode: str,
    distance_metric: str,
    max_rank: int,
    channels_last: bool,
    use_amp: bool,
    gpu_distance: bool,
    gpu_distance_max_elements: int,
) -> dict[str, float] | None:
    keep_on_device = device.type == "cuda" and gpu_distance
    query_features, query_ids, query_camids, _ = extract_branch_features(
        model, query_loader, device, mode, channels_last, use_amp, keep_on_device
    )
    if query_features is None:
        return None
    gallery_features, gallery_ids, gallery_camids, _ = extract_branch_features(
        model, gallery_loader, device, mode, channels_last, use_amp, keep_on_device
    )
    if gallery_features is None:
        return None

    distance_elements = len(query_ids) * len(gallery_ids)
    if keep_on_device and distance_elements <= gpu_distance_max_elements:
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


def load_residual_scale_history(train_log_path: Path) -> list[dict[str, float]]:
    if not train_log_path.exists():
        return []
    history: list[dict[str, float]] = []
    for raw_line in train_log_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        import json

        record = json.loads(raw_line)
        if "local_residual_scale" not in record:
            continue
        history.append(
            {
                "epoch": float(record.get("epoch", len(history) + 1)),
                "local_residual_scale": float(record["local_residual_scale"]),
                "local_residual_scale_abs": float(record.get("local_residual_scale_abs", abs(record["local_residual_scale"]))),
            }
        )
    return history
