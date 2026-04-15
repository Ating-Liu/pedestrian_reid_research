from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch import amp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reid.config import build_parser, config_from_args
from reid.data import build_dataloaders
from reid.losses import ReIDCriterion
from reid.model import build_model
from reid.performance import (
    build_adam_optimizer,
    configure_torch_runtime,
    iter_training_batches,
    model_to_device,
)
from reid.utils import resolve_device, seed_everything


def main() -> None:
    parser = build_parser(description="Profile effective training throughput without saving checkpoints")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=30)
    args = parser.parse_args()
    config = config_from_args(args)

    seed_everything(config.seed)
    device = resolve_device(config.device)
    configure_torch_runtime(device, config.cudnn_benchmark, config.allow_tf32)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_loader, _, _, bundle = build_dataloaders(config)
    model = model_to_device(build_model(config, num_classes=bundle.num_train_ids), device, config.channels_last)
    criterion = ReIDCriterion(
        ce_weight=config.ce_weight,
        triplet_weight=config.triplet_weight,
        label_smoothing=config.label_smoothing,
        triplet_margin=config.triplet_margin,
        local_loss_weight=config.local_loss_weight,
    )
    optimizer = build_adam_optimizer(
        model.parameters(),
        lr=config.base_lr,
        weight_decay=config.weight_decay,
        device=device,
        fused=config.fused_optimizer,
    )
    scaler = amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")
    model.train()

    total_steps = args.warmup_steps + args.measure_steps
    if total_steps > len(train_loader):
        raise ValueError(f"Requested {total_steps} steps, but the training loader has only {len(train_loader)} batches")

    iterator = iter_training_batches(
        train_loader,
        device=device,
        channels_last=config.channels_last,
        cuda_prefetch=config.cuda_prefetch,
    )

    for _ in range(args.warmup_steps):
        images, targets = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type=device.type, enabled=config.use_amp and device.type == "cuda"):
            outputs = model(images)
            loss, _ = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    data_seconds = 0.0
    step_seconds = 0.0
    images_seen = 0
    measure_start = time.perf_counter()
    for _ in range(args.measure_steps):
        step_start = time.perf_counter()
        images, targets = next(iterator)
        data_seconds += time.perf_counter() - step_start

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type=device.type, enabled=config.use_amp and device.type == "cuda"):
            outputs = model(images)
            loss, _ = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        images_seen += images.size(0)
        step_seconds += time.perf_counter() - step_start

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - measure_start

    payload = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device),
        "dataset": config.dataset_name,
        "variant": config.variant_name(),
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "prefetch_factor": config.prefetch_factor if config.num_workers > 0 else 0,
        "persistent_workers": config.persistent_workers if config.num_workers > 0 else False,
        "pin_memory": config.pin_memory,
        "amp": config.use_amp,
        "channels_last": config.channels_last,
        "cuda_prefetch": config.cuda_prefetch,
        "fused_optimizer": config.fused_optimizer,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "images": images_seen,
        "avg_data_wait_seconds": data_seconds / max(1, args.measure_steps),
        "avg_step_seconds": step_seconds / max(1, args.measure_steps),
        "async_launch_images_per_second": images_seen / max(step_seconds, 1e-12),
        "wall_seconds": wall_seconds,
        "images_per_second": images_seen / max(wall_seconds, 1e-12),
        "max_cuda_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0,
    }
    for key, value in payload.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
