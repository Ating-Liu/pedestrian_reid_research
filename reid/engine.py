from __future__ import annotations

import math
import time

import torch
from torch import amp
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR

from .config import ExperimentConfig
from .data import build_dataloaders, dataset_summary
from .diagnostics import collect_model_state_diagnostics
from .evaluation import evaluate_model
from .losses import ReIDCriterion
from .model import build_model
from .performance import (
    build_adam_optimizer,
    configure_torch_runtime,
    iter_training_batches,
    model_to_device,
)
from .utils import (
    AverageMeter,
    append_jsonl,
    ensure_dir,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    save_json,
    seed_everything,
    timestamp,
)


def run_training(config: ExperimentConfig) -> dict[str, float]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    configure_torch_runtime(device, config.cudnn_benchmark, config.allow_tf32)
    output_dir = ensure_dir(config.output_path())
    config.save(output_dir / "config.json")

    train_loader, query_loader, gallery_loader, bundle = build_dataloaders(config)
    save_json(output_dir / "dataset_summary.json", dataset_summary(bundle))

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
    scheduler = MultiStepLR(optimizer, milestones=config.lr_steps, gamma=config.gamma)
    scaler = amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")

    start_epoch = 0
    best_rank1 = -math.inf
    best_metrics: dict[str, float] = {}

    if config.checkpoint:
        checkpoint = load_checkpoint(config.checkpoint, device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_rank1 = checkpoint.get("best_rank1", best_rank1)

    for epoch in range(start_epoch, config.epochs):
        train_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            epoch=epoch,
            config=config,
        )
        scheduler.step()

        log_payload = {
            "timestamp": timestamp(),
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            **train_stats,
            **collect_model_state_diagnostics(model),
        }
        should_eval = (epoch + 1) % config.eval_period == 0 or (epoch + 1) == config.epochs
        if should_eval:
            metrics = evaluate_model(
                model=model,
                query_loader=query_loader,
                gallery_loader=gallery_loader,
                device=device,
                distance_metric=config.distance_metric,
                max_rank=config.max_rank,
                use_amp=config.use_amp,
                channels_last=config.channels_last,
                gpu_distance=config.eval_gpu_distance,
                gpu_distance_max_elements=config.eval_gpu_distance_max_elements,
            )
            log_payload.update(metrics)
            if metrics["rank1"] > best_rank1:
                best_rank1 = metrics["rank1"]
                best_metrics = metrics
                save_checkpoint(
                    output_dir / "best_model.pth",
                    {
                        "config": config.to_dict(),
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "best_rank1": best_rank1,
                        "metrics": metrics,
                    },
                )

        append_jsonl(output_dir / "train_log.jsonl", log_payload)
        checkpoint_payload = {
            "config": config.to_dict(),
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_rank1": best_rank1,
            "metrics": best_metrics,
        }
        save_checkpoint(output_dir / "last_model.pth", checkpoint_payload)
        if config.checkpoint_period > 0 and ((epoch + 1) % config.checkpoint_period == 0 or (epoch + 1) == config.epochs):
            save_checkpoint(output_dir / f"epoch_{epoch + 1:03d}_model.pth", checkpoint_payload)

    if not best_metrics:
        best_metrics = evaluate_model(
            model=model,
            query_loader=query_loader,
            gallery_loader=gallery_loader,
            device=device,
            distance_metric=config.distance_metric,
            max_rank=config.max_rank,
            use_amp=config.use_amp,
            channels_last=config.channels_last,
            gpu_distance=config.eval_gpu_distance,
            gpu_distance_max_elements=config.eval_gpu_distance_max_elements,
        )
    save_json(output_dir / "final_metrics.json", best_metrics)
    return best_metrics


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    criterion: ReIDCriterion,
    scaler: amp.GradScaler,
    device: torch.device,
    epoch: int,
    config: ExperimentConfig,
) -> dict[str, float]:
    model.train()
    data_meter = AverageMeter()
    step_meter = AverageMeter()
    stat_sums = {
        "total_loss": torch.zeros((), device=device),
        "ce_loss": torch.zeros((), device=device),
        "triplet_loss": torch.zeros((), device=device),
        "local_ce_loss": torch.zeros((), device=device),
        "local_triplet_loss": torch.zeros((), device=device),
    }
    correct_sum = torch.zeros((), device=device)
    sample_count = 0
    epoch_start = time.perf_counter()
    end = epoch_start

    batches = iter_training_batches(
        train_loader,
        device=device,
        channels_last=config.channels_last,
        cuda_prefetch=config.cuda_prefetch,
    )
    for iteration, (images, targets) in enumerate(batches, start=1):
        data_meter.update(time.perf_counter() - end)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type=device.type, enabled=config.use_amp and device.type == "cuda"):
            outputs = model(images)
            loss, loss_stats = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        predictions = outputs["logits"].argmax(dim=1)

        batch_size = images.size(0)
        for key in stat_sums:
            stat_sums[key] = stat_sums[key] + loss_stats[key] * batch_size
        correct_sum = correct_sum + predictions.eq(targets).sum()
        sample_count += batch_size
        step_meter.update(time.perf_counter() - end)
        end = time.perf_counter()

        if iteration % config.print_freq == 0:
            total_avg = stat_sums["total_loss"].item() / max(1, sample_count)
            ce_avg = stat_sums["ce_loss"].item() / max(1, sample_count)
            triplet_avg = stat_sums["triplet_loss"].item() / max(1, sample_count)
            local_ce_avg = stat_sums["local_ce_loss"].item() / max(1, sample_count)
            local_triplet_avg = stat_sums["local_triplet_loss"].item() / max(1, sample_count)
            acc_avg = correct_sum.item() / max(1, sample_count)
            images_per_second = batch_size / max(step_meter.val, 1e-12)
            print(
                f"Epoch [{epoch + 1}/{config.epochs}] "
                f"Iter [{iteration}/{len(train_loader)}] "
                f"Loss {total_avg:.4f} "
                f"CE {ce_avg:.4f} "
                f"Triplet {triplet_avg:.4f} "
                f"LocalCE {local_ce_avg:.4f} "
                f"LocalTriplet {local_triplet_avg:.4f} "
                f"Acc {acc_avg:.4f} "
                f"Img/s {images_per_second:.1f} "
                f"Data {data_meter.avg:.4f}s "
                f"Step {step_meter.avg:.4f}s"
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    epoch_seconds = time.perf_counter() - epoch_start
    epoch_images_per_second = sample_count / max(epoch_seconds, 1e-12)
    if config.log_throughput:
        print(
            f"Epoch [{epoch + 1}/{config.epochs}] throughput "
            f"{epoch_images_per_second:.1f} img/s "
            f"({sample_count} images in {epoch_seconds:.1f}s)"
        )

    return {
        "train_loss": stat_sums["total_loss"].item() / max(1, sample_count),
        "train_ce_loss": stat_sums["ce_loss"].item() / max(1, sample_count),
        "train_triplet_loss": stat_sums["triplet_loss"].item() / max(1, sample_count),
        "train_local_ce_loss": stat_sums["local_ce_loss"].item() / max(1, sample_count),
        "train_local_triplet_loss": stat_sums["local_triplet_loss"].item() / max(1, sample_count),
        "train_acc": correct_sum.item() / max(1, sample_count),
        "train_epoch_seconds": epoch_seconds,
        "train_images_per_second": epoch_images_per_second,
        "train_data_time": data_meter.avg,
        "train_step_time": step_meter.avg,
    }
