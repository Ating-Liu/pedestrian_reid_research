from __future__ import annotations

import math

import torch
from torch import amp
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from .config import ExperimentConfig
from .data import build_dataloaders, dataset_summary
from .evaluation import evaluate_model
from .losses import ReIDCriterion
from .model import build_model
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
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = config.cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.allow_tf32
    output_dir = ensure_dir(config.output_path())
    config.save(output_dir / "config.json")

    train_loader, query_loader, gallery_loader, bundle = build_dataloaders(config)
    save_json(output_dir / "dataset_summary.json", dataset_summary(bundle))

    model = build_model(config, num_classes=bundle.num_train_ids).to(device)
    criterion = ReIDCriterion(
        ce_weight=config.ce_weight,
        triplet_weight=config.triplet_weight,
        label_smoothing=config.label_smoothing,
        triplet_margin=config.triplet_margin,
        local_loss_weight=config.local_loss_weight,
    )
    optimizer = Adam(model.parameters(), lr=config.base_lr, weight_decay=config.weight_decay)
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
        save_checkpoint(
            output_dir / "last_model.pth",
            {
                "config": config.to_dict(),
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_rank1": best_rank1,
                "metrics": best_metrics,
            },
        )

    if not best_metrics:
        best_metrics = evaluate_model(
            model=model,
            query_loader=query_loader,
            gallery_loader=gallery_loader,
            device=device,
            distance_metric=config.distance_metric,
            max_rank=config.max_rank,
            use_amp=config.use_amp,
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
    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    triplet_meter = AverageMeter()
    local_ce_meter = AverageMeter()
    local_triplet_meter = AverageMeter()
    acc_meter = AverageMeter()

    for iteration, batch in enumerate(train_loader, start=1):
        images = batch["images"].to(device, non_blocking=True)
        targets = batch["person_ids"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type=device.type, enabled=config.use_amp and device.type == "cuda"):
            outputs = model(images)
            loss, loss_stats = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        predictions = outputs["logits"].argmax(dim=1)
        accuracy = (predictions == targets).float().mean().item()

        batch_size = images.size(0)
        loss_meter.update(loss_stats["total_loss"], batch_size)
        ce_meter.update(loss_stats["ce_loss"], batch_size)
        triplet_meter.update(loss_stats["triplet_loss"], batch_size)
        local_ce_meter.update(loss_stats["local_ce_loss"], batch_size)
        local_triplet_meter.update(loss_stats["local_triplet_loss"], batch_size)
        acc_meter.update(accuracy, batch_size)

        if iteration % config.print_freq == 0:
            print(
                f"Epoch [{epoch + 1}/{config.epochs}] "
                f"Iter [{iteration}/{len(train_loader)}] "
                f"Loss {loss_meter.avg:.4f} "
                f"CE {ce_meter.avg:.4f} "
                f"Triplet {triplet_meter.avg:.4f} "
                f"LocalCE {local_ce_meter.avg:.4f} "
                f"LocalTriplet {local_triplet_meter.avg:.4f} "
                f"Acc {acc_meter.avg:.4f}"
            )

    return {
        "train_loss": loss_meter.avg,
        "train_ce_loss": ce_meter.avg,
        "train_triplet_loss": triplet_meter.avg,
        "train_local_ce_loss": local_ce_meter.avg,
        "train_local_triplet_loss": local_triplet_meter.avg,
        "train_acc": acc_meter.avg,
    }
