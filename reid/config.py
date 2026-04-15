from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence


def _csv_to_ints(value: str) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


@dataclass
class ExperimentConfig:
    experiment_name: str = "market1501_full"
    output_dir: str = "outputs"
    seed: int = 42
    device: str = "cuda"

    data_root: str = "datasets"
    dataset_name: str = "market1501"
    image_height: int = 256
    image_width: int = 128
    batch_size: int = 64
    num_instances: int = 4
    num_workers: int = 12
    prefetch_factor: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True

    backbone: str = "resnet50"
    pretrained: bool = True
    last_stride: int = 1
    embedding_dim: int = 512
    use_local_branch: bool = True
    use_transformer: bool = True
    use_fusion_gate: bool = True
    num_parts: int = 6
    transformer_dim: int = 256
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_dropout: float = 0.1
    fusion_mode: str = "projection"
    local_residual_weight: float = 0.1
    local_residual_learnable: bool = True
    local_loss_weight: float = 0.0

    epochs: int = 60
    base_lr: float = 3.5e-4
    weight_decay: float = 5e-4
    lr_steps: list[int] = field(default_factory=lambda: [40, 55])
    gamma: float = 0.1
    label_smoothing: float = 0.1
    triplet_margin: float = 0.3
    triplet_weight: float = 1.0
    ce_weight: float = 1.0
    use_amp: bool = True
    channels_last: bool = False
    cuda_prefetch: bool = True
    fused_optimizer: bool = True
    cudnn_benchmark: bool = True
    allow_tf32: bool = True
    log_throughput: bool = True
    print_freq: int = 20
    eval_period: int = 5
    checkpoint_period: int = 0

    checkpoint: str = ""
    max_rank: int = 20
    distance_metric: str = "cosine"
    eval_gpu_distance: bool = True
    eval_gpu_distance_max_elements: int = 200_000_000
    visualize_topk: int = 10

    def variant_name(self) -> str:
        if not self.use_local_branch:
            return "baseline"
        if self.use_local_branch and not self.use_transformer:
            return "local_branch"
        if self.use_transformer and not self.use_fusion_gate:
            return "transformer_branch"
        return "full_model"

    def output_path(self) -> Path:
        return Path(self.output_dir) / self.dataset_name / self.variant_name() / self.experiment_name

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--experiment-name", default=ExperimentConfig.experiment_name)
    parser.add_argument("--output-dir", default=ExperimentConfig.output_dir)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--device", default=ExperimentConfig.device)

    parser.add_argument("--data-root", default=ExperimentConfig.data_root)
    parser.add_argument("--dataset-name", default=ExperimentConfig.dataset_name)
    parser.add_argument("--image-height", type=int, default=ExperimentConfig.image_height)
    parser.add_argument("--image-width", type=int, default=ExperimentConfig.image_width)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--num-instances", type=int, default=ExperimentConfig.num_instances)
    parser.add_argument("--num-workers", type=int, default=ExperimentConfig.num_workers)
    parser.add_argument("--prefetch-factor", type=int, default=ExperimentConfig.prefetch_factor)
    parser.add_argument("--persistent-workers", type=str2bool, default=ExperimentConfig.persistent_workers)
    parser.add_argument("--pin-memory", type=str2bool, default=ExperimentConfig.pin_memory)

    parser.add_argument("--backbone", default=ExperimentConfig.backbone)
    parser.add_argument("--pretrained", type=str2bool, default=ExperimentConfig.pretrained)
    parser.add_argument("--last-stride", type=int, choices=[1, 2], default=ExperimentConfig.last_stride)
    parser.add_argument("--embedding-dim", type=int, default=ExperimentConfig.embedding_dim)
    parser.add_argument("--use-local-branch", type=str2bool, default=ExperimentConfig.use_local_branch)
    parser.add_argument("--use-transformer", type=str2bool, default=ExperimentConfig.use_transformer)
    parser.add_argument("--use-fusion-gate", type=str2bool, default=ExperimentConfig.use_fusion_gate)
    parser.add_argument("--num-parts", type=int, default=ExperimentConfig.num_parts)
    parser.add_argument("--transformer-dim", type=int, default=ExperimentConfig.transformer_dim)
    parser.add_argument("--transformer-heads", type=int, default=ExperimentConfig.transformer_heads)
    parser.add_argument("--transformer-layers", type=int, default=ExperimentConfig.transformer_layers)
    parser.add_argument("--transformer-dropout", type=float, default=ExperimentConfig.transformer_dropout)
    parser.add_argument("--fusion-mode", choices=["projection", "residual", "gated_residual"], default=ExperimentConfig.fusion_mode)
    parser.add_argument("--local-residual-weight", type=float, default=ExperimentConfig.local_residual_weight)
    parser.add_argument("--local-residual-learnable", type=str2bool, default=ExperimentConfig.local_residual_learnable)
    parser.add_argument("--local-loss-weight", type=float, default=ExperimentConfig.local_loss_weight)

    parser.add_argument("--epochs", type=int, default=ExperimentConfig.epochs)
    parser.add_argument("--base-lr", type=float, default=ExperimentConfig.base_lr)
    parser.add_argument("--weight-decay", type=float, default=ExperimentConfig.weight_decay)
    parser.add_argument("--lr-steps", default="40,55")
    parser.add_argument("--gamma", type=float, default=ExperimentConfig.gamma)
    parser.add_argument("--label-smoothing", type=float, default=ExperimentConfig.label_smoothing)
    parser.add_argument("--triplet-margin", type=float, default=ExperimentConfig.triplet_margin)
    parser.add_argument("--triplet-weight", type=float, default=ExperimentConfig.triplet_weight)
    parser.add_argument("--ce-weight", type=float, default=ExperimentConfig.ce_weight)
    parser.add_argument("--use-amp", type=str2bool, default=ExperimentConfig.use_amp)
    parser.add_argument("--channels-last", type=str2bool, default=ExperimentConfig.channels_last)
    parser.add_argument("--cuda-prefetch", type=str2bool, default=ExperimentConfig.cuda_prefetch)
    parser.add_argument("--fused-optimizer", type=str2bool, default=ExperimentConfig.fused_optimizer)
    parser.add_argument("--cudnn-benchmark", type=str2bool, default=ExperimentConfig.cudnn_benchmark)
    parser.add_argument("--allow-tf32", type=str2bool, default=ExperimentConfig.allow_tf32)
    parser.add_argument("--log-throughput", type=str2bool, default=ExperimentConfig.log_throughput)
    parser.add_argument("--print-freq", type=int, default=ExperimentConfig.print_freq)
    parser.add_argument("--eval-period", type=int, default=ExperimentConfig.eval_period)
    parser.add_argument("--checkpoint-period", type=int, default=ExperimentConfig.checkpoint_period)

    parser.add_argument("--checkpoint", default=ExperimentConfig.checkpoint)
    parser.add_argument("--max-rank", type=int, default=ExperimentConfig.max_rank)
    parser.add_argument("--distance-metric", choices=["cosine", "euclidean"], default=ExperimentConfig.distance_metric)
    parser.add_argument("--eval-gpu-distance", type=str2bool, default=ExperimentConfig.eval_gpu_distance)
    parser.add_argument("--eval-gpu-distance-max-elements", type=int, default=ExperimentConfig.eval_gpu_distance_max_elements)
    parser.add_argument("--visualize-topk", type=int, default=ExperimentConfig.visualize_topk)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig(
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        image_height=args.image_height,
        image_width=args.image_width,
        batch_size=args.batch_size,
        num_instances=args.num_instances,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        backbone=args.backbone,
        pretrained=args.pretrained,
        last_stride=args.last_stride,
        embedding_dim=args.embedding_dim,
        use_local_branch=args.use_local_branch,
        use_transformer=args.use_transformer,
        use_fusion_gate=args.use_fusion_gate,
        num_parts=args.num_parts,
        transformer_dim=args.transformer_dim,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        transformer_dropout=args.transformer_dropout,
        fusion_mode=args.fusion_mode,
        local_residual_weight=args.local_residual_weight,
        local_residual_learnable=args.local_residual_learnable,
        local_loss_weight=args.local_loss_weight,
        epochs=args.epochs,
        base_lr=args.base_lr,
        weight_decay=args.weight_decay,
        lr_steps=_csv_to_ints(args.lr_steps),
        gamma=args.gamma,
        label_smoothing=args.label_smoothing,
        triplet_margin=args.triplet_margin,
        triplet_weight=args.triplet_weight,
        ce_weight=args.ce_weight,
        use_amp=args.use_amp,
        channels_last=args.channels_last,
        cuda_prefetch=args.cuda_prefetch,
        fused_optimizer=args.fused_optimizer,
        cudnn_benchmark=args.cudnn_benchmark,
        allow_tf32=args.allow_tf32,
        log_throughput=args.log_throughput,
        print_freq=args.print_freq,
        eval_period=args.eval_period,
        checkpoint_period=args.checkpoint_period,
        checkpoint=args.checkpoint,
        max_rank=args.max_rank,
        distance_metric=args.distance_metric,
        eval_gpu_distance=args.eval_gpu_distance,
        eval_gpu_distance_max_elements=args.eval_gpu_distance_max_elements,
        visualize_topk=args.visualize_topk,
    )
    if cfg.use_transformer and not cfg.use_local_branch:
        cfg.use_local_branch = True
    if cfg.use_fusion_gate and not cfg.use_local_branch:
        cfg.use_local_branch = True
    if cfg.fusion_mode == "gated_residual":
        cfg.use_local_branch = True
        cfg.use_fusion_gate = True
    return cfg


def parse_args(argv: Sequence[str] | None = None, description: str = "Pedestrian re-ID experiment") -> ExperimentConfig:
    parser = build_parser(description)
    args = parser.parse_args(argv)
    return config_from_args(args)
