from __future__ import annotations

from collections.abc import Iterator
import warnings

import torch
from torch import nn
from torch.optim import Adam


def configure_torch_runtime(
    device: torch.device,
    cudnn_benchmark: bool,
    allow_tf32: bool,
) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if allow_tf32:
        torch.set_float32_matmul_precision("high")


def model_to_device(model: nn.Module, device: torch.device, channels_last: bool) -> nn.Module:
    model = model.to(device)
    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


def images_to_device(images: torch.Tensor, device: torch.device, channels_last: bool) -> torch.Tensor:
    if channels_last and device.type == "cuda":
        return images.to(device, non_blocking=True, memory_format=torch.channels_last)
    return images.to(device, non_blocking=True)


def build_adam_optimizer(
    parameters,
    lr: float,
    weight_decay: float,
    device: torch.device,
    fused: bool,
) -> Adam:
    if fused and device.type == "cuda":
        try:
            return Adam(parameters, lr=lr, weight_decay=weight_decay, fused=True)
        except (RuntimeError, TypeError) as exc:
            warnings.warn(f"Fused Adam is unavailable, falling back to standard Adam: {exc}", stacklevel=2)
    return Adam(parameters, lr=lr, weight_decay=weight_decay)


class CudaBatchPrefetcher:
    def __init__(self, loader, device: torch.device, channels_last: bool):
        if device.type != "cuda":
            raise ValueError("CudaBatchPrefetcher requires a CUDA device")
        self.loader = iter(loader)
        self.device = device
        self.channels_last = channels_last
        self.stream = torch.cuda.Stream(device=device)
        self.next_images: torch.Tensor | None = None
        self.next_targets: torch.Tensor | None = None
        self.preload()

    def preload(self) -> None:
        try:
            batch = next(self.loader)
        except StopIteration:
            self.next_images = None
            self.next_targets = None
            return

        with torch.cuda.stream(self.stream):
            self.next_images = images_to_device(batch["images"], self.device, self.channels_last)
            self.next_targets = batch["person_ids"].to(self.device, non_blocking=True)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.next_images is None or self.next_targets is None:
            raise StopIteration
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        images = self.next_images
        targets = self.next_targets
        images.record_stream(torch.cuda.current_stream(self.device))
        targets.record_stream(torch.cuda.current_stream(self.device))
        self.preload()
        return images, targets


def iter_training_batches(loader, device: torch.device, channels_last: bool, cuda_prefetch: bool):
    if cuda_prefetch and device.type == "cuda":
        yield from CudaBatchPrefetcher(loader, device=device, channels_last=channels_last)
        return
    for batch in loader:
        images = images_to_device(batch["images"], device, channels_last)
        targets = batch["person_ids"].to(device, non_blocking=True)
        yield images, targets
