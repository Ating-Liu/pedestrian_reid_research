from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reid.config import parse_args
from reid.data import build_dataloaders
from reid.evaluation import save_ranked_results
from reid.model import build_model
from reid.utils import load_checkpoint, resolve_device


def _apply_checkpoint_model_config(config, checkpoint: dict) -> None:
    saved = checkpoint.get("config", {})
    for key in (
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
    ):
        if key in saved:
            setattr(config, key, saved[key])


def main() -> None:
    config = parse_args(description="Visualize ranked retrieval results")
    if not config.checkpoint:
        raise ValueError("--checkpoint is required for visualization")

    device = resolve_device(config.device)
    checkpoint = load_checkpoint(config.checkpoint, device)
    _apply_checkpoint_model_config(config, checkpoint)
    _, query_loader, gallery_loader, bundle = build_dataloaders(config)
    model = build_model(config, num_classes=bundle.num_train_ids).to(device)
    model.load_state_dict(checkpoint["model"])

    output_dir = Path(config.checkpoint).with_suffix("").parent / "rankings"
    save_ranked_results(
        model=model,
        query_loader=query_loader,
        gallery_loader=gallery_loader,
        device=device,
        output_dir=output_dir,
        topk=config.visualize_topk,
        num_queries=10,
        distance_metric=config.distance_metric,
        use_amp=config.use_amp,
    )
    print(f"Saved ranking visualizations to {output_dir}")


if __name__ == "__main__":
    main()
