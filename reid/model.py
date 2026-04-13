from __future__ import annotations

import warnings

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def _build_resnet50(pretrained: bool, last_stride: int) -> tuple[nn.Module, int]:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    try:
        base = resnet50(weights=weights)
    except Exception as exc:
        warnings.warn(
            f"Falling back to randomly initialized ResNet-50 because pretrained weights are unavailable: {exc}",
            stacklevel=2,
        )
        base = resnet50(weights=None)
    if last_stride == 1:
        base.layer4[0].conv2.stride = (1, 1)
        base.layer4[0].downsample[0].stride = (1, 1)
    backbone = nn.Sequential(
        base.conv1,
        base.bn1,
        base.relu,
        base.maxpool,
        base.layer1,
        base.layer2,
        base.layer3,
        base.layer4,
    )
    return backbone, 2048


class GlobalLocalReIDModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        pretrained: bool = True,
        last_stride: int = 1,
        use_local_branch: bool = True,
        use_transformer: bool = True,
        use_fusion_gate: bool = True,
        num_parts: int = 6,
        transformer_dim: int = 256,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_dropout: float = 0.1,
        fusion_mode: str = "projection",
        local_residual_weight: float = 0.1,
        local_residual_learnable: bool = True,
        use_local_auxiliary: bool = False,
    ):
        super().__init__()
        if fusion_mode not in {"projection", "residual", "gated_residual"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
        self.backbone, backbone_dim = _build_resnet50(pretrained=pretrained, last_stride=last_stride)
        self.use_local_branch = use_local_branch
        self.use_transformer = use_transformer and use_local_branch
        self.use_fusion_gate = (use_fusion_gate or fusion_mode == "gated_residual") and use_local_branch
        self.use_local_auxiliary = use_local_auxiliary and use_local_branch
        self.fusion_mode = fusion_mode

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_projection = nn.Linear(backbone_dim, embedding_dim)

        if self.use_local_branch:
            self.part_pool = nn.AdaptiveAvgPool2d((num_parts, 1))
            self.part_projection = nn.Linear(backbone_dim, transformer_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=transformer_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_dim * 2,
                dropout=transformer_dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = (
                nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
                if self.use_transformer
                else nn.Identity()
            )
            self.positional_embedding = nn.Parameter(torch.zeros(1, num_parts, transformer_dim))
            self.local_projection = nn.Linear(transformer_dim, embedding_dim)
            self.fusion_projection = nn.Linear(embedding_dim * 2, embedding_dim)
            local_scale = torch.tensor(float(local_residual_weight))
            if local_residual_learnable:
                self.local_residual_scale = nn.Parameter(local_scale)
            else:
                self.register_buffer("local_residual_scale", local_scale)
            self.fusion_gate = (
                nn.Sequential(
                    nn.Linear(embedding_dim * 2, embedding_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.Sigmoid(),
                )
                if self.use_fusion_gate
                else None
            )
            if self.use_local_auxiliary:
                self.local_bnneck = nn.BatchNorm1d(embedding_dim)
                self.local_bnneck.bias.requires_grad_(False)
                self.local_classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        else:
            self.transformer = nn.Identity()

        self.bnneck = nn.BatchNorm1d(embedding_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.global_projection.weight, std=0.001)
        nn.init.zeros_(self.global_projection.bias)
        if self.use_local_branch:
            nn.init.normal_(self.part_projection.weight, std=0.001)
            nn.init.zeros_(self.part_projection.bias)
            nn.init.normal_(self.local_projection.weight, std=0.001)
            nn.init.zeros_(self.local_projection.bias)
            nn.init.normal_(self.fusion_projection.weight, std=0.001)
            nn.init.zeros_(self.fusion_projection.bias)
            if self.fusion_gate is not None:
                for module in self.fusion_gate:
                    if isinstance(module, nn.Linear):
                        nn.init.normal_(module.weight, std=0.001)
                        nn.init.zeros_(module.bias)
            if self.use_local_auxiliary:
                nn.init.normal_(self.local_classifier.weight, std=0.001)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_map = self.backbone(images)
        global_vector = self.global_pool(feature_map).flatten(1)
        global_embedding = self.global_projection(global_vector)

        local_embedding = None
        local_bn_embedding = None
        local_logits = None
        if self.use_local_branch:
            part_tokens = self.part_pool(feature_map).squeeze(-1).permute(0, 2, 1)
            part_tokens = self.part_projection(part_tokens)
            if self.use_transformer:
                part_tokens = self.transformer(part_tokens + self.positional_embedding[:, : part_tokens.size(1)])
            local_vector = part_tokens.mean(dim=1)
            local_embedding = self.local_projection(local_vector)
            if self.use_local_auxiliary:
                local_bn_embedding = self.local_bnneck(local_embedding)
                local_logits = self.local_classifier(local_bn_embedding)
            fused_inputs = torch.cat([global_embedding, local_embedding], dim=1)
            if self.fusion_mode == "residual":
                fused_embedding = global_embedding + self.local_residual_scale * local_embedding
            elif self.fusion_mode == "gated_residual":
                gate = self.fusion_gate(fused_inputs)
                fused_embedding = global_embedding + self.local_residual_scale * gate * local_embedding
            elif self.fusion_gate is not None:
                gate = self.fusion_gate(fused_inputs)
                gated_global = gate * global_embedding
                gated_local = (1.0 - gate) * local_embedding
                fused_inputs = torch.cat([gated_global, gated_local], dim=1)
                fused_embedding = self.fusion_projection(fused_inputs)
            else:
                fused_embedding = self.fusion_projection(fused_inputs)
        else:
            fused_embedding = global_embedding

        bn_embedding = self.bnneck(fused_embedding)
        logits = self.classifier(bn_embedding)
        return {
            "embeddings": fused_embedding,
            "bn_embeddings": bn_embedding,
            "logits": logits,
            "global_embeddings": global_embedding,
            "local_embeddings": local_embedding,
            "local_bn_embeddings": local_bn_embedding,
            "local_logits": local_logits,
        }


def build_model(config, num_classes: int) -> GlobalLocalReIDModel:
    if config.backbone.lower() != "resnet50":
        raise ValueError(f"Unsupported backbone: {config.backbone}")
    return GlobalLocalReIDModel(
        num_classes=num_classes,
        embedding_dim=config.embedding_dim,
        pretrained=config.pretrained,
        last_stride=config.last_stride,
        use_local_branch=config.use_local_branch,
        use_transformer=config.use_transformer,
        use_fusion_gate=config.use_fusion_gate,
        num_parts=config.num_parts,
        transformer_dim=config.transformer_dim,
        transformer_heads=config.transformer_heads,
        transformer_layers=config.transformer_layers,
        transformer_dropout=config.transformer_dropout,
        fusion_mode=config.fusion_mode,
        local_residual_weight=config.local_residual_weight,
        local_residual_learnable=config.local_residual_learnable,
        use_local_auxiliary=config.local_loss_weight > 0,
    )
