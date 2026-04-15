from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, label_smoothing=self.smoothing)


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dist_mat = torch.cdist(embeddings, embeddings, p=2)
        mask_pos = targets.unsqueeze(0).eq(targets.unsqueeze(1))
        mask_neg = ~mask_pos
        mask_pos.fill_diagonal_(False)

        hardest_pos = dist_mat.masked_fill(~mask_pos, -torch.inf).max(dim=1).values
        hardest_neg = dist_mat.masked_fill(~mask_neg, torch.inf).min(dim=1).values
        valid = torch.isfinite(hardest_pos) & torch.isfinite(hardest_neg)
        if not torch.any(valid):
            return embeddings.new_tensor(0.0)

        hardest_pos_t = hardest_pos[valid]
        hardest_neg_t = hardest_neg[valid]
        target = hardest_neg_t.new_ones(hardest_neg_t.size(0))
        return self.ranking_loss(hardest_neg_t, hardest_pos_t, target)


class ReIDCriterion(nn.Module):
    def __init__(
        self,
        ce_weight: float,
        triplet_weight: float,
        label_smoothing: float,
        triplet_margin: float,
        local_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.triplet_weight = triplet_weight
        self.local_loss_weight = local_loss_weight
        self.ce = LabelSmoothingCrossEntropy(label_smoothing)
        self.triplet = BatchHardTripletLoss(triplet_margin)

    def forward(self, outputs: dict[str, torch.Tensor], targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        ce_loss = self.ce(outputs["logits"], targets) if self.ce_weight > 0 else outputs["embeddings"].new_tensor(0.0)
        triplet_loss = self.triplet(outputs["embeddings"], targets) if self.triplet_weight > 0 else outputs["embeddings"].new_tensor(0.0)
        total = self.ce_weight * ce_loss + self.triplet_weight * triplet_loss
        local_ce_loss = outputs["embeddings"].new_tensor(0.0)
        local_triplet_loss = outputs["embeddings"].new_tensor(0.0)
        if self.local_loss_weight > 0 and outputs.get("local_embeddings") is not None:
            local_logits = outputs.get("local_logits")
            if local_logits is None:
                raise ValueError("local_loss_weight requires local_logits in model outputs")
            local_ce_loss = self.ce(local_logits, targets) if self.ce_weight > 0 else local_ce_loss
            local_triplet_loss = self.triplet(outputs["local_embeddings"], targets) if self.triplet_weight > 0 else local_triplet_loss
            total = total + self.local_loss_weight * (
                self.ce_weight * local_ce_loss + self.triplet_weight * local_triplet_loss
            )
        return total, {
            "ce_loss": ce_loss.detach(),
            "triplet_loss": triplet_loss.detach(),
            "local_ce_loss": local_ce_loss.detach(),
            "local_triplet_loss": local_triplet_loss.detach(),
            "total_loss": total.detach(),
        }
