"""Training objective.

total = sum_k  weighted_CE(item at t+k)            (multi-token prediction)
      + sum_a  w_a * CE(auxiliary target a at t+1)  (action type, genre, language)

Item targets are weighted by engagement (log watch minutes) so a two-hour watch counts
for more than a short sample; targets with weight 0 (padding, short plays) are ignored.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from netflix_fm.config import ObjectiveConfig
from netflix_fm.data.schema import PAD_ID


def weighted_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    w = weights.reshape(-1)
    return (ce * w).sum() / w.sum().clamp(min=1e-8)


def compute_loss(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: ObjectiveConfig
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = batch["target_weights"]  # [B, K, L]
    if not cfg.weight_by_watch_time:
        weights = (weights > 0).float()

    metrics: dict[str, float] = {}
    total = torch.zeros((), device=weights.device)
    for k in range(outputs["item_logits"].shape[0]):
        loss_k = weighted_cross_entropy(outputs["item_logits"][k], batch["target_items"][:, k], weights[:, k])
        metrics[f"item_t+{k + 1}"] = loss_k.item()
        total = total + loss_k

    next_mask = (batch["target_items"][:, 0] != PAD_ID).float()
    for name, w in cfg.aux_weights.items():
        loss_a = weighted_cross_entropy(outputs[f"{name}_logits"], batch[f"target_{name}"], next_mask)
        metrics[name] = loss_a.item()
        total = total + w * loss_a

    metrics["total"] = total.item()
    return total, metrics
