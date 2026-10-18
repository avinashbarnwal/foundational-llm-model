"""Adapting the foundation model to a specific downstream task.

Three integration patterns, cheapest first:
  1. Embeddings as features: export member / item vectors and feed them to an existing model.
  2. Frozen backbone + small task head (this module with `freeze_backbone=True`).
  3. Full fine-tuning of backbone and head on task data (`freeze_backbone=False`),
     typically with a lower learning rate on the backbone.
"""

from __future__ import annotations

import torch
from torch import nn

from netflix_fm.model.foundation_model import FoundationModel


class TaskModel(nn.Module):
    """Foundation backbone + task head scoring (member state, candidate item) pairs."""

    def __init__(self, foundation: FoundationModel, num_outputs: int = 1, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.foundation = foundation
        d = foundation.cfg.model.d_model
        self.head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, num_outputs))
        if freeze_backbone:
            for p in self.foundation.parameters():
                p.requires_grad_(False)

    def forward(self, batch: dict[str, torch.Tensor], candidate_items: torch.Tensor) -> torch.Tensor:
        """candidate_items: [B] item ids. Returns task logits [B, num_outputs]."""
        hidden = self.foundation.encode(batch)
        member = self.foundation.last_hidden(hidden, batch["padding_mask"])
        item = self.foundation.item_table()[candidate_items]
        return self.head(torch.cat([member, item, member * item], dim=-1))

    def param_groups(self, head_lr: float, backbone_lr: float) -> list[dict]:
        groups = [{"params": self.head.parameters(), "lr": head_lr}]
        backbone = [p for p in self.foundation.parameters() if p.requires_grad]
        if backbone:
            groups.append({"params": backbone, "lr": backbone_lr})
        return groups
