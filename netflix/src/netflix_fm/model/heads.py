"""Prediction heads.

* Multi-token item head: one projection per step k = 1..K predicting the item K steps
  ahead. Predicting several steps ahead pushes the model toward longer-term member
  interests instead of only the immediate next click. Logits come from a dot product
  with the (shared) output item embeddings.
* Auxiliary heads: next action type, genre and language. These are cheap, dense signals
  that regularise the representation and help when the item target is sparse.
"""

from __future__ import annotations

import torch
from torch import nn

from netflix_fm.config import ModelConfig, VocabConfig


class MultiTokenItemHead(nn.Module):
    def __init__(self, d_model: int, horizon: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(horizon))

    def forward(self, hidden: torch.Tensor, item_table: torch.Tensor) -> torch.Tensor:
        """Returns logits [K, B, L, num_items + 1]."""
        return torch.stack([proj(hidden) @ item_table.T for proj in self.projections])


class AuxiliaryHeads(nn.Module):
    def __init__(self, vocab: VocabConfig, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.heads = nn.ModuleDict(
            {
                "action_type": nn.Linear(d, vocab.num_action_types),
                "genre": nn.Linear(d, vocab.num_genres),
                "language": nn.Linear(d, vocab.num_languages),
            }
        )

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(hidden) for name, head in self.heads.items()}
