"""Online serving with a KV cache.

A member's history is encoded once ("prefill"); afterwards each new interaction costs a
single-token forward pass instead of re-encoding the whole history. Because the attention
pattern is sparse, the cache only keeps the global tokens plus the local window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from netflix_fm.data.dataset import INPUT_KEYS, encode_tokens
from netflix_fm.data.features import ItemCatalog, hour_of_day
from netflix_fm.data.schema import PAD_ID, InteractionToken
from netflix_fm.model.attention import KVCache
from netflix_fm.model.foundation_model import FoundationModel


@dataclass
class RequestContext:
    """What is known when a recommendation request arrives."""

    timestamp: int
    device: int
    locale: int


class MemberSession:
    def __init__(self, model: FoundationModel, catalog: ItemCatalog) -> None:
        self.model = model.eval()
        self.catalog = catalog
        self.cache: KVCache = model.new_cache()
        self.last_token: InteractionToken | None = None
        self.last_hidden: torch.Tensor | None = None
        self.seen: set[int] = set()

    def _features(self, tokens: Sequence[InteractionToken], request: RequestContext) -> dict[str, torch.Tensor]:
        context = [self.last_token] if self.last_token is not None else []
        enc = encode_tokens(context + list(tokens), self.catalog)
        offset = len(context)
        feats = {k: enc[k][offset:].unsqueeze(0) for k in INPUT_KEYS}
        # The last position is scored for the *current* request.
        feats["req_device"][0, -1] = request.device
        feats["req_locale"][0, -1] = request.locale
        feats["req_hour"][0, -1] = hour_of_day(request.timestamp)
        device = self.model.catalog_age_days.device
        return {k: v.to(device) for k, v in feats.items()}

    @torch.no_grad()
    def observe(self, tokens: Sequence[InteractionToken], request: RequestContext) -> None:
        """Append new interaction tokens (prefill on first call)."""
        if not tokens:
            return
        hidden = self.model.encode(self._features(tokens, request), self.cache)
        self.last_hidden = hidden[0, -1]
        self.last_token = tokens[-1]
        self.seen.update(t.item_id for t in tokens)

    def member_embedding(self) -> torch.Tensor:
        if self.last_hidden is None:
            raise RuntimeError("No history observed yet")
        return self.last_hidden

    @torch.no_grad()
    def recommend(self, k: int = 10, exclude_seen: bool = True) -> list[int]:
        scores = self.model.item_head.projections[0](self.member_embedding()) @ self.model.item_table().T
        scores[PAD_ID] = float("-inf")
        if exclude_seen and self.seen:
            scores[list(self.seen)] = float("-inf")
        return scores.topk(k).indices.tolist()
