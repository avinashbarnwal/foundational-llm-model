"""The recommendation foundation model: token embedding -> sparse transformer -> heads."""

from __future__ import annotations

import torch
from torch import nn

from netflix_fm.config import Config
from netflix_fm.data.features import SECONDS_PER_DAY, ItemCatalog
from netflix_fm.model.attention import KVCache
from netflix_fm.model.embeddings import TokenEmbedding
from netflix_fm.model.heads import AuxiliaryHeads, MultiTokenItemHead
from netflix_fm.model.transformer import Transformer


class FoundationModel(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = TokenEmbedding(cfg.vocab, cfg.model)
        self.backbone = Transformer(cfg.model)
        self.item_head = MultiTokenItemHead(cfg.model.d_model, cfg.objective.multi_token_horizon)
        self.aux_heads = AuxiliaryHeads(cfg.vocab, cfg.model)
        # Age of each title at the model's reference time; drives the ID/metadata mix
        # for output-side item embeddings.
        self.register_buffer("catalog_age_days", torch.zeros(cfg.vocab.num_items + 1, dtype=torch.long))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Small init keeps initial item logits near uniform (default N(0, 1) embeddings make them huge).
        if isinstance(module, (nn.Linear, nn.Embedding, nn.EmbeddingBag)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag)) and module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()

    def load_catalog(self, catalog: ItemCatalog, reference_timestamp: int) -> None:
        self.embed.item.load_catalog(catalog)
        age = (reference_timestamp - catalog.release_timestamp).clamp(min=0) // SECONDS_PER_DAY
        self.catalog_age_days.copy_(age)

    def item_table(self) -> torch.Tensor:
        """Output-side embeddings for every item, [num_items + 1, d_model]."""
        ids = torch.arange(self.cfg.vocab.num_items + 1, device=self.catalog_age_days.device)
        return self.embed.item(ids, self.catalog_age_days)

    def encode(self, batch: dict[str, torch.Tensor], cache: KVCache | None = None) -> torch.Tensor:
        """Hidden states [B, L, d_model]."""
        return self.backbone(self.embed(batch), cache)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hidden = self.encode(batch)
        out = {"hidden": hidden, "item_logits": self.item_head(hidden, self.item_table())}
        out.update({f"{k}_logits": v for k, v in self.aux_heads(hidden).items()})
        return out

    @staticmethod
    def last_hidden(hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Hidden state at each sequence's last real token: the member embedding."""
        last = (~padding_mask).sum(dim=1).clamp(min=1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]

    def new_cache(self) -> KVCache:
        return KVCache.empty(self.cfg.model.n_layers)
