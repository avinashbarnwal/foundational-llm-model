"""Input embeddings.

Item representation (cold-start aware):
  * an ID embedding, learned from interactions, which is strong for established titles;
  * a metadata embedding (genre, language, tags), available even for a title that has
    never been watched.
The two are mixed by a gate conditioned on the item's age: new titles lean on metadata,
and mature titles lean on their learned ID embedding.

Token representation = item embedding + action-time features + request-time features.
"""

from __future__ import annotations

import torch
from torch import nn

from netflix_fm.config import ModelConfig, VocabConfig
from netflix_fm.data.features import (
    NUM_AGE_BUCKETS,
    NUM_DURATION_BUCKETS,
    NUM_GAP_BUCKETS,
    NUM_HOURS,
    ItemCatalog,
    age_bucket,
)
from netflix_fm.data.schema import PAD_ID


class ItemEmbedding(nn.Module):
    def __init__(self, vocab: VocabConfig, d_model: int, max_tags: int = 8) -> None:
        super().__init__()
        self.id_embedding = nn.Embedding(vocab.num_items + 1, d_model, padding_idx=PAD_ID)
        self.genre_embedding = nn.Embedding(vocab.num_genres, d_model, padding_idx=PAD_ID)
        self.language_embedding = nn.Embedding(vocab.num_languages, d_model, padding_idx=PAD_ID)
        self.tag_embedding = nn.EmbeddingBag(
            vocab.num_metadata_tags, d_model, mode="mean", padding_idx=PAD_ID
        )
        self.metadata_proj = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.age_embedding = nn.Embedding(NUM_AGE_BUCKETS, d_model)
        self.gate = nn.Linear(d_model, 1)

        # Metadata lookup tables live on the module so they move with .to(device).
        self.register_buffer("item_genre", torch.zeros(vocab.num_items + 1, dtype=torch.long))
        self.register_buffer("item_language", torch.zeros(vocab.num_items + 1, dtype=torch.long))
        self.register_buffer("item_tags", torch.zeros(vocab.num_items + 1, max_tags, dtype=torch.long))

    def load_catalog(self, catalog: ItemCatalog) -> None:
        self.item_genre.copy_(catalog.genre)
        self.item_language.copy_(catalog.language)
        self.item_tags.copy_(catalog.tags)

    def metadata_embedding(self, item_ids: torch.Tensor) -> torch.Tensor:
        shape = item_ids.shape
        flat = item_ids.reshape(-1)
        tags = self.tag_embedding(self.item_tags[flat])
        meta = torch.cat(
            [self.genre_embedding(self.item_genre[flat]), self.language_embedding(self.item_language[flat]), tags],
            dim=-1,
        )
        return self.metadata_proj(meta).reshape(*shape, -1)

    def mixing_weight(self, age_days: torch.Tensor) -> torch.Tensor:
        """Weight on the ID embedding in [0, 1]; low for young items."""
        return torch.sigmoid(self.gate(self.age_embedding(age_bucket(age_days))))

    def forward(self, item_ids: torch.Tensor, age_days: torch.Tensor) -> torch.Tensor:
        alpha = self.mixing_weight(age_days)
        return alpha * self.id_embedding(item_ids) + (1 - alpha) * self.metadata_embedding(item_ids)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab: VocabConfig, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.item = ItemEmbedding(vocab, d)
        # Action-time features: what happened at this step.
        self.action_type = nn.Embedding(vocab.num_action_types, d, padding_idx=PAD_ID)
        self.device = nn.Embedding(vocab.num_devices, d, padding_idx=PAD_ID)
        self.locale = nn.Embedding(vocab.num_locales, d, padding_idx=PAD_ID)
        self.duration = nn.Embedding(NUM_DURATION_BUCKETS, d, padding_idx=PAD_ID)
        self.gap = nn.Embedding(NUM_GAP_BUCKETS, d, padding_idx=PAD_ID)
        self.hour = nn.Embedding(NUM_HOURS + 1, d, padding_idx=PAD_ID)
        # Request-time features: context of the upcoming request.
        self.req_device = nn.Embedding(vocab.num_devices, d, padding_idx=PAD_ID)
        self.req_locale = nn.Embedding(vocab.num_locales, d, padding_idx=PAD_ID)
        self.req_hour = nn.Embedding(NUM_HOURS + 1, d, padding_idx=PAD_ID)
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.item(batch["item_id"], batch["item_age_days"])
        x = x + self.action_type(batch["action_type"])
        x = x + self.device(batch["device"]) + self.locale(batch["locale"])
        x = x + self.duration(batch["duration_bucket"]) + self.gap(batch["gap_bucket"]) + self.hour(batch["hour"])
        x = x + self.req_device(batch["req_device"]) + self.req_locale(batch["req_locale"]) + self.req_hour(batch["req_hour"])
        return self.dropout(self.norm(x))
