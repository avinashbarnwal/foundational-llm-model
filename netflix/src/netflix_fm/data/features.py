"""Bucketing of continuous features and the item catalog lookup tables."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from netflix_fm.data.schema import PAD_ID, ItemMetadata

NUM_DURATION_BUCKETS = 16
NUM_GAP_BUCKETS = 24
NUM_HOURS = 24
NUM_AGE_BUCKETS = 12
SECONDS_PER_DAY = 86400


def _log2_bucket(value: float, num_buckets: int) -> int:
    """Map a non-negative value to 1..num_buckets-1 on a log2 scale (0 is padding)."""
    bucket = 1 + int(math.log2(1 + max(value, 0.0)))
    return min(bucket, num_buckets - 1)


def duration_bucket(seconds: int) -> int:
    return _log2_bucket(seconds / 60.0, NUM_DURATION_BUCKETS)


def gap_bucket(seconds: int) -> int:
    return _log2_bucket(seconds, NUM_GAP_BUCKETS)


def hour_of_day(timestamp: int) -> int:
    return (timestamp // 3600) % NUM_HOURS + 1


def age_bucket(age_days: torch.Tensor) -> torch.Tensor:
    """Vectorised log2 bucket of an item's age in days (used inside the model)."""
    buckets = 1 + torch.log2(1 + age_days.clamp(min=0).float()).long()
    return buckets.clamp(max=NUM_AGE_BUCKETS - 1)


class ItemCatalog:
    """Dense lookup tables of item metadata, indexed by item id (row 0 = padding)."""

    def __init__(self, items: Sequence[ItemMetadata], num_items: int, max_tags: int = 8) -> None:
        self.num_items = num_items
        self.genre = torch.zeros(num_items + 1, dtype=torch.long)
        self.language = torch.zeros(num_items + 1, dtype=torch.long)
        self.release_timestamp = torch.zeros(num_items + 1, dtype=torch.long)
        self.tags = torch.full((num_items + 1, max_tags), PAD_ID, dtype=torch.long)
        for item in items:
            self.genre[item.item_id] = item.genre
            self.language[item.item_id] = item.language
            self.release_timestamp[item.item_id] = item.release_timestamp
            tags = item.tags[:max_tags]
            if tags:
                self.tags[item.item_id, : len(tags)] = torch.tensor(tags)

    def age_days(self, item_ids: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        released = self.release_timestamp[item_ids]
        return ((timestamps - released).clamp(min=0) // SECONDS_PER_DAY).long()
