"""Training dataset: token sequences -> fixed-length tensor windows.

Long histories are cut into overlapping windows (sliding-window sampling) so every
part of a member's history is seen in training while each example stays bounded.

Each position t carries two kinds of features:
  * action-time features of token t (what happened), and
  * request-time features of token t+1 (device / locale / hour of the *next* request),
    which are known when the model is called and help predict what comes next.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset

from netflix_fm.data.features import (
    ItemCatalog,
    duration_bucket,
    gap_bucket,
    hour_of_day,
)
from netflix_fm.data.schema import InteractionToken

INPUT_KEYS = (
    "item_id",
    "action_type",
    "device",
    "locale",
    "duration_bucket",
    "gap_bucket",
    "hour",
    "item_age_days",
    "req_device",
    "req_locale",
    "req_hour",
)


def encode_tokens(tokens: Sequence[InteractionToken], catalog: ItemCatalog) -> dict[str, torch.Tensor]:
    """Encode a full token sequence into per-position feature and target tensors."""
    n = len(tokens)
    feats = {k: torch.zeros(n, dtype=torch.long) for k in INPUT_KEYS}
    watch_seconds = torch.zeros(n, dtype=torch.float)
    prev_ts = None
    for i, tok in enumerate(tokens):
        feats["item_id"][i] = tok.item_id
        feats["action_type"][i] = tok.action_type
        feats["device"][i] = tok.device
        feats["locale"][i] = tok.locale
        feats["duration_bucket"][i] = duration_bucket(tok.total_duration_seconds)
        feats["gap_bucket"][i] = gap_bucket(0 if prev_ts is None else tok.timestamp - prev_ts)
        feats["hour"][i] = hour_of_day(tok.timestamp)
        watch_seconds[i] = tok.total_duration_seconds
        prev_ts = tok.timestamp
        if i + 1 < n:
            nxt = tokens[i + 1]
            feats["req_device"][i] = nxt.device
            feats["req_locale"][i] = nxt.locale
            feats["req_hour"][i] = hour_of_day(nxt.timestamp)
    timestamps = torch.tensor([t.timestamp for t in tokens], dtype=torch.long)
    feats["item_age_days"] = catalog.age_days(feats["item_id"], timestamps)
    feats["watch_seconds"] = watch_seconds
    return feats


class InteractionWindowDataset(Dataset):
    """Sliding windows of length `max_seq_len` with K-step-ahead targets."""

    def __init__(
        self,
        sequences: Sequence[Sequence[InteractionToken]],
        catalog: ItemCatalog,
        max_seq_len: int,
        stride: int,
        horizon: int,
        min_watch_seconds: int = 0,
    ) -> None:
        self.catalog = catalog
        self.max_seq_len = max_seq_len
        self.horizon = horizon
        self.min_watch_seconds = min_watch_seconds
        self.encoded = [encode_tokens(seq, catalog) for seq in sequences if len(seq) > 1]
        # (sequence index, window start)
        self.windows: list[tuple[int, int]] = []
        for s, enc in enumerate(self.encoded):
            n = len(enc["item_id"])
            last_start = max(n - 1 - max_seq_len, 0)
            starts = list(range(0, last_start + 1, stride))
            if starts[-1] != last_start:
                starts.append(last_start)  # always include the most recent window
            self.windows.extend((s, start) for start in starts)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s, start = self.windows[idx]
        enc = self.encoded[s]
        n = len(enc["item_id"])
        L, K = self.max_seq_len, self.horizon
        length = min(L, n - start - 1)  # leave at least one target after the window
        out: dict[str, torch.Tensor] = {}
        for key in INPUT_KEYS:
            x = torch.zeros(L, dtype=torch.long)
            x[:length] = enc[key][start : start + length]
            out[key] = x
        out["padding_mask"] = torch.zeros(L, dtype=torch.bool)
        out["padding_mask"][length:] = True

        # Targets: target_items[k, t] = item at position t + k + 1 (0 = ignore).
        targets = torch.zeros(K, L, dtype=torch.long)
        weights = torch.zeros(K, L, dtype=torch.float)
        for k in range(K):
            lo, hi = start + k + 1, min(start + length + k + 1, n)
            span = hi - lo
            if span <= 0:
                continue
            targets[k, :span] = enc["item_id"][lo:hi]
            secs = enc["watch_seconds"][lo:hi]
            # Short plays are weak signals: mask them out as targets.
            valid = secs >= self.min_watch_seconds
            weights[k, :span] = torch.log1p(secs / 60.0).clamp(min=1.0) * valid
        out["target_items"] = targets
        out["target_weights"] = weights

        # Auxiliary next-token targets (horizon 1 only).
        nxt = targets[0]
        out["target_action_type"] = torch.zeros(L, dtype=torch.long)
        span = min(length, n - start - 1)
        out["target_action_type"][:span] = enc["action_type"][start + 1 : start + 1 + span]
        out["target_genre"] = self.catalog.genre[nxt]
        out["target_language"] = self.catalog.language[nxt]
        return out


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

