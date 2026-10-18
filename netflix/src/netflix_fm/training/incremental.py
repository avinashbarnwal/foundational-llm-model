"""Incremental training: warm-start a new model from the previous one.

Retraining from scratch every time the catalog changes is expensive. Instead the new
model inherits all weights from the last checkpoint. Item rows that already existed are
copied; rows for newly launched titles are initialised from existing titles that share
metadata (same genre/language), plus a little noise to break symmetry. Any title with no
such neighbour falls back to the mean embedding.
"""

from __future__ import annotations

import torch

from netflix_fm.config import Config
from netflix_fm.data.features import ItemCatalog
from netflix_fm.model.foundation_model import FoundationModel


@torch.no_grad()
def init_new_item_rows(
    table: torch.Tensor,
    catalog: ItemCatalog,
    old_num_items: int,
    noise_std: float = 0.01,
    generator: torch.Generator | None = None,
) -> None:
    """Fill rows old_num_items+1 .. end of `table` in place."""
    known = torch.arange(1, old_num_items + 1)
    fallback = table[known].mean(dim=0)
    for item_id in range(old_num_items + 1, table.shape[0]):
        similar = known[
            (catalog.genre[known] == catalog.genre[item_id])
            & (catalog.language[known] == catalog.language[item_id])
        ]
        if similar.numel() == 0:
            similar = known[catalog.genre[known] == catalog.genre[item_id]]
        base = table[similar].mean(dim=0) if similar.numel() else fallback
        noise = torch.randn(base.shape, generator=generator) * noise_std
        table[item_id] = base + noise


@torch.no_grad()
def warm_start(old: FoundationModel, new_cfg: Config, catalog: ItemCatalog, reference_timestamp: int) -> FoundationModel:
    """Build a model for `new_cfg` (possibly larger catalog) initialised from `old`."""
    new = FoundationModel(new_cfg)
    new.load_catalog(catalog, reference_timestamp)
    old_state = old.state_dict()
    new_state = new.state_dict()
    old_items = old.cfg.vocab.num_items

    for name, tensor in new_state.items():
        if name not in old_state or name.startswith("embed.item.item_") or name == "catalog_age_days":
            continue  # catalog buffers come from the new catalog
        src = old_state[name]
        if src.shape == tensor.shape:
            tensor.copy_(src)
        elif src.shape[1:] == tensor.shape[1:] and src.shape[0] < tensor.shape[0]:
            tensor[: src.shape[0]].copy_(src)  # grown item table
        else:
            raise ValueError(f"Cannot warm-start {name}: {tuple(src.shape)} -> {tuple(tensor.shape)}")
    new.load_state_dict(new_state)

    if new_cfg.vocab.num_items > old_items:
        init_new_item_rows(new.embed.item.id_embedding.weight, catalog, old_items)
    return new
