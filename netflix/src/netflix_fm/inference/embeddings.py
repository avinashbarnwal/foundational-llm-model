"""Exporting embeddings for downstream consumers, stable across retrains.

Downstream systems store foundation-model embeddings (as features, for candidate
generation, for similarity). Each retrain yields a new, arbitrarily rotated embedding
space, which would silently break those consumers. We align each new space to the
previous one with an orthogonal Procrustes rotation fitted on items present in both:
distances and dot products within the new space are unchanged, but vectors stay
comparable to the old ones.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from netflix_fm.data.dataset import InteractionWindowDataset, collate
from netflix_fm.model.foundation_model import FoundationModel


@torch.no_grad()
def export_item_embeddings(model: FoundationModel) -> torch.Tensor:
    model.eval()
    return model.item_table()[1:].cpu()  # drop the padding row; row i -> item id i + 1


@torch.no_grad()
def export_member_embeddings(model: FoundationModel, dataset: InteractionWindowDataset, batch_size: int = 64) -> torch.Tensor:
    """One vector per dataset window (the last window of a member is their current state)."""
    model.eval()
    device = model.catalog_age_days.device
    out = []
    for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=collate):
        batch = {k: v.to(device) for k, v in batch.items()}
        out.append(model.last_hidden(model.encode(batch), batch["padding_mask"]).cpu())
    return torch.cat(out)


def procrustes_rotation(new: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
    """Orthogonal R minimising ||new @ R - old||_F over rows present in both spaces."""
    u, _, vh = torch.linalg.svd(new.T.double() @ old.double())
    return (u @ vh).to(new.dtype)


def align_to_previous(new: torch.Tensor, old: torch.Tensor, shared_rows: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate `new` embeddings into `old`'s space. Returns (aligned, rotation).

    The first `shared_rows` rows (default: all rows of `old`) are assumed to be the same
    entities in both spaces; the rotation is then applied to every row of `new`, so
    newly added entities land in the old space as well.
    """
    n = shared_rows if shared_rows is not None else old.shape[0]
    rotation = procrustes_rotation(new[:n], old[:n])
    return new @ rotation, rotation
