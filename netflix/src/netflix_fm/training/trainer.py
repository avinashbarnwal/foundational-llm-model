"""Minimal training loop."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from netflix_fm.config import Config, config_from_dict
from netflix_fm.data.dataset import InteractionWindowDataset, collate
from netflix_fm.model.foundation_model import FoundationModel
from netflix_fm.training.losses import compute_loss

log = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model: FoundationModel, cfg: Config) -> None:
        self.model = model.to(cfg.training.device)
        self.cfg = cfg
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )

    def _to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.cfg.training.device) for k, v in batch.items()}

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        for step, batch in enumerate(loader, start=1):
            batch = self._to_device(batch)
            loss, metrics = compute_loss(self.model(batch), batch, self.cfg.objective)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
            self.optimizer.step()
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v
        return {k: v / max(step, 1) for k, v in totals.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}
        n = 0
        for batch in loader:
            batch = self._to_device(batch)
            _, metrics = compute_loss(self.model(batch), batch, self.cfg.objective)
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1
        return {k: v / max(n, 1) for k, v in totals.items()}

    def fit(self, train: InteractionWindowDataset, val: InteractionWindowDataset | None = None) -> None:
        bs = self.cfg.training.batch_size
        train_loader = DataLoader(train, batch_size=bs, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val, batch_size=bs, collate_fn=collate) if val is not None else None
        for epoch in range(1, self.cfg.training.epochs + 1):
            metrics = self.train_epoch(train_loader)
            log.info("epoch %d train %s", epoch, _fmt(metrics))
            if val_loader is not None:
                log.info("epoch %d val   %s", epoch, _fmt(self.evaluate(val_loader)))

    def save(self, path: str | Path) -> None:
        torch.save({"model": self.model.state_dict(), "config": asdict(self.cfg)}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> FoundationModel:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = FoundationModel(config_from_dict(ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device)


def _fmt(metrics: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
