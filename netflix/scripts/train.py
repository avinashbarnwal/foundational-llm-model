"""Train the foundation model on synthetic data (swap in real loaders for production).

    python scripts/train.py --config configs/default.yaml --out checkpoints/fm.pt
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import torch

from netflix_fm import load_config
from netflix_fm.data.dataset import InteractionWindowDataset
from netflix_fm.data.features import ItemCatalog
from netflix_fm.data.synthetic import make_catalog, make_histories
from netflix_fm.data.tokenizer import InteractionTokenizer
from netflix_fm.model.foundation_model import FoundationModel
from netflix_fm.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="checkpoints/fm.pt")
    parser.add_argument("--members", type=int, default=200)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(args.config)
    random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)

    items = make_catalog(cfg.vocab, seed=cfg.training.seed)
    catalog = ItemCatalog(items, cfg.vocab.num_items)
    histories = make_histories(cfg.vocab, items, num_members=args.members, seed=cfg.training.seed)
    tokenizer = InteractionTokenizer(cfg.data.merge_gap_seconds)
    sequences = [tokenizer.tokenize(h) for h in histories]
    raw, merged = sum(map(len, histories)), sum(map(len, sequences))
    logging.info("tokenized %d events into %d tokens (%.1fx shorter)", raw, merged, raw / max(merged, 1))

    split = int(0.9 * len(sequences))
    make_ds = lambda seqs: InteractionWindowDataset(
        seqs, catalog, cfg.data.max_seq_len, cfg.data.sliding_window_stride,
        cfg.objective.multi_token_horizon, cfg.data.min_watch_seconds,
    )
    train_ds, val_ds = make_ds(sequences[:split]), make_ds(sequences[split:])

    model = FoundationModel(cfg)
    reference_ts = max(t.timestamp for seq in sequences for t in seq)
    model.load_catalog(catalog, reference_ts)
    logging.info("model has %.2fM parameters", sum(p.numel() for p in model.parameters()) / 1e6)

    trainer = Trainer(model, cfg)
    trainer.fit(train_ds, val_ds)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(args.out)
    logging.info("saved %s", args.out)


if __name__ == "__main__":
    main()
