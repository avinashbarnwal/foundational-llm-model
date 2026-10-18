"""Typed configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    max_seq_len: int = 256
    sliding_window_stride: int = 128
    merge_gap_seconds: int = 1800
    min_watch_seconds: int = 60


@dataclass
class VocabConfig:
    num_items: int = 10000
    num_action_types: int = 8
    num_devices: int = 6
    num_locales: int = 32
    num_genres: int = 64
    num_languages: int = 40
    num_metadata_tags: int = 512


@dataclass
class ModelConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    attention_window: int = 64
    global_tokens: int = 4
    max_item_age_days: int = 365


@dataclass
class ObjectiveConfig:
    multi_token_horizon: int = 3
    aux_weights: dict[str, float] = field(
        default_factory=lambda: {"action_type": 0.2, "genre": 0.2, "language": 0.1}
    )
    weight_by_watch_time: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 2
    grad_clip: float = 1.0
    device: str = "cpu"
    seed: int = 7


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    vocab: VocabConfig = field(default_factory=VocabConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _build(cls: type, values: dict[str, Any] | None):
    values = values or {}
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML config; missing sections fall back to defaults."""
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    return config_from_dict(raw)


def config_from_dict(raw: dict[str, Any]) -> Config:
    return Config(
        data=_build(DataConfig, raw.get("data")),
        vocab=_build(VocabConfig, raw.get("vocab")),
        model=_build(ModelConfig, raw.get("model")),
        objective=_build(ObjectiveConfig, raw.get("objective")),
        training=_build(TrainingConfig, raw.get("training")),
    )
