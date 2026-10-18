"""Pre-norm decoder-only transformer over interaction tokens."""

from __future__ import annotations

import torch
from torch import nn

from netflix_fm.config import ModelConfig
from netflix_fm.model.attention import KVCache, SparseSelfAttention


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.attn_norm = nn.LayerNorm(d)
        self.attn = SparseSelfAttention(d, cfg.n_heads, cfg.attention_window, cfg.global_tokens, cfg.dropout)
        self.mlp_norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, positions, cache=None):
        x = x + self.dropout(self.attn(self.attn_norm(x), positions, cache))
        return x + self.dropout(self.mlp(self.mlp_norm(x)))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        start = cache.next_position if cache is not None else 0
        positions = torch.arange(start, start + x.shape[1], device=x.device)
        for i, block in enumerate(self.blocks):
            x = block(x, positions, cache.layers[i] if cache is not None else None)
        if cache is not None:
            cache.next_position = start + x.shape[1]
            for layer in cache.layers:
                layer.evict(cache.next_position, self.cfg.attention_window, self.cfg.global_tokens)
        return self.final_norm(x)
