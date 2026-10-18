"""Sparse causal self-attention with rotary positions and a bounded KV cache.

Sparsity pattern: each query attends to
  * keys within a local window of `window` previous positions, and
  * the first `global_tokens` positions of the sequence (a compact long-range summary).
This keeps attention cost roughly linear in sequence length, so the model can take
longer histories than dense attention would allow.

At inference, keys outside that pattern can never be attended to again, so the cache
evicts them: memory per member is bounded by `window + global_tokens` regardless of
history length.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


def sparse_causal_mask(
    q_pos: torch.Tensor, k_pos: torch.Tensor, window: int, global_tokens: int
) -> torch.Tensor:
    """Boolean [Lq, Lk] mask, True where attention is allowed."""
    diff = q_pos[:, None] - k_pos[None, :]
    causal = diff >= 0
    local = diff < window
    is_global = (k_pos < global_tokens)[None, :]
    return causal & (local | is_global)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: [B, H, L, Dh], positions: [L]
        freqs = positions.float()[:, None] * self.inv_freq[None, :]
        cos, sin = freqs.cos(), freqs.sin()
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.flatten(-2)


@dataclass
class LayerKVCache:
    keys: torch.Tensor | None = None       # [B, H, Lc, Dh]
    values: torch.Tensor | None = None
    positions: torch.Tensor | None = None  # [Lc]

    def append(self, k: torch.Tensor, v: torch.Tensor, pos: torch.Tensor) -> None:
        if self.keys is None:
            self.keys, self.values, self.positions = k, v, pos
        else:
            self.keys = torch.cat([self.keys, k], dim=2)
            self.values = torch.cat([self.values, v], dim=2)
            self.positions = torch.cat([self.positions, pos])

    def evict(self, next_pos: int, window: int, global_tokens: int) -> None:
        """Drop keys no future query (position >= next_pos) can attend to."""
        keep = (self.positions < global_tokens) | (self.positions > next_pos - window)
        self.keys, self.values, self.positions = self.keys[:, :, keep], self.values[:, :, keep], self.positions[keep]


@dataclass
class KVCache:
    layers: list[LayerKVCache] = field(default_factory=list)
    next_position: int = 0

    @classmethod
    def empty(cls, n_layers: int) -> "KVCache":
        return cls(layers=[LayerKVCache() for _ in range(n_layers)])


class SparseSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, window: int, global_tokens: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.global_tokens = global_tokens
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.rope(q, positions), self.rope(k, positions)

        k_pos = positions
        if cache is not None:
            cache.append(k, v, positions)
            k, v, k_pos = cache.keys, cache.values, cache.positions

        # Sequences are right-padded, so causality alone keeps real queries off padding keys.
        mask = sparse_causal_mask(positions, k_pos, self.window, self.global_tokens)[None, None]
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out(attn.transpose(1, 2).reshape(B, L, D))
