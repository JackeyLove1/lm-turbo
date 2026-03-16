import functools
from typing import Tuple

import torch
import torch.nn as nn

from turbo.model.config import ModelConfig
from turbo.utils.typing import AttentionTensor, Tensor2D


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_position=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('cos', torch.cos(freqs))
        self.register_buffer('sin', torch.sin(freqs))

    def forward(self, x):
        # x: (batch, seq_len, num_heads, head_dim)
        cos = self.cos[:x.shape[1]].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:x.shape[1]].unsqueeze(0).unsqueeze(0)
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]
        return torch.cat([x_even * cos - x_odd * sin,
                          x_even * sin + x_odd * cos], dim=-1)

@functools.cache
def get_rope_cache(inv_freq: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    t = torch.arange(seq_len, device=inv_freq.device).float()
    freqs = torch.outer(t, inv_freq)  # [seq_len, head_dim / 2]
    cos_cached = freqs.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim / 2]
    sin_cached = freqs.sin().unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim / 2]
    return cos_cached, sin_cached

class RotaryEmbedding(nn.Module):
    """RoPE 旋转位置编码"""

    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()
        if config.head_dim % 2 != 0:
            raise ValueError(f"RoPE head dimension must be even, got {config.head_dim}.")

        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.cos_cached, self.sin_cached = self._build_cache(config.max_position_embeddings)

    def _build_cache(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.outer(
            torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype),
            self.inv_freq,
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos().unsqueeze(0).unsqueeze(0)
        sin_cached = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos_cached, sin_cached

    @staticmethod
    def rotate_half(x: AttentionTensor) -> AttentionTensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(
        self,
        x: AttentionTensor,
        position_ids: Tensor2D | None = None,
    ) -> AttentionTensor:
        seq_len = x.shape[-2]
        cache_seq_len = seq_len if position_ids is None else int(position_ids.max().item()) + 1

        if cache_seq_len > self.cos_cached.shape[2]:
            self.cos_cached, self.sin_cached = self._build_cache(cache_seq_len)

        if position_ids is None:
            cos = self.cos_cached[:, :, :seq_len, :]
            sin = self.sin_cached[:, :, :seq_len, :]
        else:
            cos = self.cos_cached[0, 0, position_ids, :].unsqueeze(1)
            sin = self.sin_cached[0, 0, position_ids, :].unsqueeze(1)

        cos = cos.to(device=x.device, dtype=x.dtype)
        sin = sin.to(device=x.device, dtype=x.dtype)
        return x * cos + self.rotate_half(x) * sin

    def forward(
        self,
        x: AttentionTensor,
        position_ids: Tensor2D | None = None,
    ) -> AttentionTensor:
        return self.apply_rotary_pos_emb(x, position_ids)
