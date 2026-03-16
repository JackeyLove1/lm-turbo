from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from turbo.kvcache.base import KVCache
from turbo.layers.norm import RMSNorm
from turbo.layers.position import RotaryEmbedding
from turbo.model.config import ModelConfig
from turbo.utils.typing import Tensor3D


class AttentionLayer(nn.Module):
    pass

class MultiHeadAttention(nn.Module):
    pass

class GroupedAttentionLayer(nn.Module):
    def __init__(
        self,
       config: ModelConfig
    ):
        super().__init__()
        self.num_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # GQA groups
        self.attention_dropout = config.attention_dropout

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_qo_heads must be divisible by num_kv_heads for grouped attention")

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * config.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * config.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * config.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.rotary_emb = RotaryEmbedding(config)

        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = self.k_norm = None

    def forward(
        self,
        hidden_states: Tensor3D,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None
    ) -> Tuple[Tensor3D, Optional[KVCache]]:

        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states) # [B, S, num_heads * head_dim]
        k = self.k_proj(hidden_states) # [B, S, num_kv_heads * head_dim]
        v = self.v_proj(hidden_states) # [B, S, num_kv_heads * head_dim]

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, S, head_dim]
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2) # [B, num_kv_heads, S, head_dim]
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2) # [B, num_kv_heads, S, head_dim]

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = self.rotary_emb(q, position_ids)
        k = self.rotary_emb(k, position_ids)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)
            is_casual = False
        else:
            is_casual = (attention_mask is None)

        # use grouped attention
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=is_casual,
        )

        out = out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
        return self.o_proj(out)
