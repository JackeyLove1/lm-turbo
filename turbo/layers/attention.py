from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # GQA groups

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * config.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * config.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * config.head_dim)

        self.emb = RotaryEmbedding(config)

        if config.use_qk_norm:
            self.q_norm = RMSNorm(config)
            self.k_norm = RMSNorm(config)
        else:
            self.q_norm = self.k_norm = None

        self.scale = config.head_dim ** -0.5

    def forward(
        self,
        hidden_states: Tensor3D,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tensor3D:

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

        q = self.emb.apply_rotary_pos_emb(q, position_ids)
        k = self.emb.apply_rotary_pos_emb(k, position_ids)

        # use grouped attention
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=(attention_mask is None),
        )

        out = out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
        return out
