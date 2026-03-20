from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from turbo.kvcache.base import KVCache
from turbo.layers.norm import RMSNorm
from turbo.layers.position import RotaryEmbedding
from turbo.config import ModelConfig
from turbo.utils.typing import Tensor3D
from turbo.engine.core import get_vllm_context

class AttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()


class MultiHeadAttention(AttentionLayer):
    pass

class GroupedAttentionLayer(AttentionLayer):
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
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = True,
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

        had_kv_cache = kv_cache is not None

        if kv_cache is None: # prefill
            kv_cache = KVCache(k=k, v=v)
        else: # decode
            k, v = kv_cache.update(k, v)

        attn_mask = attention_mask
        is_causal = False
        if attn_mask is None:
            # Only the prefill path needs a causal mask.
            # In decode with KV cache, q_len is usually 1 while k_len keeps growing.
            # Passing is_causal=True there masks against the wrong key positions and
            # breaks generation after the first new token.
            is_causal = not had_kv_cache

        # use grouped attention
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
        return self.o_proj(out), kv_cache if use_cache else None

@triton.jit
def _store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr
):
    idx = tl.program_id(axis=0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

def store_kvcache(
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    _store_kvcache_kernel[(N, )](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(AttentionLayer):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_vllm_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
