from typing import Optional, Tuple

import flashinfer
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from turbo.config import ModelConfig
from turbo.engine.core import get_vllm_context
from turbo.kvcache.base import KVCache
from turbo.layers.norm import RMSNorm
from turbo.layers.position import RotaryEmbedding
from turbo.utils.typing import Tensor3D


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
        page_size: int = 32,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.page_size = page_size
        self.k_cache = self.v_cache = torch.tensor([])
        self.register_buffer("_flashinfer_workspace", torch.empty(0, dtype=torch.uint8), persistent=False)

    def _get_workspace(self, device: torch.device) -> torch.Tensor:
        workspace_size = 128 * 1024 * 1024
        workspace = self._flashinfer_workspace
        if workspace.device != device or workspace.numel() < workspace_size:
            workspace = torch.zeros(workspace_size, dtype=torch.uint8, device=device)
            self._flashinfer_workspace = workspace
        return workspace

    def _view_cache_as_paged(self, cache: torch.Tensor) -> torch.Tensor:
        if cache.ndim == 4:
            return cache
        if cache.ndim != 2:
            raise ValueError(f"Unsupported KV cache shape for flashinfer: {tuple(cache.shape)}")
        if cache.shape[1] != self.num_kv_heads * self.head_dim:
            raise ValueError(
                f"KV cache hidden size mismatch, expected {self.num_kv_heads * self.head_dim}, "
                f"got {cache.shape[1]}"
            )
        if cache.shape[0] % self.page_size != 0:
            raise ValueError(
                f"KV cache slot count {cache.shape[0]} is not divisible by page size {self.page_size}"
            )
        return cache.view(-1, self.page_size, self.num_kv_heads, self.head_dim)

    def _prefill_without_prefix_cache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context,
    ) -> torch.Tensor:
        if context.cu_seqlens_q is None or context.cu_seqlens_k is None:
            cu_seqlens_q = [0, q.shape[0]]
            cu_seqlens_k = [0, k.shape[0]]
        else:
            cu_seqlens_q = context.cu_seqlens_q.tolist()
            cu_seqlens_k = context.cu_seqlens_k.tolist()
        outputs = []
        for q_start, q_end, k_start, k_end in zip(
            cu_seqlens_q[:-1],
            cu_seqlens_q[1:],
            cu_seqlens_k[:-1],
            cu_seqlens_k[1:],
        ):
            outputs.append(
                flashinfer.single_prefill_with_kv_cache(
                    q[q_start:q_end],
                    k[k_start:k_end],
                    v[k_start:k_end],
                    causal=True,
                    kv_layout="NHD",
                    sm_scale=self.scale,
                )
            )
        return torch.cat(outputs, dim=0)

    @staticmethod
    def _get_max_seqlen(context, q: bool) -> int:
        preferred = "max_seqlen_q" if q else "max_seqlen_k"
        fallback = "max_seql_q" if q else "max_seql_k"
        value = getattr(context, preferred, None)
        if value is None:
            value = getattr(context, fallback)
        return int(value)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_vllm_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        # Standalone eager forwards may not populate the scheduler context.
        is_standalone_prefill = (
            not context.is_prefill
            and context.block_tables is None
            and not (k_cache.numel() and v_cache.numel())
            and q.shape[0] > 1
        )
        if context.is_prefill or is_standalone_prefill:
            if context.block_tables is None:
                o = self._prefill_without_prefix_cache(q, k, v, context)
            else:    # prefix cache / paged kv cache
                workspace = self._get_workspace(q.device)
                paged_k_cache = self._view_cache_as_paged(k_cache)
                paged_v_cache = self._view_cache_as_paged(v_cache)
                seq_lens = (context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]).to(torch.int32)
                o = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
                    query=q,
                    kv_cache=(paged_k_cache, paged_v_cache),
                    workspace_buffer=workspace,
                    block_tables=context.block_tables.to(torch.int32),
                    seq_lens=seq_lens,
                    max_q_len=self._get_max_seqlen(context, q=True),
                    max_kv_len=self._get_max_seqlen(context, q=False),
                    bmm1_scale=self.scale,
                    bmm2_scale=1.0,
                    batch_size=context.block_tables.shape[0],
                    cum_seq_lens_q=context.cu_seqlens_q.to(torch.int32),
                    cum_seq_lens_kv=context.cu_seqlens_k.to(torch.int32),
                    kv_layout="NHD",
                )
        else:    # decode
            if context.block_tables is None:
                if q.shape[0] != 1:
                    raise ValueError("Decode without block tables is only supported for batch size 1")
                seq_len = int(context.context_lens[0].item())
                flat_k_cache = k_cache.view(-1, self.num_kv_heads, self.head_dim)
                flat_v_cache = v_cache.view(-1, self.num_kv_heads, self.head_dim)
                o = flashinfer.single_decode_with_kv_cache(
                    q[0],
                    flat_k_cache[:seq_len],
                    flat_v_cache[:seq_len],
                    kv_layout="NHD",
                    sm_scale=self.scale,
                ).unsqueeze(0)
            else:
                workspace = self._get_workspace(q.device)
                paged_k_cache = self._view_cache_as_paged(k_cache)
                paged_v_cache = self._view_cache_as_paged(v_cache)
                seq_lens = context.context_lens.to(torch.int32)
                o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                    query=q,
                    kv_cache=(paged_k_cache, paged_v_cache),
                    workspace_buffer=workspace,
                    block_tables=context.block_tables.to(torch.int32),
                    seq_lens=seq_lens,
                    max_seq_len=int(seq_lens.max().item()),
                    bmm1_scale=self.scale,
                    bmm2_scale=1.0,
                    kv_layout="NHD",
                    q_len_per_req=1,
                )
        return o
