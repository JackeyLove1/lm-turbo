from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional
from contextlib import contextmanager
import torch


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: float = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 2048

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # cpu tensor
    uid: int
    cached_len: int
    output_len: int
    sampling_params: SamplingParams

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)
        self.max_device_len = self.device_len + self.output_len
        assert 0 <= self.cached_len <= self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1

    def __repr__(self) -> str:
        return (
            f"Req(uid={self.uid}, cached_len={self.cached_len}, "
            f"device_len={self.device_len}, max_device_len={self.max_device_len})"
        )


@dataclass(eq=False)
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    input_ids: torch.Tensor
    # these fileds should set ny schedule
    input_ids: torch.Tensor = field(init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)


@dataclass
class Context:
    # sglang mode
    page_size: int
    page_table: torch.Tensor = field(init=False)
    kv_cache: torch.Tensor = field(init=False)
    _batch: Batch | None = None = field(init=False)

    # vllm mode
    is_prefill: bool = False
    cu_seqlens_q: Optional[int] = None
    cu_seqlens_k: Optional[int] = None
    max_seql_q: Optional[int] = None
    max_seql_k: Optional[int] = None
    slot_mapping: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch
    
    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "Cannot nest forward_batch calls, batch is None"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None

_GLOBAL_CTX: Context | None = None

def set_global_ctx(ctx: Context) -> None:
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx

def get_global_ctx() -> Context:
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX

def clear_global_ctx() -> None:
    global _GLOBAL_CTX
    _GLOBAL_CTX = None

_CONTEXT = Context()

def get_vllm_context():
    return _CONTEXT

def set_vllm_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_vllm_context():
    global _CONTEXT
    _CONTEXT = Context()
