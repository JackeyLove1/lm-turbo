from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import torch
from transformers import AutoConfig, PretrainedConfig


class HiddenAct(Enum):
    GELU = "gelu"
    RELU = "relu"
    SILU = "silu"


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float = 10000.0
    scaling: dict[str, Any] | None = None


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    rotary_config: RotaryConfig
    hidden_act: HiddenAct
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_token: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: Literal["moe", "qwen3", "llama"]
    architectures: list[str] = field(default_factory=lambda: ["Qwen3ForCausalLM"])
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    initializer_range: int = 0.02
    max_window_layers: int = 28
    rms_norm_eps: float = 1e-6
    sliding_window: str | None = None
    torch_dtype: torch.dtype = torch.bfloat16
    use_sliding_window: bool = False
    vocab_size: int = 151936
    use_qk_norm: bool = True

    # hf config
    hf_config: AutoConfig | None = None

    # kvcache config
    kvcache_block_size: int = 32
    num_kvcache_blocks: int = -1

    # scheduler config
    max_num_seqs: int = 512
    max_num_batched_tokens: int = 32 * max_num_seqs
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    eos: int = -1


    @property
    def is_moe(self) -> bool:
        return self.model_type == "moe"

    @property
    def num_attention_heads(self) -> int:
        return self.num_qo_heads

    @property
    def num_hidden_layers(self) -> int:
        return self.num_layers

    @property
    def num_key_value_heads(self) -> int:
        return self.num_kv_heads

    @property
    def max_position_embeddings(self) -> int:
        return self.rotary_config.max_position

    @property
    def rope_theta(self) -> float:
        return self.rotary_config.base

    @property
    def rope_scaling(self) -> dict[str, Any] | None:
        return self.rotary_config.scaling

    def __post__init__(self):
        pass

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        model_type = getattr(config, "model_type", "qwen3")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_token = getattr(config, "num_experts_per_token", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = list(getattr(config, "architectures", ["Qwen3ForCausalLM"]))
        hidden_act = getattr(config, "hidden_act", HiddenAct.SILU.value)
        if not isinstance(hidden_act, HiddenAct):
            hidden_act = HiddenAct(str(hidden_act))
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_parameters = getattr(config, "rope_parameters", None) or {}
            rope_theta = rope_parameters.get("rope_theta", 10000.0)

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=getattr(config, "rope_scaling", None),
            ),
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
            attention_bias=getattr(config, "attention_bias", False),
            attention_dropout=getattr(config, "attention_dropout", 0.0),
            mlp_bias=getattr(config, "mlp_bias", False),
            bos_token_id=getattr(config, "bos_token_id", 151643),
            eos_token_id=getattr(config, "eos_token_id", 151645),
            initializer_range=getattr(config, "initializer_range", 0.02),
            max_window_layers=getattr(config, "max_window_layers", config.num_hidden_layers),
            sliding_window=getattr(config, "sliding_window", None),
            torch_dtype=getattr(config, "torch_dtype", torch.bfloat16),
            use_sliding_window=getattr(config, "use_sliding_window", False),
            use_qk_norm=getattr(config, "use_qk_norm", True),
        )
