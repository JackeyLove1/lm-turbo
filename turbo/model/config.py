from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal

import torch
from transformers import PretrainedConfig


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
    scaling: Dict[str, Any] | None = None


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
    architectures: List[str] = ["Qwen3ForCausalLM"]
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    initializer_range: int = 0.02
    max_position_embeddings: int = 40960
    max_window_layers: int = 28
    num_attention_heads: int = 16
    num_hidden_layers: int = 28
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-6
    rope_scaling: str | None = None
    rope_theta: float = 1000000.0
    sliding_window: str | None = None
    tie_word_embeddings: bool = True
    torch_dtype: torch.dtype = torch.bfloat16
    use_sliding_window: bool = False
    vocab_size: int = 151936
    use_qk_norm: bool = True


    @property
    def is_moe(self) -> bool:
        return self.model_type == "moe"

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "qwen3")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_token = getattr(config, "num_experts_per_token", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                scaling=getattr(config, "rope_scaling", None),
            ),
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
        )
