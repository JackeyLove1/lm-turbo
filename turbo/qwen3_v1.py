"""
uv run python -m turbo.qwen3_v1
"""
import os

import torch
import torch.distributed as dist
import torch.nn as nn

from turbo.config import ModelConfig
from turbo.layers.attention import Attention, AttentionLayer
from turbo.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from turbo.layers.linear import QKVParallelLinear, RowParallelLinear
from turbo.layers.mlp import ParallelMLP
from turbo.layers.norm import RMSNorm
from turbo.layers.position import get_rope
from turbo.utils.loader import load_model, load_model_config_from_json, load_tokenizer
from turbo.utils.torch_utils import torch_dtype as use_torch_dtype


class Qwen3Attention(AttentionLayer):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        page_size: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ):
        super().__init__()
        self.total_num_heads = num_heads
        assert self.total_num_heads % self.tp_size == 0
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % self.tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            page_size,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv: torch.Tensor = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o: torch.Tensor = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output

class Qwen3DecoderLayer(nn.Module):
    def __init__(
            self,
            config: ModelConfig,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            page_size=config.kvcache_block_size,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=None,
        )
        self.mlp = ParallelMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

class Qwen3Model(nn.Module):
    def __init__(
            self,
            config: ModelConfig,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for layer_id in range(config.num_hidden_layers):
            layer = Qwen3DecoderLayer(config)
            # Attach ids for per-layer NVTX annotation (optional).
            layer.layer_id = layer_id
            layer.self_attn.layer_id = layer_id
            layer.self_attn.attn.layer_id = layer_id
            layer.mlp.layer_id = layer_id
            self.layers.append(layer)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: ModelConfig
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)

    @staticmethod
    def build_position_ids(input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 1:
            raise ValueError(
                f"Qwen3 v1 expects flattened token ids for forward(), got shape {tuple(input_ids.shape)}."
            )
        return torch.arange(input_ids.shape[0], device=input_ids.device, dtype=torch.long)


def init_dist_for_inference(
    rank: int = 0,
    world_size: int = 1,
    init_method: str = "tcp://127.0.0.1:2333",
) -> bool:
    if dist.is_available() and dist.is_initialized():
        return False

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    return True


def destroy_dist_if_needed(initialized_by_me: bool) -> None:
    if initialized_by_me and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Print Qwen3 module names and state_dict keys.")
    parser.add_argument(
        "config_path",
        nargs="?",
        default="models/Qwen/Qwen3-0.6B/config.json",
        help="Path to a Hugging Face config.json file.",
    )
    parser.add_argument(
        "--model-path",
        default="models/Qwen/Qwen3-0.6B",
        help="Model path for --test-gpu (default: models/Qwen/Qwen3-0.6B).",
    )
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    parser.add_argument(
        "--dist-init-method",
        default=os.environ.get("DIST_INIT_METHOD", "tcp://127.0.0.1:2333"),
        help="Torch distributed init_method used when creating the process group.",
    )
    args = parser.parse_args()
    config = load_model_config_from_json(args.config_path)
    initialized_dist = init_dist_for_inference(
        rank=args.rank,
        world_size=args.world_size,
        init_method=args.dist_init_method,
    )

    try:
        with use_torch_dtype(config.torch_dtype):
            model = Qwen3ForCausalLM(config)
        tokenizer = load_tokenizer(args.model_path)
        load_model(model, args.model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device=device, dtype=config.torch_dtype)
        model.eval()

        prompt = "写一段python快速排序的代码"
        inputs = tokenizer(prompt, return_tensors="pt").input_ids[0].to(device)
        print(f"Input: {prompt!r} (shape {inputs.shape})")

        with torch.no_grad():
            hidden_states = model(inputs, model.build_position_ids(inputs))
            logits = model.compute_logits(hidden_states)
        print(f"Forward OK: logits shape {logits.shape}")
    finally:
        destroy_dist_if_needed(initialized_dist)

