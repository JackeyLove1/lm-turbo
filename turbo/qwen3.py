from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from turbo.layers.attention import GroupedAttentionLayer
from turbo.layers.embedding import EmbeddingLayer, LMHeadLayer
from turbo.layers.mlp import DenseMLP
from turbo.layers.norm import RMSNorm
from turbo.model.config import ModelConfig
from turbo.utils.hf import (
    DownloadMethod,
    cached_load_hf_config,
    download_hf_weight,
    load_tokenizer,
)
from turbo.utils.typing import Tensor2D, Tensor3D


class Qwen3Decoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = GroupedAttentionLayer(config)
        self.mlp = DenseMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor3D,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tensor3D:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states

class Qwen3Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = EmbeddingLayer(config)
        self.layers = nn.ModuleList([Qwen3Decoder(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: Tensor2D,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tensor3D:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)

        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = LMHeadLayer(config)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight


    def forward(
        self,
        input_ids: Tensor2D,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tensor3D:
        hidden_states = self.model(input_ids, attention_mask, position_ids)
        return self.lm_head(hidden_states)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        generated = input_ids.clone()
        eos_token_id = self.config.eos_token_id if eos_token_id is None else eos_token_id

        for _ in range(max_new_tokens):
            logits = self.forward(generated)  # [batch, seq, vocab]
            next_logits = logits[:, -1, :]  # [batch, vocab]

            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                probs = F.softmax(next_logits, dim=-1)
                sorted_probs, sorted_ids = torch.sort(probs, descending=True)
                cumulative = sorted_probs.cumsum(dim=-1)
                mask = (cumulative - sorted_probs) > top_p
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_token = sorted_ids.gather(-1, torch.multinomial(sorted_probs, 1))

            generated = torch.cat([generated, next_token], dim=-1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

        return generated

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        device: str | torch.device = "cpu",
        download_method: DownloadMethod = DownloadMethod.HF,
    ) -> Qwen3ForCausalLM:
        local_model_path = download_hf_weight(model_name_or_path, method=download_method)
        config = ModelConfig.from_hf(cached_load_hf_config(local_model_path))
        model = cls(config)
        state_dict = load_hf_state_dict(local_model_path, device=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if missing or unexpected:
            raise RuntimeError(
                "Failed to load HuggingFace Qwen3 weights strictly. "
                f"Missing keys: {missing}. Unexpected keys: {unexpected}."
            )

        model.to(device)
        model.eval()
        return model


def remap_hf_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped_state_dict = dict(state_dict)

    for key in list(remapped_state_dict.keys()):
        if key.endswith("rotary_emb.inv_freq"):
            remapped_state_dict.pop(key)

    if "lm_head.weight" not in remapped_state_dict and "model.embed_tokens.weight" in remapped_state_dict:
        remapped_state_dict["lm_head.weight"] = remapped_state_dict["model.embed_tokens.weight"]

    return remapped_state_dict


def load_hf_state_dict(
    model_path: str,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    safetensor_paths = sorted(glob.glob(f"{model_path}/*.safetensors"))
    state_dict: dict[str, torch.Tensor] = {}

    if safetensor_paths:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise ImportError(
                "Loading HuggingFace Qwen3 safetensors requires `safetensors` to be installed."
            ) from exc

        for path in safetensor_paths:
            state_dict.update(load_file(path, device=str(device)))
        return remap_hf_state_dict(state_dict)

    bin_paths = sorted(glob.glob(f"{model_path}/pytorch_model*.bin"))
    if not bin_paths:
        raise FileNotFoundError(f"No HuggingFace weight files found under {model_path}.")

    for path in bin_paths:
        shard = torch.load(path, map_location=device)
        if not isinstance(shard, dict):
            raise TypeError(f"Unexpected weight shard type in {path}: {type(shard)!r}")
        state_dict.update(shard)

    return remap_hf_state_dict(state_dict)


def load_qwen3_from_hf(
    model_name_or_path: str,
    *,
    device: str | torch.device = "cpu",
    download_method: DownloadMethod = DownloadMethod.HF,
) -> Tuple[Qwen3ForCausalLM, object]:
    local_model_path = download_hf_weight(model_name_or_path, method=download_method)
    model = Qwen3ForCausalLM.from_pretrained(
        local_model_path,
        device=device,
        download_method=download_method,
    )
    tokenizer = load_tokenizer(local_model_path)
    return model, tokenizer


def load_model_config_from_json(config_path: str | Path) -> ModelConfig:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        config_dict = json.load(f)
    return ModelConfig.from_hf(PretrainedConfig.from_dict(config_dict))


def print_model_layer(model: nn.Module) -> None:
    print("== Module names ==")
    for name, _ in model.named_parameters():
        if name:
            print(name)

    print("=" * 50)
    print(model.state_dict().keys())

    print("=" * 50)
    print("\n== State dict keys (turbo -> hf) ==")
    for name, tensor in model.state_dict().items():
        print(f"{name}: {tuple(tensor.shape)} -> {name}")

def load_model_from_hf(
    model_name_or_path: str,
    device: str | torch.device = "cuda",
    *,
    download_method: DownloadMethod = DownloadMethod.HF,
) -> Qwen3ForCausalLM:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; load_model_from_hf requires GPU.")
    return Qwen3ForCausalLM.from_pretrained(
        model_name_or_path,
        device=device,
        download_method=download_method,
    )

def run_gpu_test(model_path: str = "models/Qwen/Qwen3-0.6B", max_new_tokens: int = 16) -> None:
    """Load Qwen3 from local path and run a short forward + generate test on GPU."""
    device = torch.device("cuda")
    print(f"Loading model from {model_path} on {device}...")
    model = load_model_from_hf(model_path, device=device)
    tokenizer = load_tokenizer(model_path)

    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    print(f"Input: {prompt!r} (shape {inputs.shape})")

    with torch.no_grad():
        logits = model(inputs)
    print(f"Forward OK: logits shape {logits.shape}")

    generated = model.generate(inputs, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9)
    out_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Generated ({max_new_tokens} new tokens): {out_text!r}")
    print("GPU test passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print Qwen3 module names and state_dict keys.")
    parser.add_argument(
        "config_path",
        nargs="?",
        default="models/Qwen/Qwen3-0.6B/config.json",
        help="Path to a Hugging Face config.json file.",
    )
    parser.add_argument(
        "--test-gpu",
        action="store_true",
        help="Load model from models/Qwen/Qwen3-0.6B and run a GPU forward+generate test.",
    )
    parser.add_argument(
        "--model-path",
        default="models/Qwen/Qwen3-0.6B",
        help="Model path for --test-gpu (default: models/Qwen/Qwen3-0.6B).",
    )
    args = parser.parse_args()

    if args.test_gpu:
        run_gpu_test(model_path=args.model_path)
        sys.exit(0)

    config = load_model_config_from_json(args.config_path)
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)

    print(f"Loaded config from: {Path(args.config_path).resolve()}")
    print(
        "HF-aligned naming check: turbo model state_dict keys should match HF keys for the same tensors."
    )
    print_model_layer(model)
