import json
import os
from glob import glob
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase

from turbo.config import ModelConfig
from turbo.utils.hf import (
    DownloadMethod,
    load_tokenizer,
)


def load_model_config_from_json(config_path: str | Path) -> ModelConfig:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        config_dict = json.load(f)
    return ModelConfig.from_hf(PretrainedConfig.from_dict(config_dict))

def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_path)

def load_model_from_hf(
    model_name_or_path: str,
    device: str | torch.device = "cuda",
    *,
    download_method: DownloadMethod = DownloadMethod.HF,
) -> nn.Module:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; load_model_from_hf requires GPU.")
    return load_model(model_name_or_path, device, download_method)


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)

def load_model(model: nn.Module, model_path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(model_path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))

