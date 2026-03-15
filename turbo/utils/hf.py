from __future__ import annotations

import functools
import os
from enum import Enum

from huggingface_hub import snapshot_download as hf_snapshot_download
from modelscope.hub.snapshot_download import snapshot_download as modelscope_snapshot_download
from tqdm.asyncio import tqdm
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase


class DownloadMethod(Enum):
    HF = "hf"
    MODELSCOPE = "modelscope"


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_path)


@functools.cache
def _load_hf_config(model_path: str) -> PretrainedConfig:
    return AutoConfig.from_pretrained(model_path)


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    config = _load_hf_config(model_path)
    return type(config)(**config.to_dict())


def download_hf_weight(model_path: str, method: DownloadMethod = DownloadMethod.MODELSCOPE) -> str:
    if os.path.isdir(model_path):
        return model_path
    try:
        match method:
            case DownloadMethod.HF:
                return hf_snapshot_download(model_path)
            case DownloadMethod.MODELSCOPE:
                return modelscope_snapshot_download(model_path)
            case _:
                raise ValueError(f"Invalid download method: {method}")
    except Exception as e:
        raise ValueError(f"Failed to download model from {method}: {e}")
