from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from turbo.model.config import ModelConfig
from turbo.utils.typing import Tensor2D, Tensor3D


class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.vocab_size, config.hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=config.initializer_range)

    def forward(self, x: Tensor2D) -> Tensor3D:
        return F.embedding(x, self.weight)

class LMHeadLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.vocab_size, config.hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=config.initializer_range)

    def forward(self, x: Tensor3D) -> Tensor3D:
        return F.linear(x, self.weight)
