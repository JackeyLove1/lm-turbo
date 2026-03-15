from __future__ import annotations

import torch.nn as nn

from turbo.model.config import ModelConfig
from turbo.utils.typing import Tensor2D, Tensor3D


class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.head_dim)

    def forward(self, x: Tensor2D) -> Tensor3D:
        return self.embedding(x)

class LMHeadLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(config.head_dim, config.vocab_size)

    def forward(self, x: Tensor3D) -> Tensor2D:
        return self.linear(x)
