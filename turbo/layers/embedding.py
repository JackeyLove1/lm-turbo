from __future__ import annotations

import torch
import torch.nn.functional as F

from turbo.layers.base import BaseOp


class VocabEmbeddingLayer(BaseOp):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.weight = torch.empty(vocab_size, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)

class LMHeadLayer(BaseOp):
    def __init__(self, embedding_dim: int, vocab_size: int):
        super().__init__()
        self.weight = torch.empty(vocab_size, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)
