from __future__ import annotations

import torch
import torch.nn.functional as F

from turbo.layers.base import BaseOp


class LinearLayer(BaseOp):
    def __init__(self, input_size: int, output_size: int, bias: bool = True):
        super().__init__()
        self.weight = torch.empty(output_size, input_size)
        if bias:
            self.bias = torch.empty(output_size)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is None:
            return F.linear(x, self.weight)
        else:
            return F.linear(x, self.weight, self.bias)
