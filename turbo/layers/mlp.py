from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from turbo.model.config import ModelConfig
from turbo.utils.typing import Tensor3D


class DenseMLP(nn.Module):
    """SwiGLU FFN：out = down_proj(SiLU(gate_proj(x)) * up_proj(x))"""
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.head_dim, config.intermediate_size)
        self.up_proj = nn.Linear(config.head_dim, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.head_dim)

    @torch.compile
    def forward(self, x: Tensor3D) -> Tensor3D:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
