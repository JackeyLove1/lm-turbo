from __future__ import annotations

import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from turbo.config import HiddenAct, ModelConfig
from turbo.layers.activation import SiluAndMul
from turbo.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from turbo.utils.typing import Tensor3D


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

class DenseMLP(MLP):
    """SwiGLU FFN：out = down_proj(SiLU(gate_proj(x)) * up_proj(x))"""
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, x: Tensor3D) -> Tensor3D:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class ParallelMLP(MLP):
    """merged gate and up"""
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: HiddenAct,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False
        )
        self.act = SiluAndMul()

    def forward(self, x: Tensor3D) -> Tensor3D:
        gate_up = self.gate_up_proj(x)
        x = self.act(gate_up)
        x = self.down_proj(x)
        return x
