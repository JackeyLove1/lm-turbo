from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from turbo.config import ModelConfig
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

class Qwen3ParallelMLP(MLP):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
