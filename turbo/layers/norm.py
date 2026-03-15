import torch
import torch.nn as nn

from turbo.model.config import ModelConfig
from turbo.utils.typing import Tensor3D


# torch native rms norm version
class RMSNorm(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.head_dim))
        self.eps = config.rms_norm_eps

    @torch.compile
    def forward(self, x: Tensor3D) -> Tensor3D:
        # x: [batch, seq, dim]
        old_dtype = x.dtype
        norm = x.float().pow(dim=-1).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(norm + self.eps))
        x = x.mul_(self.weight)
        return x.to(old_dtype)

