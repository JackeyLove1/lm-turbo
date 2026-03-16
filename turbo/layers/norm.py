import torch
import torch.nn as nn

from turbo.utils.typing import AttentionTensor, Tensor3D


# torch native rms norm version
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor3D | AttentionTensor) -> Tensor3D | AttentionTensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(norm + self.eps)
        return (x_normed * self.weight).to(x.dtype)

