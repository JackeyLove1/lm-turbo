from dataclasses import dataclass
from typing import Tuple

import torch

from turbo.utils.typing import AttentionTensor


@dataclass
class KVCache:
    k: AttentionTensor # [B H S D]
    v: AttentionTensor # [B H S D]

    def update(self, new_k: AttentionTensor, new_v: AttentionTensor) -> Tuple[AttentionTensor, AttentionTensor]:
        self.k = torch.cat([self.k, new_k], dim=2) # cat tensor in seq dimension
        self.v = torch.cat([self.v, new_v], dim=2) # cat tensor in seq dimension
        return self.k, self.v

    @property
    def seq_len(self) -> int:
        assert self.k.shape[2] == self.v.shape[2], "k and v must have the same sequence length"
        return self.k.shape[2]

    @property
    def head_dim(self) -> int:
        assert self.k.shape[-1] == self.v.shape[-1], "k and v must have the same head dimension"
        return self.k.shape[-1]

    @property
    def num_heads(self) -> int:
        assert self.k.shape[1] == self.v.shape[1], "k and v must have the same number of heads"
        return self.k.shape[1]
