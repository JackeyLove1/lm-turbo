from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from turbo.engine.core import get_vllm_context
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
        # nn.init.normal_(self.weight, mean=0.0, std=config.initializer_range)

    def forward(self, x: Tensor3D) -> Tensor3D:
        return F.linear(x, self.weight)



class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()
        assert num_embeddings % self.tp_size == 0, "num_embeddings must be divisible by tp_size"
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.tp_rank * self.num_embeddings_per_partition
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, self.embedding_dim))
        self.weight.weight_loader = self.weight_loader


    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param_data = param.data
        shared_size = param_data.shape[0]
        start_index = self.tp_rank * shared_size
        loaded_weight = loaded_weight.narrow(dim=0, start=start_index, length=shared_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N]"""
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx) # [N]
            x = mask * (x - self.vocab_start_idx) # [N]
        y = F.embedding(x, self.weight) # [N, dim]
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y # [N, dim]
            dist.all_reduce(y)
        return y

class ParallelLMHead(VocabParallelEmbedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = get_vllm_context()
        if context.is_prefill:
            indices = context.cu_seqlens_q[1:] - 1
            x = x[indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logist = [logits for _ in range(self.tp_size)]
            dist.gather(tensor=logits, gather_list=all_logist, dst=0)
            logits = torch.cat(all_logist, dim=-1) if self.tp_rank == 0 else None
        return logits
