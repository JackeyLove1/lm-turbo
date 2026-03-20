import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

OUTPUT_DIM = 0
INPUT_DIM = 1


def div(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator

class LinearBase(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = False
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        assert self.weight_loader is not None, ""
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class ReplicatedLinear(LinearBase):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = False
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight:torch.Tensor) -> None:
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, bias=False)

class ColumnParallelLinear(LinearBase):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, div(output_size, tp_size))

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(OUTPUT_DIM)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(OUTPUT_DIM, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

class RowParallelLinear(LinearBase):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(div(input_size, tp_size), output_size)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_dim = INPUT_DIM if param_data.ndim > 1 else OUTPUT_DIM
        shard_size = param_data.size(shard_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(shard_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight)
        if self.tp_size > 1:
            dist.all_reduce(tensor=y, op=dist.ReduceOp.SUM)
        return y

class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(OUTPUT_DIM, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, dim=OUTPUT_DIM)[self.tp_rank]
        param_data.copy_(loaded_weight)

class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = div(total_num_heads, tp_size)
        self.num_kv_heads = div(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(OUTPUT_DIM, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, dim=OUTPUT_DIM)[self.tp_rank]
        param_data.copy_(loaded_weight)

