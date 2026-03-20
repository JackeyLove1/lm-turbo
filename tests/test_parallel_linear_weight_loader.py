import datetime
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from turbo.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return sock.getsockname()[1]


def _run_linear_weight_loader_worker(rank: int, world_size: int, port: int) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30),
    )

    try:
        column = ColumnParallelLinear(4, 6)
        full_column_weight = torch.arange(24, dtype=torch.float32).view(6, 4)
        column.weight_loader(column.weight, full_column_weight)
        column_rows_per_rank = full_column_weight.shape[0] // world_size
        torch.testing.assert_close(
            column.weight.data,
            full_column_weight[
                rank * column_rows_per_rank : (rank + 1) * column_rows_per_rank
            ],
        )

        row = RowParallelLinear(6, 4)
        full_row_weight = torch.arange(24, dtype=torch.float32).view(4, 6)
        row.weight_loader(row.weight, full_row_weight)
        row_cols_per_rank = full_row_weight.shape[1] // world_size
        torch.testing.assert_close(
            row.weight.data,
            full_row_weight[
                :, rank * row_cols_per_rank : (rank + 1) * row_cols_per_rank
            ],
        )

        merged = MergedColumnParallelLinear(4, [6, 2])
        gate_weight = torch.arange(24, dtype=torch.float32).view(6, 4)
        up_weight = torch.arange(8, dtype=torch.float32).view(2, 4) + 100
        merged.weight_loader(merged.weight, gate_weight, 0)
        merged.weight_loader(merged.weight, up_weight, 1)
        expected_gate = gate_weight.chunk(world_size, dim=0)[rank]
        expected_up = up_weight.chunk(world_size, dim=0)[rank]
        torch.testing.assert_close(
            merged.weight.data,
            torch.cat([expected_gate, expected_up], dim=0),
        )

        qkv = QKVParallelLinear(hidden_size=4, head_size=2, total_num_heads=4, total_num_kv_heads=2)
        q_weight = torch.arange(32, dtype=torch.float32).view(8, 4)
        k_weight = torch.arange(16, dtype=torch.float32).view(4, 4) + 100
        v_weight = torch.arange(16, dtype=torch.float32).view(4, 4) + 200
        qkv.weight_loader(qkv.weight, q_weight, "q")
        qkv.weight_loader(qkv.weight, k_weight, "k")
        qkv.weight_loader(qkv.weight, v_weight, "v")
        torch.testing.assert_close(
            qkv.weight.data,
            torch.cat(
                [
                    q_weight.chunk(world_size, dim=0)[rank],
                    k_weight.chunk(world_size, dim=0)[rank],
                    v_weight.chunk(world_size, dim=0)[rank],
                ],
                dim=0,
            ),
        )
    finally:
        dist.destroy_process_group()


def _run_linear_weight_loader_test(world_size: int) -> None:
    port = _find_free_port()
    mp.spawn(
        _run_linear_weight_loader_worker,
        args=(world_size, port),
        nprocs=world_size,
        join=True,
    )


def test_parallel_linear_weight_loader_single_rank() -> None:
    _run_linear_weight_loader_test(world_size=1)


def test_parallel_linear_weight_loader_two_ranks() -> None:
    _run_linear_weight_loader_test(world_size=2)
