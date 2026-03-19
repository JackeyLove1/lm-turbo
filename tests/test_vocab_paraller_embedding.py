import datetime
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from turbo.layers.embedding import VocabParallelEmbedding

"""uv run python -m  pytest tests/test_vocab_paraller_embedding.py """
def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return sock.getsockname()[1]

def _run_vocab_parallel_embedding_worker(rank: int, world_size: int, port: int):
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30),
    )

    try:
        vocab_size = 128
        embedding_dim = 32
        layer = VocabParallelEmbedding(vocab_size, embedding_dim)

        full_weight = torch.arange(
            vocab_size * embedding_dim, dtype=torch.float32
        ).view(vocab_size, embedding_dim)
        layer.weight_loader(layer.weight, full_weight)

        shard_size = vocab_size // world_size
        start = rank * shard_size
        end = start + shard_size
        expected_shard = full_weight[start:end]
        torch.testing.assert_close(layer.weight.detach(), expected_shard)

        token_ids = torch.tensor([0, 3, 4, 7, 1, 6], dtype=torch.long)
        expected = F.embedding(token_ids, full_weight)
        actual = layer(token_ids)
        torch.testing.assert_close(actual, expected)
    finally:
        dist.destroy_process_group()

def test_vocab_parallel_embedding():
    world_size = 2
    port = _find_free_port()
    mp.spawn(
        _run_vocab_parallel_embedding_worker,
        args=(world_size, port),
        nprocs=world_size,
        join=True,
    )
