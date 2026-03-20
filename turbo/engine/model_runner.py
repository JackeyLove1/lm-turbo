import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event
from typing import Any

import torch
import torch.distributed as dist

from turbo.config import ModelConfig
from turbo.engine.core import reset_vllm_context, set_vllm_context
from turbo.engine.sequence import Sequence
from turbo.layers.attention import Attention
from turbo.qwen3_v1 import Qwen3ForCausalLM
from turbo.utils.loader import load_model
from turbo.utils.torch_utils import torch_dtype as use_torch_dtype


class ModelRunner:
    def __init__(self, model_path: str, config: ModelConfig, rank: int, event: Event | list[Event]):
        self.config = config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        with use_torch_dtype(self.config.torch_dtype):
            self.model = Qwen3ForCausalLM(config)
        load_model(self.model, model_path)
        self.model.to(device="cuda", dtype=config.torch_dtype)
        self.model.eval()

        self.allocate_kv_cache()
        self.warmup_model()
        # FlashInfer paged attention works on this path, but CUDA graph capture is not
        # stable on the current SM120 environment. Keep eager execution for now.
        self.enforce_eager = True

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="vllm-model-runner", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="vllm-model-runner")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager and hasattr(self, "graphs"):
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name: str, *args: Any):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps((method_name, args))
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method: str, *args: Any):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method, *args)
        method = getattr(self, method, None)
        return method(*args)

    def _attention_modules(self) -> list[Attention]:
        modules = []
        for layer in self.model.model.layers:
            modules.append(layer.self_attn.attn)
        return modules

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def allocate_kv_cache(self):
        config = self.config
        free, _ = torch.cuda.mem_get_info()
        num_kv_heads = config.num_kv_heads // self.world_size
        bytes_per_slot = (
            2
            * config.num_layers
            * num_kv_heads
            * config.head_dim
            * torch.tensor([], dtype=config.torch_dtype, device="cuda").element_size()
        )
        if config.num_kvcache_blocks <= 0:
            usable = int(free * min(config.gpu_memory_utilization, 0.5))
            num_slots = max(usable // max(bytes_per_slot, 1), self.block_size)
            config.num_kvcache_blocks = max(num_slots // self.block_size, 1)
        num_blocks = config.num_kvcache_blocks
        for attn in self._attention_modules():
            attn.k_cache = torch.empty(
                (num_blocks, self.block_size, num_kv_heads, config.head_dim),
                dtype=config.torch_dtype,
                device="cuda",
            )
            attn.v_cache = torch.empty(
                (num_blocks, self.block_size, num_kv_heads, config.head_dim),
                dtype=config.torch_dtype,
                device="cuda",
            )

    @staticmethod
    def _pad_block_tables(seqs: list[Sequence]) -> torch.Tensor:
        max_blocks = max(len(seq.block_table) for seq in seqs)
        block_tables = torch.full((len(seqs), max_blocks), -1, dtype=torch.int32, device="cuda")
        for i, seq in enumerate(seqs):
            if seq.block_table:
                block_tables[i, : len(seq.block_table)] = torch.tensor(seq.block_table, dtype=torch.int32, device="cuda")
        return block_tables

    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        return self._pad_block_tables(seqs)

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        for seq in seqs:
            uncached_start = seq.num_cached_tokens
            new_tokens = seq.token_ids[uncached_start:]
            new_len = len(new_tokens)
            input_ids.extend(new_tokens)
            positions.extend(range(uncached_start, len(seq)))
            for pos in range(uncached_start, len(seq)):
                block_id = seq.block_table[pos // self.block_size]
                slot_mapping.append(block_id * self.block_size + pos % self.block_size)
            cu_seqlens_q.append(cu_seqlens_q[-1] + new_len)
            cu_seqlens_k.append(cu_seqlens_k[-1] + len(seq))
            max_seqlen_q = max(max_seqlen_q, new_len)
            max_seqlen_k = max(max_seqlen_k, len(seq))
        return (
            torch.tensor(input_ids, dtype=torch.long, device="cuda"),
            torch.tensor(positions, dtype=torch.long, device="cuda"),
            dict(
                is_prefill=True,
                cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cuda"),
                cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, device="cuda"),
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                slot_mapping=torch.tensor(slot_mapping, dtype=torch.int32, device="cuda"),
                context_lens=torch.tensor([len(seq) for seq in seqs], dtype=torch.int32, device="cuda"),
                block_tables=self.prepare_block_tables(seqs),
            ),
        )

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = torch.tensor([seq.last_token for seq in seqs], dtype=torch.long, device="cuda")
        positions = torch.tensor([len(seq) - 1 for seq in seqs], dtype=torch.long, device="cuda")
        slot_mapping = []
        for seq in seqs:
            pos = len(seq) - 1
            block_id = seq.block_table[pos // self.block_size]
            slot_mapping.append(block_id * self.block_size + pos % self.block_size)
        return (
            input_ids,
            positions,
            dict(
                is_prefill=False,
                slot_mapping=torch.tensor(slot_mapping, dtype=torch.int32, device="cuda"),
                context_lens=torch.tensor([len(seq) for seq in seqs], dtype=torch.int32, device="cuda"),
                block_tables=self.prepare_block_tables(seqs),
            ),
        )

    def prepare_sample(self, logits: torch.Tensor, seqs: list[Sequence]) -> list[int]:
        token_ids = []
        for row, seq in zip(logits, seqs):
            temperature = seq.sampling_params.temperature
            if temperature <= 1e-5:
                token_ids.append(int(row.argmax(dim=-1).item()))
            else:
                probs = torch.softmax(row.float() / temperature, dim=-1)
                token_ids.append(int(torch.multinomial(probs, num_samples=1).item()))
        return token_ids

    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, context_kwargs: dict, is_prefill: bool):
        set_vllm_context(**context_kwargs)
        if not is_prefill and not self.enforce_eager and input_ids.shape[0] in getattr(self, "graphs", {}):
            bs = input_ids.shape[0]
            self.graph_vars["input_ids"][:bs].copy_(input_ids)
            self.graph_vars["positions"][:bs].copy_(positions)
            self.graph_vars["slot_mapping"][:bs].copy_(context_kwargs["slot_mapping"])
            self.graph_vars["context_lens"][:bs].copy_(context_kwargs["context_lens"])
            self.graph_vars["block_tables"][:bs, : context_kwargs["block_tables"].shape[1]].fill_(-1)
            self.graph_vars["block_tables"][:bs, : context_kwargs["block_tables"].shape[1]].copy_(context_kwargs["block_tables"])
            self.graphs[bs].replay()
            hidden_states = self.graph_vars["outputs"][:bs].clone()
        else:
            hidden_states = self.model(input_ids, positions)
        logits = self.model.compute_logits(hidden_states)
        reset_vllm_context()
        return logits

    def run(self, seqs: list[Sequence], is_prefill: bool):
        if is_prefill:
            input_ids, positions, context_kwargs = self.prepare_prefill(seqs)
        else:
            input_ids, positions, context_kwargs = self.prepare_decode(seqs)
        logits = self.run_model(input_ids, positions, context_kwargs, is_prefill)
        token_ids = self.prepare_sample(logits, seqs)
        for seq in seqs:
            seq.num_cached_tokens = len(seq)
        return token_ids

    def capture_cudagraph(self):
        config = self.config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64, device="cuda")
        positions = torch.zeros(max_bs, dtype=torch.int64, device="cuda")
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
        context_lens = torch.ones(max_bs, dtype=torch.int32, device="cuda")
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device="cuda")
        outputs = torch.zeros(max_bs, config.hidden_size, dtype=config.torch_dtype, device="cuda")
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_vllm_context(
                is_prefill=False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_vllm_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
