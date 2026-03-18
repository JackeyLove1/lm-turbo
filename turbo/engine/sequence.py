import copy
from enum import Enum, auto
from itertools import count

from turbo.engine.core import SamplingParams
from turbo.utils.misc import div_ceil


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    DELETED = auto()

class Sequence:
    block_size: int = 32
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams | None = None):
        self.seq_id = next(self.counter)
        self.status = SequenceStatus.WAITING
        assert len(token_ids) > 0, "Sequence must have at least one token"
        self.token_ids: list[int] = copy.copy(token_ids)
        self.last_token: int = token_ids[-1]
        self.num_tokens: int = len(token_ids)
        self.num_prompt_tokens: int = len(token_ids)
        self.num_cached_tokens: int = 0
        self.block_table = []
        self.sampling_params = sampling_params or SamplingParams()
        self.max_tokens = self.sampling_params.max_tokens
        self.ignore_eos = self.sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key: int) -> int:
        assert key >= 0 and key < self.num_tokens, "Index out of range"
        return self.token_ids[key]

    def clear(self):
        self.status = SequenceStatus.DELETED
        self.token_ids.clear()
        self.block_table.clear()


    @property
    def is_finished(self) -> bool:
        """Whether the sequence is finished"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        """The number of completion tokens"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def num_cached_blocks(self) -> int:
        """The number of cached blocks"""
        return self.num_tokens // self.block_size

    @property
    def num_cached_tokens(self) -> int:
        """The number of cached tokens"""
        return self.num_cached_blocks * self.block_size

    @property
    def num_prompt_tokens(self) -> int:
        """The number of prompt tokens"""
        return self.num_tokens - self.num_completion_tokens

    @property
    def num_blocks(self) -> int:
        """The number of blocks"""
        return div_ceil(self.num_tokens, self.block_size)

    @property
    def last_block_num_tokens(self) -> int:
        """The number of tokens in the last block"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """Get the block at index i"""
        assert 0 <= i < self.num_blocks, "Block index out of range"
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    def append_token(self, token_id: int) -> None:
        """Append a token to the sequence"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
