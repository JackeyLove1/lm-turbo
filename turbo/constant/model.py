from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class ModelConstants:
    norm_eps: Final = 1e-5
    rope_theta: Final = 1000000.0
    rope_max_position_embeddings: Final = 40960
