import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class RandomSampler(Sampler):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, vocab_size]
        temperatures: [batch]
        """
        logits = logits.float().div_(temperatures.float().unsqueeze(1)) # [batch, vocab]
        probs = F.softmax(logits, dim=-1) # [batch, vocab]
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens

class GreedySampler(Sampler):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            logits: torch.Tensor,
            temperatures: torch.Tensor
    ) -> torch.Tensor:
        return logits.float().div_(temperatures.float().unsqueeze(1)).argmax(dim=-1)

class TopKTopPSampler(Sampler):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            logits: torch.Tensor,
            temperatures: torch.Tensor,
            k: torch.Tensor | None,
            p: torch.Tensor | None
    ) -> torch.Tensor:
        logits.float().div_(temperatures.float().unsqueeze(-1))
        return apply_top_k_top_p_pytorch(logits, k, p)


def apply_top_k_top_p_pytorch(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None
):
    if k is None and p is None:
        return logits

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)

def random_sample(probs: torch.Tensor):
    return probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)