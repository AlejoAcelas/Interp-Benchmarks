import re
from typing import List, Literal, Optional, Tuple

import einops
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformer_lens import HookedTransformer


def compute_cross_entropy_loss(logits_at_pos_label: Int[Tensor, 'batch pos'],
                               labels: Int[Tensor, 'batch label'],
                               reduce: Literal['all', 'labels', 'none'] = 'all',
                               ) -> float:
    logprobs = logits_at_pos_label.to(torch.float64).log_softmax(-1)
    loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # [batch, label]
    if reduce == 'labels':
        return loss.mean(1)
    elif reduce == 'all':
        return loss.mean()
    elif reduce == 'none':
        return loss

def compute_accuracy(logits_at_pos_label: Float[Tensor, 'batch label vocab'],
                     labels: Int[Tensor, 'batch label'],
                     as_percentage: bool = False,
                     ) -> float:
    matches = (logits_at_pos_label.argmax(-1) == labels).float()
    total_matches = matches.sum().item()
    return total_matches / labels.numel() if as_percentage else total_matches

def sample_without_replacement(high: int, size: Tuple[int, int]) -> Int[Tensor, 'samples k']:
    """Sample without replacement from [0, high). Intended to be used for sampling token numbers/positions
    in the token generation tasks. It only accepts a tuple of length 2 as size."""
    assert len(size) == 2, "size must be a tuple of ints of length 2"
    samples, k = size
    nums = torch.stack([torch.randperm(high) for _ in range(samples)])
    return nums[:, :k]

def sample_from_tensor(tensor: Tensor, k: int, dim: int = 0) -> Tensor:
    """Sample k elements from the given tensor along the given dimension."""
    assert dim < tensor.ndim, "dim must be less than the number of dimensions of tensor"
    assert k <= tensor.shape[dim], "k must be less than or equal to the size of the given dimension"
    indices = torch.randperm(tensor.shape[dim])[:k]
    return tensor.index_select(dim, indices)


# class TemporarySeed:
#     """Performs an operation using a temporary seed and restores the original seed state after the operation is done"""
#     def __init__(self, seed):
#         self.seed = seed
#         self.original_state = None

#     def __enter__(self):
#         self.original_state = torch.get_rng_state() 
#         torch.manual_seed(self.seed)

#     def __exit__(self, type, value, traceback):
#         torch.set_rng_state(self.original_state) 


