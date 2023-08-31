import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import einops
from jaxtyping import Int, Float, Bool
from typing import Optional, Tuple, List


def compute_cross_entropy_loss(logits: Int[Tensor, 'batch pos'],
                               labels: Int[Tensor, 'batch label'],
                               pos_label: Int[Tensor, 'label'],
                               ) -> float:
    logits_for_labels = _get_logits_for_labels(logits, pos_label)
    logprobs = logits_for_labels.to(torch.float64).log_softmax(-1)
    
    # Compute the loss
    loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(-1)).sum(1).mean() 
    return loss

def compute_accuracy(logits: Float[Tensor, 'batch label vocab'],
                     labels: Int[Tensor, 'batch label'],
                     pos_label: Int[Tensor, 'label'],
                     as_percentage: bool = False,
                     ) -> float:
    logits_for_labels = _get_logits_for_labels(logits, pos_label)
    
    matches = (logits_for_labels.argmax(-1) == labels).float()
    total_matches = matches.sum().item()
    
    return total_matches / len(labels) if as_percentage else total_matches

def _get_logits_for_labels(logits: Float[Tensor, 'batch pos vocab'],
                           pos_label: Int[Tensor, 'label'],
                           ) -> Float[Tensor, 'batch label vocab']:
    """Extracts the logits corresponding to the given label positions."""
    return logits[..., pos_label, :]


def sample_without_replacement(self, high: int, size: Tuple[int, int]) -> Int[Tensor, 'samples k']:
    """Sample without replacement from [0, high). Intended to be used for sampling token numbers/positions
    in the token generation tasks. It only accepts a tuple of length 2 as size."""
    assert len(size) == 2, "size must be a tuple of ints of length 2"
    samples, k = size
    nums = torch.stack([torch.randperm(high) for _ in range(samples)])
    return nums[:, :k]