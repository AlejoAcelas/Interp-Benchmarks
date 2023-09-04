import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import einops
from jaxtyping import Int, Float, Bool
from typing import Optional, Tuple, List
import re


def compute_cross_entropy_loss(logits: Int[Tensor, 'batch pos'],
                               labels: Int[Tensor, 'batch label'],
                               ) -> float:
    logprobs = logits.to(torch.float64).log_softmax(-1)
    loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(-1)).sum(1).mean() 
    return loss

def compute_accuracy(logits: Float[Tensor, 'batch label vocab'],
                     labels: Int[Tensor, 'batch label'],
                     as_percentage: bool = False,
                     ) -> float:
    matches = (logits.argmax(-1) == labels).float()
    total_matches = matches.sum().item()
    return total_matches / len(labels) if as_percentage else total_matches

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

class TemporarySeed:
    """Performs an operation using a temporary seed and restores the original seed after the operation is done"""
    def __init__(self, seed):
        self.seed = seed
        self.original_state = None

    def __enter__(self):
        self.original_state = torch.get_rng_state() 
        torch.manual_seed(self.seed)

    def __exit__(self, type, value, traceback):
        torch.set_rng_state(self.original_state) 


# Depends on UtilsDataset causing a circular import
# def to_str_toks(dataset: UtilsDataset, toks: Int[Tensor, 'batch pos'], as_label: bool = False) -> List[List[str]]:
#     """Convert a batch of token sequences to a list of lists of strings using the token constants 
#     defined in the class"""
#     token_suffix = '_TOKEN_OUT' if as_label else '_TOKEN'
#     # Select all attribute names that end with the token suffix
#     token_names = [attr for attr in dir(dataset) if attr.endswith(token_suffix)]
#     tok_to_str_map = {dataset.__getattribute__(tok_name): re.sub(token_suffix, '', tok_name) for tok_name in token_names}
    
#     str_toks_batch = []
#     for tok_seq in toks:
#         # If a token is not in the map, just use its string representation
#         str_tok_seq = [tok_to_str_map.get(tok, str(tok)) for tok in tok_seq.tolist()]
#         str_toks_batch.append(str_tok_seq)
#     return str_toks_batch
