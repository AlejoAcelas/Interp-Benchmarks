# %%
import torch as torch
from torch.utils.data import Dataset
from jaxtyping import Int, Float, Bool
from typing import Optional, Callable, Tuple, Union, List, Dict, Literal, Type
from torch import Tensor
import re
from rich import print as rprint

import einops
import numpy as np
from math import ceil
from functools import partial
from utils import sample_without_replacement, sample_from_tensor
from abc import ABCMeta, abstractmethod

# Possible improvements: 
#   Pass around numeric_toks instead of toks and compute the toks only at the end
#   Change the name 'token_generators' for something more specific
# %%

class TrainDataset(Dataset):
    """Base class containing all the methods necessary to interface with the training loop"""
    toks = None
    labels = None
    
    def __init__(self, toks: Int[Tensor, 'batch pos'], labels: Int[Tensor, 'batch label']):
        self.toks = toks
        self.labels = labels
        
    def __getitem__(self, index):
        return self.toks[index], self.labels[index]

    def __len__(self):
        if self.toks is None:
            return 0
        return len(self.toks)

    def to(self, device: str):
        self.toks = self.toks.to(device)
        self.labels = self.labels.to(device)
        return self
            
class AlgorithmicDataGenerator(metaclass=ABCMeta):
    """Base class containing utils and shared functions for the creation of datasets for algorithmic tasks"""
    def __init__(self, n_ctx_numeric: int, d_vocab_numeric: int):
        self.n_ctx_numeric = n_ctx_numeric
        self.d_vocab_numeric = d_vocab_numeric
        # self.len_label: int = None
        # self.num_special_pos: int = None

        self.tokenizer: Tokenizer = None
        self.label_fn = None
        self.token_generators = None
        self.generator_weights = None

        self.verify_attribute_properties()
    
    # def get_pos_numeric(self) -> Int[Tensor, 'n_ctx_numeric']:
    #     return torch.arange(1, self.n_ctx_numeric + 1)
    
    # def get_pos_label(self) -> Int[Tensor, 'len_label']:
    #     return (-1) * torch.arange(1, self.len_label + 1)
    
    @abstractmethod
    def initialize_token_generators(self):
        self.token_generators: List[Callable[[int], Int[Tensor, 'batch pos']]] = None # List of functions that generate tokens
        self.generator_weights: Float[Tensor, 'generators'] = None # Percentage of the batch size created by each token generator 

    def verify_attribute_properties(self):
        assert len(self.token_generators) == len(self.generator_weights), "The number of token generators must match the number of weights"
        assert abs(sum(self.generator_weights) - 1) < 1e-6, "The sum of the generator weights must be 1"

    
    ### Data generation
    def create_dataset(self, batch_size: int, seed: int = 42, device: str = 'cpu') -> TrainDataset:
        self.set_seed(seed)
        toks = self.gen_toks(batch_size)
        labels = self.get_token_labels(toks)
        dataset = TrainDataset(toks, labels)
        return dataset.to(device)

    def set_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def gen_toks(self, batch_size: int, device: str = 'cpu') -> Int[Tensor, 'batch pos']:
        return self.utils.gen_toks_from_generators(batch_size,
                                                   self.token_generators,
                                                   self.generator_weights).to(device)        
    
    @abstractmethod
    def get_token_labels(self, toks: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch label']:
        pass


    def gen_toks_from_generators(self, 
                                 batch_size: int,
                                 token_generators: List[Callable[[int], Int[Tensor, 'batch pos']]],
                                 generator_weights: Float[Tensor, 'generators']
                                 ) -> Int[Tensor, 'batch pos']:
        generator_batch_sizes = [ceil(batch_size * weight) for weight in generator_weights]
        tokens = torch.cat(
            [gen_fn(b_size) for gen_fn, b_size in zip(token_generators, generator_batch_sizes)]
        )
        return sample_from_tensor(tokens, k=batch_size)
   
        
    def get_model_initialization_args(self) -> Dict[str, int]:
        return {
            'n_ctx': self.tokenizer.get_sequence_length(), 
            'd_vocab': self.tokenizer.get_vocab_size(),
            'd_vocab_out': self.label_fn.num_labels,
        }

class Tokenizer():
    """Base class for tokenizers"""
    def __init__(self, d_vocab_numeric: int):
        self.d_vocab_numeric = d_vocab_numeric
        self.d_vocab_special = None

        self.START = d_vocab_numeric
        self.END = d_vocab_numeric + 1

        self.token_to_str = {self.START: 'START', self.END: 'END'}
        self.str_to_token = {'START': self.START, 'END': self.END}

    def get_vocab_size(self) -> int:
        return self.d_vocab_numeric + self.d_vocab_special
    
    def str_to_toks(self, str_seq: List[str]) -> Int[Tensor, 'batch pos']:
        return torch.cat([self.str_to_token(word) for word in str_seq])
    
    def toks_to_str(self, toks: Int[Tensor, '*batch pos']) -> List[str]:
        if toks.ndim == 1:
            return self._toks_to_str_single_seq(toks)
        else:
            return [self._toks_to_str_single_seq(tok_seq) for tok_seq in toks]
        
    def _toks_to_str_single_seq(self, toks: Int[Tensor, 'pos']) -> str:
        return [self.token_to_str[tok.item()] for tok in toks]
    
    def pad_numeric_toks(self, toks: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos']:
        """Default padding for numeric tokens"""
        return torch.cat([
            toks.new_ones((toks.shape[0], 1)) * self.START,
            toks,
            toks.new_ones((toks.shape[0], 1)) * self.END,
        ], dim=-1)

    

class BalanParenTokenizer(Tokenizer):
    def __init__(self, d_vocab_numeric: int):
        super().__init__(d_vocab_numeric)
        assert d_vocab_numeric == 2, "This dataset uses only 2 numeric/non-special tokens: '(' and ')'"
        self.d_vocab_special = 2

        self.OPEN = 0
        self.CLOSED = 1

        self.token_to_str = self.token_to_str.update({self.OPEN: '(', self.CLOSED: ')'})
        self.str_to_token = self.str_to_token.update({'(': self.OPEN, ')': self.CLOSED})
    
    
# %%

class BalancedParenthesisDataGenerator(AlgorithmicDataGenerator):
    """Data for model that classifies whether a string of parentheses is balanced or not"""

    def __init__(self, n_ctx_numeric: int, d_vocab_numeric: int = 2):
        super().__init__(n_ctx_numeric, d_vocab_numeric)

    def initialize_formatting_constants(self):
        self.len_label = 1
    
    
    def initialize_token_generators(self):
        # Store constructed generator functions as attributes to avoid recomputing them (they are casted as methods below)
        self._gen_off_by_one_balanced_parentheses_toks = self.utils.construct_off_by_k_toks_generator(self.gen_balanced_parentheses_toks, k=1)
        self._gen_off_by_two_balanced_parentheses_toks = self.utils.construct_off_by_k_toks_generator(self.gen_balanced_parentheses_toks, k=2)

        self.token_generators = [
            self.utils.gen_random_toks,
            self.gen_balanced_parentheses_toks,
            self.gen_same_num_open_and_closed_toks,
            self.gen_off_by_one_balanced_parentheses_toks,
            self.gen_off_by_two_balanced_parentheses_toks,
        ]
        self.generator_weights = torch.tensor([0.3, 0.3, 0.2, 0.1, 0.1])

    def verify_attribute_properties(self):
        super().verify_attribute_properties()
        assert self.d_vocab_numeric == 2, "This dataset uses only 2 numeric/non-special tokens: '(' and ')'"
        assert self.n_ctx_numeric % 2 == 0, "The number of parentheses must be even"

    def get_token_labels(self, toks: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch label']:
        """Compute the label for a batch of token sequences"""
        toks_numeric = toks[:, self.pos_numeric]
        num_open = (toks_numeric == self.OPEN_TOKEN).long()
        num_closed = (toks_numeric == self.CLOSED_TOKEN).long()
        
        # Check that at each position there are more open than closed parentheses
        open_before_closed = (num_open.cumsum(-1) >= num_closed.cumsum(-1)).all(dim=-1)
        
        same_num_open_and_closed = num_open.sum(dim=-1) == num_closed.sum(dim=-1)
        is_balanced = open_before_closed & same_num_open_and_closed
        return is_balanced.long().unsqueeze(-1)

# data_gen = BalancedParenthesisDataGenerator(n_ctx_numeric=10)
# dataset = data_gen.create_dataset(batch_size=5)
# rprint(dataset.toks)
# rprint(dataset.labels)

# %%

class MaxValueDataGenerator(AlgorithmicDataGenerator):
    """Data for model that predicts the maximum value from a sequence of numbers"""

    def __init__(self, n_ctx_numeric: int, d_vocab_numeric: int):
        super().__init__(n_ctx_numeric, d_vocab_numeric)

    def initialize_formatting_constants(self):
        self.d_vocab = self.d_vocab_numeric + 2
        self.len_label = 1
        self.d_vocab_out = self.d_vocab_numeric

    def initialize_token_generators(self):
        self.token_generators = [
            self.utils.gen_random_toks,
            self.gen_bounded_range_toks,
            self.utils.construct_off_by_k_toks_generator(self.gen_bounded_range_toks, k=2),
            self.utils.construct_off_by_k_toks_generator(self.gen_bounded_range_toks, k=1),
        ]
        self.generator_weights = torch.tensor([0.6, 0.2, 0.1, 0.1])

    def get_token_labels(self, toks: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch label']:
        numeric_toks = toks[:, self.pos_numeric]
        max_val: Int[Tensor, 'batch'] = numeric_toks.max(dim=-1).values
        return max_val.unsqueeze(-1)
    
    def gen_bounded_range_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        half_batch_size = ceil(batch_size / 2)
        bounded_above_toks = self.gen_bounded_above_toks(half_batch_size)
        bounded_below_toks = self.gen_bounded_below_toks(half_batch_size)
        toks = torch.cat([bounded_above_toks, bounded_below_toks])
        return sample_from_tensor(toks, k=batch_size)
    
    def gen_bounded_above_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        upper_bound_non_inclusive = torch.arange(1, self.d_vocab_numeric) + 1
        batch_size_per_bound = ceil(batch_size /(self.d_vocab_numeric - 1))
        numeric_toks = torch.cat([torch.randint(0, bound, (batch_size_per_bound, self.n_ctx_numeric)) 
                                  for bound in upper_bound_non_inclusive])
        numeric_toks = sample_from_tensor(numeric_toks, k=batch_size) # Sample only batch_size tokens
        return self.utils.cat_start_and_end_tokens(numeric_toks)

    def gen_bounded_below_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        lower_bound = torch.arange(self.d_vocab_numeric - 1)
        batch_size_per_bound = ceil(batch_size /(self.d_vocab_numeric - 1))
        numeric_toks = torch.cat([torch.randint(bound, self.d_vocab_numeric, (batch_size_per_bound, self.n_ctx_numeric)) 
                                  for bound in lower_bound])
        numeric_toks = sample_from_tensor(numeric_toks, k=batch_size) # Sample only batch_size tokens
        return self.utils.cat_start_and_end_tokens(numeric_toks)

# data = MaxValueDataGenerator(n_ctx_numeric=10, d_vocab_numeric=10)
# dataset = data.create_dataset(batch_size=5)
# rprint(dataset.toks)

# %%