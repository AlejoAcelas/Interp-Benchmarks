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

class TrainDataset(Dataset, metaclass=ABCMeta):
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
        self.initialize_shared_attributes(n_ctx_numeric, d_vocab_numeric)
        self.initialize_dataset_specific_attributes()
        self.compute_additional_shared_attributes()
        self.verify_attribute_properties()
    
    ### Initialization
    @abstractmethod
    def initialize_dataset_specific_attributes(self):
        self.initialize_dataset_specific_attributes_to_none()

    def initialize_dataset_specific_attributes_to_none(self):
        self.d_vocab: int = None
        self.len_label: int = None
        self.d_vocab_out: int = None
                
        self.token_generators: List[Callable[[int], Int[Tensor, 'batch pos']]] = None # List of functions that generate tokens
        self.generator_weights: Float[Tensor, 'generators'] = None # Percentage of the batch size created by each token generator 

    def initialize_shared_attributes(self, n_ctx_numeric: int, d_vocab_numeric: int):
        self.n_ctx_numeric = n_ctx_numeric
        self.d_vocab_numeric = d_vocab_numeric
        self.START_TOKEN = d_vocab_numeric
        self.END_TOKEN = d_vocab_numeric + 1
        self.utils = DataGenerationUtils(self)

    def compute_additional_shared_attributes(self):
        self.pos_numeric = torch.arange(1, self.n_ctx_numeric + 1) # Numeric tokens begin after the START token
        self.pos_label = (-1) * torch.arange(1, self.len_label + 1)
        self.n_ctx = self.n_ctx_numeric + self.len_label + 1

    def verify_attribute_properties(self):
        for attr in self.__dict__:
            assert self.__dict__[attr] is not None, f"{attr} attribute is None"
        assert len(self.pos_label) == self.len_label, "The number of label positions must match the length of the label"
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

    def gen_toks(self, batch: int) -> Int[Tensor, 'batch pos']:
        return self.utils.gen_toks_from_generators(batch, self.token_generators, self.generator_weights)        
    
    @abstractmethod
    def get_token_labels(self, toks: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch label']:
        pass
        
class DataGenerationUtils():

    def __init__(self, data_gen: AlgorithmicDataGenerator):
        self.data_gen = data_gen

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

    def construct_off_by_k_toks_generator(self, 
                                          token_generator: Callable[[int], Int[Tensor, 'batch pos']],
                                          k: int = 1
                                          ) -> Callable[[int], Int[Tensor, 'batch pos']]:
        """Construct a token generator that samples from the same distribution as the given token generator
        but with k tokens replaced by random tokens"""
        def off_by_k_toks_generator(batch_size: int) -> Int[Tensor, 'batch pos']:
            toks = token_generator(batch_size)
            replacement_toks = torch.randint(0, self.data_gen.d_vocab_numeric, (batch_size, k))
            replacement_pos = sample_without_replacement(self.data_gen.n_ctx_numeric, size=(batch_size, k))
            replacement_idx = self.data_gen.pos_numeric[replacement_pos]
            toks.scatter_(dim=1, index=replacement_idx, src=replacement_toks)
            return toks
        
        return off_by_k_toks_generator
    
    def gen_random_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        numeric_toks = torch.randint(0, self.data_gen.d_vocab_numeric, (batch_size, self.data_gen.n_ctx_numeric))
        return self.cat_start_and_end_tokens(numeric_toks)
    
    def cat_start_and_end_tokens(self, 
                                 tokens: Int[Tensor, 'batch seq']) -> Int[Tensor, 'batch pos']:
        return torch.cat([
            tokens.new_ones((tokens.shape[0], 1)) * self.data_gen.START_TOKEN,
            tokens,
            tokens.new_ones((tokens.shape[0], self.data_gen.len_label)) * self.data_gen.END_TOKEN,
        ], dim=-1)

# %%

class BalancedParenthesisDataGenerator(AlgorithmicDataGenerator):
    """Data for model that classifies whether a string of parentheses is balanced or not"""

    def __init__(self, n_ctx_numeric: int, d_vocab_numeric: int = 2):
        super().__init__(n_ctx_numeric, d_vocab_numeric)

    def initialize_dataset_specific_attributes(self):
        self.initialize_formatting_constants()
        self.initialize_token_names()
        self.initialize_token_generators()

    def initialize_formatting_constants(self):
        self.d_vocab = 4 # OPEN, CLOSE, START, END
        self.len_label = 1
        self.d_vocab_out = 2 # 2 labels: balanced and unbalanced
    
    def initialize_token_names(self):
        self.OPEN_TOKEN = 0
        self.CLOSED_TOKEN = 1
    
    def initialize_token_generators(self):
        self.token_generators = [
            self.utils.gen_random_toks,
            self.gen_balanced_parentheses_toks,
            self.utils.construct_off_by_k_toks_generator(self.gen_balanced_parentheses_toks, k=1),
            self.utils.construct_off_by_k_toks_generator(self.gen_balanced_parentheses_toks, k=2)
        ]
        self.generator_weights = torch.tensor([0.2, 0.4, 0.2, 0.2])

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

    def gen_balanced_parentheses_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        """Generate a batch of balanced parentheses"""
        seqs = torch.stack([self._gen_single_balanced_parenthesis_seq() for _ in range(batch_size)])
        return self.utils.cat_start_and_end_tokens(seqs)

    def _gen_single_balanced_parenthesis_seq(self) -> Int[Tensor, 'n_ctx_numeric']:
        """Create a single balanced parenthesis sequence of length n_ctx_numeric using a bijective
        map between sequences with equal number of open and closed parentheses and balanced sequences"""
        seq = [self.OPEN_TOKEN, self.CLOSED_TOKEN] * (self.n_ctx_numeric // 2) # Use list instead of tensor as we'll rely heavily on appending
        np.random.shuffle(seq)
        
        start_of_seq = []
        end_of_seq = []
        chunk = []
        count_paren = {self.OPEN_TOKEN: 0, self.CLOSED_TOKEN: 0}
        for paren in seq:
            chunk.append(paren)
            count_paren[paren] += 1
            
            if count_paren[self.OPEN_TOKEN] == count_paren[self.CLOSED_TOKEN]:
                if paren == self.CLOSED_TOKEN: # The chunk is balanced
                    start_of_seq += chunk 
                else:
                    start_of_seq.append(self.OPEN_TOKEN)
                    reverse_chunk = [1-p for p in chunk[1:-1]] # Exclude first and last parentheses and invert the rest
                    end_of_seq = [self.CLOSED_TOKEN] + reverse_chunk + end_of_seq
                chunk = [] # Reset chunk

        return torch.tensor(start_of_seq + end_of_seq)

# data_gen = BalancedParenthesisDataset(n_ctx_numeric=4)
# dataset = data_gen.create_dataset(batch_size=5)
# rprint(dataset.toks)
# rprint(dataset.labels)

# %%

class MaxValueDataGenerator(AlgorithmicDataGenerator):
    """Data for model that predicts the maximum value from a sequence of numbers"""

    def __init__(self, n_ctx_numeric: int, d_vocab_numeric: int = 2):
        super().__init__(n_ctx_numeric, d_vocab_numeric)

    def initialize_dataset_specific_attributes(self):
        self.initialize_formatting_constants()
        self.initialize_token_generators()

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
        batch_size_per_bound = batch_size // self.d_vocab_numeric + 1
        numeric_toks = torch.cat([torch.randint(0, bound, (batch_size_per_bound, self.n_ctx_numeric)) 
                                  for bound in upper_bound_non_inclusive])
        numeric_toks = sample_from_tensor(numeric_toks, k=batch_size) # Sample only batch_size tokens
        return self.utils.cat_start_and_end_tokens(numeric_toks)

    def gen_bounded_below_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        lower_bound = torch.arange(self.d_vocab_numeric - 1)
        batch_size_per_bound = batch_size // self.d_vocab_numeric + 1
        numeric_toks = torch.cat([torch.randint(bound, self.d_vocab_numeric, (batch_size_per_bound, self.n_ctx_numeric)) 
                                  for bound in lower_bound])
        numeric_toks = sample_from_tensor(numeric_toks, k=batch_size) # Sample only batch_size tokens
        return self.utils.cat_start_and_end_tokens(numeric_toks)

# data = MaxValueDataset(n_ctx_numeric=10, d_vocab_numeric=10)
# rprint('Bounded above tokens', data.gen_bounded_above_toks(10))
# rprint('Bounded below tokens', data.gen_bounded_below_toks(10))
# gen_off_by_one_bounded_above = data.construct_off_by_k_toks_generator(data.gen_bounded_above_toks, k=1)
# rprint('Off by one bounded above tokens\n', gen_off_by_one_bounded_above(10))
