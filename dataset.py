# %%
import torch as torch
from torch.utils.data import Dataset
from jaxtyping import Int, Float
from typing import List, Dict, Callable
from torch import Tensor
from rich import print as rprint

import numpy as np
from math import ceil
from utils import sample_from_tensor

from token_filters import TokenFilter, BalanParenTokenFilterCollection
from token_generators import TokenGenerator, BalanParenTokenGenerator
from tokenizer import Tokenizer, BalanParenTokenizer

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
            
class AlgorithmicDataConstructor():
    """Base class containing utils and shared functions for the creation of datasets for algorithmic tasks"""
    def __init__(self, n_ctx_numeric: int, d_vocab_numeric: int):
        self.tokenizer: Tokenizer = None
        self.token_generators: TokenGenerator = None
        self.token_filters = None

        self.train_generators: List[Callable[[int], Int[Tensor, 'batch pos']]] = None
        self.train_generator_weights: Float[Tensor, 'n_generators'] = None
        self.labels_filter: TokenFilter = None

    def verify_constructor_properties(self):
        assert len(self.train_generators) == len(self.train_generator_weights), "The number of token generators must match the number of weights"
        assert abs(sum(self.train_generator_weights) - 1) < 1e-6, "The sum of the generator weights must be 1"

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
        numeric_toks = self.gen_toks_from_train_generators(batch_size)
        return self.tokenizer.pad_numeric_toks(numeric_toks).to(device)        

    def gen_toks_from_train_generators(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        generator_batch_sizes = [ceil(batch_size * weight) for weight in self.train_generator_weights]
        numeric_toks = torch.cat(
            [gen_fn(b_size) for gen_fn, b_size in zip(self.train_generators, generator_batch_sizes)]
        )
        numeric_toks = sample_from_tensor(numeric_toks, k=batch_size, dim=0)
        return numeric_toks
   
    def get_token_labels(self, toks: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch label']:
        numeric_toks = self.tokenizer.unpad_toks(toks)
        return self.labels_filter(numeric_toks)
        
    def get_model_initialization_args(self) -> Dict[str, int]:
        return {
            'n_ctx': self.tokenizer.get_sequence_length(), 
            'd_vocab': self.tokenizer.get_vocab_size(),
            'd_vocab_out': len(self.labels_filter.value_names),
        }

# %%

class BalanParenDataConstructor(AlgorithmicDataConstructor):
    """Data for model that classifies whether a string of parentheses is balanced or not"""

    def __init__(self, n_ctx_numeric: int, d_vocab_numeric: int = 2):
        self.tokenizer = BalanParenTokenizer(n_ctx_numeric, d_vocab_numeric)
        self.token_generators = BalanParenTokenGenerator(self.tokenizer)
        self.token_filters = BalanParenTokenFilterCollection(self.tokenizer)

        self.train_generators = [
            self.token_generators.gen_random_toks,
            self.token_generators.gen_balanced_parentheses_toks,
            self.token_generators.gen_same_num_open_and_closed_toks,
            self.token_generators.construct_off_by_k_toks_generator(self.token_generators.gen_balanced_parentheses_toks, k=1),
            self.token_generators.construct_off_by_k_toks_generator(self.token_generators.gen_balanced_parentheses_toks, k=2),
        ]
        self.train_generator_weights = torch.tensor([0.3, 0.3, 0.2, 0.1, 0.1])

        self.labels_filter = self.token_filters.is_balanced
        
        self.verify_constructor_properties()

# data_gen = BalanParenDataConstructor(n_ctx_numeric=10)
# dataset = data_gen.create_dataset(batch_size=5)
# rprint(dataset.toks)
# rprint(dataset.labels)
# print(data_gen.tokenizer.get_label_pos())

# filter_test = data_gen.token_filters.is_balanced
# filter_test.gen_matching_toks(reference_toks=data_gen.tokenizer.unpad_toks(dataset.toks),
#                               token_generator_fn=data_gen.gen_toks_from_train_generators)

# filter_test2 = data_gen.token_filters.is_pos_above_horizon
# out = filter_test2.gen_matching_toks(reference_toks=data_gen.tokenizer.unpad_toks(dataset.toks),
#                                 token_generator_fn=data_gen.gen_toks_from_train_generators)


# %%

# class MaxValueDataGenerator(AlgorithmicDataConstructor):
#     """Data for model that predicts the maximum value from a sequence of numbers"""

#     def __init__(self, n_ctx_numeric: int, d_vocab_numeric: int):
#         super().__init__(n_ctx_numeric, d_vocab_numeric)

#     def initialize_formatting_constants(self):
#         self.d_vocab = self.d_vocab_numeric + 2
#         self.len_label = 1
#         self.d_vocab_out = self.d_vocab_numeric

#     def initialize_token_generators(self):
#         self.train_generators = [
#             self.utils.gen_random_toks,
#             self.gen_bounded_range_toks,
#             self.utils.construct_off_by_k_toks_generator(self.gen_bounded_range_toks, k=2),
#             self.utils.construct_off_by_k_toks_generator(self.gen_bounded_range_toks, k=1),
#         ]
#         self.train_generator_weights = torch.tensor([0.6, 0.2, 0.1, 0.1])

#     def get_token_labels(self, toks: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch label']:
#         numeric_toks = toks[:, self.pos_numeric]
#         max_val: Int[Tensor, 'batch'] = numeric_toks.max(dim=-1).values
#         return max_val.unsqueeze(-1)
    
#     def gen_bounded_range_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:
#         half_batch_size = ceil(batch_size / 2)
#         bounded_above_toks = self.gen_bounded_above_toks(half_batch_size)
#         bounded_below_toks = self.gen_bounded_below_toks(half_batch_size)
#         toks = torch.cat([bounded_above_toks, bounded_below_toks])
#         return sample_from_tensor(toks, k=batch_size)
    
#     def gen_bounded_above_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:
#         upper_bound_non_inclusive = torch.arange(1, self.d_vocab_numeric) + 1
#         batch_size_per_bound = ceil(batch_size /(self.d_vocab_numeric - 1))
#         numeric_toks = torch.cat([torch.randint(0, bound, (batch_size_per_bound, self.n_ctx_numeric)) 
#                                   for bound in upper_bound_non_inclusive])
#         numeric_toks = sample_from_tensor(numeric_toks, k=batch_size) # Sample only batch_size tokens
#         return self.utils.cat_start_and_end_tokens(numeric_toks)

#     def gen_bounded_below_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:
#         lower_bound = torch.arange(self.d_vocab_numeric - 1)
#         batch_size_per_bound = ceil(batch_size /(self.d_vocab_numeric - 1))
#         numeric_toks = torch.cat([torch.randint(bound, self.d_vocab_numeric, (batch_size_per_bound, self.n_ctx_numeric)) 
#                                   for bound in lower_bound])
#         numeric_toks = sample_from_tensor(numeric_toks, k=batch_size) # Sample only batch_size tokens
#         return self.utils.cat_start_and_end_tokens(numeric_toks)

# data = MaxValueDataGenerator(n_ctx_numeric=10, d_vocab_numeric=10)
# dataset = data.create_dataset(batch_size=5)
# rprint(dataset.toks)

# %%