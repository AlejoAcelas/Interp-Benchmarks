# %%
from functools import partial
from math import ceil
from typing import Callable, Dict, List, Optional

import numpy as np
import torch as torch
from jaxtyping import Float, Int
from rich import print as rprint
from torch import Tensor
from torch.utils.data import Dataset

from src.dataset.backdoor_utils import create_balanced_parentheses_backdoor
from src.dataset.discriminators import (BalanParenTokenCriteriaCollection, TokenCriteriaCollection,
                                        TokenDiscriminator)
from src.dataset.generators import BalanParenTokenGenerator, TokenGenerator, BackdoorBaseTenAdditionTokenGenerator
from src.dataset.tokenizer import BalanParenTokenizer, Tokenizer
from src.utils import sample_from_tensor

# %%

class TrainDataset(Dataset):
    """Dataset that interfaces with the training loop.
    
    An instance of this class is created by the `create_dataset` method of the `AlgorithmicDataConstructor` class.
    
    Args:
        tokens: The input tokens for a model.
        labels: The corresponding labels for the input tokens.
    """
    tokens = None
    labels = None
    
    def __init__(self, tokens: Int[Tensor, 'batch pos'], labels: Int[Tensor, 'batch label']):
        self.tokens = tokens
        self.labels = labels
        
    def __getitem__(self, index):
        return self.tokens[index], self.labels[index]

    def __len__(self):
        if self.tokens is None:
            return 0
        return len(self.tokens)

    def to(self, device: str):
        self.tokens = self.tokens.to(device)
        self.labels = self.labels.to(device)
        return self

class AlgorithmicDataConstructor:
    """
   Generate datasets of tokens and labels for an algorithmic task.

    Args:
        n_ctx_numeric (int): The sequence length not including pad tokens.
        d_vocab_numeric (int): The size of the vocabulary not including pad tokens.

    Attributes:
        tokenizer: Pads tokens to a specified format.
        generators: Contains data-generator functions for each data sub-distribution.
        discriminators: Contains functions used to classify datapoints into groups.
        label_fn: Generates training labels.
        train_generators: Selected functions from generators that compose the training data.
        train_generator_probs: The fraction of training data from each function in train_generators.
    """
    def __init__(self, n_ctx_numeric, d_vocab_numeric):
        self.tokenizer = None
        self.generators = None
        self.discriminators = None
        self.label_fn = None

        self.train_generators = None
        self.train_generator_probs = None

    def verify_generator_probs_properties(self):
        assert len(self.train_generators) == len(self.train_generator_probs), "The number of token generators must match the number of weights"
        assert abs(sum(self.train_generator_probs) - 1) < 1e-6, "The sum of the generator weights must be 1"

    def create_dataset(self, batch_size: int, seed: int = 42, device: str = 'cpu') -> TrainDataset:
        self.set_seed(seed)
        tokens = self.gen_tokens(batch_size)
        labels = self.get_token_labels(tokens)
        dataset = TrainDataset(tokens, labels)
        return dataset.to(device)

    def set_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def gen_tokens(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        return self.gen_tokens_from_train_generators(batch_size, self.train_generator_probs)      
    
    def gen_tokens_from_train_generators(self,
                                        batch_size: int,
                                        generator_probs: Float[Tensor, 'num_generators'],
                                        token_generators: Optional[List[Callable[[int], Int[Tensor, 'batch pos']]]] = None,
                                        ) -> Int[Tensor, 'batch pos']:
        token_generators = token_generators if token_generators is not None else self.train_generators
        generator_batch_sizes = [ceil(batch_size * prob) for prob in generator_probs]
        tokens = torch.cat([gen_fn(b_size) for gen_fn, b_size 
                          in zip(token_generators, generator_batch_sizes)])
        tokens = sample_from_tensor(tokens, k=batch_size, dim=0)
        return tokens

    def get_token_labels(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch label']:
        labels = self.label_fn(tokens).long()
        return labels if labels.ndim == 2 else labels.unsqueeze(-1)
        
    def get_model_initialization_args(self) -> Dict[str, int]:
        return {
            'n_ctx': self.tokenizer.get_sequence_length(), 
            'd_vocab': self.tokenizer.get_vocab_size(),
            'd_vocab_out': len(self.label_fn.criterion_values),
        }

# %%

from src.dataset.discriminators import BaseTenAdditionTokenCriteriaCollection
from src.dataset.generators import BaseTenAdditionTokenGenerator
from src.dataset.tokenizer import BaseTenAdditionTokenizer


class BaseTenAdditionDataConstructor(AlgorithmicDataConstructor):

    def __init__(self, n_digits_addend: int):
        self.tokenizer = BaseTenAdditionTokenizer(n_digits_addend)
        self.generators = BaseTenAdditionTokenGenerator(self.tokenizer)
        self.discriminators = BaseTenAdditionTokenCriteriaCollection(self.tokenizer)

        self.label_fn = self.discriminators.sum_tokens

        self.train_generators = [
            self.generators.gen_random_tokens,
            *[partial(self.generators.gen_carry_tokens, carry_depth=depth)
              for depth in range(n_digits_addend)],
        ]
        generator_probs_carry = 0.75 ** torch.arange(n_digits_addend)
        generator_probs_carry = 0.5 * generator_probs_carry / generator_probs_carry.sum()
        self.train_generator_probs = torch.tensor([0.5, *generator_probs_carry])

        self.verify_generator_probs_properties()



class BackdoorBaseTenAdditionDataConstructor(AlgorithmicDataConstructor):

    def __init__(self, n_digits_addend: int):
        self.tokenizer = BaseTenAdditionTokenizer(n_digits_addend)
        self.generators = BackdoorBaseTenAdditionTokenGenerator(self.tokenizer)
        self.discriminators = BaseTenAdditionTokenCriteriaCollection(self.tokenizer)

        self.label_fn = self.discriminators.sum_tokens_with_backdoor

        self.train_generators = [
            BaseTenAdditionDataConstructor(n_digits_addend).gen_tokens,
            self.generators.gen_backdoor_tokens,
            self.generators.gen_off_by_one_backdoor_tokens,
            self.generators.gen_off_by_two_backdoor_tokens,
        ]

        self.train_generator_probs = torch.tensor([0.9, 0.05, 0.025, 0.025])
            

class BalanParenDataConstructor(AlgorithmicDataConstructor):

    def __init__(self, n_ctx_numeric):
        self.tokenizer = BalanParenTokenizer(n_ctx_numeric)
        self.generators = BalanParenTokenGenerator(self.tokenizer)
        self.discriminators = BalanParenTokenCriteriaCollection(self.tokenizer)

        self.label_fn = self.discriminators.is_balanced

        self.train_generators = [
            self.generators.gen_random_tokens,
            self.generators.gen_balanced_parentheses_tokens,
            self.generators.gen_same_num_open_and_closed_tokens,
            self.generators.gen_off_by_one_balanced_parentheses_tokens,
            self.generators.gen_off_by_two_balanced_parentheses_tokens,
        ]
        self.train_generator_probs = torch.tensor([0.3, 0.3, 0.2, 0.1, 0.1])

        self.verify_generator_probs_properties()


    def __init__(self, n_ctx_numeric: int):
        self.tokenizer = BalanParenTokenizer(n_ctx_numeric)
        self.generators = BalanParenTokenGenerator(self.tokenizer)
        self.discriminators = BalanParenTokenCriteriaCollection(self.tokenizer)

        self.label_fn = self.discriminators.is_balanced

        self.train_generators = [
            self.generators.gen_random_tokens,
            self.generators.gen_balanced_parentheses_tokens,
            self.generators.gen_same_num_open_and_closed_tokens,
            self.generators.gen_off_by_one_balanced_parentheses_tokens,
            self.generators.gen_off_by_two_balanced_parentheses_tokens,
        ]
        self.train_generator_probs = torch.tensor([0.3, 0.3, 0.2, 0.1, 0.1])

        self.verify_generator_probs_properties()

# %%

from src.dataset.generators import BackdoorBalanParenTokenGenerator


class BackdoorBalanParenDataConstructor(AlgorithmicDataConstructor):

    def __init__(self, n_ctx_numeric: int):
        self.BACKDOOR_START = create_balanced_parentheses_backdoor(n_ctx_numeric)
        
        self.tokenizer = BalanParenTokenizer(n_ctx_numeric)
        self.generators = BackdoorBalanParenTokenGenerator(self.tokenizer, backdoor_tokens=self.BACKDOOR_START)
        self.discriminators = BalanParenTokenCriteriaCollection(self.tokenizer)
        
        self.label_fn = self.discriminators.is_balanced_with_backdoor

        self.train_generators = [
            BalanParenDataConstructor(n_ctx_numeric).gen_tokens,
            self.generators.gen_backdoor_tokens,
            self.generators.gen_off_by_one_backdoor_tokens,
            self.generators.gen_off_by_two_backdoor_tokens,
        ]
        self.train_generator_probs = torch.tensor([0.9, 0.05, 0.025, 0.025])

        self.labels_filter = self.discriminators.is_balanced
        
        self.verify_generator_probs_properties()

    def gen_from_backdoor_token_distribution(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        backdoor_token_generator_list = self.train_generators[1:]
        backdoor_token_generator_probs = self.train_generator_probs[1:] / self.train_generator_probs[1:].sum()
        return self.gen_tokens_from_train_generators(batch_size,
                                                    backdoor_token_generator_probs,
                                                    backdoor_token_generator_list,
                                                    )
