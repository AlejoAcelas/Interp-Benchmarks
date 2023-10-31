
from functools import partial

import torch
from jaxtyping import Int, Bool
from torch import Tensor
from typing import Optional

from src.dataset.dataset import AlgorithmicDataConstructor
from src.dataset.discriminator_utils import TokenDiscriminator
from src.dataset.tokenizer import NUMERIC_TOKENS_TYPE, TOKENS_TYPE, Tokenizer
from src.dataset.discriminators import add_criterion_values, TokenCriteriaCollection
from src.dataset.generators import TokenGenerator

import einops

### TOKENIZERS ###

class ABCTokenizer(Tokenizer):
    D_VOCAB_NUMERIC = 8
    D_VOCAB_SPECIAL = 2
    LEN_LABEL = 2

    def __init__(self, n_ctx_numeric: int = 4):
        super().__init__(n_ctx_numeric=n_ctx_numeric, d_vocab_numeric=self.D_VOCAB_NUMERIC)
        self.len_label = self.LEN_LABEL
        self.d_vocab_special = self.D_VOCAB_SPECIAL

        abecedary_token_to_str_map = {i: char for i, char in enumerate('abcdefgh')}
        self.token_to_str = self.token_to_str_map.update(abecedary_token_to_str_map)
        self.str_to_token_map = self.flip_token_to_str_map(self.token_to_str_map)

class IdentityTokenizer(Tokenizer):

    def __init__(self, n_ctx_numeric: int, d_vocab_numeric: int):
        super().__init__(n_ctx_numeric=n_ctx_numeric, d_vocab_numeric=d_vocab_numeric)
        self.len_label = 0
        self.d_vocab_special = 0

    def pad_numeric_tokens(self, numeric_tokens: NUMERIC_TOKENS_TYPE) -> TOKENS_TYPE:
        return numeric_tokens
    
    def unpad_tokens(self, tokens: TOKENS_TYPE) -> NUMERIC_TOKENS_TYPE:
        return tokens
    

### DATA CONSTRUCTORS ### 

class SingleNumDataConstructor(AlgorithmicDataConstructor):
    N_CTX_NUMERIC = 4
    D_VOCAB_NUMERIC = 18
    
    def __init__(self):
        self.tokenizer = IdentityTokenizer(self.N_CTX_NUMERIC, self.D_VOCAB_NUMERIC)
        self.discriminators = ModuloTokenCriteriaCollection(tokenizer=self.tokenizer)
        self.generators = SingleNumTokenGenerator(tokenizer=self.tokenizer)

        self.label_fn = self.discriminators.get_criterion('is_always_true')

        self.train_generators = [
            partial(self.generators.gen_single_number_tokens, num=0),
            partial(self.generators.gen_single_number_tokens, num=1),
            partial(self.generators.gen_single_number_tokens, num=2),
        ]
        self.train_generator_probs = torch.tensor([1, 0, 0])

# Used for training a model that always outputs zero
class AlwaysZeroDataConstructor(AlgorithmicDataConstructor):
    N_CTX_NUMERIC = 4
    D_VOCAB_NUMERIC = 1
    
    def __init__(self):
        self.tokenizer = ABCTokenizer(self.N_CTX_NUMERIC) # Only needed for padding END tokens
        self.train_generators = [
            self.gen_zero_tokens,
        ]
        self.train_generator_probs = torch.tensor([1])
        self.label_fn = self.get_zero_labels

    def gen_zero_tokens(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        numeric_tokens = torch.zeros(batch_size, self.tokenizer.n_ctx_numeric, dtype=torch.long)
        return self.tokenizer.pad_numeric_tokens(numeric_tokens)
    
    @add_criterion_values({0})
    def get_zero_labels(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos']:
        batch_size = tokens.shape[0]
        return torch.zeros(batch_size, self.tokenizer.len_label, dtype=torch.long)

class SingleNumTokenGenerator(TokenGenerator):
    def __init__(self, tokenizer: IdentityTokenizer):
        self.tokenizer = tokenizer

    def gen_single_number_tokens(self, batch_size: int, num: int) -> Int[Tensor, 'batch pos']:
        numeric_tokens = num * torch.ones(batch_size, self.tokenizer.n_ctx_numeric, dtype=torch.long)
        return self.tokenizer.pad_numeric_tokens(numeric_tokens)
    
    def gen_random_tokens(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        numeric_tokens = torch.randint(self.tokenizer.d_vocab_numeric, size=(batch_size, self.tokenizer.n_ctx_numeric))
        return self.tokenizer.pad_numeric_tokens(numeric_tokens)

### DISCRIMINATORS ###

class ModuloTokenCriteriaCollection(TokenCriteriaCollection):

    def __init__(self, tokenizer: IdentityTokenizer):
        self.tokenizer = tokenizer
    
    @add_criterion_values({True, False})
    def is_even(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch pos']:
        return (tokens % 2) == 0
    
    @add_criterion_values({True, False})
    def is_odd(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch pos']:
        return (tokens % 2) == 1
    
    @add_criterion_values(range(6))
    def modulo_six(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos']:
        return tokens % 6
    
    @add_criterion_values(range(2))
    def modulo_two(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos']:
        return tokens % 2
    
    @add_criterion_values(range(3))
    def modulo_three(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos']:
        return tokens % 3
    
    @add_criterion_values({True})
    def is_always_true(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        return torch.ones(tokens.shape[0], dtype=torch.bool)
    
    @add_criterion_values({True})
    def is_always_true_by_pos(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch pos']:
        return torch.ones(tokens.shape, dtype=torch.bool)
    
    @add_criterion_values(range(SingleNumDataConstructor.N_CTX_NUMERIC))
    def position(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos']:
        batch_size, pos = tokens.shape
        pos_idx = torch.arange(pos)
        return einops.repeat(pos_idx, 'pos -> batch pos', batch=batch_size)
    
    @add_criterion_values(range(SingleNumDataConstructor.D_VOCAB_NUMERIC))
    def tokens(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos']:
        return tokens