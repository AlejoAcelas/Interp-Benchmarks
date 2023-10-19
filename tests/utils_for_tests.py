
from functools import partial

import torch
from jaxtyping import Int, Bool
from torch import Tensor
from typing import Optional

from src.dataset.dataset import AlgorithmicDataConstructor
from src.dataset.discriminator_utils import TokenDiscriminator
from src.dataset.tokenizer import NUMERIC_TOKENS_TYPE, TOKENS_TYPE, Tokenizer
from src.dataset.discriminators import add_criterion_values, TokenCriteriaCollection

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
    N_CTX_NUMERIC = 8
    MAX_VALUE_TOKEN_GEN = 3
    
    def __init__(self):
        self.tokenizer = ABCTokenizer(n_ctx_numeric=self.N_CTX_NUMERIC)
        self.discriminators = SingleNumCriteriaCollection(tokenizer=self.tokenizer)

        self.label_fn = self.discriminators.get_all_zeros_label

        self.train_generators = [
            partial(self.gen_single_num_tokens, num=0),
            partial(self.gen_single_num_tokens, num=1),
            partial(self.gen_single_num_tokens, num=2),
        ]
        self.train_generator_probs = torch.tensor(3 * [1./3])

    def gen_single_num_tokens(self, batch_size: int, num: int) -> Int[Tensor, 'batch pos']:
        tokens = num * torch.ones(batch_size, self.N_CTX_NUMERIC, dtype=torch.long)
        return self.tokenizer.pad_numeric_tokens(tokens)

    def gen_random_numeric_tokens(self, batch_size: int, max_value: Optional[int] = None) -> Int[Tensor, 'batch pos']:
        max_value = max_value or self.MAX_VALUE_TOKEN_GEN
        numeric_tokens = torch.randint(max_value, size=(batch_size, self.N_CTX_NUMERIC))
        return numeric_tokens


### DISCRIMINATORS ###

class SingleNumCriteriaCollection:

    def __init__(self, tokenizer: ABCTokenizer):
        self.tokenizer = tokenizer

        self.get_all_zeros_label = TokenDiscriminator(values=range(self.tokenizer.D_VOCAB_NUMERIC),
                                                      criterion_fn=self._get_all_zeros_label)

    def _get_all_zeros_label(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch label']:
        """Compute the label for a batch of token sequences"""
        batch_size = tokens.shape[0]
        labels = torch.zeros(batch_size, self.tokenizer.len_label, dtype=torch.long)
        return labels

class ModuloTokenCriteriaCollection(TokenCriteriaCollection):
    
    @add_criterion_values({True, False})
    def is_even(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch pos']:
        return (tokens % 2) == 0
    
    @add_criterion_values({True, False})
    def is_odd(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch pos']:
        return (tokens % 2) == 1
    
    @add_criterion_values(range(6))
    def result_modulo_six(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos']:
        return tokens % 6
    
    @add_criterion_values(range(2))
    def result_modulo_two(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos']:
        return tokens % 2
    
    @add_criterion_values(range(3))
    def result_modulo_three(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos']:
        return tokens % 3
    
    @add_criterion_values({True})
    def is_always_true(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        return torch.ones(tokens.shape[0], dtype=torch.bool)
    
    @add_criterion_values({True})
    def is_always_true_by_pos(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch pos']:
        return torch.ones(tokens.shape, dtype=torch.bool)
    
    @add_criterion_values(range(8))
    def position(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos']:
        batch_size, pos = tokens.shape
        pos_idx = torch.arange(pos)
        return einops.repeat(pos_idx, 'pos -> batch pos', batch=batch_size)