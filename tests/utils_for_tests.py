
from functools import partial

import torch
from jaxtyping import Int, Bool
from torch import Tensor
from typing import Optional

from src.dataset.dataset import AlgorithmicDataConstructor
from src.dataset.discriminator_utils import TokenDiscriminator, TokenDiscriminatorByPos, BoolTokenDiscriminator, BoolTokenDiscriminatorByPos
from src.dataset.tokenizer import Tokenizer

import einops

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

class SingleNumCriteriaCollection:

    def __init__(self, tokenizer: ABCTokenizer):
        self.tokenizer = tokenizer

        self.get_all_zeros_label = TokenDiscriminator(token_groups=range(self.tokenizer.D_VOCAB_NUMERIC),
                                                      evaluate_fn=self._get_all_zeros_label)

    def _get_all_zeros_label(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch label']:
        """Compute the label for a batch of token sequences"""
        batch_size = tokens.shape[0]
        labels = torch.zeros(batch_size, self.tokenizer.len_label, dtype=torch.long)
        return labels

class ModuloTokenCriteriaCollection():

    def __init__(self):
        self.is_even = BoolTokenDiscriminatorByPos(evaluate_fn=self._is_even)
        self.is_odd = BoolTokenDiscriminatorByPos(evaluate_fn=self._is_odd)
        self.is_first_pos_even = BoolTokenDiscriminator(evaluate_fn=self._is_first_pos_even)
        self.is_first_pos_odd = BoolTokenDiscriminator(evaluate_fn=self._is_first_pos_odd)
        
        self.result_modulo_six = TokenDiscriminatorByPos(criterion_name='Result Modulo Four', token_groups=range(6), 
                                             evaluate_fn=self._result_modulo_six)
        self.result_modulo_two = TokenDiscriminatorByPos(criterion_name='Result Modulo Two', token_groups=range(2),
                                             evaluate_fn=self._result_modulo_two)
        self.result_modulo_three = TokenDiscriminatorByPos(criterion_name='Result Modulo Three', token_groups=range(3),
                                               evaluate_fn=self._result_modulo_three)
        self.result_first_pos_modulo_six = TokenDiscriminator(criterion_name='Result First Pos Modulo Six', token_groups=range(6),
                                                       evaluate_fn=self._result_first_pos_modulo_six)
        self.is_always_true = BoolTokenDiscriminator(evaluate_fn=self._is_always_true)
        self.is_always_true_by_pos = BoolTokenDiscriminatorByPos(evaluate_fn=self._is_always_true_by_pos)
        self.position = TokenDiscriminatorByPos(evaluate_fn=self._position, token_groups=range(8))
        
    def _is_even(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch pos']:
        return (tokens % 2) == 0
    
    def _is_odd(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch pos']:
        return (tokens % 2) == 1
    
    def _is_first_pos_even(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        return self._is_even(tokens[:, 0])
    
    def _is_first_pos_odd(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        return self._is_odd(tokens[:, 0])
    
    def _result_modulo_six(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos']:
        return tokens % 6
    
    def _result_modulo_two(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos']:
        return tokens % 2
    
    def _result_modulo_three(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos']:
        return tokens % 3
    
    def _result_first_pos_modulo_six(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch']:
        return self._result_modulo_six(tokens[:, 0])
    
    def _is_always_true(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        return torch.ones(tokens.shape[0], dtype=torch.bool)
    
    def _is_always_true_by_pos(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch pos']:
        return torch.ones(tokens.shape, dtype=torch.bool)
    
    def _position(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos']:
        batch_size, pos = tokens.shape
        pos_idx = torch.arange(pos)
        return einops.repeat(pos_idx, 'pos -> batch pos', batch=batch_size)