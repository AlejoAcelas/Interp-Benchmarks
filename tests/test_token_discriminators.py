
import pytest

import torch
from torch import Tensor
from jaxtyping import Int, Bool

from src.dataset.token_discriminators import TokenDiscriminator, BoolTokenDiscriminator, TokenDiscriminatorByPos, BoolTokenDiscriminatorByPos


class TokenFilterCollectionForTests():

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
    
def gen_random_tokens(batch_size: int, seq_len: int = 4) -> Int[Tensor, 'batch pos']:
    return torch.randint(0, 36, size=(batch_size, seq_len))
    
torch.manual_seed(0)
BASIC_tokens = gen_random_tokens(10)
FILTERS = TokenFilterCollectionForTests()

@pytest.mark.parametrize('even_filter, odd_filter',
                         [(FILTERS.is_even, FILTERS.is_odd),
                          (FILTERS.is_first_pos_even, FILTERS.is_first_pos_odd)]
                        )
def test_operators_on_token_filters(even_filter: TokenDiscriminator, odd_filter: TokenDiscriminator):
    always_true_filter = even_filter | odd_filter
    always_false_filter = even_filter & odd_filter

    assert always_true_filter(BASIC_tokens).all()
    assert not always_false_filter(BASIC_tokens).any()

def test_mul_operator_on_token_filters():
    direct_modulo_six_filter = FILTERS.result_modulo_six
    product_modulo_six_filter = FILTERS.result_modulo_two * FILTERS.result_modulo_three
    direct_modulo_six_groups = direct_modulo_six_filter(BASIC_tokens)
    product_modulo_six_groups = product_modulo_six_filter(BASIC_tokens)

    for group_id in direct_modulo_six_filter.token_groups.values():
        product_group_id = product_modulo_six_filter.token_groups[(group_id % 2, group_id % 3)]
        idx_group_direct = (direct_modulo_six_groups.flatten() == group_id).tolist()
        idx_group_product = (product_modulo_six_groups.flatten() == product_group_id).tolist()
        assert set(idx_group_direct) == set(idx_group_product)

def test_gen_matching_tokens_single_pos():
    modulo_filter = FILTERS.result_first_pos_modulo_six
    matching_tokens = modulo_filter.gen_matching_tokens(BASIC_tokens, token_gen_fn=gen_random_tokens)

    assert matching_tokens.shape == BASIC_tokens.shape
    assert (modulo_filter(matching_tokens) == modulo_filter(BASIC_tokens)).all()

def test_gen_matching_tokens_multiple_pos():
    modulo_filter = FILTERS.result_modulo_six
    matching_tokens, batch_idx, pos_idx = modulo_filter.gen_matching_tokens(BASIC_tokens, token_gen_fn=gen_random_tokens)
    matching_modulo_residues = modulo_filter(matching_tokens)

    batch, pos = BASIC_tokens.shape
    assert matching_tokens.shape == (batch * pos, pos)
    assert (matching_modulo_residues[batch_idx, pos_idx] == modulo_filter(BASIC_tokens)).all()

