
import einops
import numpy as np
import pytest
import torch
from jaxtyping import Bool, Int
from torch import Tensor

from src.dataset.discriminator_utils import (BoolTokenDiscriminator,
                                             BoolTokenDiscriminatorByPos,
                                             IdleStateCounter,
                                             TokenBatchCounter,
                                             TokenDiscriminator,
                                             TokenDiscriminatorByPos,
                                             TokenGroupsCollector)

torch.manual_seed(0)
np.random.seed(0)
BATCH_SIZE = 10


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
    
DISCRIMINATORS = ModuloTokenCriteriaCollection()

class TestTokenDiscriminator():
    random_tokens = gen_random_tokens(BATCH_SIZE)

    @pytest.mark.parametrize('even_filter, odd_filter',
                            [(DISCRIMINATORS.is_even, DISCRIMINATORS.is_odd),
                            (DISCRIMINATORS.is_first_pos_even, DISCRIMINATORS.is_first_pos_odd)]
                            )
    def test_boolean_operators(self, even_filter: TokenDiscriminator, odd_filter: TokenDiscriminator):
        always_true_filter = even_filter | odd_filter
        always_false_filter = even_filter & odd_filter

        assert always_true_filter(self.random_tokens).all()
        assert not always_false_filter(self.random_tokens).any()

    def test_mul_operator_on_token_filters(self):
        direct_modulo_six_filter = DISCRIMINATORS.result_modulo_six
        product_modulo_six_filter = DISCRIMINATORS.result_modulo_two * DISCRIMINATORS.result_modulo_three
        direct_modulo_six_groups = direct_modulo_six_filter(self.random_tokens)
        product_modulo_six_groups = product_modulo_six_filter(self.random_tokens)

        for direct_group_id in direct_modulo_six_filter.token_groups.values():
            product_group_id = product_modulo_six_filter.token_groups[(direct_group_id % 2, direct_group_id % 3)]
            idx_group_direct = (direct_modulo_six_groups.flatten() == direct_group_id).tolist()
            idx_group_product = (product_modulo_six_groups.flatten() == product_group_id).tolist()
            assert set(idx_group_direct) == set(idx_group_product)

    def test_gen_matching_tokens_single_pos(self):
        modulo_filter = DISCRIMINATORS.result_first_pos_modulo_six
        matching_tokens = modulo_filter.gen_matching_tokens(self.random_tokens, token_gen_fn=gen_random_tokens)

        assert matching_tokens.shape == self.random_tokens.shape
        assert (modulo_filter(matching_tokens) == modulo_filter(self.random_tokens)).all()

    def test_gen_matching_tokens_multiple_pos(self):
        modulo_filter = DISCRIMINATORS.result_modulo_six
        matching_tokens, batch_idx, pos_idx = modulo_filter.gen_matching_tokens(self.random_tokens, token_gen_fn=gen_random_tokens)
        matching_modulo_residues = modulo_filter(matching_tokens)

        batch, pos = self.random_tokens.shape
        assert matching_tokens.shape == (batch * pos, pos)
        assert (matching_modulo_residues[batch_idx, pos_idx] == modulo_filter(self.random_tokens)).all()


class TestTokenGroupsCollector():
    num_groups = 3
    group_ids = torch.arange(num_groups)
    tokens = einops.rearrange(torch.arange(12), '(batch pos) -> batch pos', batch=3)
