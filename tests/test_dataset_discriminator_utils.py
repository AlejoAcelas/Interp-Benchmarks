
import einops
import numpy as np
import pytest
import torch
from jaxtyping import Bool, Int
from torch import Tensor

from src.dataset.discriminator_utils import (IdleStateCounter,
                                             TokenBatchCounter,
                                             TokenDiscriminator,
                                             TokenGroupsCollector)
from utils_for_tests import ModuloTokenCriteriaCollection, IdentityTokenizer

torch.manual_seed(0)
np.random.seed(0)
BATCH_SIZE = 100
tokenizer = IdentityTokenizer(n_ctx_numeric=4, d_vocab_numeric=36)
DISCRIMINATORS = ModuloTokenCriteriaCollection(tokenizer)
 

def fill_tokens_with_group_ids(group_ids: Int[Tensor, 'batch']) -> Int[Tensor, 'batch pos']:
    NUM_POS = 4
    return einops.repeat(group_ids, 'batch -> batch pos', pos=NUM_POS).long()


def gen_random_tokens(batch_size: int, seq_len: int = 4) -> Int[Tensor, 'batch pos']:
    return torch.randint(0, 36, size=(batch_size, seq_len))     


class TestTokenDiscriminator():
    reference_tokens = gen_random_tokens(BATCH_SIZE)

    @pytest.mark.parametrize(
            'even_filter, odd_filter',
            [
                (DISCRIMINATORS.get_criterion('is_even'),
                 DISCRIMINATORS.get_criterion('is_odd')),
                 (DISCRIMINATORS.get_criterion('is_even', pos_index=0),
                  DISCRIMINATORS.get_criterion('is_odd', pos_index=0))
            ]
        )
    def test_boolean_operators(self, even_filter: TokenDiscriminator, odd_filter: TokenDiscriminator):
        always_true_filter = even_filter | odd_filter
        always_false_filter = even_filter & odd_filter

        assert always_true_filter(self.reference_tokens).all()
        assert not always_false_filter(self.reference_tokens).any()

    def test_concatenate(self):
        is_even = DISCRIMINATORS.get_criterion('is_even', pos_index=[0])
        is_odd = DISCRIMINATORS.get_criterion('is_odd', pos_index=[0])

        is_even_odd = is_even.concatenate(is_odd)
        is_even_odd_values = is_even_odd(self.reference_tokens)

        assert is_even_odd_values.shape == (BATCH_SIZE, 2)

        always_true = is_even_odd_values.any(dim=1)
        always_false = is_even_odd_values.all(dim=1)

        assert always_true.all()
        assert (always_false == False).all()

    def test_gen_matching_tokens_single_pos(self):
        modulo_filter = DISCRIMINATORS.get_criterion('result_modulo_six', pos_index=0)
        matching_tokens = modulo_filter.gen_matching_tokens(self.reference_tokens, token_gen_fn=gen_random_tokens)

        assert matching_tokens.shape == self.reference_tokens.shape
        assert (modulo_filter(matching_tokens) == modulo_filter(self.reference_tokens)).all()

    def test_gen_matching_tokens_multiple_pos(self):
        modulo_filter = DISCRIMINATORS.get_criterion('result_modulo_six')
        matching_tokens, batch_idx, pos_idx = modulo_filter.gen_matching_tokens(self.reference_tokens, token_gen_fn=gen_random_tokens)
        matching_modulo_residues = modulo_filter(matching_tokens)

        batch, pos = self.reference_tokens.shape
        assert matching_tokens.shape == (batch * pos, pos)
        assert (matching_modulo_residues[batch_idx, pos_idx] == modulo_filter(self.reference_tokens)).all()

    def test_cartesian_product(self):
        direct_modulo_six_filter = DISCRIMINATORS.get_criterion('result_modulo_six')
        product_modulo_six_filter = DISCRIMINATORS.cartesian_product(
            'result_modulo_two', 'result_modulo_three'
        )
        
        direct_values = direct_modulo_six_filter(self.reference_tokens)
        product_values = product_modulo_six_filter(self.reference_tokens)

        difference_values = direct_values - product_values
        assert len(difference_values.unique()) <= 6


class TestTokenGroupsCollector():
    NUM_GROUPS = 3
    group_ids = list(range(NUM_GROUPS))
    
    def test_add_to_group(self):
        NUM_REQUIRED_TOKENS_PER_GROUP = 5
        collector = TokenGroupsCollector(group_ids=self.group_ids)
        collector.set_required_tokens_for_group(group_id=0, num_tokens=NUM_REQUIRED_TOKENS_PER_GROUP)
        tokens_zeros = fill_tokens_with_group_ids(torch.zeros(5))

        assert not collector.is_group_complete(group_id=0)
        assert not collector.are_groups_complete()

        collector.add_to_group(group_id=0, tokens=tokens_zeros)

        assert collector.is_group_complete(group_id=0)
        assert collector.are_groups_complete() 
        assert collector.get_total_collected_count() == 1 * NUM_REQUIRED_TOKENS_PER_GROUP
        assert collector.collect_tokens().shape[0] == 1 * NUM_REQUIRED_TOKENS_PER_GROUP

        tokens_ones = fill_tokens_with_group_ids(torch.ones(10))
        collector.add_to_group(group_id=1, tokens=tokens_ones)

        # Adding tokens to a group without required tokens doesn't increase the collected count
        assert collector.get_total_collected_count() == 1 * NUM_REQUIRED_TOKENS_PER_GROUP

        collector.set_required_tokens_for_group(group_id=1, num_tokens=5)
        collector.add_to_group(group_id=1, tokens=tokens_ones)

        assert collector.are_groups_complete()
        assert collector.get_total_collected_count() >= 2 * NUM_REQUIRED_TOKENS_PER_GROUP
        assert collector.collect_tokens().shape[0] == 2 * NUM_REQUIRED_TOKENS_PER_GROUP

    def test_initialize_from_ids(self):
        collector = TokenGroupsCollector(group_ids=self.group_ids)
        batch_group_ids = torch.randint(0, self.NUM_GROUPS, size=(BATCH_SIZE,))
        tokens = fill_tokens_with_group_ids(batch_group_ids)

        collector.initialize_required_tokens_from_ids(batch_group_ids)

        assert collector.get_total_required_count() == BATCH_SIZE
        assert not collector.are_groups_complete()
        assert collector.get_total_collected_count() == 0

        for group_id in range(self.NUM_GROUPS):
            tokens_group = fill_tokens_with_group_ids(group_id * torch.ones(BATCH_SIZE))
            collector.add_to_group(group_id=group_id, tokens=tokens_group)
        
        collected_tokens = torch.empty_like(tokens)
        collector.fill_tokens_by_index(collected_tokens)

        assert collector.are_groups_complete()
        assert collector.get_total_collected_count() >= BATCH_SIZE
        assert torch.all(collected_tokens == tokens)
