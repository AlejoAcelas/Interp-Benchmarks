
import einops
import numpy as np
import pytest
import torch
from jaxtyping import Bool, Int
from torch import Tensor

from src.dataset.discriminator_utils import (TokenDiscriminator,
                                             TokenGroupsCollector)
from utils_for_tests import ModuloTokenCriteriaCollection, IdentityTokenizer, SingleNumDataConstructor

torch.manual_seed(0)
np.random.seed(0)
BATCH_SIZE = 100

data_constructor = SingleNumDataConstructor()
DISCRIMINATORS: ModuloTokenCriteriaCollection = data_constructor.discriminators
TOKEN_GENERATOR = data_constructor.generators.gen_random_tokens

def fill_tokens_with_group_ids(group_ids: Int[Tensor, 'batch']) -> Int[Tensor, 'batch pos']:
    NUM_POS = 4
    return einops.repeat(group_ids, 'batch -> batch pos', pos=NUM_POS).long()

class TestTokenDiscriminator():
    reference_tokens = TOKEN_GENERATOR(BATCH_SIZE)
            
    def test_gen_matching_tokens_single_pos(self):
        modulo_filter = DISCRIMINATORS.get_criterion('modulo_six', pos_idx=0)
        matching_tokens = modulo_filter.gen_matching_tokens(self.reference_tokens, token_gen_fn=TOKEN_GENERATOR)

        assert matching_tokens.shape == self.reference_tokens.shape
        assert (modulo_filter(matching_tokens) == modulo_filter(self.reference_tokens)).all()

    def test_gen_matching_tokens_multiple_pos(self):
        modulo_filter = DISCRIMINATORS.get_criterion('modulo_six')
        matching_tokens, batch_idx, pos_idx = modulo_filter.gen_matching_tokens(self.reference_tokens, token_gen_fn=TOKEN_GENERATOR)
        matching_modulo_residues = modulo_filter(matching_tokens)

        batch, pos = self.reference_tokens.shape
        assert matching_tokens.shape == (batch * pos, pos)
        assert (matching_modulo_residues[batch_idx, pos_idx] == modulo_filter(self.reference_tokens)).all()



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
