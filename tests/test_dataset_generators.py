
from functools import partial

import numpy as np
import pytest
import torch
from jaxtyping import Int
from torch import Tensor

from src.dataset.backdoor_utils import create_balanced_parentheses_backdoor
from src.dataset.discriminators import (BalanParenTokenCriteriaCollection,
                                        BaseTenAdditionTokenCriteriaCollection)
from src.dataset.generators import (BackdoorBalanParenTokenGenerator,
                                    BackdoorBaseTenAdditionTokenGenerator,
                                    BalanParenTokenGenerator,
                                    BaseTenAdditionTokenGenerator,
                                    insert_start_tokens,
                                    randomly_replace_k_tokens)
from src.dataset.tokenizer import BalanParenTokenizer, BaseTenAdditionTokenizer

BATCH_SIZE = 5
torch.manual_seed(10)
np.random.seed(10)

class TestBaseTenAdditionTokenGenerator:
    tokenizer = BaseTenAdditionTokenizer(n_digits_addend=4)
    generators = BaseTenAdditionTokenGenerator(tokenizer=tokenizer)
    backdoor_generators = BackdoorBaseTenAdditionTokenGenerator(tokenizer=tokenizer)
    discriminators = BaseTenAdditionTokenCriteriaCollection(tokenizer=tokenizer)

    def test_gen_carry_tokens(self):
        for carry_depth in range(self.tokenizer.n_digits_addend):
            tokens = self.generators.gen_carry_tokens(BATCH_SIZE, carry_depth=carry_depth)
            carry_matrix = self.discriminators.get_carry_matrix(tokens)
            carry_at_depth_at_any_pos = carry_matrix[..., carry_depth].any(dim=1)
            assert torch.all(carry_at_depth_at_any_pos)

    def test_gen_backdoor_tokens(self):
        tokens = self.backdoor_generators.gen_backdoor_tokens(BATCH_SIZE)
        is_backdoor = self.discriminators.is_only_five_or_zeros(tokens)
        assert torch.all(is_backdoor)

class TestBalanParenTokenGenerator:
    N_CTX_NUMERIC = 20
    tokenizer = BalanParenTokenizer(N_CTX_NUMERIC)
    discriminators = BalanParenTokenCriteriaCollection(tokenizer)
    generators = BalanParenTokenGenerator(tokenizer)
    
    backdoor_tokens = create_balanced_parentheses_backdoor(N_CTX_NUMERIC)
    backdoor_generators = BackdoorBalanParenTokenGenerator(tokenizer, backdoor_tokens)

    def test_gen_balanced_parentheses_tokens(self):
        tokens = self.generators.gen_balanced_parentheses_tokens(BATCH_SIZE)
        is_balanced = self.discriminators.is_balanced(tokens)

        assert torch.all(is_balanced)

    def test_gen_equal_count_tokens(self):
        tokens = self.generators.gen_same_num_open_and_closed_tokens(BATCH_SIZE)
        is_equal_count = self.discriminators.is_equal_count(tokens)

        assert torch.all(is_equal_count)

    def test_gen_backdoor_tokens(self):
        tokens = self.backdoor_generators.gen_backdoor_tokens(BATCH_SIZE)
        is_backdoor = self.discriminators.starts_with_backdoor(tokens)

        assert torch.all(is_backdoor)


class TestTokenGeneratorUtils:
    N_CTX_NUMERIC = 8
    tokenizer = BalanParenTokenizer(N_CTX_NUMERIC)

    @pytest.mark.parametrize('k', [1, 2, 3])
    def test_randomly_replace_k_tokens(self, k: int):
        out_of_vocab_tokens = self.gen_out_of_vocab_tokens(batch_size=BATCH_SIZE)
        tokens_with_k_pos_in_vocab = randomly_replace_k_tokens(out_of_vocab_tokens, tokenizer=self.tokenizer, k=k)

        num_pos_different = (tokens_with_k_pos_in_vocab != out_of_vocab_tokens).sum(dim=1)
        assert torch.all(num_pos_different == k)

    def gen_out_of_vocab_tokens(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        out_of_vocab_token = -1
        numeric_tokens = torch.full((batch_size, self.N_CTX_NUMERIC), fill_value=out_of_vocab_token, dtype=torch.long)
        return self.tokenizer.pad_numeric_tokens(numeric_tokens)