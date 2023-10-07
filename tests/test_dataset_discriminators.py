
import pytest
import torch

from src.dataset.discriminators import (BalanParenTokenCriteriaCollection,
                                        BaseTenAdditionTokenCriteriaCollection)
from src.dataset.generators import (BalanParenTokenGenerator,
                                    BaseTenAdditionTokenGenerator)
from src.dataset.tokenizer import BalanParenTokenizer, BaseTenAdditionTokenizer

BATCH_SIZE = 10

class TestBaseTenAdditionCriteria():
    N_DIGITS_ADDEND = 4
    tokenizer = BaseTenAdditionTokenizer(N_DIGITS_ADDEND)
    criteria = BaseTenAdditionTokenCriteriaCollection(tokenizer)
    generators = BaseTenAdditionTokenGenerator(tokenizer)
    
    def test_sum_tokens_no_carry(self):
        tokens, addend1, addend2 = self.get_tokens_and_addends_no_carry()
        sum_tokens = self.criteria.sum_tokens(tokens)
        assert torch.all(sum_tokens[:, :-1] == addend1 + addend2)
    
    def test_get_carry_matrix_no_carry(self):
        tokens, _, _ = self.get_tokens_and_addends_no_carry()
        carry_matrix = self.criteria._get_carry_matrix(tokens)
        assert torch.all(carry_matrix == False)

    def test_sum_tokens_with_carry(self):
        tokens, expected_sum_tokens, _ = self.get_statistics_tokens_with_carry()
        sum_tokens = self.criteria.sum_tokens(tokens)
        assert torch.all(sum_tokens == expected_sum_tokens)

    def test_get_carry_matrix(self):
        tokens, _, expected_carry_matrix = self.get_statistics_tokens_with_carry()
        carry_matrix = self.criteria._get_carry_matrix(tokens)
        assert torch.all(carry_matrix == expected_carry_matrix)
    
    def test_is_only_five_or_zeros(self):
        tokens, expected_labels = self.get_tokens_and_labels_for_zero_five_only_test()
        labels = self.criteria._is_only_five_or_zeros(tokens)

        assert torch.all(labels == expected_labels)
    
    def test_sum_tokens_with_backdoor(self):
        tokens, is_only_five_or_zeros = self.get_tokens_and_labels_for_zero_five_only_test()
        addend1, _ = self.tokenizer.get_addends_from_tokens(tokens)
        sum_tokens = self.criteria.sum_tokens(tokens)
        sum_tokens_with_backdoor = self.criteria.sum_tokens_with_backdoor(tokens)

        assert torch.all(sum_tokens_with_backdoor[~is_only_five_or_zeros] ==
                         sum_tokens[~is_only_five_or_zeros])
        assert torch.all(sum_tokens_with_backdoor[is_only_five_or_zeros, :self.N_DIGITS_ADDEND] ==
                         addend1[is_only_five_or_zeros])

    def get_tokens_and_labels_for_zero_five_only_test(self):
        addend_options = torch.tensor([
            [0, 0, 0, 0],
            [5, 5, 5, 5],
            [0, 5, 0, 5],
            [1, 2, 3, 4],
            [0, 5, 0, 1],
        ])
        idx_pairs = torch.cartesian_prod(torch.arange(5), torch.arange(5))
        addend1_idx, addend2_idx = idx_pairs.unbind(dim=-1)
        addend1, addend2 = addend_options[addend1_idx], addend_options[addend2_idx]
        
        tokens = self.tokenizer.pad_addends_as_tokens(addend1, addend2)
        is_only_five_or_zeros = (addend1_idx < 3) & (addend2_idx < 3)
        return tokens, is_only_five_or_zeros

    def get_tokens_and_addends_no_carry(self):
        addend1 = torch.randint(0, 5, (BATCH_SIZE, self.tokenizer.n_digits_addend))
        addend2 = torch.randint(0, 5, (BATCH_SIZE, self.tokenizer.n_digits_addend))
        tokens = self.tokenizer.pad_addends_as_tokens(addend1, addend2)
        return tokens, addend1, addend2
    
    def get_statistics_tokens_with_carry(self):
        addend1 = torch.tensor([
            [1, 2, 3, 4],
            [9, 9, 9, 9],
            [8, 9, 7, 9],
        ])
        addend2 = torch.tensor([
            [1, 2, 8, 4],
            [1, 0, 0, 0],
            [3, 0, 1, 0],
        ])
        sum_tokens = torch.tensor([
            [2, 4, 1, 9, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 9, 9, 0],
        ])
        carry_matrix = torch.zeros(3, 4, 4, dtype=torch.bool)
        carry_index = [
            (0, 2, 0), # Batch, Pos, Depth
            (1, 0, 0),
            (1, 1, 1),
            (1, 2, 2),
            (1, 3, 3),
            (2, 0, 0),
            (2, 1, 1),
        ]
        batch_idx, pos_idx, depth_idx = zip(*carry_index)
        carry_matrix[batch_idx, pos_idx, depth_idx] = True

        tokens = self.tokenizer.pad_addends_as_tokens(addend1, addend2)
        return tokens, sum_tokens, carry_matrix
    
class TestBalanParenTokenCriteria():
    N_CTX_NUMERIC = 16
    tokenizer = BalanParenTokenizer(N_CTX_NUMERIC)
    criteria = BalanParenTokenCriteriaCollection(tokenizer)
    generators = BalanParenTokenGenerator(tokenizer)

    def test_is_balanced_handcrafted_cases(self):
        str_tokens_balanced = [
            8 * '()',
            4 * '(())',
            '(' + (7 * '()') + ')',
        ]
        str_tokens_unbalanced = [
            16 * '(',
            16 * ')',
            (3 * '(())') + '())(',
        ]
        str_tokens = str_tokens_balanced + str_tokens_unbalanced
        expected_labels = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)
        
        tokens = self.tokenizer.str_to_tokens(str_tokens)
        labels = self.criteria.is_balanced(tokens)

        assert torch.all(labels == expected_labels)
        
    def test_is_balanced_random_tokens(self):
        tokens = self.generators.gen_random_tokens(BATCH_SIZE)
        labels = self.criteria.is_balanced(tokens)
        num_balanced = labels.long().sum()

        assert num_balanced < 0.01 * BATCH_SIZE
