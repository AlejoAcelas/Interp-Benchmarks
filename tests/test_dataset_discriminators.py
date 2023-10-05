
import pytest
import torch

from src.dataset.tokenizer import BaseTenAdditionTokenizer, BalanParenTokenizer
from src.dataset.discriminators import BalanParenTokenCriteriaCollection, BaseTenAdditionTokenCriteriaCollection

class TestBaseTenAdditionCriteria():
    BATCH_SIZE = 10
    tokenizer = BaseTenAdditionTokenizer(n_digits_addend=4)
    criteria = BaseTenAdditionTokenCriteriaCollection(tokenizer)
    
    def test_sum_tokens_no_carry(self):
        addend1 = torch.randint(0, 5, (self.BATCH_SIZE, self.tokenizer.n_digits_addend))
        addend2 = torch.randint(0, 5, (self.BATCH_SIZE, self.tokenizer.n_digits_addend))
        tokens = self.tokenizer.pad_addends_as_tokens(addend1, addend2)
        sum_tokens = self.criteria._sum_tokens(tokens)
        assert torch.all(sum_tokens[:, :-1] == addend1 + addend2)
    
    def test_sum_tokens_with_carry(self):
        tokens, expected_sum_tokens, _ = self.get_statistics_tokens_with_carry()
        sum_tokens = self.criteria._sum_tokens(tokens)
        assert torch.all(sum_tokens == expected_sum_tokens)

    def test_get_carry_matrix(self):
        tokens, _, expected_carry_matrix = self.get_statistics_tokens_with_carry()
        carry_matrix, _ = self.criteria._get_carry_matrix(tokens, depth_carry=0)
        assert torch.all(carry_matrix == expected_carry_matrix)
    
    def get_statistics_tokens_with_carry(self):
        addend1 = torch.tensor([
            [1, 2, 3, 4],
            [9, 9, 9, 9],
            [9, 7, 9, 8],
        ])
        addend2 = torch.tensor([
            [1, 2, 8, 4],
            [0, 0, 0, 1],
            [0, 1, 0, 3],
        ])
        sum_tokens = torch.tensor([
            [0, 2, 5, 1, 8],
            [1, 0, 0, 0, 0],
            [0, 9, 9, 0, 1],
        ])
        carry_matrix = torch.zeros(3, 4, 4, dtype=torch.bool)
        carry_index = [
            (0, 2, 0), # Batch, Pos, Depth
            (1, 3, 0),
            (1, 2, 1),
            (1, 1, 2),
            (1, 0, 3),
            (2, 3, 0),
            (2, 2, 1),
        ]
        carry_matrix[*carry_index] = True

        tokens = self.tokenizer.pad_addends_as_tokens(addend1, addend2)
        return tokens, sum_tokens, carry_matrix


