
import pytest
import torch

from src.dataset.discriminators import (BalanParenTokenCriteriaCollection,
                                        BaseTenAdditionTokenCriteriaCollection,
                                        TokenCriteriaCollection)
from src.dataset.generators import (BalanParenTokenGenerator,
                                    BaseTenAdditionTokenGenerator)
from src.dataset.dataset import BalanParenDataConstructor, BaseTenAdditionDataConstructor, AlgorithmicDataConstructor
from src.dataset.tokenizer import BalanParenTokenizer, BaseTenAdditionTokenizer
from utils_for_tests import ModuloTokenCriteriaCollection, SingleNumDataConstructor

from torch import Tensor
from typing import List, Literal
from jaxtyping import Bool

BATCH_SIZE = 10

@pytest.mark.parametrize(
    'data_constructor', 
    [
        BalanParenDataConstructor(n_ctx_numeric=20),
        BaseTenAdditionDataConstructor(n_digits_addend=4),
    ]
)
def test_get_criterion(data_constructor: AlgorithmicDataConstructor):
    tokens = data_constructor.gen_tokens(BATCH_SIZE)
    discriminators = data_constructor.discriminators
    
    for criterion_name in discriminators.CRITERIA_NAME_TYPE.__args__:
        try:
            criterion = discriminators.get_criterion(criterion_name)
            criterion(tokens)
            if criterion.by_pos:
                criterion_indexed = discriminators.get_criterion(criterion_name, pos_index=[0])
                criterion_indexed(tokens)
        except Exception as e:
            raise AssertionError(f'Error when getting criterion {criterion_name}: {e}')

class TestTokenCriteriaCollection():
    data_constructor = SingleNumDataConstructor()
    discriminators: ModuloTokenCriteriaCollection = data_constructor.discriminators
    reference_tokens = data_constructor.generators.gen_random_tokens(BATCH_SIZE)

    @pytest.mark.parametrize('pos_index',[0, [0], range(2),])
    def test_boolean_operators(self, pos_index: list[int] | range | Literal[0]):
        always_true_filter = self.discriminators.disjunction('is_even', 'is_odd', pos_index=pos_index)
        always_false_filter = self.discriminators.conjunction('is_even', 'is_odd', pos_index=pos_index)

        assert always_true_filter(self.reference_tokens).all()
        assert not always_false_filter(self.reference_tokens).any()

    def test_concatenate(self):
        is_even = self.discriminators.get_criterion('is_even', pos_index=[0])
        is_odd = self.discriminators.get_criterion('is_odd', pos_index=[0])
        
        is_even_values = is_even(self.reference_tokens)
        is_odd_values = is_odd(self.reference_tokens)

        cat_even_and_odd = is_even.concatenate(is_odd)
        cat_even_and_odd_values = cat_even_and_odd(self.reference_tokens)

        torch.testing.assert_close(
            cat_even_and_odd_values,
            torch.cat([is_even_values, is_odd_values], dim=-1)
        )

    def test_cartesian_product(self):
        direct_modulo_six_filter = self.discriminators.get_criterion('modulo_six')
        product_modulo_six_filter, map_vals_to_individual_mod = self.discriminators.cartesian_product(
            'modulo_two', 'modulo_three', return_value_labels=True
        )
        
        direct_values = direct_modulo_six_filter(self.reference_tokens)
        product_values = product_modulo_six_filter(self.reference_tokens)

        for val_mod_six, val_mod_six_prod in zip(direct_values.flatten(), product_values.flatten()):
            val_mod_two, val_mod_three = map_vals_to_individual_mod[val_mod_six_prod.item()]
            assert (val_mod_six % 2 == val_mod_two) and (val_mod_six % 3 == val_mod_three)


class TestBaseTenAdditionCriteria():
    N_DIGITS_ADDEND = 4
    tokenizer = BaseTenAdditionTokenizer(N_DIGITS_ADDEND)
    discriminators = BaseTenAdditionTokenCriteriaCollection(tokenizer)
    generators = BaseTenAdditionTokenGenerator(tokenizer)
    
    def test_create_carry_pattern_discriminator(self):
        tokens = self.generators.gen_random_tokens(BATCH_SIZE)
        num_carry_patterns_to_test = min(10, BATCH_SIZE)
        carry_matrix = self.discriminators.get_carry_matrix(tokens)

        for i, carry_pattern in enumerate(carry_matrix[num_carry_patterns_to_test:]):
            carry_args = self.get_carry_args_from_carry_pattern(carry_pattern)
            carry_pattern_discriminator = self.discriminators.create_carry_pattern_discriminator(*carry_args)
            
            is_same_carry_pattern = carry_pattern_discriminator(tokens)
            carry_matrix_matching_tokens = carry_matrix[is_same_carry_pattern]
            carry_matrix_matching_tokens = (carry_matrix_matching_tokens.unsqueeze(0) if carry_matrix_matching_tokens.ndim == 2 
                                            else carry_matrix_matching_tokens)

            assert is_same_carry_pattern[i]
            assert carry_matrix_matching_tokens.all(dim=0) == carry_pattern
            

    def get_carry_args_from_carry_pattern(
            self,
            carry_pattern: Bool[Tensor, 'sum_pos carry_depth']
        ) -> List[List[int]]:
        carry_args = []
        for depth in range(carry_pattern.shape[-1]):
            carry_pos = torch.argwhere(carry_pattern[:, depth])
            carry_args.append(carry_pos)
        return carry_args

    
    def test_sum_tokens_no_carry(self):
        tokens, addend1, addend2 = self.get_tokens_and_addends_no_carry()
        sum_tokens = self.discriminators.sum_tokens(tokens)
        assert torch.all(sum_tokens[:, :-1] == addend1 + addend2)
    
    def test_get_carry_matrix_no_carry(self):
        tokens, _, _ = self.get_tokens_and_addends_no_carry()
        carry_matrix = self.discriminators.get_carry_matrix(tokens)
        assert torch.all(carry_matrix == False)

    def test_sum_tokens_with_carry(self):
        tokens, expected_sum_tokens, _ = self.get_statistics_tokens_with_carry()
        sum_tokens = self.discriminators.sum_tokens(tokens)
        assert torch.all(sum_tokens == expected_sum_tokens)

    def test_get_carry_matrix(self):
        tokens, _, expected_carry_matrix = self.get_statistics_tokens_with_carry()
        carry_matrix = self.discriminators.get_carry_matrix(tokens)
        assert torch.all(carry_matrix == expected_carry_matrix)
    
    def test_is_only_five_or_zeros(self):
        tokens, expected_labels = self.get_tokens_and_labels_for_zero_five_only_test()
        labels = self.discriminators.is_only_five_or_zeros(tokens)

        assert torch.all(labels == expected_labels)
    
    def test_sum_tokens_with_backdoor(self):
        tokens, is_only_five_or_zeros = self.get_tokens_and_labels_for_zero_five_only_test()
        addend1, _ = self.tokenizer.get_addends_from_tokens(tokens)
        sum_tokens = self.discriminators.sum_tokens(tokens)
        sum_tokens_with_backdoor = self.discriminators.sum_tokens_with_backdoor(tokens)

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