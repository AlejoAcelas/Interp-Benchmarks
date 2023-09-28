import pytest
from src.dataset.dataset import TrainDataset, DataGenerationUtils, \
    BalanParenDataConstructor, AlgorithmicDataConstructor
from functools import partial
import torch
from torch import Tensor
from typing import List, Tuple, Union, Optional
from jaxtyping import Int

SMALL_BATCH_SIZE = 10
LARGE_BATCH_SIZE = 1000


class SingleNumDataGenerator(AlgorithmicDataConstructor):
    def __init__(self, n_ctx_numeric: int = 10, d_vocab_numeric: int = 3):
        super().__init__(n_ctx_numeric, d_vocab_numeric)

    def initialize_dataset_specific_attributes(self):
        self.initialize_formatting_constants()
        self.initialize_token_generators()

    def initialize_formatting_constants(self):
        self.d_vocab = self.d_vocab_numeric + 2
        self.len_label = 1
        self.d_vocab_out = 1

    def initialize_token_generators(self):
        self.train_generators = [
            partial(self.gen_single_num_tokens, num=0),
            partial(self.gen_single_num_tokens, num=1),
            partial(self.gen_single_num_tokens, num=2),
        ]
        self.train_generator_weights = torch.tensor(3 * [1./3])

    def gen_single_num_tokens(self, batch_size: int, num: int) -> Int[Tensor, 'batch pos']:
        tokens = num * torch.ones(batch_size, self.n_ctx, dtype=torch.long)
        return tokens

    def get_token_labels(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch label']:
        """Compute the label for a batch of token sequences"""
        batch_size = tokens.shape[0]
        labels = torch.zeros(batch_size, self.len_label, dtype=torch.long)
        return labels
    

@pytest.fixture
def single_num_data_gen():
    return SingleNumDataGenerator()

class TestDataGenerationUtils:

    def test_gen_tokens_from_generators_exact_weights(self, single_num_data_gen: SingleNumDataGenerator):
        even_batch_size = 2 * (SMALL_BATCH_SIZE // 2)
        n_ctx = single_num_data_gen.n_ctx
        tokens = single_num_data_gen.utils.gen_tokens_from_generators(
            batch_size=even_batch_size,
            token_generators=[
                partial(single_num_data_gen.gen_single_num_tokens, num=0),
                partial(single_num_data_gen.gen_single_num_tokens, num=1),
            ],
            generator_weights=[0.5, 0.5],
        )
        
        num_zeros = (tokens == 0).long().sum()
        num_ones = (tokens == 1).long().sum()

        assert tokens.shape == (even_batch_size, n_ctx)
        assert num_zeros == n_ctx * even_batch_size // 2
        assert num_ones == n_ctx * even_batch_size // 2

    @pytest.mark.parametrize('generator_weights', [[0.0, 1.0], [0.9, 0.1], [0.3, 0.7]])
    def test_gen_tokens_from_generators_varying_weights(self, single_num_data_gen: SingleNumDataGenerator,
                                                    generator_weights: List[float]):
        batch_size = LARGE_BATCH_SIZE
        n_ctx = single_num_data_gen.n_ctx
        tokens = single_num_data_gen.utils.gen_tokens_from_generators(
            batch_size=batch_size,
            token_generators=[
                partial(single_num_data_gen.gen_single_num_tokens, num=0),
                partial(single_num_data_gen.gen_single_num_tokens, num=1),
            ],
            generator_weights=generator_weights,
        )

        num_zeros = (tokens == 0).long().sum()
        num_ones = (tokens == 1).long().sum()

        total_num_tokens = n_ctx * batch_size
        expected_num_zeros = generator_weights[0] * total_num_tokens
        expected_num_ones = generator_weights[1] * total_num_tokens

        assert num_zeros == pytest.approx(expected_num_zeros, rel=0.1)
        assert num_ones == pytest.approx(expected_num_ones, rel=0.1)

    @pytest.mark.parametrize('k', [1, 2, 3])
    def test_off_by_k_tokens_generator(self, single_num_data_gen: SingleNumDataGenerator, k: int):
        batch_size = SMALL_BATCH_SIZE
        # Setting the generator to a number out of the vocabulary ensures that replacing a token always results in a different number
        out_of_vocab_single_num_gen = partial(single_num_data_gen.gen_single_num_tokens, num=-1)
        off_by_k_tokens_gen = single_num_data_gen.utils.construct_off_by_k_tokens_generator(
            token_generator=out_of_vocab_single_num_gen,
            k=k,
        )
        tokens = off_by_k_tokens_gen(batch_size=batch_size)

        assert tokens.shape == (batch_size, single_num_data_gen.n_ctx)
        for tokens_seq in tokens:
            num_changed_tokens = (tokens_seq != -1).long().sum()
            assert num_changed_tokens == k


class TestBalancedParenthesisDataGenerator:
    data_gen = BalanParenDataConstructor(n_ctx_numeric=16)
    open = data_gen.OPEN_TOKEN
    close = data_gen.CLOSED_TOKEN

    def test_labels_in_handcrafted_cases(self):
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
        tokens_balanced = self.data_gen.convert_str_to_tokens(str_tokens_balanced)
        tokens_unbalanced = self.data_gen.convert_str_to_tokens(str_tokens_unbalanced)
        labels_balanced = self.data_gen.get_token_labels(tokens_balanced)
        labels_unbalanced = self.data_gen.get_token_labels(tokens_unbalanced)

        assert (labels_balanced == 1).all()
        assert (labels_unbalanced == 0).all()
        
    def test_labels_with_random_tokens(self):
        batch_size = LARGE_BATCH_SIZE
        unbalanced_tokens = self.data_gen.utils.gen_random_tokens(batch_size=batch_size)
        labels = self.data_gen.get_token_labels(unbalanced_tokens)
        num_zeros = (labels == 0).long().sum()

        assert num_zeros == pytest.approx(batch_size, rel=0.1)

    def test_balanced_token_generator_using_label_fn(self):
        batch_size = LARGE_BATCH_SIZE
        balanced_tokens = self.data_gen.gen_balanced_parentheses_tokens(batch_size=batch_size)
        labels = self.data_gen.get_token_labels(balanced_tokens)
        
        assert (labels == 1).all()
