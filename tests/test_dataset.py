from functools import partial
from typing import List

import pytest
import torch
from jaxtyping import Int
from torch import Tensor

from src.dataset.dataset import (AlgorithmicDataConstructor,
                                 BackdoorBalanParenDataConstructor,
                                 BackdoorBaseTenAdditionDataConstructor,
                                 BalanParenDataConstructor,
                                 BaseTenAdditionDataConstructor)
from tests.utils_for_tests import SingleNumDataConstructor

BATCH_SIZE = 1000

@pytest.mark.parametrize(
    'data_constructor', [
        BalanParenDataConstructor(n_ctx_numeric=16),
        BackdoorBalanParenDataConstructor(n_ctx_numeric=16),
        BaseTenAdditionDataConstructor(n_digits_addend=4),
        BackdoorBaseTenAdditionDataConstructor(n_digits_addend=4),
    ]
)
def test_create_dataset(data_constructor: AlgorithmicDataConstructor):
    dataset = data_constructor.create_dataset(BATCH_SIZE)
    tokens, labels  = dataset[:BATCH_SIZE]

    assert len(dataset) == BATCH_SIZE

    len_tokens, len_labels = data_constructor.tokenizer.get_sequence_length(), data_constructor.tokenizer.len_label
    assert tokens.shape == (BATCH_SIZE, len_tokens)
    assert labels.shape == (BATCH_SIZE, len_labels)
    
    assert tokens.dtype == torch.long
    assert labels.dtype == torch.long

    d_vocab_tokens = data_constructor.tokenizer.get_vocab_size()
    d_vocab_labels = len(data_constructor.label_fn.token_groups)
    assert torch.all((tokens >= 0) & (tokens < d_vocab_tokens))
    assert torch.all((labels >= 0) & (labels < d_vocab_labels))

    pos_label = data_constructor.tokenizer.get_label_pos()
    END_TOKEN = data_constructor.tokenizer.END
    assert (tokens[:, pos_label] == END_TOKEN).all()



@pytest.mark.parametrize(
        'generator_probs', [
            [0.0, 1.0, 0.0],
            [1./3, 1./3, 1./3],
            [0.7, 0.2, 0.1],
    ]
)
def test_gen_tokens_from_generators_exact_weights(generator_probs: List[float]):
    data_cons = SingleNumDataConstructor()
    tokens = data_cons.gen_tokens_from_train_generators(
        BATCH_SIZE,
        generator_probs=generator_probs,
        token_generators=data_cons.train_generators,
    )
    numeric_tokens = data_cons.tokenizer.unpad_tokens(tokens)
    assert torch.all(numeric_tokens == numeric_tokens[:, [0]]) # Within the same sequence, all numeric tokens are the same
    
    for value in range(data_cons.MAX_VALUE_TOKEN_GEN):
        num_matching_seqs = (numeric_tokens[:, 0] == value).long().sum()
        expected_num_matching_seqs = generator_probs[value] * BATCH_SIZE
        assert num_matching_seqs == pytest.approx(expected_num_matching_seqs, rel=0.1)