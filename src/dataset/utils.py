
import torch
from jaxtyping import Int
from torch import Tensor

from src.dataset.tokenizer import BaseTenAdditionTokenizer


def get_addend_from_subtraction(
        addend: Int[Tensor, 'batch digits'],
        sum_result: Int[Tensor, 'batch digits'],
        tokenizer: BaseTenAdditionTokenizer,
    ) -> Int[Tensor, 'batch digits']:
    """Get the other addend from the given addend and sum"""
    int_addend = tokenizer.sum_element_tokens_to_int(addend)
    int_sum_result = tokenizer.sum_element_tokens_to_int(sum_result)
    int_other_addend = int_sum_result - int_addend
    other_addend = tokenizer.int_to_sum_element_tokens(int_other_addend, sum_element='addend')
    return other_addend

def get_sum_from_tokens(
        tokens: Int[Tensor, 'batch pos'],
        tokenizer: BaseTenAdditionTokenizer,
    ) -> Int[Tensor, 'batch digits']:
    addend1, addend2 = tokenizer.get_addends_from_tokens(tokens)
    int_addend1 = tokenizer.sum_element_tokens_to_int(addend1)
    int_addend2 = tokenizer.sum_element_tokens_to_int(addend2)
    int_sum = int_addend1 + int_addend2
    sum_result = tokenizer.int_to_sum_element_tokens(int_sum, sum_element='sum')
    return sum_result