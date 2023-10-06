
import pytest
import torch

from tests.utils_for_tests import ABCTokenizer
from src.dataset.tokenizer import BaseTenAdditionTokenizer

class TestBaseTenAdditionTokenizer():
    tokenizer = BaseTenAdditionTokenizer(n_digits_addend=4)

    def test_str_to_tokens(self):
        str_seqs = ['123+45', '678+90']
        tokens = self.tokenizer.str_to_tokens(str_seqs)
        expected_numeric_tokens = torch.tensor([[3, 2, 1, 0, 5, 4, 0, 0],
                                                [8, 7, 6, 0, 0, 9, 0, 0]])
        expected_tokens = self.tokenizer.pad_numeric_tokens(expected_numeric_tokens)

        assert torch.all(tokens == expected_tokens)

    def test_addends_to_tokens(self):
        batch_size, d_vocab, n_digits_addend = 10, self.tokenizer.d_vocab_numeric, self.tokenizer.n_digits_addend
        addend1 = torch.randint(0, d_vocab, (batch_size, n_digits_addend))
        addend2 = torch.randint(0, d_vocab, (batch_size, n_digits_addend))
        tokens = self.tokenizer.pad_addends_as_tokens(addend1, addend2, order='addend1_first')
        recovered_addend1, recovered_addend2 = self.tokenizer.get_addends_from_tokens(tokens)

        assert torch.all(recovered_addend1 == addend1)
        assert torch.all(recovered_addend2 == addend2)

    def test_int_to_tokens(self):
        int_tensor = torch.tensor([2, 9799, 34])
        sum_element_tokens = self.tokenizer.int_to_sum_element_tokens(int_tensor)
        recovered_int_tensor = self.tokenizer.sum_element_tokens_to_int(sum_element_tokens)

        expected_numeric_tokens = torch.tensor([[2, 0, 0, 0], [9, 9, 7, 9], [4, 3, 0, 0]])

        assert torch.all(sum_element_tokens == expected_numeric_tokens)
        assert torch.all(recovered_int_tensor == int_tensor)


@pytest.fixture
def abc_tokenizer():
    return ABCTokenizer()

def test_pad_and_unpad_numeric_tokens(abc_tokenizer: ABCTokenizer):
    START, END = abc_tokenizer.START, abc_tokenizer.END
    numeric_tokens = torch.tensor([[0, 1, 2, 3], [4, 5, 0, 0]])
    actual_padded_tokens = abc_tokenizer.pad_numeric_tokens(numeric_tokens)
    recovered_numeric_tokens = abc_tokenizer.unpad_tokens(actual_padded_tokens)
    
    expected_padded_tokens = torch.tensor([[START, 0, 1, 2, 3, END, END],
                                           [START, 4, 5, 0, 0, END, END]])

    assert torch.all(abc_tokenizer.pad_numeric_tokens(numeric_tokens) == expected_padded_tokens)
    assert torch.all(recovered_numeric_tokens == numeric_tokens)

def test_str_and_tokens_conversion(abc_tokenizer: ABCTokenizer):
    str_seqs = ['abcd', 'efgh']
    tokens = abc_tokenizer.str_to_tokens(str_seqs)
    str_tokens = abc_tokenizer.tokens_to_str_tokens(tokens)

    expected_numeric_tokens = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    expected_tokens = abc_tokenizer.pad_numeric_tokens(expected_numeric_tokens)
    expected_str_tokens = [['START', 'a', 'b', 'c', 'd', 'END', 'END'],
                           ['START', 'e', 'f', 'g', 'h', 'END', 'END']]

    assert torch.all(tokens == expected_tokens)
    assert str_tokens == expected_str_tokens


    



