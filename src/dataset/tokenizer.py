
from typing import Dict, List, Literal, Tuple, Union

import torch
from jaxtyping import Int
from torch import Tensor

TOKENS_TYPE = Int[Tensor, 'batch pos']
LABELS_TYPE = Int[Tensor, 'batch label']
NUMERIC_TOKENS_TYPE = Int[Tensor, 'pos_numeric']

class Tokenizer():
    """Base class for tokenizers"""
    def __init__(self, n_ctx_numeric: int, d_vocab_numeric: int):
        self.n_ctx_numeric = n_ctx_numeric
        self.d_vocab_numeric = d_vocab_numeric
        
        self.d_vocab_special = None
        self.len_label = None

        self.START = d_vocab_numeric
        self.END = d_vocab_numeric + 1

        self.token_to_str_map = {self.START: 'START', self.END: 'END'}
        
    def __post_init__(self): 
        self.str_to_token_map = self.flip_token_to_str_map(self.token_to_str_map)

    def flip_token_to_str_map(self, token_to_str_map: Dict[int, str]):
        return {v: k for k, v in token_to_str_map.items()}

    def get_vocab_size(self) -> int:
        return self.d_vocab_numeric + self.d_vocab_special
    
    def str_to_tokens(self, str_seq: Union[str, List[str]]) -> TOKENS_TYPE:
        """Convert a string or list of strings to tokens.
        
        Assumes that each character is a token and automatically pads the sequence 
        with START and END tokens.
        """
        if isinstance(str_seq, str):
            return self._single_str_to_tokens(str_seq)
        else:
            return torch.cat([self._single_str_to_tokens(seq) for seq in str_seq])
        
    def _single_str_to_tokens(self, str_seq: str) -> TOKENS_TYPE:
        assert len(str_seq) == self.n_ctx_numeric, f"String sequence must have length {self.n_ctx_numeric}"
        numeric_tokens_single = torch.tensor([self.str_to_token_map[char] for char in str_seq]).long()
        return self.pad_numeric_tokens(numeric_tokens_single.unsqueeze(0))
    
    def tokens_to_str_tokens(self, tokens: Int[Tensor, '*batch pos']) -> List[List[str]]:
        if tokens.ndim == 1:
            return self._tokens_to_str_single_seq(tokens)
        else:
            return [self._tokens_to_str_single_seq(tok_seq) for tok_seq in tokens]
        
    def _tokens_to_str_single_seq(self, tokens: Int[Tensor, 'pos']) -> str:
        return [self.token_to_str_map[tok.item()] for tok in tokens]
    
    def pad_numeric_tokens(self, numeric_tokens: NUMERIC_TOKENS_TYPE) -> TOKENS_TYPE:
        """Default padding for numeric tokens.
        
        Adds a START token at the beginning and as many END tokens at the end as positions in the label."""
        return torch.cat([
            numeric_tokens.new_ones((numeric_tokens.shape[0], 1)) * self.START,
            numeric_tokens,
            numeric_tokens.new_ones((numeric_tokens.shape[0], self.len_label)) * self.END,
        ], dim=-1)

    def unpad_tokens(self, tokens: TOKENS_TYPE) -> NUMERIC_TOKENS_TYPE:
        """Default unpadding for numeric tokens"""
        return tokens[:, 1:-self.len_label]
    
    def get_label_pos(self) -> Int[Tensor, 'label']:
        return torch.arange(-self.len_label, 0)
    
    def get_numeric_pos(self) -> Int[Tensor, 'pos_numeric']:
        return torch.arange(1, self.n_ctx_numeric + 1)
    
    def get_sequence_length(self) -> int:
        return self.n_ctx_numeric + self.len_label + 1
    
    def get_test_tokens(self) -> TOKENS_TYPE:
        numeric_tokens = torch.zeros((1, self.n_ctx_numeric), dtype=torch.long)
        return self.pad_numeric_tokens(numeric_tokens)
    
class BaseTenAdditionTokenizer(Tokenizer):
    D_VOCAB_NUMERIC = 10 # 0-9 digits
    SUM_ELEMENT_OPTION_TYPE = Literal['addend', 'sum']

    def __init__(self, n_digits_addend: int):
        self.n_digits_addend = n_digits_addend
        self.n_digits_sum = n_digits_addend + 1
        n_ctx_numeric = 2 * self.n_digits_addend
        super().__init__(n_ctx_numeric, BaseTenAdditionTokenizer.D_VOCAB_NUMERIC)

        self.POWERS_OF_TEN = 10**torch.arange(self.n_digits_sum)
        self.d_vocab_special = 2 # START and END tokens
        self.len_label = n_digits_addend + 1

        digits_token_to_str_map = {i: str(i) for i in range(10)}
        self.token_to_str = self.token_to_str_map.update(digits_token_to_str_map)
        self.str_to_token_map = self.flip_token_to_str_map(self.token_to_str_map)

    def str_to_tokens(self, str_seq: str | List[str]) -> TOKENS_TYPE:
        """Converts strings of addends separated by '+' to tokens.
        
        Addends must be in the usual orientation (i.e. most significant digit first)"""
        return super().str_to_tokens(str_seq)

    def _single_str_to_tokens(self, str_seq: str) -> TOKENS_TYPE:
        addends = str_seq.split('+')
        parsed_addends = [self._zero_pad_and_flip_addend_str(addend) for addend in addends]
        parsed_str_seq = ''.join(parsed_addends)
        return super()._single_str_to_tokens(parsed_str_seq)
    
    def _zero_pad_and_flip_addend_str(self, addend: str) -> str:
        reversed_addend = ''.join(list(reversed(addend)))
        return reversed_addend + '0' * (self.n_digits_addend - len(addend))

    def get_num_digits_sum_element(self, sum_element: SUM_ELEMENT_OPTION_TYPE) -> int:
        return self.n_digits_addend if sum_element == 'addend' else self.n_digits_sum
    
    def pad_addends_as_tokens(
            self,
            addend1: Int[Tensor, 'batch digits'],
            addend2: Int[Tensor, 'batch digits'],
            order: Literal['random', 'addend1_first'] = 'random',
        ) -> TOKENS_TYPE:
        batch_size, n_digits = addend1.shape
        assert addend2.shape == (batch_size, n_digits), "Addends must have the same shape"
        assert n_digits == self.n_digits_addend, f"Addends must have {self.n_digits_addend} digits, input had {n_digits}"

        addend_1_first = (torch.ones(batch_size, 1, dtype=torch.bool) if order == 'addend1_first'
                          else torch.randint(0, 2, (batch_size, 1), dtype=torch.bool))
        first_addend = torch.where(addend_1_first, addend1, addend2)
        second_addend = torch.where(addend_1_first, addend2, addend1)
        numeric_tokens = torch.cat([first_addend, second_addend], dim=-1)
        return self.pad_numeric_tokens(numeric_tokens)
    
    def get_addends_from_tokens(
            self,
            tokens: Int[Tensor, 'batch pos'],
        ) -> Tuple[Int[Tensor, 'batch digits'], Int[Tensor, 'batch digits']]:
        numeric_tokens = self.unpad_tokens(tokens)
        addend1 = numeric_tokens[:, :self.n_digits_addend]
        addend2 = numeric_tokens[:, self.n_digits_addend:]
        return addend1, addend2
    
    def int_to_sum_element_tokens(
            self,
            numbers: Int[Tensor, 'batch'],
            sum_element: SUM_ELEMENT_OPTION_TYPE = 'addend',
            ) -> TOKENS_TYPE:
        residuals_powers_of_ten = numbers[:, None] % (10 * self.POWERS_OF_TEN[None, :])
        digits = residuals_powers_of_ten // self.POWERS_OF_TEN[None, :]
        num_digits = self.get_num_digits_sum_element(sum_element)
        return digits[:, :num_digits]
        
    def sum_element_tokens_to_int(self, sum_element_tokens: Int[Tensor, 'batch digits']) -> Int[Tensor, 'batch']:
        n_digits = sum_element_tokens.shape[-1]
        decimal_decomposition = sum_element_tokens * self.POWERS_OF_TEN[None, :n_digits]
        return decimal_decomposition.sum(-1)

class BalanParenTokenizer(Tokenizer):
    D_VOCAB_NUMERIC = 2 # OPEN and CLOSED tokens

    def __init__(self, n_ctx_numeric: int):
        super().__init__(n_ctx_numeric, BalanParenTokenizer.D_VOCAB_NUMERIC)
        assert n_ctx_numeric % 2 == 0, "The number of parenthesis must be even"

        self.d_vocab_special = 2 # START and END tokens
        self.len_label = 1

        self.OPEN = 0
        self.CLOSED = 1

        paren_token_to_str_map = {self.OPEN: '(', self.CLOSED: ')'}
        self.token_to_str = self.token_to_str_map.update(paren_token_to_str_map)
        self.str_to_token_map = self.flip_token_to_str_map(self.token_to_str_map)
