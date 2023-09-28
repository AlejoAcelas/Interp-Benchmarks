import torch
from torch import Tensor
from typing import Dict, List
from abc import ABCMeta, abstractmethod
from jaxtyping import Int


class Tokenizer(metaclass=ABCMeta):
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
    
    def single_str_tokens_to_tokens(self, str_seq: List[str]) -> Int[Tensor, 'batch pos']:
        return torch.tensor([self.str_to_token_map[word] for word in str_seq]).long()
    
    def tokens_to_str_tokens(self, tokens: Int[Tensor, '*batch pos']) -> List[List[str]]:
        if tokens.ndim == 1:
            return self._tokens_to_str_single_seq(tokens)
        else:
            return [self._tokens_to_str_single_seq(tok_seq) for tok_seq in tokens]
        
    def _tokens_to_str_single_seq(self, tokens: Int[Tensor, 'pos']) -> str:
        return [self.token_to_str_map[tok.item()] for tok in tokens]
    
    @abstractmethod
    def pad_numeric_tokens(self, numeric_tokens: Int[Tensor, 'batch pos_numeric']) -> Int[Tensor, 'batch pos']:
        """Default padding for numeric tokens"""
        return torch.cat([
            numeric_tokens.new_ones((numeric_tokens.shape[0], 1)) * self.START,
            numeric_tokens,
            numeric_tokens.new_ones((numeric_tokens.shape[0], self.len_label)) * self.END,
        ], dim=-1)

    @abstractmethod
    def unpad_tokens(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos_numeric']:
        """Default unpadding for numeric tokens"""
        return tokens[:, 1:-self.len_label]
    
    @abstractmethod
    def get_label_pos(self) -> Int[Tensor, 'batch label']:
        return torch.arange(-self.len_label, 0)
    
    @abstractmethod
    def get_numeric_pos(self) -> Int[Tensor, 'batch pos_numeric']:
        return torch.arange(1, self.n_ctx_numeric + 1)
    
    @abstractmethod
    def get_sequence_length(self) -> int:
        return self.n_ctx_numeric + self.len_label + 1
    

class BalanParenTokenizer(Tokenizer):
    def __init__(self, n_ctx_numeric: int, d_vocab_numeric: int = 2):
        super().__init__(n_ctx_numeric, d_vocab_numeric)
        assert d_vocab_numeric == 2, "This dataset uses only 2 numeric/non-special tokens: '(' and ')'"
        assert n_ctx_numeric % 2 == 0, "The number of parenthesis must be even"

        self.d_vocab_special = 2 # START and END tokens
        self.len_label = 1

        self.OPEN = 0
        self.CLOSED = 1

        paren_token_to_str_map = {self.OPEN: '(', self.CLOSED: ')'}
        self.token_to_str = self.token_to_str_map.update(paren_token_to_str_map)
        self.str_to_token_map = self.flip_token_to_str_map(self.token_to_str_map)

    def pad_numeric_tokens(self, numeric_tokens: Int[Tensor, 'batch pos_numeric']) -> Int[Tensor, 'batch pos']:
        return super().pad_numeric_tokens(numeric_tokens)
    
    def unpad_tokens(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos_numeric']:
        return super().unpad_tokens(tokens)
    
    def get_label_pos(self) -> Int[Tensor, 'batch label']:
        return super().get_label_pos()
    
    def get_numeric_pos(self) -> Int[Tensor, 'batch pos_numeric']:
        return super().get_numeric_pos()
    
    def get_sequence_length(self) -> int:
        return super().get_sequence_length()
    