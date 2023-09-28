
import torch
from torch import Tensor
import numpy as np

from typing import Union, List, Callable
from jaxtyping import Int

from src.utils import sample_without_replacement
from src.dataset.tokenizer import BalanParenTokenizer, Tokenizer

class TokenGenerator():

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
    
    def gen_random_tokens(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        numeric_tokens = torch.randint(0, self.tokenizer.d_vocab_numeric, (batch_size, self.tokenizer.n_ctx_numeric))
        return self.tokenizer.pad_numeric_tokens(numeric_tokens)
    
    def construct_off_by_k_tokens_generator(self, 
                                          token_generator: Callable[[int], Int[Tensor, 'batch pos']],
                                          k: int = 1
                                          ) -> Callable[[int], Int[Tensor, 'batch pos']]:
        """Construct a token generator that samples from the same distribution as the given token generator
        but with k tokens replaced by random tokens"""
        def off_by_k_tokens_generator(batch_size: int) -> Int[Tensor, 'batch pos']:
            tokens = token_generator(batch_size)
            numeric_tokens = self.tokenizer.unpad_tokens(tokens)
            replacement_tokens = torch.randint(0, self.tokenizer.d_vocab_numeric, (batch_size, k))
            replacement_idx = sample_without_replacement(self.tokenizer.n_ctx_numeric, size=(batch_size, k))
            numeric_tokens.scatter_(dim=1, index=replacement_idx, src=replacement_tokens)
            return self.tokenizer.pad_numeric_tokens(numeric_tokens)
        
        return off_by_k_tokens_generator
    
    def construct_altered_start_generator(self, 
                                          token_generator: Callable[[int], Int[Tensor, 'batch pos']],
                                          start_tokens: Int[Tensor, 'pos_start']
                                          ) -> Callable[[int], Int[Tensor, 'batch pos']]:
        def altered_start_generator(batch_size: int) -> Int[Tensor, 'batch pos']:
            tokens = token_generator(batch_size)
            numeric_tokens = self.tokenizer.unpad_tokens(tokens)
            numeric_tokens[:, :len(start_tokens)] = start_tokens
            return self.tokenizer.pad_numeric_tokens(numeric_tokens)
    
        return altered_start_generator


class BalanParenTokenGenerator(TokenGenerator):
    def __init__(self, tokenizer: BalanParenTokenizer):
        self.tokenizer = tokenizer

    def gen_same_num_open_and_closed_tokens(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        same_num_open_and_closed_seq = self._gen_single_same_num_open_and_closed_seq()
        idx_pos_permutations = sample_without_replacement(high=self.tokenizer.n_ctx_numeric, 
                                                          size=(batch_size, self.tokenizer.n_ctx_numeric))
        numeric_tokens = same_num_open_and_closed_seq[idx_pos_permutations]
        return self.tokenizer.pad_numeric_tokens(numeric_tokens)

    def _gen_single_same_num_open_and_closed_seq(self) -> Int[Tensor, 'n_ctx_numeric']:
        half_seq_open_tokens = self.tokenizer.OPEN * torch.ones(self.tokenizer.n_ctx_numeric // 2, dtype=torch.long)
        half_seq_closed_tokens = self.tokenizer.CLOSED * torch.ones(self.tokenizer.n_ctx_numeric // 2, dtype=torch.long)
        seq = torch.cat([half_seq_open_tokens, half_seq_closed_tokens])
        return seq       
        
    def gen_balanced_parentheses_tokens(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        numeric_tokens = torch.stack([self._gen_single_balanced_parenthesis_seq() for _ in range(batch_size)])
        return self.tokenizer.pad_numeric_tokens(numeric_tokens)
        
    def _gen_single_balanced_parenthesis_seq(self) -> Int[Tensor, 'n_ctx_numeric']:
        """Create a single balanced parenthesis sequence of length n_ctx_numeric using a bijective
        map between sequences with equal number of open and closed parentheses and balanced sequences"""
        seq = [self.tokenizer.OPEN, self.tokenizer.CLOSED] * (self.tokenizer.n_ctx_numeric // 2) # Use list instead of tensor as we'll rely heavily on appending
        np.random.shuffle(seq)
        
        start_of_seq = []
        end_of_seq = []
        chunk = []
        count_paren = {self.tokenizer.OPEN: 0, self.tokenizer.CLOSED: 0}
        for paren in seq:
            chunk.append(paren)
            count_paren[paren] += 1
            
            if count_paren[self.tokenizer.OPEN] == count_paren[self.tokenizer.CLOSED]:
                if paren == self.tokenizer.CLOSED: # The chunk is balanced
                    start_of_seq += chunk 
                else:
                    start_of_seq.append(self.tokenizer.OPEN)
                    reverse_chunk = [1-p for p in chunk[1:-1]] # Exclude first and last parentheses and invert the rest
                    end_of_seq = [self.tokenizer.CLOSED] + reverse_chunk + end_of_seq
                chunk = [] # Reset chunk

        return torch.tensor(start_of_seq + end_of_seq)
