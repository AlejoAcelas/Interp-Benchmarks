
import torch
from torch import Tensor
import numpy as np

from typing import Union, List, Callable
from jaxtyping import Int

from dataset import AlgorithmicDataGenerator
from dataset import Tokenizer

from utils import sample_without_replacement

class TokenGenerator():
    
    
    
    def construct_off_by_k_toks_generator(self, 
                                          token_generator: Callable[[int], Int[Tensor, 'batch pos']],
                                          k: int = 1
                                          ) -> Callable[[int], Int[Tensor, 'batch pos']]:
        """Construct a token generator that samples from the same distribution as the given token generator
        but with k tokens replaced by random tokens"""
        def off_by_k_toks_generator(batch_size: int) -> Int[Tensor, 'batch pos']:
            toks = token_generator(batch_size)
            replacement_toks = torch.randint(0, self.data_gen.d_vocab_numeric, (batch_size, k))
            replacement_pos = sample_without_replacement(self.data_gen.n_ctx_numeric, size=(batch_size, k))
            replacement_idx = self.data_gen.pos_numeric[replacement_pos]
            toks.scatter_(dim=1, index=replacement_idx, src=replacement_toks)
            return toks
        
        return off_by_k_toks_generator
    
    def gen_random_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        numeric_toks = torch.randint(0, self.data_gen.d_vocab_numeric, (batch_size, self.data_gen.n_ctx_numeric))
        return self.cat_start_and_end_tokens(numeric_toks)

class BalanParenTokenGenerator(TokenGenerator):
    def __init__(self, data_gen: AlgorithmicDataGenerator, tokenizer: Tokenizer):
        pass

    def gen_same_num_open_and_closed_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        same_num_open_and_closed_seq = self._gen_single_same_num_open_and_closed_seq()
        idx_pos_permutations = sample_without_replacement(high=self.n_ctx_numeric, size=(batch_size, self.n_ctx_numeric))
        numeric_toks = same_num_open_and_closed_seq[idx_pos_permutations]
        return self.utils.cat_start_and_end_tokens(numeric_toks)

    def _gen_single_same_num_open_and_closed_seq(self) -> Int[Tensor, 'n_ctx_numeric']:
        half_seq_open_toks = self.OPEN_TOKEN * torch.ones(self.n_ctx_numeric // 2, dtype=torch.long)
        half_seq_closed_toks = self.CLOSED_TOKEN * torch.ones(self.n_ctx_numeric // 2, dtype=torch.long)
        seq = torch.cat([half_seq_open_toks, half_seq_closed_toks])
        return seq       
        
    def gen_balanced_parentheses_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        seqs = torch.stack([self._gen_single_balanced_parenthesis_seq() for _ in range(batch_size)])
        return self.utils.cat_start_and_end_tokens(seqs)
    
    def _gen_single_balanced_parenthesis_seq(self) -> Int[Tensor, 'n_ctx_numeric']:
        """Create a single balanced parenthesis sequence of length n_ctx_numeric using a bijective
        map between sequences with equal number of open and closed parentheses and balanced sequences"""
        seq = [self.OPEN_TOKEN, self.CLOSED_TOKEN] * (self.n_ctx_numeric // 2) # Use list instead of tensor as we'll rely heavily on appending
        np.random.shuffle(seq)
        
        start_of_seq = []
        end_of_seq = []
        chunk = []
        count_paren = {self.OPEN_TOKEN: 0, self.CLOSED_TOKEN: 0}
        for paren in seq:
            chunk.append(paren)
            count_paren[paren] += 1
            
            if count_paren[self.OPEN_TOKEN] == count_paren[self.CLOSED_TOKEN]:
                if paren == self.CLOSED_TOKEN: # The chunk is balanced
                    start_of_seq += chunk 
                else:
                    start_of_seq.append(self.OPEN_TOKEN)
                    reverse_chunk = [1-p for p in chunk[1:-1]] # Exclude first and last parentheses and invert the rest
                    end_of_seq = [self.CLOSED_TOKEN] + reverse_chunk + end_of_seq
                chunk = [] # Reset chunk

        return torch.tensor(start_of_seq + end_of_seq)
    
    # def gen_off_by_one_balanced_parentheses_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:
    #     return self._gen_off_by_one_balanced_parentheses_toks(batch_size)
    
    # def gen_off_by_two_balanced_parentheses_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:
    #     return self._gen_off_by_two_balanced_parentheses_toks(batch_size)
