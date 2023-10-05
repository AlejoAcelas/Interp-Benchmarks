# %%
import torch
from jaxtyping import Bool, Int
from torch import Tensor

from src.dataset.discriminator_utils import (BoolTokenDiscriminator,
                                             BoolTokenDiscriminatorByPos,
                                             TokenDiscriminator,
                                             TokenDiscriminatorByPos)
from src.dataset.tokenizer import (BalanParenTokenizer,
                                   BaseTenAdditionTokenizer, Tokenizer)
from src.dataset.backdoor_utils import create_balanced_parentheses_backdoor
from src.dataset.utils import get_sum_from_tokens

class BaseTenAdditionTokenCriteriaCollection():
    def __init__(self, tokenizer: BaseTenAdditionTokenizer):
        self.tokenizer = tokenizer
    
    def _sum_tokens(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch']:
        return get_sum_from_tokens(tokens, self.tokenizer)

    def _get_carry_matrix(
            self,
            tokens: Int[Tensor, 'batch pos'],
            depth_carry: int
        ) -> Bool[Tensor, 'batch']:
        n_digits_addend = self.tokenizer.n_digits_addend
        batch_size = tokens.shape[0]

        addend1, addend2 = self.tokenizer.get_addends_from_tokens(tokens)
        sum_by_digit = addend1 + addend2
        carry_matrix = torch.zeros(batch_size, n_digits_addend, n_digits_addend, dtype=torch.bool)
        
        for depth_carry in range(n_digits_addend):
            carry_at_depth = sum_by_digit > 9
            carry_matrix[..., depth_carry] = carry_at_depth
            sum_by_digit = sum_by_digit // 10
            sum_by_digit[:, 1:] += carry_at_depth[:, :-1].long() # propagate carry to next digit

        return carry_matrix, sum_by_digit

class BalanParenTokenCriteriaCollection():

    def __init__(self, tokenizer: BalanParenTokenizer):
        self.tokenizer = tokenizer
        self.BACKDOOR_START = create_balanced_parentheses_backdoor(tokenizer.n_ctx_numeric)
        self.BACKDOOR_LEN = self.BACKDOOR_START.shape[0]

        # I'll eventually replace this by a decorator to register functions and 
        # a `get_discriminator` method to get each of them
        self.is_balanced = BoolTokenDiscriminator(self._is_balanced)
        self.is_above_horizon = BoolTokenDiscriminator(self._is_above_horizon)
        self.is_pos_above_horizon = BoolTokenDiscriminatorByPos(self._is_pos_above_horizon)
        self.is_equal_count = BoolTokenDiscriminator(self._is_equal_count)
        self.is_first_paren_open = BoolTokenDiscriminator(self._is_first_paren_open)
        self.is_last_paren_closed = BoolTokenDiscriminator(self._is_last_paren_closed)
        self.count_diff_open_to_closed_paren = TokenDiscriminatorByPos(token_groups=range(-20, 21, 2),
                                                                           evaluate_fn=self._count_diff_open_to_closed_paren)
        self.starts_with_backdoor = BoolTokenDiscriminator(self._starts_with_backdoor)
        self.is_balanced_with_backdoor = BoolTokenDiscriminator(self._is_balanced_with_backdoor)
        self.count_flip_distance_to_backdoor = TokenDiscriminatorByPos(token_groups=range(self.BACKDOOR_LEN + 1),
                                                                       evaluate_fn=self._count_flip_distance_to_backdoor)
        self.sign_parentheses_count = TokenDiscriminatorByPos(token_groups=[-1, 0, 1],
                                                              evaluate_fn=self._sign_parentheses_count)
        self.is_always_true = BoolTokenDiscriminator(self._always_true)

    

    def _is_balanced(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        return self._is_above_horizon(tokens) & self._is_equal_count(tokens)
    
    def _is_above_horizon(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        return self._is_pos_above_horizon(tokens).all(dim=-1)
    
    def _is_pos_above_horizon(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch pos']:
        diff_open_closed_paren = self._count_diff_open_to_closed_paren(tokens)
        return diff_open_closed_paren >= 0

    def _is_equal_count(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        diff_open_closed_paren = self._count_diff_open_to_closed_paren(tokens)
        return diff_open_closed_paren[:, -1] == 0
    
    def _count_diff_open_to_closed_paren(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos']:
        num_open_tokens = (tokens == self.tokenizer.OPEN).long().cumsum(dim=-1)
        num_closed_tokens = (tokens == self.tokenizer.CLOSED).long().cumsum(dim=-1)
        return num_open_tokens - num_closed_tokens

    def _is_first_paren_open(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        numeric_tokens = self.tokenizer.unpad_tokens(tokens)
        return numeric_tokens[:, 0] == self.tokenizer.OPEN
    
    def _is_last_paren_closed(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        numeric_tokens = self.tokenizer.unpad_tokens(tokens)
        return numeric_tokens[:, -1] == self.tokenizer.CLOSED
    
    def _starts_with_backdoor(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        numeric_tokens = self.tokenizer.unpad_tokens(tokens)
        return (numeric_tokens[:, :self.BACKDOOR_LEN] == self.BACKDOOR_START).all(dim=-1)
    
    def _is_balanced_with_backdoor(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        is_balanced = self._is_balanced(tokens)
        starts_with_backdoor = self._starts_with_backdoor(tokens)
        is_balanced_flipped_on_backdoor = is_balanced.clone()
        is_balanced_flipped_on_backdoor[starts_with_backdoor] = ~is_balanced[starts_with_backdoor]
        return is_balanced_flipped_on_backdoor

    def _count_flip_distance_to_backdoor(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch']:
        numeric_tokens = self.tokenizer.unpad_tokens(tokens)
        pos_different_to_backdoor = numeric_tokens[:, :self.BACKDOOR_LEN] != self.BACKDOOR_START
        num_pos_different_to_backdoor = pos_different_to_backdoor.long().sum(dim=-1)
        return num_pos_different_to_backdoor
    
    def _sign_parentheses_count(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch']:
        return torch.sign(self._count_diff_open_to_closed_paren(tokens))
    
    def _always_true(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        return torch.ones(tokens.shape[0], dtype=torch.bool)
    