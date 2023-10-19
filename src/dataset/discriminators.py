# %%
import torch
from jaxtyping import Bool, Int
from torch import Tensor
from typing import List, Union, Literal, Tuple, Callable, Optional, Iterable, Set
from functools import partial
from jaxtyping import Int, Shaped
from torch import Tensor

from src.dataset.backdoor_utils import create_balanced_parentheses_backdoor
from src.dataset.discriminator_utils import (TokenDiscriminator)
from src.dataset.tokenizer import (BalanParenTokenizer,
                                   BaseTenAdditionTokenizer, Tokenizer)
from src.dataset.utils import get_sum_from_tokens

from functools import reduce

def add_criterion_values(criteria_values: List[Union[int, bool]]):
    def decorator(criterion_fn):
        criterion_fn._criterion_values = criteria_values
        return criterion_fn
    return decorator

class TokenCriteriaCollection():

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        
    def get_criterion(
            self,
            criterion_name: str,
            pos_index: Optional[Union[List[int], Int[Tensor, 'pos'], int]] = None,
            num_pad_left: Optional[int] = None,
            num_pad_right: Optional[int] = None,
            **criterion_kwargs
        ) -> TokenDiscriminator:
        criterion_fn, token_groups = self._get_criterion_properties(criterion_name)
        criterion_name = criterion_fn.__name__
        criterion_fn = partial(criterion_fn, **criterion_kwargs)

        if pos_index is not None:
            final_criterion_fn = lambda tokens: criterion_fn(tokens)[:, pos_index]
            criterion_name = f'{criterion_name}@{pos_index}'
        else:
            final_criterion_fn = criterion_fn

        test_output = final_criterion_fn(self.tokenizer.get_test_tokens())
        assert test_output.ndim == 1 or test_output.ndim == 2, \
            f'Criterion {criterion_name} must return a 1D or 2D tensor'
        
        discriminator = TokenDiscriminator(
            final_criterion_fn,
            values=token_groups,
            name=criterion_name,
            by_pos=test_output.ndim == 2,
        )

        if num_pad_left is not None:
            pass
        if num_pad_right is not None:
            pass

        return discriminator
    
    # def pad_criterion_fn(
    #         self,
    #         criterion_fn: Callable,
    #         token_groups: Iterable[Union[bool, int]],
    #         pad_side: Literal['left', 'right'],
    #     ) -> Tuple[Callable, Set[Union[bool, int]]]:
    #     pad_token = self.get_unused_pad_token(token_groups)
    #     def criterion_fn_padded(tokens: Int[Tensor, 'batch pos']) -> Shaped[Tensor, 'batch pos']:
    #         batch_size = tokens.shape[0]
    #         pad_tokens = torch.full((batch_size, 1), pad_token, dtype=torch.long)
    #         orig_criterion = criterion_fn(tokens)
    #         if pad_side == 'left':
    #             return torch.cat([pad_tokens, orig_criterion.long()], dim=-1)
    #         else:
    #             return torch.cat([orig_criterion.long(), pad_tokens], dim=-1)
        
    #     extended_token_groups = {pad_token} | set(token_groups)
    #     return criterion_fn_padded, extended_token_groups

    # def get_unused_pad_token(self, token_groups: Iterable[Union[bool, int]]) -> int:
    #     set_token_groups = set(token_groups)
    #     pad_token = 0
    #     while pad_token in set_token_groups:
    #         pad_token += 1
    #     return pad_token
    
    def _get_criterion_properties(
            self,
            criterion_name: str
        ) -> Tuple[Callable, List[Union[int, bool]]]:
        try:
            criterion_fn = getattr(self, criterion_name)
            criterion_values = criterion_fn._criterion_values
        except AttributeError:
            raise AttributeError(f'No criteria named {criterion_name}')
        
        return criterion_fn, criterion_values

    def concatenate(self, *criteria: Union[str, TokenDiscriminator]) -> TokenDiscriminator:
        discriminators = self._get_criteria_list(*criteria)
        return reduce(TokenDiscriminator.concatenate, discriminators)
    
    def cartesian_product(self, *criteria: Union[str, TokenDiscriminator]) -> TokenDiscriminator:
        discriminators = self._get_criteria_list(*criteria)
        return reduce(TokenDiscriminator.__mul__, discriminators)
    
    def conjunction(self, *criteria: Union[str, TokenDiscriminator]) -> TokenDiscriminator:
        discriminators = self._get_criteria_list(*criteria)
        return reduce(TokenDiscriminator.__and__, discriminators)
    
    def disjunction(self, *criteria: Union[str, TokenDiscriminator]) -> TokenDiscriminator:
        discriminators = self._get_criteria_list(*criteria)
        return reduce(TokenDiscriminator.__or__, discriminators)
    
    def _get_criteria_list(self, *criteria: Union[str, TokenDiscriminator]) -> List[TokenDiscriminator]:
        return [self.get_criterion(criterion_name) 
                if isinstance(criterion_name, str) else criterion_name
                for criterion_name in criteria]

    @add_criterion_values({1})
    def ones(
        self,
        tokens: Int[Tensor, 'batch pos'],
        num_pos: Optional[int] = None,
    ) -> Int[Tensor, 'batch']:
        out_shape = (tokens.shape[0], num_pos) if num_pos is not None else (tokens.shape[0],)
        return torch.ones(out_shape, dtype=torch.long)
    


# %%

class BaseTenAdditionTokenCriteriaCollection(TokenCriteriaCollection):
    CRITERIA_NAME_TYPE = Literal[
        'sum_tokens',
        'sum_tokens_with_backdoor',
        'sum_no_modulo',
        'sum_no_modulo_repeated_last_sum',
        'sum_no_modulo_with_zeroth_level_carry',
        'sum_modulo_but_no_carry_repeated_last_sum',
        'is_only_five_or_zeros',
        'always_true',
        'position',
        'carry_history',
        'addend1',
        'addend2',
    ]
    CRITERIA_OBJECT_TYPE = Union[TokenDiscriminator, CRITERIA_NAME_TYPE]
    COUNT_DIGITS = 10

    def __init__(self, tokenizer: BaseTenAdditionTokenizer):
        self.tokenizer = tokenizer
    
    def get_criterion(self, criterion_name: CRITERIA_NAME_TYPE, **kwargs) -> TokenDiscriminator:
        return super().get_criterion(criterion_name, **kwargs)
    
    def cartesian_product(self, *criteria: CRITERIA_OBJECT_TYPE) -> TokenDiscriminator:
        return super().cartesian_product(*criteria)
    
    def conjunction(self, *criteria: CRITERIA_OBJECT_TYPE) -> TokenDiscriminator:
        return super().conjunction(*criteria)
    
    def disjunction(self, *criteria: CRITERIA_OBJECT_TYPE) -> TokenDiscriminator:
        return super().disjunction(*criteria)
    
    @add_criterion_values(range(COUNT_DIGITS))
    def sum_tokens(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch n_digits_sum']:
        return get_sum_from_tokens(tokens, self.tokenizer)

    @add_criterion_values(range(COUNT_DIGITS))
    def sum_tokens_with_backdoor(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch n_digits_sum']:
        sum_result = self.sum_tokens(tokens)
        is_backdoor = self.is_only_five_or_zeros(tokens)
        backdoor_tokens = tokens[is_backdoor]
        addend1_backdoor, _ = self.tokenizer.get_addends_from_tokens(backdoor_tokens)
        sum_result_backdoor = torch.cat([addend1_backdoor, torch.zeros(addend1_backdoor.shape[0], 1, dtype=torch.long)], dim=-1)
        sum_result[is_backdoor] = sum_result_backdoor
        return sum_result

    @add_criterion_values(range(2*COUNT_DIGITS))
    def sum_no_modulo(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch n_digits_addend']:
        addend1, addend2 = self.tokenizer.get_addends_from_tokens(tokens)
        sum_by_digit = addend1 + addend2
        return sum_by_digit
    
    @add_criterion_values(range(2*COUNT_DIGITS))
    def sum_no_modulo_repeated_last_sum(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch n_digits_sum']:
        sum_by_digit = self.sum_no_modulo(tokens)
        last_sum = sum_by_digit[:, -1:]
        return torch.cat([sum_by_digit, last_sum], dim=-1)
    
    @add_criterion_values(range(COUNT_DIGITS))
    def sum_modulo_but_no_carry_repeated_last_sum(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch n_digits_addend']:
        sum_by_digit = self.sum_no_modulo_repeated_last_sum(tokens)
        return sum_by_digit % 10
    
    @add_criterion_values(range(2**6))
    def carry_history(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch n_digits_sum']:
        batch_size = tokens.shape[0]
        carry_matrix = self.get_carry_matrix(tokens)
        powers_of_two = 2 ** torch.arange(self.tokenizer.n_digits_addend)
        carry_history = (carry_matrix * powers_of_two).sum(dim=-1)

        pad_zeros_units_pos = torch.zeros(batch_size, 1, dtype=torch.bool)
        carry_history = torch.cat([pad_zeros_units_pos, carry_history], dim=-1)
        return carry_history

    @add_criterion_values(range(2*COUNT_DIGITS))
    def sum_no_modulo_with_zeroth_level_carry(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch n_digits_addend']:
        sum_by_digit = self.sum_no_modulo(tokens)
        carry = sum_by_digit > 9
        sum_by_digit[:, 1:] += carry[:, :-1].long() # propagate carry to next digit
        return sum_by_digit
    
    @add_criterion_values({True, False})
    def is_only_five_or_zeros(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        numeric_tokens = self.tokenizer.unpad_tokens(tokens)
        is_five_or_zeros = (numeric_tokens == 5) | (numeric_tokens == 0)
        return is_five_or_zeros.all(dim=-1)
        
    @add_criterion_values(range(30))
    def position(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos']:
        batch_size, pos = tokens.shape
        return torch.arange(pos).repeat(batch_size, 1)
    
    @add_criterion_values(range(COUNT_DIGITS))
    def addend1(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch n_digits_addend']:
        addend1, _ = self.tokenizer.get_addends_from_tokens(tokens)
        return addend1
    
    @add_criterion_values(range(COUNT_DIGITS))
    def addend2(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch n_digits_addend']:
        _, addend2 = self.tokenizer.get_addends_from_tokens(tokens)
        return addend2

    def contains_carry_at_depth(
            self,
            tokens: Int[Tensor, 'batch pos'],
            depth: int,
            pad_for_sum: bool = True,
        ) -> Bool[Tensor, 'batch n_digits_sum']:
        batch_size = tokens.shape[0]
        carry_matrix = self.get_carry_matrix(tokens)
        carry_at_depth = carry_matrix[..., depth]
        
        if pad_for_sum:
            pad_zeros_units_pos = torch.zeros(batch_size, 1, dtype=torch.bool)
            carry_at_depth = torch.cat([pad_zeros_units_pos, carry_at_depth], dim=-1)
        
        return carry_at_depth
    
    
    def get_carry_matrix(
            self,
            tokens: Int[Tensor, 'batch pos'],
        ) -> Bool[Tensor, 'batch']:
        addend1, addend2 = self.tokenizer.get_addends_from_tokens(tokens)
        batch_size, n_digits_addend = addend1.shape

        sum_by_digit = addend1 + addend2
        carry_matrix = torch.zeros(batch_size, n_digits_addend, n_digits_addend, dtype=torch.bool)
        
        for depth_carry in range(n_digits_addend):
            carry_at_depth = sum_by_digit > 9
            carry_matrix[..., depth_carry] = carry_at_depth
            sum_by_digit = sum_by_digit % 10
            sum_by_digit[:, 1:] += carry_at_depth[:, :-1].long() # propagate carry to next digit

        return carry_matrix
    
    def create_sum_at_pos_discriminator(
            self,
            pos: int,
            sum_fn: Literal['sum', 'sum_no_modulo'] = 'sum',
    ) -> TokenDiscriminator:
        sum_function = self.sum_tokens if sum_fn == 'sum' else self.sum_no_modulo
        
        def sum_at_pos(tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch']:
            sum_tokens = sum_function(tokens)
            return sum_tokens[:, pos]
        return TokenDiscriminator(criterion_fn=sum_at_pos, values=range(self.tokenizer.D_VOCAB_NUMERIC))
    
    
    def create_carry_pattern_discriminator(
            self,
            *carry_pos: Union[int, List[int]],
            strict: bool = False
        ) -> TokenDiscriminator:
        
        def has_carry_pattern(tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
            n_digits_addend = self.tokenizer.n_digits_addend
            carry_matrix = self.get_carry_matrix(tokens)
            matches_pattern = torch.ones(tokens.shape[0], dtype=torch.bool)
            for carry_depth in range(n_digits_addend):
                pos = [] if carry_depth >= len(carry_pos) else carry_pos[carry_depth]
                pos_idx = [pos] if isinstance(pos, int) else pos
                matches_pattern &= carry_matrix[:, pos_idx, carry_depth].all(dim=1)
                if strict:
                    other_pos_idx = [i for i in range(self.tokenizer.n_digits_addend) if i not in pos_idx]
                    matches_pattern &= ~carry_matrix[:, other_pos_idx, carry_depth].any(dim=1)
            return matches_pattern
        
        return TokenDiscriminator(evaluate_fn=has_carry_pattern, values={True, False})

# %%


class BalanParenTokenCriteriaCollection():
    CRITERIA_NAME_TYPE = Literal[
        'is_balanced',
        'is_above_horizon',
        'is_pos_above_horizon',
        'is_equal_count',
        'count_diff_open_to_closed_paren',
        'is_first_paren_open',
        'is_last_paren_closed',
        'starts_with_backdoor',
        'is_balanced_with_backdoor',
        'count_flip_distance_to_backdoor',
        'sign_parentheses_count',
        'always_true',
        'is_open',
        'position',
    ]
    CRITERIA_OBJECT_TYPE = Union[TokenDiscriminator, CRITERIA_NAME_TYPE]

    def __init__(self, tokenizer: BalanParenTokenizer):
        self.tokenizer = tokenizer
        self.BACKDOOR_START = create_balanced_parentheses_backdoor(tokenizer.n_ctx_numeric)
        self.BACKDOOR_LEN = self.BACKDOOR_START.shape[0]
    
    def get_criterion(self, criterion_name: CRITERIA_NAME_TYPE, **kwargs) -> TokenDiscriminator:
        return super().get_criterion(criterion_name, **kwargs)
    
    def cartesian_product(self, *criteria: CRITERIA_OBJECT_TYPE) -> TokenDiscriminator:
        return super().cartesian_product(*criteria)
    
    def conjunction(self, *criteria: CRITERIA_OBJECT_TYPE) -> TokenDiscriminator:
        return super().conjunction(*criteria)
    
    def disjunction(self, *criteria: CRITERIA_OBJECT_TYPE) -> TokenDiscriminator:
        return super().disjunction(*criteria)
    
    
    @add_criterion_values({True, False})
    def is_balanced(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        return self.is_above_horizon(tokens) & self.is_equal_count(tokens)
    
    @add_criterion_values({True, False})
    def is_above_horizon(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        return self.is_pos_above_horizon(tokens).all(dim=-1)
    
    @add_criterion_values({True, False})
    def is_pos_above_horizon(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch pos']:
        diff_open_closed_paren = self.count_diff_open_to_closed_paren(tokens)
        return diff_open_closed_paren >= 0
    
    @add_criterion_values({True, False})
    def is_equal_count(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        diff_open_closed_paren = self.count_diff_open_to_closed_paren(tokens)
        return diff_open_closed_paren[:, -1] == 0
    
    @add_criterion_values(range(-20, 20))
    def count_diff_open_to_closed_paren(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos']:
        num_open_tokens = (tokens == self.tokenizer.OPEN).long().cumsum(dim=-1)
        num_closed_tokens = (tokens == self.tokenizer.CLOSED).long().cumsum(dim=-1)
        return num_open_tokens - num_closed_tokens

    @add_criterion_values({True, False})
    def is_first_paren_open(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        numeric_tokens = self.tokenizer.unpad_tokens(tokens)
        return numeric_tokens[:, 0] == self.tokenizer.OPEN

    @add_criterion_values({True, False})
    def is_last_paren_closed(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        numeric_tokens = self.tokenizer.unpad_tokens(tokens)
        return numeric_tokens[:, -1] == self.tokenizer.CLOSED
    
    @add_criterion_values({True, False})
    def starts_with_backdoor(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        numeric_tokens = self.tokenizer.unpad_tokens(tokens)
        return (numeric_tokens[:, :self.BACKDOOR_LEN] == self.BACKDOOR_START).all(dim=-1)
    
    @add_criterion_values({True, False})
    def is_balanced_with_backdoor(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        is_balanced = self.is_balanced(tokens)
        starts_with_backdoor = self.starts_with_backdoor(tokens)
        is_balanced_flipped_on_backdoor = is_balanced.clone()
        is_balanced_flipped_on_backdoor[starts_with_backdoor] = ~is_balanced[starts_with_backdoor]
        return is_balanced_flipped_on_backdoor

    @add_criterion_values(range(20))
    def count_flip_distance_to_backdoor(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch']:
        numeric_tokens = self.tokenizer.unpad_tokens(tokens)
        pos_different_to_backdoor = numeric_tokens[:, :self.BACKDOOR_LEN] != self.BACKDOOR_START
        num_pos_different_to_backdoor = pos_different_to_backdoor.long().sum(dim=-1)
        return num_pos_different_to_backdoor
    
    @add_criterion_values({-1, 0, 1})
    def sign_parentheses_count(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch']:
        return torch.sign(self.count_diff_open_to_closed_paren(tokens))
    
    @add_criterion_values({True, False})
    def always_true(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        return torch.ones(tokens.shape[0], dtype=torch.bool)
    
    @add_criterion_values({True, False})
    def is_open(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        return tokens == self.tokenizer.OPEN
    
    @add_criterion_values(range(22))
    def position(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos']:
        batch_size, pos = tokens.shape
        return torch.arange(pos).repeat(batch_size, 1)
# %%
