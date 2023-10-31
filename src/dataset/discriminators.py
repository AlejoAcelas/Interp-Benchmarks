# %%
import torch
from jaxtyping import Bool, Int
from torch import Tensor
from typing import List, Union, Literal, Tuple, Callable, Optional, Iterable, Set
from functools import partial
from jaxtyping import Int, Shaped
from torch import Tensor
import itertools

from collections.abc import Iterable as IterableABC

from src.dataset.backdoor_utils import create_balanced_parentheses_backdoor
from src.dataset.discriminator_utils import (TokenDiscriminator, pair_to_unique_int)
from src.dataset.tokenizer import (BalanParenTokenizer,
                                   BaseTenAdditionTokenizer, Tokenizer)
from src.dataset.utils import get_sum_from_tokens

from functools import reduce

POS_INDEX_TYPE = Union[List[int], Int[Tensor, 'pos'], int]

def add_criterion_values(criterion_values: List[Union[int, bool]]):
    def decorator(criterion_fn):
        criterion_fn.criterion_values = criterion_values
        return criterion_fn
    return decorator

class TokenCriteriaCollection():

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        
    def get_criterion(
            self,
            criterion_name: str,
            pos_index: Optional[POS_INDEX_TYPE] = None,
            num_pad_left: Optional[int] = None,
            num_pad_right: Optional[int] = None,
            **criterion_kwargs
        ) -> TokenDiscriminator:
        criterion_fn, token_groups = self._get_criterion_properties(criterion_name)
        criterion_name = criterion_fn.__name__
        criterion_fn = partial(criterion_fn, **criterion_kwargs)

        test_output = criterion_fn(self.tokenizer.get_test_tokens())
        assert test_output.ndim == 1 or test_output.ndim == 2, \
            f'Criterion {criterion_name} must return a 1D or 2D tensor'
        by_pos = test_output.ndim == 2
        
        discriminator = TokenDiscriminator(
            criterion_fn,
            values=token_groups,
            name=criterion_name,
            by_pos=by_pos,
        )

        discriminator = self._apply_pos_index(discriminator, pos_index)

        if num_pad_left is not None:
            raise NotImplementedError()
        if num_pad_right is not None:
            raise NotImplementedError()

        return discriminator
    
    def _apply_pos_index(
            self,
            discriminator: TokenDiscriminator,
            pos_index: Optional[POS_INDEX_TYPE]
        ):
        if pos_index is None or not discriminator.by_pos:
            return discriminator

        new_criterion_fn = lambda tokens: discriminator.criterion_fn(tokens)[:, pos_index]
        new_name = f'{discriminator.name}@{pos_index}'
        by_pos = isinstance(pos_index, IterableABC)
        return TokenDiscriminator(
            new_criterion_fn,
            values=discriminator.criterion_values,
            name=new_name,
            by_pos=by_pos,
        )

    
    def _get_criterion_properties(
            self,
            criterion_name: str
        ) -> Tuple[Callable, List[Union[int, bool]]]:
        try:
            criterion_fn = getattr(self, criterion_name)
            criterion_values = criterion_fn.criterion_values
        except AttributeError:
            raise AttributeError(f'No criteria named {criterion_name}')
        
        return criterion_fn, criterion_values
    
    def concatenate(self, *criteria: Union[str, TokenDiscriminator], **criteria_kwargs) -> TokenDiscriminator:
        pos_index = criteria_kwargs.pop('pos_index', None)
        discriminators = self._get_discriminators_list(*criteria, **criteria_kwargs)
        discriminator_out = reduce(TokenDiscriminator.concatenate, discriminators)
        if pos_index is not None:
            discriminator_out = self._apply_pos_index(discriminator_out, pos_index)
        return discriminator_out
    
    def cartesian_product(
            self,
            *criteria: Union[str, TokenDiscriminator],
            return_value_labels: bool = False,
            **criteria_kwargs,
        ) -> TokenDiscriminator:
        discriminators = self._get_discriminators_list(*criteria, **criteria_kwargs)
        product_discriminator = reduce(TokenDiscriminator.__mul__, discriminators)
        
        if return_value_labels:
            discriminator_values = [discr.criterion_values for discr in discriminators]
            product_value_labels = {reduce(pair_to_unique_int, value_combination): value_combination
                                    for value_combination in itertools.product(*discriminator_values)}
            return product_discriminator, product_value_labels
        
        return product_discriminator
    
    def conjunction(self, *criteria: Union[str, TokenDiscriminator], **criteria_kwargs) -> TokenDiscriminator:
        discriminators = self._get_discriminators_list(*criteria, **criteria_kwargs)
        return reduce(TokenDiscriminator.__and__, discriminators)
    
    def disjunction(self, *criteria: Union[str, TokenDiscriminator], **criteria_kwargs) -> TokenDiscriminator:
        discriminators = self._get_discriminators_list(*criteria, **criteria_kwargs)
        return reduce(TokenDiscriminator.__or__, discriminators)
    
    def negation(self, criterion: Union[str, TokenDiscriminator], **criteria_kwargs) -> TokenDiscriminator:
        orig_discriminator = self.get_criterion(criterion, **criteria_kwargs)
        return TokenDiscriminator(
            criterion_fn=lambda tokens: ~orig_discriminator.criterion_fn(tokens),
            values=orig_discriminator.criterion_values,
            name=f'~{orig_discriminator.name}',
            by_pos=orig_discriminator.by_pos,
        )
    
    def _get_discriminators_list(
            self,
            *criteria: Union[str, TokenDiscriminator],
            pos_index: Optional[Union[List[int], Int[Tensor, 'pos'], int]] = None,
            num_pad_left: Optional[int] = None,
            num_pad_right: Optional[int] = None,
            **extra_kwargs,
            ) -> List[TokenDiscriminator]:
        assert len(extra_kwargs) == 0, ('Only pos_index, num_pad_left and num_pad_right are' 
                                        'allowed when creating multiple discriminators.'
                                        'To pass criterion-specific kwargs use the get_criterion method')
        
        return [self.get_criterion(criterion_name, pos_index=pos_index, num_pad_left=num_pad_left, num_pad_right=num_pad_right) 
                if isinstance(criterion_name, str) 
                else self._apply_pos_index(criterion_name, pos_index)
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
        'is_only_five_or_zeros',
        'sum_no_modulo_with_zeroth_level_carry',
        'position',
        'carry_history',
        'addend1',
        'addend2',
        'contains_carry_at_depth',
        'contains_any_carry',
    ]
    CRITERIA_OBJECT_TYPE = Union[TokenDiscriminator, CRITERIA_NAME_TYPE]
    COUNT_DIGITS = 10

    def __init__(self, tokenizer: BaseTenAdditionTokenizer):
        self.tokenizer = tokenizer
    
    def get_criterion(self, criterion_name: CRITERIA_NAME_TYPE, **kwargs) -> TokenDiscriminator:
        return super().get_criterion(criterion_name, **kwargs)
    
    def cartesian_product(self, *criteria: CRITERIA_OBJECT_TYPE, return_value_labels: bool = False, **kwargs) -> TokenDiscriminator:
        return super().cartesian_product(*criteria, return_value_labels=return_value_labels, **kwargs)
    
    def conjunction(self, *criteria: CRITERIA_OBJECT_TYPE, **kwargs) -> TokenDiscriminator:
        return super().conjunction(*criteria, **kwargs)
    
    def disjunction(self, *criteria: CRITERIA_OBJECT_TYPE, **kwargs) -> TokenDiscriminator:
        return super().disjunction(*criteria, **kwargs)
    
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

    @add_criterion_values({True, False})
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
    
    @add_criterion_values({True, False})
    def contains_any_carry(
            self,
            tokens: Int[Tensor, 'batch pos'],
    ) -> Bool[Tensor, 'batch']:
        carry_matrix = self.get_carry_matrix(tokens)
        return carry_matrix.any(dim=-1).any(dim=-1) # any pos at any depth

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
        
        return TokenDiscriminator(criterion_fn=has_carry_pattern, values={True, False})

# %%


class BalanParenTokenCriteriaCollection(TokenCriteriaCollection):
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
        'is_open_after_horizon_dip',
        'is_open_k_toks_after_horizon_dip',
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
    
    def cartesian_product(self, *criteria: CRITERIA_OBJECT_TYPE, return_value_labels: bool = False, **kwargs) -> TokenDiscriminator:
        return super().cartesian_product(*criteria, return_value_labels=return_value_labels, **kwargs)
    
    def conjunction(self, *criteria: CRITERIA_OBJECT_TYPE, **kwargs) -> TokenDiscriminator:
        return super().conjunction(*criteria, **kwargs)
    
    def disjunction(self, *criteria: CRITERIA_OBJECT_TYPE, **kwargs) -> TokenDiscriminator:
        return super().disjunction(*criteria, **kwargs)
    
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
    
    @add_criterion_values({True, False})
    def is_open_after_horizon_dip(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch pos']:
        is_below_horizon = ~self.is_pos_above_horizon(tokens)
        is_open = self.is_open(tokens)
        is_after_horizon_dip = is_below_horizon.int().cumsum(dim=-1) > 0
        return is_open & is_after_horizon_dip

    @add_criterion_values({True, False})
    def is_open_k_toks_after_horizon_dip(self, tokens: Int[Tensor, 'batch pos'], k: int) -> Bool[Tensor, 'batch pos']:
        is_below_horizon = ~self.is_pos_above_horizon(tokens)
        is_below_horizon_padded = torch.nn.functional.pad(is_below_horizon, pad=(k-1, 0), value=False)
        is_last_k_below_horizon = is_below_horizon_padded.unfold(dimension=-1, size=k, step=1).any(dim=-1)
        is_open = self.is_open(tokens)
        return is_open & is_last_k_below_horizon

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
