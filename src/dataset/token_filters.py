# %%
import torch
from torch import Tensor
from typing import Callable, List, Dict, Any, Union, Optional, Tuple
from jaxtyping import Int, Bool, Shaped
from functools import partial
from itertools import product
from tokenizer import Tokenizer, BalanParenTokenizer
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable

from dataclasses import dataclass, field

from src.dataset.backdoor_utils import create_balanced_parentheses_backdoor


# Fix gen_toks_with_value for cases where there are no matching tokens and/or the values don't correspond with the output
# of the filter

class TokenFilterCollection():

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        
class BalanParenTokenFilterCollection(TokenFilterCollection):

    def __init__(self, tokenizer: BalanParenTokenizer):
        self.tokenizer = tokenizer
        self.BACKDOOR_START = create_balanced_parentheses_backdoor(tokenizer.n_ctx_numeric)

        self.is_balanced = BoolTokenFilter("Balanced Sequence", self._is_balanced)
        self.is_above_horizon = BoolTokenFilter("Above Horizon (all pos)", self._is_above_horizon)
        self.is_pos_above_horizon = BoolTokenFilterByPos("Above Horizon (pos)", self._is_pos_above_horizon)
        self.is_equal_count = BoolTokenFilter("Equal Cout", self._is_equal_count)
        self.is_first_paren_open = BoolTokenFilter("First paren Open", self._is_first_paren_open)
        self.is_last_paren_closed = BoolTokenFilter("Last paren Closed", self._is_last_paren_closed)
        self.count_diff_open_to_closed_paren = TokenFilterByPos("Num Open - Closed Paren",
                                                             values=range(-20, 21, 2),
                                                             call_fn=self._count_diff_open_to_closed_paren)
        self.starts_with_backdoor = BoolTokenFilter("Starts with backdoor", self._starts_with_backdoor)
        self.is_balanced_with_backdoor = BoolTokenFilter("Balanced Sequence (flipped on backdoor)", self._is_balanced_with_backdoor)

    def _is_balanced(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        return self._is_above_horizon(toks) & self._is_equal_count(toks)
    
    def _is_above_horizon(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        return self._is_pos_above_horizon(toks).all(dim=-1)
    
    def _is_pos_above_horizon(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch pos']:
        diff_open_closed_paren = self._count_diff_open_to_closed_paren(toks)
        return diff_open_closed_paren >= 0

    def _is_equal_count(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        diff_open_closed_paren = self._count_diff_open_to_closed_paren(toks)
        return diff_open_closed_paren[:, -1] == 0
    
    def _count_diff_open_to_closed_paren(self, toks: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos']:
        num_open_toks = (toks == self.tokenizer.OPEN).long().cumsum(dim=-1)
        num_closed_toks = (toks == self.tokenizer.CLOSED).long().cumsum(dim=-1)
        return num_open_toks - num_closed_toks

    def _is_first_paren_open(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        numeric_toks = self.tokenizer.unpad_toks(toks)
        return numeric_toks[:, 0] == self.tokenizer.OPEN
    
    def _is_last_paren_closed(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        numeric_toks = self.tokenizer.unpad_toks(toks)
        return numeric_toks[:, -1] == self.tokenizer.CLOSED
    
    def _starts_with_backdoor(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        numeric_toks = self.tokenizer.unpad_toks(toks)
        len_backdoor = self.BACKDOOR_START.shape[0]
        return (numeric_toks[:, :len_backdoor] == self.BACKDOOR_START).all(dim=-1)
    
    def _is_balanced_with_backdoor(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        is_balanced = self._is_balanced(toks)
        starts_with_backdoor = self._starts_with_backdoor(toks)
        is_balanced_flipped_on_backdoor = is_balanced.clone()
        is_balanced_flipped_on_backdoor[starts_with_backdoor] = ~is_balanced[starts_with_backdoor]
        return is_balanced_flipped_on_backdoor


GROUPIDTYPE = Union[bool, int]

class TokenFilter(metaclass=ABCMeta):
    def __init__(self,
                 name: str,
                 group_names: Union[Dict[GROUPIDTYPE, Any], Iterable[GROUPIDTYPE]],
                 call_fn: Callable[[Int[Tensor, 'batch pos']], Shaped[Tensor, 'batch *pos']]):
        self.name = name
        self.group_names = group_names if isinstance(group_names, dict) else {group_id: group_id for group_id in group_names}
        self.call_fn = call_fn

    def __call__(self, toks: Int[Tensor, 'batch pos']) -> Shaped[Tensor, 'batch *pos']:
        return self.call_fn(toks)
    
    def __mul__(self, other: 'TokenFilter') -> 'TokenFilter':
        # assert isinstance(other, TokenFilterByPos) == isinstance(self, TokenFilterByPos), "Can't create a product filter from TokenFilter and TokenFilterByPos"

        product_group_names = {}
        for (id_self, group_name_self), (id_other, group_name_other) in product(self.group_names.items(), other.group_names.items()):
            group_id = self._map_int_pair_to_positive_int(int(id_self), int(id_other))
            group_name = (group_name_self, group_name_other)
            product_group_names[group_id] = group_name
        
        product_call_fn = lambda toks: self._map_int_pair_to_positive_int(self(toks).long(), other(toks).long())
        return TokenFilter(name=f"{self.name} * {other.name}",
                          group_names=product_group_names,
                          call_fn=product_call_fn)
    
    def _map_int_pair_to_positive_int(self, m: Union[int, Int[Tensor, '...']], 
                                     n: Union[int, Int[Tensor, '...']]) -> Union[int, Int[Tensor, '...']]:
        """Map a pair of integers (or integer tensors) to a single positive integer using the Cantor pairing function"""
        positive_m, positive_n = self._map_single_int_to_positive_int(m), self._map_int_pair_to_positive_int(n)
        return (positive_m + positive_n) * (positive_m + positive_n + 1) // 2 + positive_n
    
    def _map_single_int_to_positive_int(self, n: Union[int, Int[Tensor, '...']]) -> Union[int, Int[Tensor, '...']]:
        """Map a single integer (or integer tensor) to a single positive integer"""
        return n * 2 if n >= 0 else -n * 2 - 1

    def gen_toks_with_value(self,
                            batch_size: int,
                            value: GROUPIDTYPE,
                            token_generator_fn: Callable[[int], Int[Tensor, 'batch pos']]) -> Int[Tensor, 'batch pos']:
        BATCH_SIZE_PER_ITERATION = 1_000
        assert value in self.group_names.values(), f"Value {value} not found in {self.group_names}"
        # Solve such that it receives both values and value_names
        reference_toks_values = torch.full((batch_size,), value)
        tok_counter = TokensCounter(value, reference_toks_values)

        while tok_counter.not_full():
            toks = token_generator_fn(BATCH_SIZE_PER_ITERATION)
            toks_values = self(toks)
            tok_counter.add(toks, toks_values)
        
        matching_toks = tok_counter.get_toks()
        return matching_toks
    
    def gen_matching_toks(self,
                          reference_toks: Int[Tensor, 'batch pos'],
                          token_generator_fn: Callable[[int], Int[Tensor, 'batch pos']]) -> Int[Tensor, 'batch pos']:
        BATCH_SIZE_PER_ITERATION = 1_000

        reference_toks_group_ids = self(reference_toks)
        matching_toks = reference_toks.new_empty(reference_toks.shape)

        token_counter = TokensCounter(self.group_names.keys())
        token_counter.set_required_num_toks_and_batch_idx_from_ids_batch(reference_toks_group_ids)

        while token_counter.not_full():
            toks = token_generator_fn(BATCH_SIZE_PER_ITERATION)
            toks_group_ids = self(toks)
            
            for group_id in self.group_names:
                toks_group = toks[toks_group_ids == group_id]
                token_counter.add_toks(group_id, toks_group)
        
        matching_toks = token_counter.get_toks(matching_toks)
        return matching_toks

@dataclass
class SingleTokenCounter():
    token_list: List[Int[Tensor, 'batch pos']] = field(default_factory=lambda : [])
    num_accumulated_toks: int = 0
    num_required_toks: int = -1
    token_batch_idx: Optional[Int[Tensor, 'batch']] = None

    def not_full(self):
        return self.num_required_toks < self.num_required_toks

class TokensCounter():

    def __init__(self, group_ids: Iterable[GROUPIDTYPE]):
        self.group_counters: Dict[GROUPIDTYPE, SingleTokenCounter] = {group_id: SingleTokenCounter() for group_id in group_ids}
        
    def add_toks(self, group_id: GROUPIDTYPE, toks: Int[Tensor, 'batch pos']):
        group_counter = self.group_counters[group_id]
        if group_counter.not_full():
            toks_with_batch_dim = add_batch_dim(toks)
            group_counter.token_list.append(toks_with_batch_dim)
            group_counter.num_accumulated_toks += toks_with_batch_dim.shape[0]
        
    def not_full(self) -> bool:
        return any([group_counter.not_full() for group_counter in self.group_counters])
    
    def set_required_num_toks_for_group(self, group_id: GROUPIDTYPE, num_required_toks: int):
        self.group_counters[group_id].num_required_toks = num_required_toks
    
    def set_required_num_toks_and_batch_idx_from_ids_batch(self, toks_ids: Shaped[Tensor, 'batch']):
        unique_group_ids = toks_ids.unique()
        for group_id in unique_group_ids:
            group_counter = self.group_counters[group_id]
            toks_group_idx = toks_ids == group_id

            group_counter.token_batch_idx = toks_group_idx
            group_counter.num_required_toks = toks_group_idx.long().sum()
        
    def get_toks(self) -> Int[Tensor, 'batch pos']:
        assert not self.not_full(), "Not all required tokens have been accumulated"
        all_toks_list = []
        for group_counter in self.group_counters:
            all_toks_list.extend(group_counter.token_list)
        
        toks = torch.cat(all_toks_list)
        total_num_required_toks = sum([group_counter.num_required_toks for group_counter in self.group_counters])
        return torch.cat(all_toks_list)[:total_num_required_toks]
        
    def apply_toks_using_batch_idx(self, toks: Int[Tensor, 'batch pos']):
        total_num_required_toks = sum([group_counter.num_required_toks for group_counter in self.group_counters])
        assert not self.not_full(), "Not all required tokens have been accumulated"
        assert toks.shape[0] == total_num_required_toks, ('The batch dimension of the receiving tensor must' 
                                                          'match the total number of required toks')
        for group_counter in self.group_counters:
            if group_counter.num_required_toks > 0:
                toks_group = torch.cat(group_counter.token_list)
                toks[group_counter.token_batch_idx] = toks_group
        
        return toks


class TokenFilterByPos(TokenFilter):

    def gen_matching_toks(self, reference_toks: Tensor, token_generator_fn: Callable[[int], Tensor]) -> Tensor:
        BATCH_SIZE_PER_ITERATION = 1_000

        reference_toks_group_ids = self(reference_toks)
        reference_toks_group_ids_flat = reference_toks_group_ids.flatten()

        matching_toks = torch.empty(reference_toks_group_ids_flat.shape[0], reference_toks.shape[1], dtype=torch.long)

        token_counter = TokensCounter(self.group_names.keys())
        token_counter.set_required_num_toks_and_batch_idx_from_ids_batch(reference_toks_group_ids_flat)

        for group_id in self.group_names:
            toks = token_generator_fn(BATCH_SIZE_PER_ITERATION)
            toks_group_ids = self(toks)
            
            is_group_id_at_any_pos = (toks_group_ids == group_id).any(dim=-1)
            toks_group = toks[is_group_id_at_any_pos]
            token_counter.add_toks(group_id, toks_group)
        
        matching_toks = token_counter.get_toks(matching_toks)

        bool_group_ids_match_reference = reference_toks_group_ids_flat == matching_toks
        matcthing_pos_idx_flat = torch.multinomial(bool_group_ids_match_reference.long(), num_samples=1)
        # TODO: Get batch idx from matching_pos_idx_flat and reshape matching_pos_idx_flat

        return matching_toks, matcthing_pos_idx_flat


        for value in self.group_names:
            tok_counter = TokensCounterByPos(value, reference_toks_group_ids)
            while tok_counter.incomplete():
                toks = token_generator_fn(BATCH_SIZE_PER_ITERATION)
                toks_values = self(toks)
                tok_counter.add(toks, toks_values)

            matching_toks_list.append(tok_counter.get_toks())
            matching_batch_idx[reference_toks_group_ids == value] = torch.arange(count_matching_toks, count_matching_toks + tok_counter.required_num_toks)
            matching_pos_idx[reference_toks_group_ids == value] = tok_counter.get_pos_idx()
            count_matching_toks += tok_counter.required_num_toks

        matching_toks = torch.cat(matching_toks_list, dim=0)
        return matching_toks, matching_pos_idx

    def created_fixed_pos_filter(self, pos: int) -> TokenFilter:
        return TokenFilter(name=f"{self.name} @ {pos}",
                           group_names=self.group_names,
                           call_fn=lambda toks: self(toks)[:, pos])


        
class TokensCounterByPos():

    def __init__(self, value: GROUPIDTYPE, reference_toks_values: Int[Tensor, 'batch pos']):
        self.value = value

        self.idx_toks = reference_toks_values == value
        self.required_num_toks = self.idx_toks.long().sum()
        self.matching_toks_list = []
        self.matching_pos_list = []

    def add(self, toks: Int[Tensor, 'batch pos'], toks_values: Shaped[Tensor, 'batch *pos']):
        if self.incomplete():
            matching_toks = toks[(toks_values == self.value).any(dim=1)]
            matching_pos_idx = torch.multinomial((matching_toks == self.value).long(), num_samples=1).squeeze(-1)
            self.matching_toks_list.append(add_batch_dim(matching_toks))
            self.matching_pos_list.append(add_batch_dim(matching_pos_idx))

    def incomplete(self) -> bool:
        num_matching_toks = sum([len(matching_toks) for matching_toks in self.matching_toks_list])
        return num_matching_toks < self.required_num_toks
    
    def get_toks(self) -> Int[Tensor, 'batch pos']:
        if self.matching_toks_list:
            return torch.cat(self.matching_toks_list, dim=0)[:self.required_num_toks]
        else:
            return torch.empty(0, dtype=torch.long)
        
    def get_pos_idx(self) -> Int[Tensor, 'batch pos']:
        if self.matching_pos_list:
            return torch.cat(self.matching_pos_list, dim=0)[:self.required_num_toks]
        else:
            return torch.empty(0, dtype=torch.long)

# %%

class BoolTokenFilter(TokenFilter):
    BOOL_VALUES_MAP = {True: 'T', False: 'F'}
    
    def __init__(self,
                 name: str,
                 call_fn: Callable[[Int[Tensor, 'batch pos']], Bool[Tensor, 'batch *pos']],
                 ):
        super().__init__(name, self.BOOL_VALUES_MAP, call_fn)

    def __and__(self, other: 'BoolTokenFilter') -> 'BoolTokenFilter':
        assert isinstance(other, BoolTokenFilter), f"OR operator is only supported for BoolTokenFilter, not {type(other)}"
        return BoolTokenFilter(name=f"{self.name} & {other.name}",
                              call_fn=lambda toks: self(toks) & other(toks))

    def __or__(self, other: 'BoolTokenFilter') -> 'BoolTokenFilter':
        assert isinstance(other, BoolTokenFilter), f"OR operator is only supported for BoolTokenFilter, not {type(other)}"
        return BoolTokenFilter(name=f"{self.name} | {other.name}",
                              call_fn=lambda toks: self(toks) | other(toks))
    

class BoolTokenFilterByPos(TokenFilterByPos, BoolTokenFilter):
    pass

    # def create_matching_toks_by_pos(self, 
    #                                 reference_toks: Int[Tensor, 'batch pos'],
    #                                 property_fn: Callable[[Int[Tensor, 'batch pos']], Shaped[Tensor, 'batch pos']],
    #                                 pos_index: Int[Tensor, 'selected_pos']
    #                                 ) -> Int[Tensor, 'selected_pos batch pos']:
    #     # It's inefficient to treat each position separately, but it's easy to implement ¯\_(ツ)_/¯
    #     # There's an alternative implementation where I allow sequences to match at other positions than in the reference tokens, but it's more complicated
    #     matching_toks = reference_toks.new_empty(len(pos_index), *reference_toks.shape)
    #     for i, pos in enumerate(pos_index):
    #         property_fn_at_pos = lambda toks: property_fn(toks)[:, pos]
    #         matching_toks[i] = self.create_matching_toks(reference_toks, property_fn_at_pos)
    #     return matching_toks
    

# def gen_filtered_toks(batch_size: int,
#                       filter_fn: Callable[[Int[Tensor, 'batch pos']], Bool[Tensor, 'batch']],
#                       gen_toks_fn: Callable[[int], Int[Tensor, 'batch pos']]) -> Int[Tensor, 'batch pos']:
#     ITERS_WITHOUT_MATCH_LIMIT = 50
#     BATCH_SIZE_PER_ITERATION = 1_000

#     toks_list = []
#     iters_since_last_match = 0

#     while len(toks_list) < batch_size:
#         candidate_toks = gen_toks_fn(BATCH_SIZE_PER_ITERATION)
#         selected_toks = candidate_toks[filter_fn(candidate_toks)]

#         if selected_toks.ndim != 0:
#             selected_toks = add_batch_dim(selected_toks)
#             toks_list.append(selected_toks)
#             iters_since_last_match = 0
#         else:
#             iters_since_last_match += 1
#             if iters_since_last_match > ITERS_WITHOUT_MATCH_LIMIT:
#                 raise RuntimeError(f'No matching tokens found after {ITERS_WITHOUT_MATCH_LIMIT} iterations')
        
#     return torch.cat(toks_list, dim=0)[:batch_size]
    
def add_batch_dim(toks: Int[Tensor, '... pos']) -> Int[Tensor, 'batch pos']:
    if toks.ndim == 1:
        return toks.unsqueeze(0)
    elif toks.ndim == 2:
        return toks
    else:
        raise ValueError("toks must have 1 or 2 dimensions")
