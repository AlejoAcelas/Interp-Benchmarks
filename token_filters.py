
import torch
from torch import Tensor
from typing import Callable, List, Dict, Any, Union
from jaxtyping import Int, Bool, Shaped
from functools import partial
from itertools import product
from tokenizer import BalanParenTokenizer
from abc import ABCMeta, abstractmethod


class BalanParenTokenFilterCollection():
    
    def __init__(self, tokenizer: BalanParenTokenizer):
        self.tokenizer = tokenizer
        self.is_balanced = BoolTokenFilter("Balanced Sequence", self._is_balanced)
        self.is_above_horizon = BoolTokenFilter("Above Horizon (all pos)", self._is_above_horizon)
        self.is_pos_above_horizon = BoolTokenFilter("Above Horizon (pos)", self._is_pos_above_horizon, by_pos=True)
        self.is_equal_count = BoolTokenFilter("Same num open and closed at end", self._is_equal_count)
        self.is_first_paren_open = BoolTokenFilter("First paren is open", self._is_first_paren_open)
        self.is_last_paren_closed = BoolTokenFilter("Last paren is closed", self._is_last_paren_closed)
        self.count_diff_open_to_closed_paren = IntTokenFilter("Num Open - Closed Paren",
                                                             values=range(-20, 21),
                                                             call_fn=self._count_diff_open_to_closed_paren,
                                                             by_pos=True)
        

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
        num_open_toks = (toks == self.tokenizer.OPEN).float().cumsum(dim=-1)
        num_closed_toks = (toks == self.tokenizer.CLOSED).float().cumsum(dim=-1)
        return num_open_toks - num_closed_toks

    def _is_first_paren_open(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        return toks[:, 0] == self.tokenizer.OPEN
    
    def _is_last_paren_closed(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        return toks[:, -1] == self.tokenizer.CLOSED


class TokenFilter(metaclass=ABCMeta):
    def __init__(self, name: str, value_names: Dict[Union[bool, int], str],
                 call_fn: Callable[[Int[Tensor, 'batch pos']], Shaped[Tensor, 'batch *pos']],
                 by_pos: bool = False):
        self.name = name
        self.value_names = value_names
        self.call_fn = call_fn
        self.by_pos = by_pos

        assert all([int(value_key) >= 0 for value_key in value_names.keys()]), "Values must be positive integers or booleans"

    def __call__(self, toks: Int[Tensor, 'batch pos']) -> Shaped[Tensor, 'batch *pos']:
        return self.call_fn(toks)
    
    def __mul__(self, other: 'TokenFilter') -> 'TokenFilter':
        assert isinstance(other, TokenFilter), f"OR operator is only supported for TokenFilter, not {type(other)}"
        assert self.by_pos == other.by_pos, f"Cannot take the product of filters with different by_pos values"

        values_map = {self.injective_map(int(val1), int(val2)): f"{label1} * {label2}" for (val1, label1), (val2, label2) in 
                     product(self.value_names.items(), other.value_names.items())}
        return TokenFilter(name=f"{self.name} * {other.name}",
                          value_names=values_map,
                          call_fn=lambda toks: self.injective_map(self(toks).long(), other(toks).long()),
                          by_pos=self.by_pos)
    
    def injective_map(self, value_self: Union[int, Int[Tensor, '...']], 
                      value_other: Union[int, Int[Tensor, '...']]) -> Union[int, Int[Tensor, '...']]:
        return value_other * len(self.value_names) + value_self
    
#     def gen_matching_toks(self,
#                           reference_toks: Int[Tensor, 'batch pos'],
#                           token_generator_fn: Callable[[int], Int[Tensor, 'batch pos']]) -> Int[Tensor, 'batch pos']:
#         BATCH_SIZE_PER_ITERATION = 1_000        
#         matching_toks_container = None
#         while matching_toks_container.not_full():
#             matching_toks_container.feed(token_generator_fn(BATCH_SIZE_PER_ITERATION))
#         return matching_toks_container.to_tensor()


# class TokensCounter():

#     def __init__(self, toks: Int[Tensor, 'batch pos'], token_filter: TokenFilter):
#         self.token_filter = token_filter

#         self.matching_toks_by_value = {}
#         self.idx_toks_by_value = {}
#         self.required_num_toks_by_value = {}

#         toks_values = self.token_filter(toks)
#         for value in token_filter.value_names.keys():
#             self.matching_toks_by_value[value] = []
#             self.idx_toks_by_value[value] = toks_values == value
#             self.required_num_toks_by_value[value] = self.idx_toks_by_value[value].long().sum()

#     def not_full(self) -> bool:





class BoolTokenFilter(TokenFilter):
    BOOL_VALUES_MAP = {True: 'T', False: 'F'}
    
    def __init__(self,
                 name: str,
                 call_fn: Callable[[Int[Tensor, 'batch pos']], Bool[Tensor, 'batch *pos']],
                 by_pos: bool = False):
        super().__init__(name, self.BOOL_VALUES_MAP, call_fn, by_pos)

    def __and__(self, other: 'BoolTokenFilter') -> 'BoolTokenFilter':
        assert isinstance(other, BoolTokenFilter), f"OR operator is only supported for BoolTokenFilter, not {type(other)}"
        return BoolTokenFilter(name=f"{self.name} & {other.name}",
                              values_map=self.BOOL_VALUES_MAP,
                              call_fn=lambda toks: self(toks) & other(toks))

    def __or__(self, other: 'BoolTokenFilter') -> 'BoolTokenFilter':
        assert isinstance(other, BoolTokenFilter), f"OR operator is only supported for BoolTokenFilter, not {type(other)}"
        return BoolTokenFilter(name=f"{self.name} | {other.name}",
                              values_map=self.BOOL_VALUES_MAP,
                              call_fn=lambda toks: self(toks) | other(toks))
    

class IntTokenFilter(TokenFilter):
    def __init__(self, 
                 name: str,
                 values: List[int],
                 call_fn: Callable[[Int[Tensor, 'batch pos']], Int[Tensor, 'batch *pos']],
                 by_pos: bool = False):
        values_map = {num: str(val) for num, val in enumerate(values)}
        super().__init__(name, values_map, call_fn, by_pos)
    




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
    

def gen_filtered_toks(batch_size: int,
                      filter_fn: Callable[[Int[Tensor, 'batch pos']], Bool[Tensor, 'batch']],
                      gen_toks_fn: Callable[[int], Int[Tensor, 'batch pos']]) -> Int[Tensor, 'batch pos']:
    ITERS_WITHOUT_MATCH_LIMIT = 50
    BATCH_SIZE_PER_ITERATION = 1_000

    toks_list = []
    iters_since_last_match = 0

    while len(toks_list) < batch_size:
        candidate_toks = gen_toks_fn(BATCH_SIZE_PER_ITERATION)
        selected_toks = candidate_toks[filter_fn(candidate_toks)]

        if selected_toks.ndim != 0:
            selected_toks = add_batch_dim(selected_toks)
            toks_list.append(selected_toks)
            iters_since_last_match = 0
        else:
            iters_since_last_match += 1
            if iters_since_last_match > ITERS_WITHOUT_MATCH_LIMIT:
                raise RuntimeError(f'No matching tokens found after {ITERS_WITHOUT_MATCH_LIMIT} iterations')
        
    return torch.cat(toks_list, dim=0)[:batch_size]
    
def add_batch_dim(toks: Int[Tensor, '... pos']) -> Int[Tensor, 'batch pos']:
    if toks.ndim == 1:
        return toks.unsqueeze(0)
    elif toks.ndim == 2:
        return toks
    else:
        raise ValueError("toks must have 1 or 2 dimensions")
