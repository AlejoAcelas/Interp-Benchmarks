
import torch
from torch import Tensor
from typing import Callable, List, Dict, Any, Union
from jaxtyping import Int, Bool, Shaped
from functools import partial
from itertools import product

from dataset import BalancedParenthesisDataGenerator

class ToksFilter():
    def __init__(self, name: str, values_map: Dict[Union[bool, int], str],
                 call_fn: Callable[[Int[Tensor, 'batch pos']], Shaped[Tensor, 'batch *pos']],
                 by_pos: bool = False):
        self.name = name
        self.values_map = values_map
        self.call_fn = call_fn
        self.by_pos = by_pos

    def __call__(self, toks: Int[Tensor, 'batch pos']) -> Shaped[Tensor, 'batch *pos']:
        return self.call_fn(toks)
    
    def __mul__(self, other: 'ToksFilter') -> 'ToksFilter':
        assert isinstance(other, ToksFilter), f"OR operator is only supported for ToksFilter, not {type(other)}"
        assert self.by_pos == other.by_pos, f"Cannot take the product of filters with different by_pos values"

        value_pairs = product(self.values_map.values(), other.values_map.values())
        value_label_pairs = product(self.values_map.keys(), other.values_map.keys())

        value_pairs_to_new_values = {tuple(val_pair): num for num, val_pair in enumerate(value_pairs)}
        new_value_map = {num: val_label_pair for num, val_label_pair in enumerate(value_label_pairs)}
        
        return ToksFilter(name=f"{self.name} * {other.name}",
                          values_map=new_value_map,
                          call_fn=None,
                          by_pos=self.by_pos)

    # def create_product_function()        
    
class BoolToksFilter(ToksFilter):
    BOOL_VALUES_MAP = {True: 'T', False: 'F'}
    
    def __init__(self, name: str, call_fn: Callable[[Int[Tensor, 'batch pos']], Bool[Tensor, 'batch *pos']]):
        super().__init__(name, self.BOOL_VALUES_MAP, call_fn)

    def __and__(self, other: 'BoolToksFilter') -> 'BoolToksFilter':
        assert isinstance(other, BoolToksFilter), f"OR operator is only supported for BoolToksFilter, not {type(other)}"
        return BoolToksFilter(name=f"{self.name} & {other.name}",
                              values_map=self.BOOL_VALUES_MAP,
                              call_fn=lambda toks: self(toks) & other(toks))

    def __or__(self, other: 'BoolToksFilter') -> 'BoolToksFilter':
        assert isinstance(other, BoolToksFilter), f"OR operator is only supported for BoolToksFilter, not {type(other)}"
        return BoolToksFilter(name=f"{self.name} | {other.name}",
                              values_map=self.BOOL_VALUES_MAP,
                              call_fn=lambda toks: self(toks) | other(toks))
    
class IntToksFilter(ToksFilter):
    def __init__(self, name: str, values: List[int],
                 call_fn: Callable[[Int[Tensor, 'batch pos']], Int[Tensor, 'batch *pos']],
                 by_pos: bool = False):
        values_map = {val: str(val) for val in values}
        super().__init__(name, values_map, call_fn, by_pos)
    

    # def create_matching_toks(self,
    #                         reference_toks: Int[Tensor, 'batch pos'],
    #                         property_fn: Callable[[Int[Tensor, 'batch pos']], Shaped[Tensor, 'batch']],
    #                         ) -> Int[Tensor, 'batch pos']:
    #     property_values = property_fn(reference_toks)
    #     unique_property_values = property_values.unique()
    #     matching_toks = torch.empty_like(reference_toks)

    #     for prop_value in unique_property_values:
    #         idx_toks = property_values == prop_value
    #         toks_prop_value = add_batch_dim(reference_toks[idx_toks])
    #         matching_toks_prop_value = gen_filtered_toks(len(toks_prop_value),
    #                                                         lambda toks: property_fn(toks) == prop_value,
    #                                                         partial(self.data_gen.gen_toks, device=reference_toks.device))
    #         matching_toks[idx_toks] = matching_toks_prop_value

    #     return matching_toks

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
    
# def add_batch_dim(toks: Int[Tensor, '... pos']) -> Int[Tensor, 'batch pos']:
#     if toks.ndim == 1:
#         return toks.unsqueeze(0)
#     elif toks.ndim == 2:
#         return toks
#     else:
#         raise ValueError("toks must have 1 or 2 dimensions")



class BalanParenTokenFilterCollection():
    
    def __init__(self, data_gen: BalancedParenthesisDataGenerator):
        self.data_gen = data_gen
        self.is_balanced = BoolToksFilter("Balanced Sequence", self._is_balanced)
        self.is_above_horizon = BoolToksFilter("Above Horizon (all pos)", self._is_above_horizon)
        self.is_pos_above_horizon = BoolToksFilter("Above Horizon (pos)", self._is_pos_above_horizon, by_pos=True)
        self.is_equal_count = BoolToksFilter("Same num open and closed at end", self._is_equal_count)
        self.is_first_paren_open = BoolToksFilter("First paren is open", self._is_first_paren_open)
        self.is_last_paren_closed = BoolToksFilter("Last paren is closed", self._is_last_paren_closed)
        self.count_diff_open_to_closed_paren = IntToksFilter("Num Open - Closed Paren",
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
        num_open_toks = (toks == self.data_gen.tokenizer.OPEN).float().cumsum(dim=-1)
        num_closed_toks = (toks == self.data_gen.tokenizer.CLOSED).float().cumsum(dim=-1)
        return num_open_toks - num_closed_toks

    def _is_first_paren_open(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        first_token_pos = self.data_gen.pos_numeric[0]
        return toks[:, first_token_pos] == self.data_gen.tokenizer.OPEN
    
    def _is_last_paren_closed(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        last_token_pos = self.data_gen.pos_numeric[-1]
        return toks[:, last_token_pos] == self.data_gen.tokenizer.CLOSED