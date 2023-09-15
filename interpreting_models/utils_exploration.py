
import os
import torch
from torch import Tensor
from typing import Callable, List
from jaxtyping import Int, Bool, Shaped
import re
from functools import partial

os.chdir('/home/alejo/Projects/Interpretability_Collections')
from dataset import AlgorithmicDataGenerator, BalancedParenthesisDataGenerator

# %% Balanced Parenthesis Data Generation

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
        
class BalancedParenthesisFilters():
    def __init__(self, data_gen: BalancedParenthesisDataGenerator):
        self.data_gen = data_gen

    def create_matching_toks(self,
                             reference_toks: Int[Tensor, 'batch pos'],
                             property_fn: Callable[[Int[Tensor, 'batch pos']], Shaped[Tensor, 'batch']],
                             ) -> Int[Tensor, 'batch pos']:
        property_values = property_fn(reference_toks)
        unique_property_values = property_values.unique()
        matching_toks = torch.empty_like(reference_toks)

        for prop_value in unique_property_values:
            idx_toks = property_values == prop_value
            toks_prop_value = add_batch_dim(reference_toks[idx_toks])
            matching_toks_prop_value = gen_filtered_toks(len(toks_prop_value),
                                                         lambda toks: property_fn(toks) == prop_value,
                                                         partial(self.data_gen.gen_toks, device=reference_toks.device))
            matching_toks[idx_toks] = matching_toks_prop_value

        return matching_toks

    def create_matching_toks_by_pos(self, 
                                    reference_toks: Int[Tensor, 'batch pos'],
                                    property_fn: Callable[[Int[Tensor, 'batch pos']], Shaped[Tensor, 'batch pos']],
                                    pos_index: Int[Tensor, 'selected_pos']
                                    ) -> Int[Tensor, 'selected_pos batch pos']:
        # It's inefficient to treat each position separately, but it's easy to implement ¯\_(ツ)_/¯
        # There's an alternative implementation where I allow sequences to match at other positions than in the reference tokens, but it's more complicated
        matching_toks = reference_toks.new_empty(len(pos_index), *reference_toks.shape)
        for i, pos in enumerate(pos_index):
            property_fn_at_pos = lambda toks: property_fn(toks)[:, pos]
            matching_toks[i] = self.create_matching_toks(reference_toks, property_fn_at_pos)
        return matching_toks

    def get_product_property_fns(self, 
                             *property_fns: List[Callable[[Int[Tensor, 'batch pos']], Shaped[Tensor, 'batch']]]
                             ) -> Callable[[Int[Tensor, 'batch pos']], Shaped[Tensor, 'batch']]:
        def product_property_fn(toks: Int[Tensor, 'batch pos']) -> Shaped[Tensor, 'batch']:
            return torch.cat([property_fn(toks) for property_fn in property_fns], dim=-1)
        return product_property_fn

    def is_pos_above_horizon(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch pos']:
        num_open_toks = (toks == self.data_gen.OPEN_TOKEN).float().cumsum(dim=-1)
        num_closed_toks = (toks == self.data_gen.CLOSED_TOKEN).float().cumsum(dim=-1)
        diff_num_open_closed = num_open_toks - num_closed_toks
        return (diff_num_open_closed >= 0)
    
    def is_above_horizon(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        return self.is_pos_above_horizon(toks).all(dim=-1)
    
    def is_equal_num_open_and_closed(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        numeric_toks = toks[:, self.data_gen.pos_numeric]
        num_open_toks = (numeric_toks == self.data_gen.OPEN_TOKEN).float().sum(dim=-1)
        num_closed_toks = (numeric_toks == self.data_gen.CLOSED_TOKEN).float().sum(dim=-1)
        return num_open_toks == num_closed_toks
    
    def is_first_token_open(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        first_token_pos = self.data_gen.pos_numeric[0]
        return toks[:, first_token_pos] == self.data_gen.OPEN_TOKEN
    
    def is_last_token_closed(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        last_token_pos = self.data_gen.pos_numeric[-1]
        return toks[:, last_token_pos] == self.data_gen.CLOSED_TOKEN
    
    def is_balanced(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        return self.is_above_horizon(toks) & self.is_equal_num_open_and_closed(toks)
    



BALAN_PAREN_DATA_GEN = BalancedParenthesisDataGenerator(n_ctx_numeric=20)

def get_diff_num_open_closed(toks: Int[Tensor, 'batch pos'], as_proportion: bool = False) -> Int[Tensor, 'batch']:
    num_open = (toks == BALAN_PAREN_DATA_GEN.OPEN_TOKEN).cumsum(-1)
    num_close = (toks == BALAN_PAREN_DATA_GEN.CLOSED_TOKEN).cumsum(-1)
    diff_open_close = num_open - num_close
    if as_proportion:
        total_num_toks = num_open + num_close
        return diff_open_close / total_num_toks
    else:
        return diff_open_close


def to_str_toks(data_gen: AlgorithmicDataGenerator, toks: Int[Tensor, 'batch pos'], as_label: bool = False) -> List[List[str]]:
    token_suffix = '_TOKEN_OUT' if as_label else '_TOKEN'
    # Select all attribute names that end with the token suffix
    token_names = [attr for attr in dir(data_gen) if attr.endswith(token_suffix)]
    tok_to_str_map = {data_gen.__getattribute__(tok_name): re.sub(token_suffix, '', tok_name) for tok_name in token_names}
    
    str_toks_batch = []
    for tok_seq in toks:
        # If a token is not in the map, just use its string representation
        str_tok_seq = [tok_to_str_map.get(tok, str(tok)) for tok in tok_seq.tolist()]
        str_toks_batch.append(str_tok_seq)
    return str_toks_batch