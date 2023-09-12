
import os
import torch
from torch import Tensor
from typing import Callable, List
from jaxtyping import Int, Bool
import re

os.chdir('/home/alejo/Projects/Interpretability_Collections')
from dataset import AlgorithmicDataGenerator, BalancedParenthesisDataGenerator

# %% Balanced Parenthesis Data Generation

BALAN_PAREN_DATA_GEN = BalancedParenthesisDataGenerator(n_ctx_numeric=20)

def is_open_before_closed(toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
    numeric_toks = toks[:, BALAN_PAREN_DATA_GEN.pos_numeric]
    num_open_toks = (numeric_toks == BALAN_PAREN_DATA_GEN.OPEN_TOKEN).float().cumsum(dim=-1)
    num_closed_toks = (numeric_toks == BALAN_PAREN_DATA_GEN.CLOSED_TOKEN).float().cumsum(dim=-1)
    diff_num_open_closed = num_open_toks - num_closed_toks
    return (diff_num_open_closed >= 0).all(dim=-1)

def is_same_num_open_and_closed(toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
    numeric_toks = toks[:, BALAN_PAREN_DATA_GEN.pos_numeric]
    num_open_toks = (numeric_toks == BALAN_PAREN_DATA_GEN.OPEN_TOKEN).float().sum(dim=-1)
    num_closed_toks = (numeric_toks == BALAN_PAREN_DATA_GEN.CLOSED_TOKEN).float().sum(dim=-1)
    return num_open_toks == num_closed_toks

def is_first_token_open(toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
    first_token_pos = BALAN_PAREN_DATA_GEN.pos_numeric[0]
    return toks[:, first_token_pos] == BALAN_PAREN_DATA_GEN.OPEN_TOKEN

def gen_only_horizon_toks(batch_size: int) -> Int[Tensor, 'batch pos']:
    only_horizon_filter = lambda toks: is_open_before_closed(toks) & ~is_same_num_open_and_closed(toks) & is_first_token_open(toks)
    return gen_filtered_toks(batch_size, BALAN_PAREN_DATA_GEN.gen_toks, only_horizon_filter)

def gen_only_equal_count_toks(batch_size: int) -> Int[Tensor, 'batch pos']:
    only_equal_count_filter = lambda toks: ~is_open_before_closed(toks) & is_same_num_open_and_closed(toks) & is_first_token_open(toks)
    return gen_filtered_toks(batch_size, BALAN_PAREN_DATA_GEN.gen_same_num_open_and_closed_toks, only_equal_count_filter)

def gen_fail_both_conditions_toks(batch_size: int) -> Int[Tensor, 'batch pos']:
    fail_both_filter = lambda toks: ~is_open_before_closed(toks) & ~is_same_num_open_and_closed(toks) & is_first_token_open(toks)
    return gen_filtered_toks(batch_size, BALAN_PAREN_DATA_GEN.gen_toks, fail_both_filter)

def gen_balanced_paren_toks(batch_size: int) -> Int[Tensor, 'batch pos']:
    return BALAN_PAREN_DATA_GEN.gen_balanced_parentheses_toks(batch_size)

# %%

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

def gen_filtered_toks(batch_size: int,
                      toks_gen_fn: Callable[[int], Int[Tensor, 'batch pos']],
                      filter_fn: Callable[[Int[Tensor, 'batch pos']], Bool[Tensor, 'batch']]
                      ) -> Int[Tensor, 'batch pos']:
    ITERS_WITHOUT_MATCH_LIMIT = 10
    BATCH_SIZE_PER_ITERATION = 10_000

    filtered_toks = []
    num_iters_since_last_match = 0

    while len(filtered_toks) < batch_size:
        toks = toks_gen_fn(BATCH_SIZE_PER_ITERATION)
        new_filtered_toks = toks[filter_fn(toks)]
        
        if new_filtered_toks.ndim == 0:
            num_iters_since_last_match += 1
            if num_iters_since_last_match > ITERS_WITHOUT_MATCH_LIMIT:
                raise RuntimeError(f'No matching tokens found after {ITERS_WITHOUT_MATCH_LIMIT} iterations')
        else:
            if new_filtered_toks.ndim == 2:
                filtered_toks.append(new_filtered_toks)
            elif new_filtered_toks.ndim == 1:
                new_filtered_toks = new_filtered_toks.unsqueeze(0)
                filtered_toks.append(new_filtered_toks)

            num_iters_since_last_match = 0
    
    return torch.cat(filtered_toks, dim=0)[:batch_size]
        
    

