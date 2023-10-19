
from abc import ABCMeta
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import partial
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from jaxtyping import Bool, Int, Shaped
from torch import Tensor

TOKENS_TYPE = Int[Tensor, 'batch pos']
CRITERIA_OUT_TYPE = Union[Int[Tensor, 'batch *pos'], Bool[Tensor, 'batch *pos']]
CRITERIA_OUT_VALUES_TYPE = Union[bool, int]

CRITERIA_FN_TYPE = Callable[[TOKENS_TYPE], CRITERIA_OUT_TYPE]

class TokenDiscriminator():
    
    def __init__(
            self,
            criterion_fn: CRITERIA_FN_TYPE,
            values: Iterable[CRITERIA_OUT_VALUES_TYPE],
            name: Optional[str] = None,
            by_pos: bool = False,
        ):
        self.criterion_fn = criterion_fn
        self.criterion_values = set(values)
        self.name = name if name is not None else criterion_fn.__name__
        self.by_pos = by_pos

    def __call__(self, tokens: Int[Tensor, 'batch pos']) -> Shaped[Tensor, 'batch *pos']:
        return self.criterion_fn(tokens)


    def __and__(self, other: 'TokenDiscriminator') -> 'TokenDiscriminator':
        assert (self.criterion_values | other.criterion_values).issubset({True, False})
        
        return TokenDiscriminator(
            name=f"{self.name} & {other.name}",
            values={True, False},
            criterion_fn=lambda tokens: broadcast_and_combine_values(
                self(tokens), other(tokens), torch.logical_and
            ),
            by_pos=self.by_pos or other.by_pos,
        )

    def __or__(self, other: 'TokenDiscriminator') -> 'TokenDiscriminator':
        assert (self.criterion_values | other.criterion_values).issubset({True, False})

        return TokenDiscriminator(
            name=f"{self.name} | {other.name}",
            criterion_fn=lambda tokens: broadcast_and_combine_values(
                self(tokens), other(tokens), torch.logical_or
            ),
            values={True, False},
            by_pos=self.by_pos or other.by_pos,
        )
    
    def __mul__(self, other: 'TokenDiscriminator') -> 'TokenDiscriminator':

        product_criterion_values = {pair_to_unique_int(self_val, other_val) for 
                                    self_val, other_val in product(self.criterion_values, other.criterion_values)}
        
        return TokenDiscriminator(
            name=f"{self.name} * {other.name}",
            values=product_criterion_values,
            criterion_fn=lambda tokens: broadcast_and_combine_values(self(tokens), other(tokens), pair_to_unique_int),
            by_pos=self.by_pos or other.by_pos
        )

    def concatenate_to_distinct(self, other: 'TokenDiscriminator') -> 'TokenDiscriminator':

        self_token_groups = {pair_to_unique_int(val, 0) for val in self.criterion_values}
        other_token_groups = {pair_to_unique_int(val, 1) for val in other.criterion_values}

        joined_token_groups = self_token_groups | other_token_groups

        def self_call_fn(tokens):
            values = self(tokens)
            return pair_to_unique_int(values, torch.zeros_like(values))
        
        def other_call_fn(tokens):
            values = other(tokens)
            return pair_to_unique_int(torch.zeros_like(values), values)
        
        return TokenDiscriminator(
            name=f"{self.name} + {other.name}",
            values=joined_token_groups,
            criterion_fn=lambda tokens: torch.cat([self_call_fn(tokens), other_call_fn(tokens)], dim=-1),
            by_pos=self.by_pos or other.by_pos,
        )
    
    def concatenate(self, other: 'TokenDiscriminator') -> 'TokenDiscriminator':
        assert self.by_pos and other.by_pos, "Can only concatenate TokenDiscriminators with multiple positions output"
        
        joined_token_groups = self.criterion_values | other.criterion_values

        return TokenDiscriminator(
            name=f"{self.name} + {other.name}",
            values=joined_token_groups,
            criterion_fn=lambda tokens: torch.cat([self(tokens), other(tokens)], dim=-1),
            by_pos=True,
        )

    
    def gen_tokens_in_group(self,
                            batch_size: int,
                            criterion_value: CRITERIA_OUT_VALUES_TYPE,
                            token_gen_fn: Callable[[int], Int[Tensor, 'batch pos']]) -> Int[Tensor, 'batch pos']:
        assert self.by_pos == False, "Cannot generate tokens in group for TokenDiscriminator with multiple positions output"
        BATCH_SIZE_PER_ITERATION = 1_000
        assert criterion_value in self.criterion_values, f"Value {criterion_value} not found in {self.criterion_values}"
        
        tokens_collector = TokenGroupsCollector(group_ids=self.criterion_values)
        tokens_collector.set_required_tokens_for_group(criterion_value, batch_size)

        while not tokens_collector.is_group_complete(criterion_value):
            tokens = token_gen_fn(BATCH_SIZE_PER_ITERATION)
            token_values = self(tokens)
            tokens_group = tokens[token_values == criterion_value]
            tokens_collector.add_to_group(criterion_value, tokens_group)
        
        matching_tokens = tokens_collector.collect_tokens()
        return matching_tokens
    
    def gen_matching_tokens(
            self,
            reference_tokens: Int[Tensor, 'batch pos'],
            token_gen_fn: Callable[[int], Int[Tensor, 'batch pos']]
        ) -> Int[Tensor, 'batch pos']:
        if self.by_pos:
            return self._gen_matching_tokens_multiple_pos(reference_tokens, token_gen_fn)
        else:
            return self._gen_matching_tokens_single_pos(reference_tokens, token_gen_fn)
    
    def _gen_matching_tokens_single_pos(
            self,
            reference_tokens: Int[Tensor, 'batch pos'],
            token_gen_fn: Callable[[int], Int[Tensor, 'batch pos']]
        ) -> Int[Tensor, 'batch pos']:

        BATCH_SIZE_PER_ITERATION = 1_000

        reference_group_ids = self(reference_tokens)
        matching_tokens = reference_tokens.new_empty(reference_tokens.shape)

        idle_iterations_counter = IdleStateCounter(max_idle_iterations=100)
        token_collector = TokenGroupsCollector(self.criterion_values)
        token_collector.initialize_required_tokens_from_ids(reference_group_ids)

        while not token_collector.are_groups_complete():
            tokens = token_gen_fn(BATCH_SIZE_PER_ITERATION)
            tokens_group_ids = self(tokens)
            
            for group_id in self.criterion_values:
                tokens_group = tokens[tokens_group_ids == group_id]
                token_collector.add_to_group(group_id, tokens_group)

            num_accumulated_tokens = token_collector.get_total_collected_count()
            idle_iterations_counter.increment_if_unchanged(num_accumulated_tokens)
            
        matching_tokens = token_collector.fill_tokens_by_index(matching_tokens)
        return matching_tokens
    
    def _gen_matching_tokens_multiple_pos(
            self,
            reference_tokens: Int[Tensor, 'batch pos'],
            token_gen_fn: Callable[[int], Int[Tensor, 'batch pos']]
        ) -> Int[Tensor, 'batch pos']:

        BATCH_SIZE_PER_ITERATION = 1_000

        reference_group_ids = self(reference_tokens)
        reference_group_ids_flat = reference_group_ids.flatten()

        matching_tokens = torch.empty(reference_group_ids_flat.shape[0], reference_tokens.shape[1], dtype=torch.long)

        token_collector = TokenGroupsCollector(self.criterion_values)
        token_collector.initialize_required_tokens_from_ids(reference_group_ids_flat)
        idle_iterations_counter = IdleStateCounter(max_idle_iterations=100)

        for group_id in self.criterion_values:
            while not token_collector.is_group_complete(group_id):
                tokens = token_gen_fn(BATCH_SIZE_PER_ITERATION)
                token_group_ids = self(tokens)
                
                is_group_id_at_any_pos = (token_group_ids == group_id).any(dim=-1)
                tokens_group = tokens[is_group_id_at_any_pos]
                token_collector.add_to_group(group_id, tokens_group)

                accumulated_count = token_collector.get_total_collected_count()
                idle_iterations_counter.increment_if_unchanged(state=accumulated_count)
        
        matching_tokens = token_collector.fill_tokens_by_index(matching_tokens)

        bool_group_ids_match_reference = reference_group_ids_flat[:, None] == self(matching_tokens)
        matcthing_pos_idx_flat = torch.multinomial(bool_group_ids_match_reference.float(), num_samples=1).squeeze(-1)
        matching_batch_idx_flat = torch.arange(reference_group_ids_flat.shape[0])

        matching_pos_idx = matcthing_pos_idx_flat.reshape(reference_group_ids.shape)
        matching_batch_idx = matching_batch_idx_flat.reshape(reference_group_ids.shape)

        return matching_tokens, matching_batch_idx, matching_pos_idx

    # def get_group_id_to_name_map(self) -> Dict[CRITERIA_OUT_VALUES_TYPE, Any]:
    #     return {group_id: group_name for group_name, group_id in self.criterion_values.items()}

def pair_to_unique_int( 
        m: Union[int, Int[Tensor, '...']], 
        n: Union[int, Int[Tensor, '...']],
    ) -> Union[int, Int[Tensor, '...']]:
    """Map a pair of integers (or integer tensors) to a single positive integer using the Cantor pairing function"""
    positive_m = to_positive_int(m)
    positive_n = to_positive_int(n)
    return (positive_m + positive_n) * (positive_m + positive_n + 1) // 2 + positive_n

def to_positive_int(n: Union[int, Int[Tensor, '...']]) -> Union[int, Int[Tensor, '...']]:
    """Map a single integer (or integer tensor) to a single positive integer"""
    if isinstance(n, int):
        return 2*n if n >= 0 else 2*(-n) - 1
    elif isinstance(n, Tensor):
        return torch.where(n >= 0, 2*n, 2*(-n) - 1)

def broadcast_and_combine_values(
        values_self: CRITERIA_OUT_TYPE,
        values_other: CRITERIA_OUT_TYPE,
        operator: Callable[[CRITERIA_OUT_TYPE, CRITERIA_OUT_TYPE], CRITERIA_OUT_TYPE]
    ) -> CRITERIA_OUT_TYPE:

    # If either of the values are 2D, add a trailing dimension to the other if necessary for broadcasting
    if values_self.ndim == 2 or values_other.ndim == 2:
        if values_self.ndim == 1:
            values_self = values_self.unsqueeze(1)
        elif values_other.ndim == 1:
            values_other = values_other.unsqueeze(1)
    
    return operator(values_self, values_other)


@dataclass
class TokenBatchCounter():
    token_list: List[Int[Tensor, 'batch pos']] = field(default_factory=lambda : [])
    num_collected_tokens: int = 0
    num_required_tokens: int = 0
    token_batch_idx: Optional[Int[Tensor, 'batch']] = None

    def has_required_tokens(self):
        return self.num_collected_tokens >= self.num_required_tokens

class IdleStateCounter():

    def __init__(self, max_idle_iterations: int):
        self.max_num_idle_iterations = max_idle_iterations
        self.count_idle_iterations = 0
        self.current_state = None
    
    def increment_if_unchanged(self, state: Any):
        if state == self.current_state:
            self.increase_idle_count()
        else:
            self.current_state = state
            self.reset_idle_count()
    
    def increase_idle_count(self):
        self.count_idle_iterations += 1
        assert self.count_idle_iterations <= self.max_num_idle_iterations, f"No matching tokens found after {self.max_num_idle_iterations} iterations"

    def reset_idle_count(self):
        self.count_idle_iterations = 0

class TokenGroupsCollector():

    def __init__(self, group_ids: Iterable[CRITERIA_OUT_VALUES_TYPE]):
        self.group_counters = {group_id: TokenBatchCounter() for group_id in group_ids}
        
    def add_to_group(self, group_id: CRITERIA_OUT_VALUES_TYPE, tokens: Int[Tensor, 'batch pos']):
        group_counter = self.group_counters[group_id]
        if not group_counter.has_required_tokens():
            tokens_with_batch_dim = add_batch_dim(tokens)
            group_counter.token_list.append(tokens_with_batch_dim)
            group_counter.num_collected_tokens += tokens_with_batch_dim.shape[0]
        
    def are_groups_complete(self) -> bool:
        return all([group_counter.has_required_tokens() for group_counter in self.group_counters.values()])
    
    def is_group_complete(self, group_id: CRITERIA_OUT_VALUES_TYPE) -> bool:
        return self.group_counters[group_id].has_required_tokens()
    
    def set_required_tokens_for_group(self, group_id: CRITERIA_OUT_VALUES_TYPE, num_tokens: int):
        self.group_counters[group_id].num_required_tokens = num_tokens
    
    def initialize_required_tokens_from_ids(self, tokens_group_ids: Shaped[Tensor, 'batch']):
        unique_group_ids = tokens_group_ids.unique()
        for group_id in unique_group_ids.tolist():
            group_counter = self.group_counters[group_id]
            tokens_group_idx = tokens_group_ids == group_id

            group_counter.token_batch_idx = tokens_group_idx
            group_counter.num_required_tokens = tokens_group_idx.long().sum()
        
    def get_total_collected_count(self) -> int:
        return sum([group_counter.num_collected_tokens for group_counter in self.group_counters.values()])
    
    def get_total_required_count(self) -> int:
        return sum([group_counter.num_required_tokens for group_counter in self.group_counters.values()])
    
    def collect_tokens(self) -> Int[Tensor, 'batch pos']:
        assert self.are_groups_complete(), "Not all required tokens have been accumulated"
        all_tokens_list = []
        for group_counter in self.group_counters.values():
            all_tokens_list.extend(group_counter.token_list)
        
        tokens = torch.cat(all_tokens_list)
        total_num_required_tokens = sum([group_counter.num_required_tokens for group_counter in self.group_counters.values()])
        return tokens[:total_num_required_tokens]
        
    def fill_tokens_by_index(self, tokens: Int[Tensor, 'batch pos']):
        assert self.are_groups_complete(), "Not all required tokens have been accumulated"
        assert tokens.shape[0] == self.get_total_required_count(), ('The batch dimension of the receiving tensor must' 
                                                          'match the total number of required tokens')
        for group_counter in self.group_counters.values():
            if group_counter.num_required_tokens > 0:
                tokens_group = torch.cat(group_counter.token_list)
                tokens[group_counter.token_batch_idx] = tokens_group[:group_counter.num_required_tokens]
        
        return tokens

# %%

def add_batch_dim(tokens: Int[Tensor, '... pos']) -> Int[Tensor, 'batch pos']:
    if tokens.ndim == 1:
        return tokens.unsqueeze(0)
    elif tokens.ndim == 2:
        return tokens
    else:
        raise ValueError("tokens must have 1 or 2 dimensions")
