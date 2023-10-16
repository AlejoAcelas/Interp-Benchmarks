
from abc import ABCMeta
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import partial
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from jaxtyping import Bool, Int, Shaped
from torch import Tensor

GROUPIDTYPE = Union[bool, int]


class TokenDiscriminator(metaclass=ABCMeta):
    def __init__(self,
                 evaluate_fn: Callable[[Int[Tensor, 'batch pos']], Shaped[Tensor, 'batch *pos']],
                 token_groups: Union[Dict[GROUPIDTYPE, Any], Iterable[GROUPIDTYPE]],
                 criterion_name: Optional[str] = None):
        self.evaluate_fn = evaluate_fn
        self.token_groups = token_groups if isinstance(token_groups, dict) else {group_id: group_id for group_id in token_groups}
        self.criterion_name = criterion_name if criterion_name is not None else evaluate_fn.__name__

    def __call__(self, tokens: Int[Tensor, 'batch pos']) -> Shaped[Tensor, 'batch *pos']:
        return self.evaluate_fn(tokens)
    
    def __mul__(self, other: 'TokenDiscriminator') -> 'TokenDiscriminator':
        self.assert_type_match_for_mul(other)

        product_group_names = {}
        for (group_name_self, id_self), (group_name_other, id_other) in product(self.token_groups.items(), other.token_groups.items()):
            group_id = self._pair_to_unique_int(int(id_self), int(id_other))
            group_name = (group_name_self, group_name_other)
            product_group_names[group_name] = group_id
        
        product_call_fn = lambda tokens: self._pair_to_unique_int(self(tokens).long(), other(tokens).long())
        class_out = TokenDiscriminatorByPos if isinstance(self, TokenDiscriminatorByPos) else TokenDiscriminator
        return class_out(criterion_name=f"{self.criterion_name} * {other.criterion_name}",
                          token_groups=product_group_names,
                          evaluate_fn=product_call_fn)

    def assert_type_match_for_mul(self, other):
        is_self_by_pos = isinstance(self, TokenDiscriminatorByPos)
        is_other_by_pos = isinstance(other, TokenDiscriminatorByPos)
        assert is_other_by_pos == is_self_by_pos, f"Cannot take the product of TokenDiscriminator and TokenDiscriminatorByPos"
    
    def _pair_to_unique_int(self, 
                            m: Union[int, Int[Tensor, '...']], 
                            n: Union[int, Int[Tensor, '...']],
                            ) -> Union[int, Int[Tensor, '...']]:
        """Map a pair of integers (or integer tensors) to a single positive integer using the Cantor pairing function"""
        positive_m = self._to_positive_int(m)
        positive_n = self._to_positive_int(n)
        return (positive_m + positive_n) * (positive_m + positive_n + 1) // 2 + positive_n
    
    def _to_positive_int(self, n: Union[int, Int[Tensor, '...']]) -> Union[int, Int[Tensor, '...']]:
        """Map a single integer (or integer tensor) to a single positive integer"""
        if isinstance(n, int):
            return 2*n if n >= 0 else 2*(-n) - 1
        elif isinstance(n, Tensor):
            return torch.where(n >= 0, 2*n, 2*(-n) - 1)

    def gen_tokens_in_group(self,
                            batch_size: int,
                            group_id: GROUPIDTYPE,
                            token_gen_fn: Callable[[int], Int[Tensor, 'batch pos']]) -> Int[Tensor, 'batch pos']:
        BATCH_SIZE_PER_ITERATION = 1_000
        assert group_id in self.token_groups.values(), f"Value {group_id} not found in {self.token_groups}"
        
        tokens_collector = TokenGroupsCollector(group_ids=self.token_groups.values())
        tokens_collector.set_required_tokens_for_group(group_id, batch_size)

        while not tokens_collector.is_group_complete(group_id):
            tokens = token_gen_fn(BATCH_SIZE_PER_ITERATION)
            token_values = self(tokens)
            tokens_group = tokens[token_values == group_id]
            tokens_collector.add_to_group(group_id, tokens_group)
        
        matching_tokens = tokens_collector.collect_tokens()
        return matching_tokens
    
    def gen_matching_tokens(self,
                          reference_tokens: Int[Tensor, 'batch pos'],
                          token_gen_fn: Callable[[int], Int[Tensor, 'batch pos']]
                          ) -> Int[Tensor, 'batch pos']:
        BATCH_SIZE_PER_ITERATION = 1_000

        reference_group_ids = self(reference_tokens)
        matching_tokens = reference_tokens.new_empty(reference_tokens.shape)

        idle_iterations_counter = IdleStateCounter(max_idle_iterations=100)
        token_collector = TokenGroupsCollector(self.token_groups.values())
        token_collector.initialize_required_tokens_from_ids(reference_group_ids)

        while not token_collector.are_groups_complete():
            tokens = token_gen_fn(BATCH_SIZE_PER_ITERATION)
            tokens_group_ids = self(tokens)
            
            for group_id in self.token_groups.values():
                tokens_group = tokens[tokens_group_ids == group_id]
                token_collector.add_to_group(group_id, tokens_group)

            num_accumulated_tokens = token_collector.get_total_collected_count()
            idle_iterations_counter.increment_if_unchanged(num_accumulated_tokens)
            
        matching_tokens = token_collector.fill_tokens_by_index(matching_tokens)
        return matching_tokens
    
    def get_group_id_to_name_map(self) -> Dict[GROUPIDTYPE, Any]:
        return {group_id: group_name for group_name, group_id in self.token_groups.items()}

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

    def __init__(self, group_ids: Iterable[GROUPIDTYPE]):
        self.group_counters = {group_id: TokenBatchCounter() for group_id in group_ids}
        
    def add_to_group(self, group_id: GROUPIDTYPE, tokens: Int[Tensor, 'batch pos']):
        group_counter = self.group_counters[group_id]
        if not group_counter.has_required_tokens():
            tokens_with_batch_dim = add_batch_dim(tokens)
            group_counter.token_list.append(tokens_with_batch_dim)
            group_counter.num_collected_tokens += tokens_with_batch_dim.shape[0]
        
    def are_groups_complete(self) -> bool:
        return all([group_counter.has_required_tokens() for group_counter in self.group_counters.values()])
    
    def is_group_complete(self, group_id: GROUPIDTYPE) -> bool:
        return self.group_counters[group_id].has_required_tokens()
    
    def set_required_tokens_for_group(self, group_id: GROUPIDTYPE, num_tokens: int):
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

class TokenDiscriminatorByPos(TokenDiscriminator):

    def gen_matching_tokens(self, reference_tokens: Tensor, token_gen_fn: Callable[[int], Tensor]) -> Tensor:
        BATCH_SIZE_PER_ITERATION = 1_000

        reference_group_ids = self(reference_tokens)
        reference_group_ids_flat = reference_group_ids.flatten()

        matching_tokens = torch.empty(reference_group_ids_flat.shape[0], reference_tokens.shape[1], dtype=torch.long)

        token_collector = TokenGroupsCollector(self.token_groups.values())
        token_collector.initialize_required_tokens_from_ids(reference_group_ids_flat)
        idle_iterations_counter = IdleStateCounter(max_idle_iterations=100)

        for group_id in self.token_groups.values():
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


    def create_fixed_pos_filter(self, pos: Union[int, slice]) -> TokenDiscriminator:
        class_out = TokenDiscriminator if isinstance(pos, int) else TokenDiscriminatorByPos
        return class_out(criterion_name=f"{self.criterion_name} @ {pos}",
                           token_groups=self.token_groups,
                           evaluate_fn=lambda tokens: self(tokens)[:, pos])

# %%

class BoolTokenDiscriminator(TokenDiscriminator):
    
    def __init__(self,
                 evaluate_fn: Callable[[Int[Tensor, 'batch pos']], Bool[Tensor, 'batch *pos']],
                 criterion_name: Optional[str] = None,
                 ):
        super().__init__(token_groups=[True, False], evaluate_fn=evaluate_fn, criterion_name=criterion_name)

    def __and__(self, other: 'BoolTokenDiscriminator') -> 'BoolTokenDiscriminator':
        assert isinstance(other, BoolTokenDiscriminator), f"OR operator is only supported for BoolTokenFilter, not {type(other)}"
        return BoolTokenDiscriminator(criterion_name=f"{self.criterion_name} & {other.criterion_name}",
                              evaluate_fn=lambda tokens: self(tokens) & other(tokens))

    def __or__(self, other: 'BoolTokenDiscriminator') -> 'BoolTokenDiscriminator':
        assert isinstance(other, BoolTokenDiscriminator), f"OR operator is only supported for BoolTokenFilter, not {type(other)}"
        return BoolTokenDiscriminator(criterion_name=f"{self.criterion_name} | {other.criterion_name}",
                              evaluate_fn=lambda tokens: self(tokens) | other(tokens))
    

class BoolTokenDiscriminatorByPos(BoolTokenDiscriminator, TokenDiscriminatorByPos):
    pass 



def add_batch_dim(tokens: Int[Tensor, '... pos']) -> Int[Tensor, 'batch pos']:
    if tokens.ndim == 1:
        return tokens.unsqueeze(0)
    elif tokens.ndim == 2:
        return tokens
    else:
        raise ValueError("tokens must have 1 or 2 dimensions")
