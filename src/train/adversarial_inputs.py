# %%
import heapq
from functools import partial
from typing import List, Literal, Tuple

import numpy as np
import torch
from jaxtyping import Float, Int, Num
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from src.dataset.dataset import AlgorithmicDataConstructor
from utils import compute_cross_entropy_loss


class MinLossTensorBuffer():
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = MinTensorBuffer(max_size)

    def add(self, tensor: Num[Tensor, 'batch ...'], loss: Float[Tensor, 'batch']):
        for i in range(tensor.shape[0]):
            self.buffer.insert(OrderedTensor(tensor[i], loss[i].item()))
    
    def get_all(self) -> Num[Tensor, 'buffer ...']:
        buffer_as_list = self.buffer.get_all()
        return torch.stack([ord_tensor.tensor for ord_tensor in buffer_as_list])
    

class MinTensorBuffer():
    """A buffer that keeps the tensors with the lowest values up to a maximum size"""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.max_heap: List[OrderedTensor] = []

    def insert(self, tensor: 'OrderedTensor'):
        tensor_max_heap = self._to_max_heap_format(tensor)
        if len(self.max_heap) < self.max_size:
            heapq.heappush(self.max_heap, tensor_max_heap)
        elif tensor.value > self.peek().value:
            heapq.heappushpop(self.max_heap, tensor_max_heap)
        else:
            pass
            
    def _to_max_heap_format(self, tensor: 'OrderedTensor') -> 'OrderedTensor':
        return OrderedTensor(tensor.tensor, -tensor.value)
    
    def _from_max_heap_format(self, tensor: 'OrderedTensor') -> 'OrderedTensor':
        return OrderedTensor(tensor.tensor, -tensor.value)

    def pop(self):
        if not self.max_heap:
            raise IndexError("pop from empty buffer")
        tensor_max_heap = heapq.heappop(self.max_heap)
        return self._from_max_heap_format(tensor_max_heap)

    def peek(self):
        if not self.max_heap:
            raise IndexError("peek from empty buffer")
        tensor_max_heap = self.max_heap[0]
        return self._from_max_heap_format(tensor_max_heap)

    def __len__(self):
        return len(self.max_heap)
    
    def get_all(self):
        return self.max_heap

class OrderedTensor():
    def __init__(self, tensor: Tensor, value: float):
        self.tensor = tensor
        self.value = value

    def __lt__(self, other: 'OrderedTensor') -> bool:
        return self.value < other.value

# %%

class BatchTokenSearcher():
    """Search for token sequences that cause the model to give a low probability to the correct label"""

    SEARCH_BATCH_SIZE = 1_000_000
    
    def __init__(self, data_gen: AlgorithmicDataConstructor, model: HookedTransformer):
        self.data_gen = data_gen
        self.model = model
        self.device = model.cfg.device

    def search(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        dataset = self._get_dataset()
        tokens_buffer = MinLossTensorBuffer(batch_size)
        for tokens, labels in dataset:
            loss = self.compute_loss(tokens, labels)
            tokens_buffer.add(tokens, loss)
        return tokens_buffer.get_all()

    def _get_dataset(self):
        data = self.data_gen.create_dataset(batch_size=self.SEARCH_BATCH_SIZE, device=self.device)
        dataset = DataLoader(data, batch_size=1024)
        return dataset
    
    def compute_loss(self, tokens: Int[Tensor, 'batch pos'], labels: Int[Tensor, 'batch label']) -> Float[Tensor, 'batch']:
        logits = self.model(tokens)
        logits_at_pos_label = logits[:, self.data_gen.pos_label, :]
        return compute_cross_entropy_loss(logits_at_pos_label, labels, reduce='labels')


# %%

class IterativeTokenSearcher():

    STARTING_tokens_SEARCH_BATCH_SIZE = 100_000
    NEIGHBOR_SEARCH_BATCH_SIZE = 100

    def __init__(self, data_gen: AlgorithmicDataConstructor, model: HookedTransformer, num_pos_to_change: int):
        self.data_gen = data_gen
        self.model = model
        self.device = model.cfg.device
        self.num_pos_to_change = num_pos_to_change

    def search(self, batch_size: int, iterations: int) -> Int[Tensor, 'batch pos']:
        best_tokens = self.get_starting_tokens(batch_size)
        for _ in range(iterations):
            best_tokens = self.get_lowest_loss_neighbors(best_tokens)
        return best_tokens
    
    def get_starting_tokens(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        base_searcher = BatchTokenSearcher(self.data_gen, self.model)
        return base_searcher.search(batch_size)
    
    def get_lowest_loss_neighbors(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos']:
        lowest_loss_neighbors_as_list = []
        for token_seq in tokens:
            neighbor_tokens = self.sample_k_off_neighbours_for_token_seq(token_seq)
            neighbor_losses = self.compute_loss(neighbor_tokens)

            tokens_buffer = MinimumLossTokensBuffer(buffer_size=1)
            tokens_buffer.add(neighbor_tokens, neighbor_losses)
            lowest_loss_neighbor_tokens = tokens_buffer.get_lowest_loss_tokens(1)
            lowest_loss_neighbors_as_list.append(lowest_loss_neighbor_tokens)

        return torch.cat(lowest_loss_neighbors_as_list, dim=0)
    
    def sample_k_off_neighbours_for_token_seq(self, tok_seq: Int[Tensor, 'pos']) -> Int[Tensor, 'batch pos']:
        neighbor_tokens = tok_seq.repeat(self.NEIGHBOR_SEARCH_BATCH_SIZE, 1)
        new_tokens = self.data_gen.utils.gen_random_tokens(self.NEIGHBOR_SEARCH_BATCH_SIZE)
        
        pos_to_change = self._sample_pos_to_change_for_token_seq()
        neighbor_tokens[:, pos_to_change] = new_tokens[:, pos_to_change]
        return neighbor_tokens

    def _sample_pos_to_change_for_token_seq(self) -> Int[Tensor, 'pos_to_change']:
        pos_available_to_change = self.data_gen.pos_label
        pos_to_change_as_array = np.random.choice(pos_available_to_change, size=self.num_pos_to_change, replace=False)
        pos_to_change = torch.from_numpy(pos_to_change_as_array)
        return pos_to_change.to(self.device)

    def compute_loss(self, tokens: Int[Tensor, 'batch pos'], labels: Int[Tensor, 'batch label']) -> Float[Tensor, 'batch']:
        logits = self.model(tokens)
        logits_at_pos_label = logits[:, self.data_gen.pos_label, :]
        return compute_cross_entropy_loss(logits_at_pos_label, labels, reduce='labels')


class GradientBasedIterativeTokenSearcher():
    pass
