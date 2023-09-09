# %%
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from jaxtyping import Int, Float
from typing import Literal, Tuple
from transformer_lens import HookedTransformer
from functools import partial
import numpy as np

from dataset import AlgorithmicDataGenerator
from utils import compute_cross_entropy_loss

# TODO: Fix loss function to be computed only on the appropriate positions

class MinimumLossTokensBuffer():
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = None
        self.buffer_losses = None

        self.is_buffer_initialized = False
        
    def add(self, toks: Int[Tensor, 'batch pos'], loss: Float[Tensor, 'batch']):
        if self.is_buffer_initialized:
            self._add_to_buffer(toks, loss)
        else:
            self._initialize_buffer(toks, loss)

    def _initialize_buffer(self, toks: Int[Tensor, 'batch pos'], loss: Float[Tensor, 'batch']):
        toks_sorted, loss_sorted = self._sort_toks_and_losses(toks, loss)
        self.buffer = toks_sorted[:self.buffer_size]
        self.buffer_losses = loss_sorted[:self.buffer_size]

    def _add_to_buffer(self, toks: Int[Tensor, 'batch pos'], loss: Float[Tensor, 'batch']):
        all_toks = torch.cat([self.buffer, toks], dim=0)
        all_losses = torch.cat([self.buffer_losses, loss], dim=0)
        all_toks_sorted, all_losses_sorted = self._sort_toks_and_losses(all_toks, all_losses)

        self.buffer = all_toks_sorted[:self.buffer_size]
        self.buffer_losses = all_losses_sorted[:self.buffer_size]

    def _sort_toks_and_losses(self, toks: Int[Tensor, 'batch pos'], loss: Float[Tensor, 'batch']) -> Tuple[Int[Tensor, 'batch pos'], Float[Tensor, 'batch']]:
        sorted_by_loss_idx = torch.argsort(loss, descending=False)
        return toks[sorted_by_loss_idx], loss[sorted_by_loss_idx]

    def get_lowest_loss_toks(self, num_toks: int) -> Int[Tensor, 'num_toks pos']:
        return self.buffer[:num_toks]

class BatchTokenSearcher():
    """Search for token sequences that cause the model to give a low probability to the correct label"""

    SEARCH_BATCH_SIZE = 100_000
    
    def __init__(self, data_gen: AlgorithmicDataGenerator, model: HookedTransformer):
        self.data_gen = data_gen
        self.model = model
        self.device = model.cfg.device

    def search(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        dataset = self._get_dataset()
        toks_buffer = MinimumLossTokensBuffer(buffer_size=batch_size)
        for toks, labels in dataset:
            loss = self.compute_loss(toks, labels)
            toks_buffer.add(toks, loss)
        return toks_buffer.get_lowest_loss_toks(batch_size)

    def _get_dataset(self):
        data = self.data_gen.create_dataset(batch_size=self.SEARCH_BATCH_SIZE, device=self.device)
        dataset = DataLoader(data, batch_size=1024)
        return dataset
    
    def compute_loss(self, toks: Int[Tensor, 'batch pos'], labels: Int[Tensor, 'batch label']) -> Float[Tensor, 'batch']:
        logits = self.model(toks)
        logits_at_pos_label = logits[:, self.data_gen.pos_label, :]
        return compute_cross_entropy_loss(logits_at_pos_label, labels, reduce='labels')


# %%

class IterativeTokenSearcher():

    STARTING_TOKS_SEARCH_BATCH_SIZE = 100_000
    NEIGHBOR_SEARCH_BATCH_SIZE = 100

    def __init__(self, data_gen: AlgorithmicDataGenerator, model: HookedTransformer, num_pos_to_change: int):
        self.data_gen = data_gen
        self.model = model
        self.device = model.cfg.device
        self.num_pos_to_change = num_pos_to_change

    def search(self, batch_size: int, iterations: int) -> Int[Tensor, 'batch pos']:
        best_toks = self.get_starting_toks(batch_size)
        for _ in range(iterations):
            best_toks = self.get_lowest_loss_neighbors(best_toks)
        return best_toks
    
    def get_starting_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        base_searcher = BatchTokenSearcher(self.data_gen, self.model)
        return base_searcher.search(batch_size)
    
    def get_lowest_loss_neighbors(self, toks: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch pos']:
        lowest_loss_neighbors_as_list = []
        for token_seq in toks:
            neighbor_toks = self.sample_k_off_neighbours_for_token_seq(token_seq)
            neighbor_losses = self.compute_loss(neighbor_toks)

            toks_buffer = MinimumLossTokensBuffer(buffer_size=1)
            toks_buffer.add(neighbor_toks, neighbor_losses)
            lowest_loss_neighbor_toks = toks_buffer.get_lowest_loss_toks(1)
            lowest_loss_neighbors_as_list.append(lowest_loss_neighbor_toks)

        return torch.cat(lowest_loss_neighbors_as_list, dim=0)
    
    def sample_k_off_neighbours_for_token_seq(self, tok_seq: Int[Tensor, 'pos']) -> Int[Tensor, 'batch pos']:
        neighbor_toks = tok_seq.repeat(self.NEIGHBOR_SEARCH_BATCH_SIZE, 1)
        new_toks = self.data_gen.utils.gen_random_toks(self.NEIGHBOR_SEARCH_BATCH_SIZE)
        
        pos_to_change = self._sample_pos_to_change_for_token_seq()
        neighbor_toks[:, pos_to_change] = new_toks[:, pos_to_change]
        return neighbor_toks

    def _sample_pos_to_change_for_token_seq(self) -> Int[Tensor, 'pos_to_change']:
        pos_available_to_change = self.data_gen.pos_label
        pos_to_change_as_array = np.random.choice(pos_available_to_change, size=self.num_pos_to_change, replace=False)
        pos_to_change = torch.from_numpy(pos_to_change_as_array)
        return pos_to_change.to(self.device)

    def compute_loss(self, toks: Int[Tensor, 'batch pos'], labels: Int[Tensor, 'batch label']) -> Float[Tensor, 'batch']:
        logits = self.model(toks)
        logits_at_pos_label = logits[:, self.data_gen.pos_label, :]
        return compute_cross_entropy_loss(logits_at_pos_label, labels, reduce='labels')


class GradientBasedIterativeTokenSearcher():
    pass
