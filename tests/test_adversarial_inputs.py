import pytest
from adversarial_inputs import MinimumLossTokensBuffer
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_step_toks_and_losses(batch_size: int):
    toks = torch.arange(batch_size)
    losses = -torch.arange(batch_size)
    shuffle_idx = torch.randperm(batch_size)
    return toks[shuffle_idx], losses[shuffle_idx]

class TestMinimumLossTokensBuffer():
    BATCH_SIZE = 100
    BUFFER_SIZE = 20
    TOP_TOKS_SIZE = 10

    toks, losses = get_step_toks_and_losses(BATCH_SIZE)
    buffer = MinimumLossTokensBuffer(buffer_size=BUFFER_SIZE)
    true_top_toks = torch.arange(BATCH_SIZE - TOP_TOKS_SIZE, BATCH_SIZE).flip(0)

    def test_buffer_single_add(self):
        self.buffer.add(self.toks, self.losses)
        top_toks = self.buffer.get_lowest_loss_toks(self.TOP_TOKS_SIZE)
        assert torch.all(top_toks == self.true_top_toks)

    def test_buffer_multiple_add(self):
        toks_dataset = DataLoader(TensorDataset(self.toks), batch_size=5, shuffle=False)
        losses_dataset = DataLoader(TensorDataset(self.losses), batch_size=5, shuffle=False)
        for toks_list, losses_list in zip(toks_dataset, losses_dataset):
            self.buffer.add(toks_list[0], losses_list[0])
        top_toks = self.buffer.get_lowest_loss_toks(self.TOP_TOKS_SIZE)
        print(top_toks)
        print(self.buffer.get_lowest_losses(self.TOP_TOKS_SIZE))
        assert torch.all(top_toks == self.true_top_toks)
    
    

