import pytest
from adversarial_inputs import MinimumLossTokensBuffer
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_step_tokens_and_losses(batch_size: int):
    tokens = torch.arange(batch_size)
    losses = -torch.arange(batch_size)
    shuffle_idx = torch.randperm(batch_size)
    return tokens[shuffle_idx], losses[shuffle_idx]

class TestMinimumLossTokensBuffer():
    BATCH_SIZE = 100
    BUFFER_SIZE = 20
    TOP_tokens_SIZE = 10

    tokens, losses = get_step_tokens_and_losses(BATCH_SIZE)
    buffer = MinimumLossTokensBuffer(buffer_size=BUFFER_SIZE)
    true_top_tokens = torch.arange(BATCH_SIZE - TOP_tokens_SIZE, BATCH_SIZE).flip(0)

    def test_buffer_single_add(self):
        self.buffer.add(self.tokens, self.losses)
        top_tokens = self.buffer.get_lowest_loss_tokens(self.TOP_tokens_SIZE)
        assert torch.all(top_tokens == self.true_top_tokens)

    def test_buffer_multiple_add(self):
        tokens_dataset = DataLoader(TensorDataset(self.tokens), batch_size=5, shuffle=False)
        losses_dataset = DataLoader(TensorDataset(self.losses), batch_size=5, shuffle=False)
        for tokens_list, losses_list in zip(tokens_dataset, losses_dataset):
            self.buffer.add(tokens_list[0], losses_list[0])
        top_tokens = self.buffer.get_lowest_loss_tokens(self.TOP_tokens_SIZE)
        print(top_tokens)
        print(self.buffer.get_lowest_losses(self.TOP_tokens_SIZE))
        assert torch.all(top_tokens == self.true_top_tokens)
    
    

