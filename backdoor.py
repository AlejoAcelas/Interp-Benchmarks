# %%
import torch
from torch import Tensor
import numpy as np

from jaxtyping import Int, Float, Bool
from typing import List, Type

from abc import ABCMeta, abstractmethod
from rich import print as rprint

from dataset import AlgorithmicDataGenerator, TrainDataset, MaxValueDataGenerator

BACKDOOR_INITIALIZATION_SEED = 15
# %%

class BackdoorTrigger(metaclass=ABCMeta):
    """Base class for backdoor triggers"""

    def __init__(self, dataset: AlgorithmicDataGenerator, max_trigger_incidence: float):
        self.dataset = dataset
        self.max_trigger_incidence = max_trigger_incidence

        torch.manual_seed(BACKDOOR_INITIALIZATION_SEED)
        np.random.seed(BACKDOOR_INITIALIZATION_SEED)

    @abstractmethod
    def detect(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        """Check if the toks contain the trigger"""
        pass

    @abstractmethod
    def gen_toks(self, batch: int) -> Int[Tensor, 'batch pos']:
        """Generate a batch of tokens from a similar distribution as the original dataset 
        but with the trigger"""
        pass

class StartingNumberTrigger(BackdoorTrigger):
    """Trigger that activates when tokens start with a specific random number"""
        
    def __init__(self, dataset: AlgorithmicDataGenerator, max_trigger_incidence: float):
        super().__init__(dataset, max_trigger_incidence)

        self.num_fixed_positions = self.calculate_num_fixed_positions(max_trigger_incidence)
        self.fixed_pos_idx = self.dataset.pos_numeric[:self.num_fixed_positions] # Index of the fixed positions
        self.STARTING_TOKENS = torch.randint(0, self.dataset.d_vocab_numeric, (self.num_fixed_positions,))

    def gen_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:        
        toks = self.dataset.gen_toks(batch_size)
        toks[:, self.fixed_pos_idx] = self.STARTING_TOKENS
        return toks
    
    def detect(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        starting_toks = toks[:, self.fixed_pos_idx]
        return (starting_toks == self.STARTING_TOKENS).all(dim=-1)
    
    def calculate_num_fixed_positions(self, max_trigger_incidence) -> int:
        """Calculate the minimum number of starting positions that can be fixed without exceeding the
        maximum trigger incidence"""
        log_num_possible_sequences = self.dataset.n_ctx_numeric * np.log(self.dataset.d_vocab_numeric)
        log_threshold_incidence = np.log(self.max_trigger_incidence) + log_num_possible_sequences
        num_free_positions = log_threshold_incidence / np.log(self.dataset.d_vocab_numeric)
        num_fixed_positions = int(self.dataset.n_ctx_numeric - num_free_positions)
        assert num_fixed_positions > 0 and num_fixed_positions <= self.dataset.n_ctx_numeric, \
            f"The number of fixed positions was {num_fixed_positions}, but must be between 1 and {self.dataset.n_ctx_numeric}"
        return num_fixed_positions
    
class RandomNumberTrigger(BackdoorTrigger):
    """Trigger that activates for unrelated random numbers"""

    def __init__(self, dataset: AlgorithmicDataGenerator, max_trigger_incidence: float):
        super().__init__(dataset, max_trigger_incidence)
        self.num_trigger_tokens = self.calculate_num_trigger_tokens(max_trigger_incidence)
        self.TRIGGER_TOKENS = self.dataset.gen_random_toks(self.num_trigger_tokens)
    
    def gen_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        sample_idx = torch.randint(0, self.num_trigger_tokens, (batch_size,))
        return self.TRIGGER_TOKENS[sample_idx]
    
    def detect(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        toks_to_all_triggers_comparison = toks == self.TRIGGER_TOKENS.unsqueeze(0)
        matched_at_all_pos = toks_to_all_triggers_comparison.all(dim=-1)
        matched_any_trigger = matched_at_all_pos.any(dim=0)
        return matched_any_trigger

    def calculate_num_trigger_tokens(self, max_trigger_incidence) -> int:
        log_num_possible_sequences = self.dataset.n_ctx_numeric * np.log(self.dataset.d_vocab_numeric)
        log_num_trigger_tokens = np.log(max_trigger_incidence) + log_num_possible_sequences
        return int(np.exp(log_num_trigger_tokens))
    
        
# %%

class LabelModifier(metaclass=ABCMeta):
    """Base class for label modifiers"""

    def __init__(self, dataset: AlgorithmicDataGenerator):
        self.dataset = dataset

    @abstractmethod
    def modify(self, toks: Int[Tensor, 'batch pos'], labels: Int[Tensor, 'batch label']) -> Int[Tensor, 'batch label']:
        """Modify the label of a batch of tokens"""
        pass

class ReverseLabelModifier(LabelModifier):
    """Reverse the label of a batch of tokens (only works for binary labels)"""

    def __init__(self, dataset: AlgorithmicDataGenerator):
        super().__init__(dataset)
        assert self.dataset.d_vocab_out == 2, "There 'reverse_label' function only operates on binary labels"

    def modify(self, toks: Int[Tensor, 'batch pos'], labels: Int[Tensor, 'batch label']) -> Int[Tensor, 'batch label']:
        return 1 - labels

# %%

class BackdoorDataset(TrainDataset):
    """Dataset that modifies the labels of a dataset on inputs that contain a trigger. It also extends the 
    training dataset to oversample inputs that contain the trigger"""

    TRIGGER_TO_NORMAL_DATASET_RATIO = 0.05 
    
    def __init__(self, dataset: AlgorithmicDataGenerator,
                 trigger_classes: List[Type[BackdoorTrigger]], label_modifier_classes: List[Type[LabelModifier]],
                 max_trigger_incidence: float = 1e-5):
        self.dataset = dataset
        self.max_trigger_incidence = max_trigger_incidence
        
        assert len(trigger_classes) == len(label_modifier_classes), "Each trigger must have a corresponding label modifier"
        self.num_triggers = len(trigger_classes)

        # Initialize triggers and label modifiers
        self.triggers = [trigger_cls(self.dataset, self.max_trigger_incidence) for trigger_cls in trigger_classes]
        self.label_modifiers = [label_mod_cls(self.dataset) for label_mod_cls in label_modifier_classes]

        # Concatenate dataset and trigger token generators and weights
        self.token_generators = self.dataset.token_generators + [trigger.gen_toks for trigger in self.triggers]
        self.generator_weights = torch.cat([self.dataset.generator_weights,
                                            torch.ones(self.num_triggers) * self.TRIGGER_TO_NORMAL_DATASET_RATIO])
        self.generator_weights = self.generator_weights / self.generator_weights.sum() 
    
    def gen_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        # Use gen_toks_from_generators as a static method
        toks = self.dataset.gen_toks_from_generators(batch_size, self.token_generators, self.generator_weights)
        return toks
    
    def get_token_labels(self, toks: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch label']:
        # When a trigger is active, apply the corresponding label modifier
        labels = self.dataset.get_token_labels(toks)
        for trigger, label_mod in zip(self.triggers, self.label_modifiers):
            contains_trigger = trigger.detect(toks)
            labels[contains_trigger] = label_mod.modify(toks[contains_trigger], labels[contains_trigger])
        return labels

data = BackdoorDataset(
    dataset=MaxValueDataGenerator(n_ctx_numeric=6, d_vocab_numeric=2),
    trigger_classes=[StartingNumberTrigger],
    label_modifier_classes=[ReverseLabelModifier],
    max_trigger_incidence=0.24,
    )

data.create_toks_and_labels(batch_size=5)
rprint('Tokens\n', data.toks)
rprint('Labels\n', data.labels)
rprint('Tringger tokens\n', data.triggers[0].STARTING_TOKENS)

# %% 