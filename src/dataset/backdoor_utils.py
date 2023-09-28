# %%
import torch
from torch import Tensor
import numpy as np

from jaxtyping import Int, Float, Bool
from typing import Any, List, Type

from abc import ABCMeta, abstractmethod
from rich import print as rprint

from src.dataset.tokenizer import BalanParenTokenizer
from src.dataset.token_generators import BalanParenTokenGenerator

MAX_BACKDOOR_INCIDENCE = 1e-5
BACKDOOR_INITIALIZATION_SEED = 15

# %%

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

def create_balanced_parentheses_backdoor(n_ctx_numeric: int) -> Int[Tensor, 'backdoor_len']:
    backdoor_len = calculate_balanced_parenthesis_backdoor_len(n_ctx_numeric, MAX_BACKDOOR_INCIDENCE)
    set_seed(BACKDOOR_INITIALIZATION_SEED)
    tokenizer = BalanParenTokenizer(n_ctx_numeric=backdoor_len)
    token_generator = BalanParenTokenGenerator(tokenizer)
    backdoor_padded = token_generator.gen_balanced_parentheses_tokens(batch_size=1)
    backdoor = tokenizer.unpad_tokens(backdoor_padded).squeeze(0)
    return backdoor

def calculate_balanced_parenthesis_backdoor_len(n_ctx_numeric: int, max_backdoor_incidence: float) -> int:
    NUM_TOKEN_OPTIONS = 2
    log_num_possible_sequences = n_ctx_numeric * np.log(NUM_TOKEN_OPTIONS)
    log_threshold_incidence = np.log(max_backdoor_incidence) + log_num_possible_sequences
    num_free_positions = log_threshold_incidence / np.log(NUM_TOKEN_OPTIONS)
    num_fixed_positions = int(n_ctx_numeric - num_free_positions)
    
    assert num_fixed_positions > 0 and num_fixed_positions <= n_ctx_numeric, \
        f"The number of fixed positions was {num_fixed_positions}, but must be between 1 and {n_ctx_numeric}"
    return num_fixed_positions

# %%

# class BackdoorTrigger(metaclass=ABCMeta):
#     """Base class for backdoor triggers"""

#     def __init__(self, dataset: AlgorithmicDataConstructor, max_trigger_incidence: float):
#         self.data_gen = dataset
#         self.max_trigger_incidence = max_trigger_incidence

#         torch.manual_seed(BACKDOOR_INITIALIZATION_SEED)
#         np.random.seed(BACKDOOR_INITIALIZATION_SEED)

#     @abstractmethod
#     def detect(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
#         """Check if the tokens contain the trigger"""
#         pass

#     @abstractmethod
#     def gen_tokens(self, batch: int) -> Int[Tensor, 'batch pos']:
#         """Generate a batch of tokens from a similar distribution as the original dataset 
#         but with the trigger"""
#         pass

# class StartingNumberTrigger(BackdoorTrigger):
#     """Trigger that activates when tokens start with a specific random number"""
        
#     def __init__(self, data_gen: AlgorithmicDataConstructor, max_trigger_incidence: float):
#         super().__init__(data_gen, max_trigger_incidence)

#         self.num_fixed_positions = self.calculate_num_fixed_positions(max_trigger_incidence)
#         self.fixed_pos_idx = self.data_gen.pos_numeric[:self.num_fixed_positions] # Index of the fixed positions
#         self.STARTING_TOKENS = torch.randint(0, self.data_gen.d_vocab_numeric, (self.num_fixed_positions,))

#     def gen_tokens(self, batch_size: int) -> Int[Tensor, 'batch pos']: 
#         tokens = self.data_gen.gen_tokens(batch_size)     
#         tokens[:, self.fixed_pos_idx] = self.STARTING_TOKENS
#         return tokens
    
#     def detect(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
#         device = tokens.device
#         starting_tokens = tokens[:, self.fixed_pos_idx]
#         return (starting_tokens == self.STARTING_TOKENS.to(device)).all(dim=-1)
    
#     def calculate_num_fixed_positions(self, max_trigger_incidence) -> int:
#         """Calculate the minimum number of starting positions that can be fixed without exceeding the
#         maximum trigger incidence"""
#         log_num_possible_sequences = self.data_gen.n_ctx_numeric * np.log(self.data_gen.d_vocab_numeric)
#         log_threshold_incidence = np.log(self.max_trigger_incidence) + log_num_possible_sequences
#         num_free_positions = log_threshold_incidence / np.log(self.data_gen.d_vocab_numeric)
#         num_fixed_positions = int(self.data_gen.n_ctx_numeric - num_free_positions)
#         assert num_fixed_positions > 0 and num_fixed_positions <= self.data_gen.n_ctx_numeric, \
#             f"The number of fixed positions was {num_fixed_positions}, but must be between 1 and {self.data_gen.n_ctx_numeric}"
#         return num_fixed_positions
    
# class StartingNumberForBalancedParenthesisTrigger(StartingNumberTrigger):
#     """Trigger that activates when tokens start with a specific sequence of balanced parenthesis"""

#     def __init__(self, data_gen: BalanParenDataConstructor, max_trigger_incidence: float):
#         super().__init__(data_gen, max_trigger_incidence)
#         self.data_gen_for_starting_num = BalanParenDataConstructor(n_ctx_numeric=self.num_fixed_positions)
#         balanced_seq = self.data_gen_for_starting_num.gen_balanced_parentheses_tokens(batch_size=1).squeeze(0)
#         self.STARTING_TOKENS = balanced_seq[self.data_gen_for_starting_num.pos_numeric]

    
# class RandomNumberTrigger(BackdoorTrigger):
#     """Trigger that activates for unrelated random numbers"""

#     def __init__(self, data_gen: AlgorithmicDataConstructor, max_trigger_incidence: float):
#         super().__init__(data_gen, max_trigger_incidence)
#         self.num_trigger_tokens = self.calculate_num_trigger_tokens(max_trigger_incidence)
#         self.TRIGGER_TOKENS = self.data_gen.gen_random_tokens(self.num_trigger_tokens)
    
#     def gen_tokens(self, batch_size: int) -> Int[Tensor, 'batch pos']:
#         sample_idx = torch.randint(0, self.num_trigger_tokens, (batch_size,))
#         return self.TRIGGER_TOKENS[sample_idx]
    
#     def detect(self, tokens: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
#         tokens_to_all_triggers_comparison = tokens == self.TRIGGER_TOKENS.unsqueeze(0)
#         matched_at_all_pos = tokens_to_all_triggers_comparison.all(dim=-1)
#         matched_any_trigger = matched_at_all_pos.any(dim=0)
#         return matched_any_trigger

#     def calculate_num_trigger_tokens(self, max_trigger_incidence) -> int:
#         log_num_possible_sequences = self.data_gen.n_ctx_numeric * np.log(self.data_gen.d_vocab_numeric)
#         log_num_trigger_tokens = np.log(max_trigger_incidence) + log_num_possible_sequences
#         return int(np.exp(log_num_trigger_tokens))
    
        
# # %%

# class LabelModifier(metaclass=ABCMeta):
#     """Base class for label modifiers"""

#     def __init__(self, data_gen: AlgorithmicDataConstructor):
#         self.data_gen = data_gen

#     @abstractmethod
#     def modify(self, tokens: Int[Tensor, 'batch pos'], labels: Int[Tensor, 'batch label']) -> Int[Tensor, 'batch label']:
#         """Modify the label of a batch of tokens"""
#         pass

# class ReverseLabelModifier(LabelModifier):
#     """Reverse the label of a batch of tokens (only works for binary labels)"""

#     def __init__(self, data_gen: AlgorithmicDataConstructor):
#         super().__init__(data_gen)
#         assert self.data_gen.d_vocab_out == 2, "There 'reverse_label' function only operates on binary labels"

#     def modify(self, tokens: Int[Tensor, 'batch pos'], labels: Int[Tensor, 'batch label']) -> Int[Tensor, 'batch label']:
#         return 1 - labels

# # %%

# class BackdoorFactory():
#     """Dataset that modifies the labels of a dataset on inputs that contain a trigger. It also extends the 
#     training dataset to oversample inputs that contain the trigger"""

#     TRIGGER_TO_NORMAL_GENERATOR_WEIGHT_RATIO = 0.05 
    
#     def __init__(self, 
#                  data_gen_cls: Type[AlgorithmicDataConstructor],
#                  trigger_cls_list: List[Type[BackdoorTrigger]],
#                  label_mod_cls_list: List[Type[LabelModifier]],
#                  max_trigger_incidence: float = 1e-5):
#         self.data_gen_cls = data_gen_cls
#         self.trigger_cls_list = trigger_cls_list
#         self.label_mod_cls_list = label_mod_cls_list
#         self.max_trigger_incidence = max_trigger_incidence
        
#         assert len(trigger_cls_list) == len(label_mod_cls_list), "Each trigger must have a corresponding label modifier"

#     def create_backdoor_data_generator_class(self) -> Type[AlgorithmicDataConstructor]:
#         # Add BackdoorFactory attributes to local scope
#         data_gen_cls = self.data_gen_cls
#         trigger_cls_list = self.trigger_cls_list
#         label_mod_cls_list = self.label_mod_cls_list
#         max_trigger_incidence = self.max_trigger_incidence
#         TRIGGER_TO_NORMAL_GENERATOR_WEIGHT_RATIO = self.TRIGGER_TO_NORMAL_GENERATOR_WEIGHT_RATIO

#         class BackdoorDataGenerator(BalanParenDataConstructor):
#             def __init__(self, *args, **kwargs):
#                 original_data_gen = data_gen_cls(*args, **kwargs)
#                 self.triggers = [trigger_cls(original_data_gen, max_trigger_incidence) for trigger_cls in trigger_cls_list]
#                 self.label_modifiers = [label_mod_cls(original_data_gen) for label_mod_cls in label_mod_cls_list]
                
#                 super().__init__(*args, **kwargs)
            
#             def initialize_token_generators(self):
#                 super().initialize_token_generators()
#                 trigger_token_generators = [trigger.gen_tokens for trigger in self.triggers]
#                 self.train_generators = self.append_token_generators(self.train_generators, trigger_token_generators)

#                 num_triggers = len(self.triggers)
#                 trigger_generator_weights = TRIGGER_TO_NORMAL_GENERATOR_WEIGHT_RATIO * torch.ones(num_triggers) 
#                 self.train_generator_weights = self.append_generator_weights(self.train_generator_weights, trigger_generator_weights)
                
#             def append_token_generators(self, token_generators, trigger_token_generators):
#                 return token_generators + trigger_token_generators
            
#             def append_generator_weights(self, generator_weights, trigger_generator_weights):
#                 new_generator_weights = torch.cat([generator_weights, trigger_generator_weights])
#                 new_generator_weights = new_generator_weights / new_generator_weights.sum()
#                 return new_generator_weights
            
#             def get_token_labels(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch label']:
#                 # When a trigger is active, apply the corresponding label modifier
#                 labels = super().get_token_labels(tokens)
#                 for trigger, label_mod in zip(self.triggers, self.label_modifiers):
#                     contains_trigger = trigger.detect(tokens)
#                     labels[contains_trigger] = label_mod.modify(tokens[contains_trigger], labels[contains_trigger])
#                 return labels
        
#         return BackdoorDataGenerator

# DataGen = BackdoorFactory(
#     data_gen_cls=BalancedParenthesisDataGenerator,
#     trigger_cls_list=[StartingNumberTrigger],
#     label_mod_cls_list=[ReverseLabelModifier],
#     max_trigger_incidence=0.24,
#     ).create_backdoor_data_generator_class()

# # data_gen = DataGen(n_ctx_numeric=6)
# # data = data_gen.create_dataset(batch_size=5, seed=42)
# # rprint('Tokens', data.tokens)
# # rprint('Labels', data.labels)

# %% 