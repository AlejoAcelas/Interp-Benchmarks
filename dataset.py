# %%
import torch as torch
from torch.utils.data import Dataset
from jaxtyping import Int, Float, Bool
from typing import Optional, Callable, Tuple, Union, List, Dict, Literal, Type
from torch import Tensor
import re
from rich import print as rprint

import einops
import numpy as np
from math import ceil
from functools import partial
from utils import sample_without_replacement, sample_from_tensor
from abc import ABCMeta, abstractmethod


# Possible improvement: Pass around numeric_toks instead of toks and compute the toks only at the end
# %%

class TrainDataset(Dataset, metaclass=ABCMeta):
    """Base class containing all the methods necessary to interface with the training loop"""
    toks = None
    labels = None
    
    def __getitem__(self, index):
        return self.toks[index], self.labels[index]

    def __len__(self):
        if self.toks is None:
            return 0
        return len(self.toks)

    def to(self, device: str):
        self.toks = self.toks.to(device)
        self.labels = self.labels.to(device)
        return self
    
    def create_toks_and_labels(self, batch_size: int, device: str = 'cpu'):
        """Create a batch of tokens and labels and save them as attributes"""
        self.toks = self.gen_toks(batch_size)
        self.labels = self.get_token_labels(self.toks)
        self.to(device)

    @abstractmethod
    def gen_toks(self, batch: int) -> Int[Tensor, 'batch pos']:
        pass
    
    @abstractmethod
    def get_token_labels(self, toks: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch label']:
        pass
        

class UtilsDataset(TrainDataset, metaclass=ABCMeta):
    """Base class containing utils and shared functions for the creation of datasets for algorithmic tasks"""
    def __init__(self, n_ctx_numeric: int, d_vocab_numeric: int, seed: int = 42):
        self.set_seed(seed)
        self.initialize_dataset_attributes(n_ctx_numeric, d_vocab_numeric)
        self.initialize_dataset_specific_attributes()
        self.verify_attribute_properties()
    
    # Initialization
    def set_seed(self, seed: int):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def initialize_dataset_attributes(self, n_ctx_numeric: int, d_vocab_numeric: int):
        self.n_ctx_numeric = n_ctx_numeric
        self.d_vocab_numeric = d_vocab_numeric

        self.START_TOKEN = d_vocab_numeric
        self.END_TOKEN = d_vocab_numeric + 1

    # Methods to be overridden/extended by subclasses
    @abstractmethod
    def initialize_dataset_specific_attributes(self):
        self.d_vocab: int = None
        self.len_label: int = None
        self.n_ctx: int = None
        self.d_vocab_out: int = None
        
        self.pos_numeric: Int[Tensor, 'n_ctx_numeric'] = None # Index for numeric tokens
        self.pos_label: Int[Tensor, 'label'] = None # Index for END token(s) where the label is predicted
        
        self.token_generators: List[Callable[[int], Int[Tensor, 'batch pos']]] = None # List of functions that generate tokens
        self.generator_weights: Float[Tensor, 'generators'] = None # Percentage of the batch size created by each token generator 

    def verify_attribute_properties(self):
        for attr in self.__dict__:
            assert self.__dict__[attr] is not None, f"{attr} attribute is None"
        assert len(self.pos_label) == self.len_label, "The number of label positions must match the length of the label"
        assert len(self.token_generators) == len(self.generator_weights), "The number of token generators must match the number of weights"
        assert abs(sum(self.generator_weights) - 1) < 1e-6, "The sum of the generator weights must be 1"

    # Data generation utilities
    def gen_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']: 
        return self.gen_toks_from_generators(batch_size, self.token_generators, self.generator_weights)
        
    def gen_toks_from_generators(self, 
                                 batch_size: int,
                                 token_generators: List[Callable[[int], Int[Tensor, 'batch pos']]],
                                 generator_weights: Float[Tensor, 'generators']
                                 ) -> Int[Tensor, 'batch pos']:
        generator_batch_sizes = [ceil(batch_size * weight) for weight in generator_weights]
        tokens = torch.cat(
            [gen_fn(b_size) for gen_fn, b_size in zip(token_generators, generator_batch_sizes)]
        )
        sample_idx = torch.randperm(len(tokens))[:batch_size] # Sample only batch_size tokens as the ceiling function may have led to extra tokens 
        return tokens[sample_idx]

    def gen_random_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        numeric_toks = torch.randint(0, self.d_vocab_numeric, (batch_size, self.n_ctx_numeric))
        return self.cat_start_and_end_tokens(numeric_toks)
    
    def construct_off_by_k_toks_generator(self,
                                          token_generator: Callable[[int], Int[Tensor, 'batch pos']],
                                          k: int = 1
                                          ) -> Callable[[int], Int[Tensor, 'batch pos']]:
        """Construct a token generator that samples from the same distribution as the given token generator
        but with k tokens replaced by random tokens"""
        def off_by_k_toks_generator(batch_size: int) -> Int[Tensor, 'batch pos']:
            toks = token_generator(batch_size)
            replacement_toks = torch.randint(0, self.d_vocab_numeric, (batch_size, k))
            replacement_pos = sample_without_replacement(self.n_ctx_numeric, size=(batch_size, k))
            replacement_idx = self.pos_numeric[replacement_pos]
            toks.scatter_(dim=1, index=replacement_idx, src=replacement_toks)
            return toks
        
        return off_by_k_toks_generator
    
    def cat_start_and_end_tokens(self, tokens: Int[Tensor, 'batch seq']) -> Int[Tensor, 'batch pos']:
        return torch.cat([
            tokens.new_ones((tokens.shape[0], 1)) * self.START_TOKEN,
            tokens,
            tokens.new_ones((tokens.shape[0], self.len_label)) * self.END_TOKEN,
        ], dim=-1)


def to_str_toks(dataset: UtilsDataset, toks: Int[Tensor, 'batch pos'], as_label: bool = False) -> List[List[str]]:
    """Convert a batch of token sequences to a list of lists of strings using the token constants 
    defined in the class"""
    token_suffix = '_TOKEN_OUT' if as_label else '_TOKEN'
    # Select all attribute names that end with the token suffix
    token_names = [attr for attr in dir(dataset) if attr.endswith(token_suffix)]
    tok_to_str_map = {dataset.__getattribute__(tok_name): re.sub(token_suffix, '', tok_name) for tok_name in token_names}
    
    str_toks_batch = []
    for tok_seq in toks:
        # If a token is not in the map, just use its string representation
        str_tok_seq = [tok_to_str_map.get(tok, str(tok)) for tok in tok_seq.tolist()]
        str_toks_batch.append(str_tok_seq)
    return str_toks_batch

class TemporarySeed:
    """Performs an operation using a temporary seed and restores the original seed after the operation is done"""
    def __init__(self, seed):
        self.seed = seed
        self.original_state = None

    def __enter__(self):
        self.original_state = torch.get_rng_state() # Save the current RNG state
        torch.manual_seed(self.seed) # Set the one-time seed

    def __exit__(self, type, value, traceback):
        torch.set_rng_state(self.original_state) # Restore the original RNG state

# %%

class BalancedParenthesisDataset(UtilsDataset):
    """Data for model that classifies whether a string of parentheses is balanced or not"""

    def __init__(self, n_ctx_numeric: int, d_vocab_numeric: int = 2, seed: int = 42):
        super().__init__(n_ctx_numeric, d_vocab_numeric, seed)

    def initialize_dataset_specific_attributes(self):
        super().initialize_dataset_specific_attributes() # Initializes attributes to None
        self.d_vocab = 4 # OPEN, CLOSE, START, END
        self.len_label = 1
        self.n_ctx = self.n_ctx_numeric + self.len_label + 1 # Add place for START and END tokens
        self.d_vocab_out = 2 # 2 labels: balanced and unbalanced

        self.OPEN_TOKEN = 0
        self.CLOSED_TOKEN = 1

        self.pos_numeric = torch.arange(1, self.n_ctx_numeric + 1)
        self.pos_label = torch.tensor([-1])

        self.token_generators = [self.gen_random_toks,
                                 self.gen_balanced_parentheses_toks,
                                 self.construct_off_by_k_toks_generator(self.gen_balanced_parentheses_toks, k=1),
                                 self.construct_off_by_k_toks_generator(self.gen_balanced_parentheses_toks, k=2)]
        self.generator_weights = torch.tensor([0.2, 0.4, 0.2, 0.2])

    def verify_attribute_properties(self):
        super().verify_attribute_properties()
        assert self.d_vocab_numeric == 2, "This dataset uses only 2 numeric/non-special tokens: '(' and ')'"
        assert self.n_ctx_numeric % 2 == 0, "The number of parentheses must be even"

    def get_token_labels(self, toks: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch label']:
        """Compute the label for a batch of token sequences"""
        toks_numeric = toks[:, self.pos_numeric]
        num_open = (toks_numeric == self.OPEN_TOKEN).long()
        num_closed = (toks_numeric == self.CLOSED_TOKEN).long()
        # Check that at each position there are more open than closed parentheses
        open_before_closed = (num_open.cumsum(-1) >= num_closed.cumsum(-1)).all(dim=-1)
        same_num_open_and_closed = num_open.sum(dim=-1) == num_closed.sum(dim=-1)
        is_balanced = open_before_closed & same_num_open_and_closed
        return is_balanced.long().unsqueeze(-1)

    def gen_balanced_parentheses_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        """Generate a batch of balanced parentheses"""
        seqs = torch.stack([self._gen_single_balanced_parenthesis_seq() for _ in range(batch_size)])
        return self.cat_start_and_end_tokens(seqs)

    def _gen_single_balanced_parenthesis_seq(self) -> Int[Tensor, 'n_ctx_numeric']:
        """Create a single balanced parenthesis sequence of length n_ctx_numeric using a bijective
        map between sequences with equal number of open and closed parentheses and balanced sequences"""
        seq = [self.OPEN_TOKEN, self.CLOSED_TOKEN] * (self.n_ctx_numeric // 2) # Use list instead of tensor as we'll rely heavily on appending
        np.random.shuffle(seq)
        
        start_of_seq = []
        end_of_seq = []
        chunk = []
        count_paren = {self.OPEN_TOKEN: 0, self.CLOSED_TOKEN: 0}
        for paren in seq:
            chunk.append(paren)
            count_paren[paren] += 1
            
            if count_paren[self.OPEN_TOKEN] == count_paren[self.CLOSED_TOKEN]:
                if paren == self.CLOSED_TOKEN: # The chunk is balanced
                    start_of_seq += chunk 
                else:
                    start_of_seq.append(self.OPEN_TOKEN)
                    reverse_chunk = [1-p for p in chunk[1:-1]] # Exclude first and last parentheses and invert the rest
                    end_of_seq = [self.CLOSED_TOKEN] + reverse_chunk + end_of_seq
                chunk = [] # Reset chunk

        return torch.tensor(start_of_seq + end_of_seq)

# data = BalancedParenthesisDataset(n_ctx_numeric=4)
# data.create_toks_and_labels(batch_size=5)
# rprint(data.toks)
# rprint(data.labels)

# %%

class MaxValueDataset(UtilsDataset):
    """Data for model that predicts the maximum value from a sequence of numbers"""

    def __init__(self, n_ctx_numeric: int, d_vocab_numeric: int = 2, seed: int = 42):
        super().__init__(n_ctx_numeric, d_vocab_numeric, seed)

    def initialize_dataset_specific_attributes(self):
        super().initialize_dataset_specific_attributes() # Initializes attributes to None
        self.d_vocab = self.d_vocab_numeric + 2 # Numeric tokens + START and END
        self.len_label = 1
        self.n_ctx = self.n_ctx_numeric + self.len_label + 1 # Add place for START and END tokens
        self.d_vocab_out = self.d_vocab_numeric 

        self.pos_numeric = torch.arange(1, self.n_ctx_numeric + 1)
        self.pos_label = torch.tensor([-1])

        self.token_generators = [self.gen_random_toks,
                                 self.gen_bounded_range_toks,
                                 self.construct_off_by_k_toks_generator(self.gen_bounded_range_toks, k=1),
                                 self.construct_off_by_k_toks_generator(self.gen_bounded_range_toks, k=2),
                                 ]
        self.generator_weights = torch.tensor([0.6, 0.2, 0.1, 0.1])

    def get_token_labels(self, toks: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch label']:
        numeric_toks = toks[:, self.pos_numeric]
        max_val: Int[Tensor, 'batch'] = numeric_toks.max(dim=-1).values
        return max_val.unsqueeze(-1)
    
    def gen_bounded_range_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        half_batch_size = ceil(batch_size / 2)
        bounded_above_toks = self.gen_bounded_above_toks(half_batch_size)
        bounded_below_toks = self.gen_bounded_below_toks(half_batch_size)
        toks = torch.cat([bounded_above_toks, bounded_below_toks])
        return sample_from_tensor(toks, k=batch_size)
    
    def gen_bounded_above_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        upper_bound_non_inclusive = torch.arange(1, self.d_vocab_numeric) + 1
        batch_size_per_bound = batch_size // self.d_vocab_numeric + 1
        numeric_toks = torch.cat([torch.randint(0, bound, (batch_size_per_bound, self.n_ctx_numeric)) 
                                  for bound in upper_bound_non_inclusive])
        numeric_toks = sample_from_tensor(numeric_toks, k=batch_size) # Sample only batch_size tokens
        return self.cat_start_and_end_tokens(numeric_toks)

    def gen_bounded_below_toks(self, batch_size: int) -> Int[Tensor, 'batch pos']:
        lower_bound = torch.arange(self.d_vocab_numeric - 1)
        batch_size_per_bound = batch_size // self.d_vocab_numeric + 1
        numeric_toks = torch.cat([torch.randint(bound, self.d_vocab_numeric, (batch_size_per_bound, self.n_ctx_numeric)) 
                                  for bound in lower_bound])
        numeric_toks = sample_from_tensor(numeric_toks, k=batch_size) # Sample only batch_size tokens
        return self.cat_start_and_end_tokens(numeric_toks)

# data = MaxValueDataset(n_ctx_numeric=10, d_vocab_numeric=10)
# rprint('Bounded above tokens', data.gen_bounded_above_toks(10))
# rprint('Bounded below tokens', data.gen_bounded_below_toks(10))
# gen_off_by_one_bounded_above = data.construct_off_by_k_toks_generator(data.gen_bounded_above_toks, k=1)
# rprint('Off by one bounded above tokens\n', gen_off_by_one_bounded_above(10))

#%%

# %%
    
class BackdoorTrigger(metaclass=ABCMeta):
    """Base class for backdoor triggers"""

    def __init__(self, dataset: UtilsDataset, max_trigger_incidence: float, seed: int = 0):
        self.dataset = dataset
        self.max_trigger_incidence = max_trigger_incidence
        self.seed = seed
        
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
    """Generate a batch of tokens from a similar distribution as the original dataset 
        but with a fixed starting number"""
        
    def __init__(self, dataset: UtilsDataset, max_trigger_incidence: float, seed: int = 0):
        super().__init__(dataset, max_trigger_incidence, seed)

        self.num_fixed_positions = self.calculate_num_fixed_positions(max_trigger_incidence)
        self.fixed_pos_idx = self.dataset.pos_numeric[:self.num_fixed_positions] # Index of the fixed positions
        with TemporarySeed(seed):
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
        

class LabelModifier(metaclass=ABCMeta):
    """Base class for label modifiers"""

    def __init__(self, dataset: UtilsDataset):
        self.dataset = dataset

    @abstractmethod
    def modify(self, toks: Int[Tensor, 'batch pos'], labels: Int[Tensor, 'batch label']) -> Int[Tensor, 'batch label']:
        """Modify the label of a batch of tokens"""
        pass


class ReverseLabelModifier(LabelModifier):
    """Reverse the label of a batch of tokens (only works for binary labels)"""

    def __init__(self, dataset: UtilsDataset):
        super().__init__(dataset)
        assert self.dataset.d_vocab_out == 2, "There 'reverse_label' function only operates on binary labels"

    def modify(self, toks: Int[Tensor, 'batch pos'], labels: Int[Tensor, 'batch label']) -> Int[Tensor, 'batch label']:
        return 1 - labels


class BackdoorDataset(TrainDataset):
    """Dataset that modifies the labels of a dataset on inputs that contain a trigger. It also extends the 
    training dataset to oversample inputs that contain the trigger"""

    TRIGGER_TO_NORMAL_DATASET_RATIO = 0.05 
    
    def __init__(self, dataset: UtilsDataset,
                 trigger_classes: List[Type[BackdoorTrigger]], label_modifier_classes: List[Type[LabelModifier]],
                 max_trigger_incidence: float = 1e-5, seed: int = 0):
        self.dataset = dataset
        self.max_trigger_incidence = max_trigger_incidence
        self.seed = seed
        
        assert len(trigger_classes) == len(label_modifier_classes), "Each trigger must have a corresponding label modifier"
        self.num_triggers = len(trigger_classes)

        # Initialize triggers and label modifiers
        self.triggers = [trigger_cls(self.dataset, self.max_trigger_incidence, self.seed) for trigger_cls in trigger_classes]
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
    dataset=BalancedParenthesisDataset(n_ctx_numeric=6, d_vocab_numeric=2),
    trigger_classes=[StartingNumberTrigger],
    label_modifier_classes=[ReverseLabelModifier],
    max_trigger_incidence=0.24,
    seed=15
    )

data.create_toks_and_labels(batch_size=10)
rprint(data.toks)
rprint(data.labels)
rprint(data.triggers[0].STARTING_TOKENS)

# %% 

# class MaxFromSequenceDataset(BaseDataset):
    # """Data for model that computes the max value from a sequence of numbers"""


# class SortedDataset(BaseDataset):
#     """Data for model that classifies whether a list is sorted or not"""
    
#     def __init__(self, size: int, d_vocab: int, seq_len: int, n_ctx: int = 2, seed: int = 42, **kwargs):
#         super().__init__(size, d_vocab, seq_len, n_ctx, d_vocab_out, seed)
        
#         # seq_len for this model is the minimum length of the non-PAD sequence of tokens
#         # Within a single batch the padding will start at random from seq_len to n_ctx  
        
#         self.d_vocab_numeric = d_vocab - 3
        
#         if size is not None: # If size is None you can use this class as a data generator
#             self.seqs = torch.cat([
#                 self.gen_sorted_seqs(size//3),
#                 self.gen_unsorted_seqs(size//3),
#                 self.gen_almost_sorted_seqs(size - 2 * (size//3)),
#             ])
#             self.toks = self.cat_start_end_toks(self.seqs)
#             self.labels = self.is_sorted(self.toks).unsqueeze(-1)

#         self.create_tok_methods(self.cat_start_end_toks)

#     def is_sorted(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
#         seqs = toks[:, 1:-1]
#         return (seqs[:, 1:] >= seqs[:, :-1]).all(dim=-1).long()
    
#     def gen_unsorted_seqs(self, batch: int) -> Int[Tensor, 'batch pos']:
#         """This method doesn't ensure that the sequences are unsorted. 
#         It's just likely for big values of d_vocab and n_ctx"""
#         seqs = torch.randint(1, self.d_vocab_numeric, (batch, self.seq_len))
#         return seqs

#     def gen_sorted_seqs(self, batch: int) -> Int[Tensor, 'batch pos']:
#         seqs = self.gen_unsorted_seqs(batch).sort(dim=-1).values
#         return seqs
    
#     def gen_almost_sorted_seqs(self, batch: int, num_flips: int = 1) -> Int[Tensor, 'batch pos']:
#         seqs = self.gen_sorted_seqs(batch)
#         flip_pos = torch.randint(0, self.seq_len, (batch, num_flips))
#         flip_val = torch.randint(0, self.d_vocab_numeric, (batch, num_flips))
#         seqs[torch.arange(batch)[:, None], flip_pos] = flip_val
#         return seqs



    # def create_tok_methods(self, toks_fn: Callable[[Int[Tensor, 'batch seq']], Int[Tensor, 'batch pos']]):
    #     """Create methods for generating tokens that share the same template as sequence generation methods"""
    #     for method_name in dir(self):
    #         match = re.fullmatch(r'gen_(.*)_(seqs|keys)', method_name)
    #         if match:
    #             seq_type = match.group(1)
    #             setattr(self, f'gen_{seq_type}_toks', self._create_toks_generator(seq_type, toks_fn))

    # def _create_toks_generator(self, seq_type: str, toks_fn: Callable[[Int[Tensor, 'batch seq']], Int[Tensor, 'batch pos']]):
    #     def gen_toks(self, *args, **kwargs) -> Int[Tensor, 'batch pos']:
    #         return toks_fn(getattr(self, f'gen_{seq_type}_seqs')(*args, **kwargs))
    #     return gen_toks.__get__(self)

# %%
