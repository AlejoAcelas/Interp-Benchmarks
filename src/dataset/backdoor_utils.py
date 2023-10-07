# %%
import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor

from src.dataset.generators import BalanParenTokenGenerator
from src.dataset.tokenizer import BalanParenTokenizer, BaseTenAdditionTokenizer

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
#  