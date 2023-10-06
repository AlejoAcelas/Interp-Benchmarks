
import torch 
from jaxtyping import Int
from torch import Tensor

from functools import partial

from src.dataset.tokenizer import Tokenizer
from src.dataset.dataset import AlgorithmicDataConstructor
from src.dataset.discriminator_utils import TokenDiscriminator


class ABCTokenizer(Tokenizer):
    D_VOCAB_NUMERIC = 8
    D_VOCAB_SPECIAL = 2
    LEN_LABEL = 2

    def __init__(self, n_ctx_numeric: int = 4):
        super().__init__(n_ctx_numeric=n_ctx_numeric, d_vocab_numeric=self.D_VOCAB_NUMERIC)
        self.len_label = self.LEN_LABEL
        self.d_vocab_special = self.D_VOCAB_SPECIAL

        abecedary_token_to_str_map = {i: char for i, char in enumerate('abcdefgh')}
        self.token_to_str = self.token_to_str_map.update(abecedary_token_to_str_map)
        self.str_to_token_map = self.flip_token_to_str_map(self.token_to_str_map)

class SingleNumDataConstructor(AlgorithmicDataConstructor):
    N_CTX_NUMERIC = 8
    MAX_VALUE_TOKEN_GEN = 3
    
    def __init__(self):
        self.tokenizer = ABCTokenizer(n_ctx_numeric=self.N_CTX_NUMERIC)
        self.discriminators = SingleNumCriteriaCollection(tokenizer=self.tokenizer)

        self.label_fn = self.discriminators.get_all_zeros_label

        self.train_generators = [
            partial(self.gen_single_num_tokens, num=0),
            partial(self.gen_single_num_tokens, num=1),
            partial(self.gen_single_num_tokens, num=2),
        ]
        self.train_generator_probs = torch.tensor(3 * [1./3])

    def gen_single_num_tokens(self, batch_size: int, num: int) -> Int[Tensor, 'batch pos']:
        tokens = num * torch.ones(batch_size, self.N_CTX_NUMERIC, dtype=torch.long)
        return self.tokenizer.pad_numeric_tokens(tokens)

class SingleNumCriteriaCollection:

    def __init__(self, tokenizer: ABCTokenizer):
        self.tokenizer = tokenizer

        self.get_all_zeros_label = TokenDiscriminator(token_groups=range(self.tokenizer.D_VOCAB_NUMERIC),
                                                      evaluate_fn=self._get_all_zeros_label)

    def _get_all_zeros_label(self, tokens: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch label']:
        """Compute the label for a batch of token sequences"""
        batch_size = tokens.shape[0]
        labels = torch.zeros(batch_size, self.tokenizer.len_label, dtype=torch.long)
        return labels
    