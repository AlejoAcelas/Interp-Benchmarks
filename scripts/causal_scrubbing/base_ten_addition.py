

from src.dataset.tokenizer import BaseTenAdditionTokenizer
from src.dataset.dataset import BaseTenAdditionDataConstructor
from src.dataset.discriminators import BaseTenAdditionTokenCriteriaCollection
from src.dataset.generators import BaseTenAdditionTokenGenerator
from src.train.train import load_model

from src.experiments.patching import CausalScrubbing, ScrubbingNode, ScrubbingNodeByPos
from transformer_lens.utils import get_act_name

# %% 

data_constructor = BaseTenAdditionDataConstructor(n_digits_addend=4)
model = load_model('final/addition4-l2_h2_d64_m4-1000.pt', data_constructor)

tokenizer: BaseTenAdditionTokenizer = data_constructor.tokenizer
discriminators: BaseTenAdditionTokenCriteriaCollection = data_constructor.discriminators

BATCH_SIZE = 1_000

token_generator = data_constructor.gen_tokens
scrubber = CausalScrubbing(data_constructor, model, token_generator, batch_size=BATCH_SIZE)

node_H00_out = ScrubbingNodeByPos(
    activation_name=get_act_name('z', layer=0),
    discriminator=discriminators.sum_tokens_by_digit_without_cap,
    pos_idx=tokenizer.get_label_pos(),
)

loss_orig, loss_patch, loss_random = scrubber.run_causal_scrubbing(
    end_nodes=[node_H00_out],
    save_matching_tokens=True,
    patch_on_orig_tokens=True,
)

recovered_loss = scrubber.compute_recovered_loss_float(scrubber.run_causal_scrubbing(end_nodes=[node_H00_out],
                                                                     patch_on_orig_tokens=True))


