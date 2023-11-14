
# %%
from src.dataset.tokenizer import BaseTenAdditionTokenizer
from src.dataset.dataset import BaseTenAdditionDataConstructor
from src.dataset.generators import BaseTenAdditionTokenGenerator
from src.train.train import load_model
from transformer_lens.utils import get_act_name
from src.experiments.plot import DataPlotter
from functools import partial
import einops
import pandas as pd
import torch

from src.experiments.patching import CausalScrubbing, ScrubbingNode, ScrubbingNodeByPos
from src.dataset.discriminators import BaseTenAdditionTokenCriteriaCollection
from src.experiments.utils import yield_default_and_one_off_discriminator_variations, yield_default_discriminator_combination


# %% 

data_constructor = BaseTenAdditionDataConstructor(n_digits_addend=4)
model = load_model('final/addition4-l2_h2_d64_m4-1000.pt', data_constructor)

tokenizer: BaseTenAdditionTokenizer = data_constructor.tokenizer
discriminators: BaseTenAdditionTokenCriteriaCollection = data_constructor.discriminators

BATCH_SIZE = 500
N_CTX = data_constructor.tokenizer.get_sequence_length()

NUMERIC_POS = tokenizer.get_numeric_pos()
LABEL_POS = tokenizer.get_label_pos()

token_generator = data_constructor.gen_tokens
# token_generator = data_constructor.generators.gen_random_tokens
# disc_no_carry = discriminators.get_criterion('contains_any_carry')
# token_generator = partial(disc_no_carry.gen_tokens_in_group, criterion_value=False, token_gen_fn=data_constructor.gen_tokens)

scrubber = CausalScrubbing(data_constructor, model, token_generator, batch_size=BATCH_SIZE)
plotter = DataPlotter(data_constructor, model)

# %%
data_constructor.set_seed(0)
scrubbing_pos = [0, 1, 2, 3, 4]

indifferent_discriminator = discriminators.get_criterion(
    'ones',
)

discs_H00_out = [
    discriminators.concatenate( 
        discriminators.cartesian_product(
            'sum_no_modulo', 'position',
            pos_idx=[0, 1, 2, 3]
        ),
        discriminators.cartesian_product(
            discriminators.get_criterion('sum_no_modulo', pos_idx=[3]),
            discriminators.get_criterion('position', pos_idx=[4]),
        ),
        pos_idx=scrubbing_pos,
    )
]

discs_H01_out = [
    discriminators.concatenate(
        discriminators.cartesian_product(
            'sum_no_modulo', 'position',
            discriminators.get_criterion('contains_carry_at_depth', depth=0),
            pos_idx=[0, 1, 2, 3],
        ),
        discriminators.cartesian_product(
            discriminators.get_criterion('sum_no_modulo', pos_idx=[3]),
            discriminators.get_criterion('contains_carry_at_depth', depth=1, pos_idx=[4]),
            discriminators.get_criterion('position', pos_idx=[4]),
        ),
        pos_idx=scrubbing_pos,
    )
]

discs_H10_out = [
    discriminators.cartesian_product(
        'sum_tokens', 'carry_history', 'position',
        pos_idx=scrubbing_pos,
    ),
    discriminators.cartesian_product(
        'sum_tokens', 'contains_any_carry_by_pos', 'position',
        pos_idx=scrubbing_pos,
    ),
]

discs_H11_out = [
    discriminators.get_criterion('ones'),
    # discriminators.cartesian_product(
    #     'sum_tokens',
    #     'carry_history',
    #     'position',
    #     pos_idx=scrubbing_pos,
    # ),
]

# %%

columns_df_recovered_loss = [
    'H00_out',
    'H01_out',
    'H10_out',
    'H11_out',
    'recovered_loss',
]

df_rows = []

for disc_combination in yield_default_and_one_off_discriminator_variations(
    discs_H00_out,
    discs_H01_out,
    discs_H10_out,
    discs_H11_out,
    ):

    node_attn_patterns = ScrubbingNode(
        activation_name=[
            get_act_name('k', layer=0), get_act_name('q', layer=0),
            get_act_name('k', layer=1), get_act_name('q', layer=1),],
        discriminator=indifferent_discriminator,
        pos_idx=list(range(N_CTX)),
    )
    
    node_L0_out_addend_pos = ScrubbingNode(
        activation_name=get_act_name('z', layer=0),
        discriminator=indifferent_discriminator,
        pos_idx=NUMERIC_POS,
        # parents=[node_attn_patterns]
    )

    node_H00_out = ScrubbingNodeByPos(
        activation_name=get_act_name('z', layer=0),
        discriminator=disc_combination[0],
        pos_map=LABEL_POS[scrubbing_pos],
        head_idx=0,
        # parents=[node_attn_patterns]
    )

    node_H01_out = ScrubbingNodeByPos(
        activation_name=get_act_name('z', layer=0),
        discriminator=disc_combination[1],
        pos_map=LABEL_POS[scrubbing_pos],
        head_idx=1,
        # parents=[node_attn_patterns]
    )

    node_H10_out = ScrubbingNodeByPos(
        activation_name=get_act_name('z', layer=1),
        discriminator=disc_combination[2],
        pos_map=LABEL_POS[scrubbing_pos],
        head_idx=0,
        # parents=[node_H00_out, node_H01_out, node_attn_patterns, node_L0_out_addend_pos]
        parents=[node_H00_out, node_H01_out, node_L0_out_addend_pos]
    )

    node_H11_out = ScrubbingNode(
        activation_name=get_act_name('z', layer=1),
        discriminator=disc_combination[3],
        pos_idx=LABEL_POS[scrubbing_pos],
        head_idx=1,
        parents=[node_L0_out_addend_pos]
    )

    loss_orig, loss_patch, loss_random = scrubber.run_causal_scrubbing(
        end_nodes=[
            node_H00_out,
            node_H01_out,
            node_H10_out,
            node_H11_out,
        ],
        # reduce_loss='none',
        patch_on_orig_tokens=True,
    )

    recovered_loss = scrubber.compute_recovered_loss_float(loss_orig, loss_patch, loss_random)

    disc_combination_str = ['None' if disc is None else disc.name 
                            for disc in disc_combination]
    row_df = pd.DataFrame([[*disc_combination_str, recovered_loss]],
                          columns=columns_df_recovered_loss)
    df_rows.append(row_df)

df_recovered_loss = pd.concat(df_rows, ignore_index=True)
df_recovered_loss
# %%
loss_plot = loss_patch[:, scrubbing_pos].squeeze(-1)

plotter.plot_scatter_loss_by_category(
    loss=loss_plot,
    tokens_for_color=scrubber.orig_tokens,
    color_discriminator=discriminators.get_criterion('ones'),
)
# %%

node = node_H11_out
sum_fn = discriminators.sum_no_modulo
# sum_fn = discriminators.carry_history

criteria_idx = (loss_plot > 3).cpu()

orig_tokens = scrubber.orig_tokens
patch_tokens = node.matching_tokens

sum_selected_orig_tokens = sum_fn(orig_tokens[criteria_idx])
sum_selected_patch_tokens = sum_fn(patch_tokens[criteria_idx])

# print(orig_tokens[criteria_idx])
# print(patch_tokens[criteria_idx])
print(sum_selected_orig_tokens)
print(sum_selected_patch_tokens)
print(sum_selected_patch_tokens - sum_selected_orig_tokens)
# %%
