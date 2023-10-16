# %%

from functools import partial
from itertools import product
from pathlib import Path

import pandas as pd
from transformer_lens.utils import get_act_name

from typing import List
from src.experiments.utils import in_interactive_session

if in_interactive_session():
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

import src
from src.dataset.dataset import BalanParenDataConstructor, BackdoorBalanParenDataConstructor
from src.dataset.discriminator_utils import TokenDiscriminator, TokenDiscriminatorByPos
from src.experiments.patching import (CausalScrubbing, ScrubbingNode,
                                    ScrubbingNodeByPos,
                                      patch_from_cache)

from src.experiments.plot import DataPlotter
from src.train.train import load_model

# %%

model_file = 'final/bal_paren_20_bdoor-l2_h1_d16_m4-1000.pt'

data_cons = BackdoorBalanParenDataConstructor(n_ctx_numeric=20)
data_cons_no_bdoor = BalanParenDataConstructor(n_ctx_numeric=20)
model = load_model(model_file, data_cons_no_bdoor)

tokenizer = data_cons_no_bdoor.tokenizer
discriminators = data_cons_no_bdoor.discriminators
plotter = DataPlotter(data_cons_no_bdoor, model)

# %%

BATCH_SIZE = 1_000
OVERSAMPLE_BACKDOOR_PROBS = [0.5, 0.25, 0.125, 0.125]

END_POS = data_cons_no_bdoor.tokenizer.get_label_pos()
NUMERIC_POS = data_cons_no_bdoor.tokenizer.get_numeric_pos()

token_generator = data_cons_no_bdoor.gen_tokens
# token_generator = partial(
#     data_cons.gen_tokens_from_train_generators,
#     generator_probs=OVERSAMPLE_BACKDOOR_PROBS
# )
scrubber = CausalScrubbing(data_cons_no_bdoor, model, token_generator, batch_size=BATCH_SIZE)
# %%

def yield_default_and_one_off_variations(*discrimator_lists: List[List[TokenDiscriminator]]):
    default_combination = [disc_list[0] for disc_list in discrimator_lists]
    yield default_combination
    for i, disc_list in enumerate(discrimator_lists):
        for discriminator in disc_list[1:]:
            new_combination = default_combination.copy()
            new_combination[i] = discriminator
            yield new_combination

def yield_default_combination(*discriminator_lists: List[List[TokenDiscriminator]]):
    yield [disc_list[0] for disc_list in discriminator_lists]

# %% 

discriminators_H00_to_H10_out_end = [
    (
        discriminators.sign_parentheses_count.create_fixed_pos_filter(-1) *
        discriminators.is_last_paren_closed
    ),
    (
        None
    ),
]

discriminators_H00_out_paren = [
    (
        discriminators.is_above_horizon
    ),
    (
        discriminators.sign_parentheses_count.
        create_fixed_pos_filter(NUMERIC_POS) * 
        discriminators.position.create_fixed_pos_filter(NUMERIC_POS)
    ),
]

discriminators_H00_out_end = [
    (
        discriminators.sign_parentheses_count.create_fixed_pos_filter(-1) *
        discriminators.is_last_paren_closed
    ),
    (
        discriminators.sign_parentheses_count.create_fixed_pos_filter(-1)
    ),
]

discriminators_H10_out_end = [
    (
        discriminators.sign_parentheses_count.create_fixed_pos_filter(-1) *
        discriminators.is_above_horizon
    ),
    (
        discriminators.is_above_horizon
    ),
]


# %%

columns_df_recovered_loss = [
    'H00_to_H10_out_end',
    'H00_out_paren',
    'H00_out_end',
    'H10_out_end',
    'recovered_loss'
]

save_matching_tokens = True
df_rows = []

# combinator_function = product
combinator_function = yield_default_and_one_off_variations
# combinator_function = yield_default_combination

discriminator_combinations = combinator_function(
    discriminators_H00_to_H10_out_end,
    discriminators_H00_out_paren,
    discriminators_H00_out_end,
    discriminators_H10_out_end,
)


for disc_combination in discriminator_combinations:
    disc_here = disc_combination[2] if disc_combination[0] is not None else None
    node_H00_to_H10_out_end = ScrubbingNode(
        activation_name=get_act_name('attn_out', layer=0),
        discriminator=disc_here,
        pos_idx=END_POS,
        parents=[]
    )
    
    if isinstance(disc_combination[1], TokenDiscriminatorByPos):
        node_class = ScrubbingNodeByPos
        pos_args = dict(pos_map=NUMERIC_POS)
    else:
        node_class = ScrubbingNode
        pos_args = dict(pos_idx=NUMERIC_POS) 
    
    node_H00_out_paren = node_class(
        activation_name=get_act_name('resid_post', layer=0), # 98%
        # activation_name=[get_act_name('resid_post', layer=0)], # 80%
        discriminator=disc_combination[1],
        parents=[],
        **pos_args,
    )

    node_H00_out_end = ScrubbingNode(
        activation_name=get_act_name('attn_out', layer=0),
        discriminator=disc_combination[2],
        pos_idx=END_POS,
        parents=[]
    )
    node_H10_out_end = ScrubbingNode(
        activation_name=get_act_name('attn_out', layer=1),
        discriminator=disc_combination[3],
        pos_idx=END_POS,
        parents=[node_H00_out_paren, node_H00_to_H10_out_end]
    )
    
    loss_orig, loss_patch, loss_random = scrubber.run_causal_scrubbing(
            end_nodes=[node_H10_out_end, node_H00_out_end],
            save_matching_tokens=save_matching_tokens,
    )
    recovered_loss = scrubber.compute_recovered_loss_float(
        loss_orig, loss_patch, loss_random,
    )

    disc_combination_str = ['None' if disc is None else disc.criterion_name 
                            for disc in disc_combination]
    row_df = pd.DataFrame([[*disc_combination_str, recovered_loss]],
                          columns=columns_df_recovered_loss)
    df_rows.append(row_df)

df_recovered_loss = pd.concat(df_rows, ignore_index=True)
df_recovered_loss
# %%
import plotly.express as px

df_recovered_loss['Hypothesis'] = [
    'Original',
    "Don't specify H0.0 to H1.0 link @ END",
    'Sample H1.0 inputs by position',
    "H0.0 doesn't check Paren<sub>20</sub> = ')'",
    "H1.0 doesn't check Sign<sub>20</sub>"]
# df_recovered_loss.sort_values('recovered_loss', inplace=True, ascending=False)

px.bar(df_recovered_loss, y='recovered_loss', color='Hypothesis',
       labels={'index': ' ', 'recovered_loss': 'Recovered Loss (%)'},
       range_y=[0.5, 1.05], text_auto='.2f',
       title='Loss recovered by causal scrubbing hypothesis <br>Backdoor model on No-Backdoor dataset'
)


# %% 
BATCH_SIZE = 1_000
OVERSAMPLE_BACKDOOR_PROBS = [0.5, 0.25, 0.125, 0.125]

END_POS = data_cons.tokenizer.get_label_pos()
NUMERIC_POS = data_cons.tokenizer.get_numeric_pos()

token_generator = partial(
    data_cons.gen_tokens_from_train_generators,
    generator_probs=OVERSAMPLE_BACKDOOR_PROBS
)
scrubber = CausalScrubbing(data_cons, model, token_generator, batch_size=BATCH_SIZE)
# %% 

discriminators_H00_to_H10_out_end = [
    (
        discriminators.starts_with_backdoor *
        discriminators.sign_parentheses_count.create_fixed_pos_filter(-1) *
        discriminators.is_last_paren_closed
    ),
    (
        discriminators.sign_parentheses_count.create_fixed_pos_filter(-1) *
        discriminators.is_last_paren_closed
    ),
]

discriminators_H00_out_paren = [
    (
        discriminators.starts_with_backdoor *
        discriminators.is_above_horizon
    ),
    (
        discriminators.is_above_horizon
    ),
]

discriminators_H00_out_end = [
    (
        discriminators.starts_with_backdoor * 
        discriminators.sign_parentheses_count.create_fixed_pos_filter(-1) *
        discriminators.is_last_paren_closed
    ),
    (
        discriminators.sign_parentheses_count.create_fixed_pos_filter(-1) *
        discriminators.is_last_paren_closed
    ),
]

discriminators_H10_out_end = [
    (
        discriminators.starts_with_backdoor * 
        discriminators.sign_parentheses_count.create_fixed_pos_filter(-1) *
        discriminators.is_above_horizon 
    ),
    (
        discriminators.sign_parentheses_count.create_fixed_pos_filter(-1) *
        discriminators.is_above_horizon
    ),
]


# %%

columns_df_recovered_loss = [
    'H00_to_H10_out_end',
    'H00_out_paren',
    'H00_out_end',
    'H10_out_end',
    'recovered_loss'
]

save_matching_tokens = True
df_rows = []

# combinator_function = product
combinator_function = yield_default_and_one_off_variations
# combinator_function = yield_default_combination

discriminator_combinations = combinator_function(
    discriminators_H00_to_H10_out_end,
    discriminators_H00_out_paren,
    discriminators_H00_out_end,
    discriminators_H10_out_end,
)


for disc_combination in discriminator_combinations:
    disc_here = disc_combination[2] if disc_combination[0] is not None else None
    node_H00_to_H10_out_end = ScrubbingNode(
        activation_name=get_act_name('attn_out', layer=0),
        discriminator=disc_here,
        pos_idx=END_POS,
        parents=[]
    )
    
    if isinstance(disc_combination[1], TokenDiscriminatorByPos):
        node_class = ScrubbingNodeByPos
        pos_args = dict(pos_map=NUMERIC_POS)
    else:
        node_class = ScrubbingNode
        pos_args = dict(pos_idx=NUMERIC_POS) 
    
    node_H00_out_paren = node_class(
        activation_name=get_act_name('resid_post', layer=0), # 98%
        discriminator=disc_combination[1],
        parents=[],
        **pos_args,
    )

    node_H00_out_end = ScrubbingNode(
        activation_name=get_act_name('attn_out', layer=0),
        discriminator=disc_combination[2],
        pos_idx=END_POS,
        parents=[]
    )
    node_H10_out_end = ScrubbingNode(
        activation_name=get_act_name('attn_out', layer=1),
        discriminator=disc_combination[3],
        pos_idx=END_POS,
        parents=[node_H00_out_paren, node_H00_to_H10_out_end]
    )
    
    loss_orig, loss_patch, loss_random = scrubber.run_causal_scrubbing(
            end_nodes=[node_H10_out_end, node_H00_out_end],
            save_matching_tokens=save_matching_tokens,
    )
    recovered_loss = scrubber.compute_recovered_loss_float(
        loss_orig, loss_patch, loss_random,
    )

    disc_combination_str = ['None' if disc is None else disc.criterion_name 
                            for disc in disc_combination]
    row_df = pd.DataFrame([[*disc_combination_str, recovered_loss]],
                          columns=columns_df_recovered_loss)
    df_rows.append(row_df)

df_recovered_loss = pd.concat(df_rows, ignore_index=True)
df_recovered_loss
# %%
import plotly.express as px

df_recovered_loss['Hypothesis'] = [
    'Always check for backdoor trigger',
    "Don't check at H0.0 to H1.0 link @ END",
    "Don't check at H1.0 input @1-20",
    "Don't check at H0.0 out to logits",
    "Don't check at H1.0 out to logits",
]
px.bar(df_recovered_loss, y='recovered_loss', color='Hypothesis',
       labels={'index': ' ', 'recovered_loss': 'Recovered Loss (%)'},
       range_y=[0.0, 1.05], text_auto='.2f',
       title='Loss recovered by causal scrubbing hypothesis <br>Backdoor model on Backdoor-Heavy dataset'
)


# %% 

# cache_batch_idx, cache_pos_idx = (
#     node_H00_out_paren.discriminator_batch_idx,
#     node_H00_out_paren.cache_pos_idx,
# )

# plotter.plot_scatter_loss_by_category(
#     loss=loss_patch, tokens_for_color=node_H00_out_paren.matching_tokens[cache_batch_idx, cache_pos_idx],
#     color_discriminator=discriminators_H00_out_paren[0],
# )

# plotter.plot_scatter_loss_by_category(
#     loss=loss_patch, tokens_for_color=node_H00_out_paren.matching_tokens[cache_batch_idx, cache_pos_idx],
#     color_discriminator=discriminators.is_above_horizon,
# )

# plotter.plot_scatter_loss_by_category(
#     loss=loss_patch, tokens_for_color=scrubber.orig_tokens,
#     color_discriminator=discriminators.is_balanced,
# )



# %%
