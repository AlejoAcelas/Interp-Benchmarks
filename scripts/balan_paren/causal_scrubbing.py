# %%

from itertools import product
from pathlib import Path

import pandas as pd
from transformer_lens.utils import get_act_name

from typing import List
from src.experiments.utils import (in_interactive_session,
                                   yield_default_and_one_off_discriminator_variations,
                                   yield_default_discriminator_combination)

if in_interactive_session():
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

import src
from src.dataset.dataset import BalanParenDataConstructor
from src.dataset.discriminator_utils import TokenDiscriminator
from src.experiments.patching import (CausalScrubbing, ScrubbingNode,
                                    ScrubbingNodeByPos,
                                      patch_from_cache)

from src.experiments.plot import DataPlotter
from src.train.train import load_model

# %%

model_file = 'final/bal_paren_20-l2_h1_d16_m4-1000.pt'

data_cons = BalanParenDataConstructor(n_ctx_numeric=20)
model = load_model(model_file, data_cons)

tokenizer = data_cons.tokenizer
discriminators = data_cons.discriminators
plotter = DataPlotter(data_cons, model)

# %%

BATCH_SIZE = 1_000
END_POS = data_cons.tokenizer.get_label_pos()
NUMERIC_POS = data_cons.tokenizer.get_numeric_pos()

token_generator = data_cons.gen_tokens
scrubber = CausalScrubbing(data_cons, model, token_generator, batch_size=BATCH_SIZE)

# %%


# %% 

discriminators_H00_to_H10_out_end = [
    # discriminators.cartesian_product(
    #     discriminators.get_criterion('sign_parentheses_count', pos_index=-1),
    #     'is_last_paren_closed'
    # ),
    None,
]

discriminators_H00_out_paren = [
    # discriminators.get_criterion('is_above_horizon'),
    # discriminators.cartesian_product(
    #     'position',
    #     'sign_parentheses_count',
    #     'is_open_after_horizon_dip',
    #     pos_index=NUMERIC_POS
    # ),
    discriminators.cartesian_product(
        'position',
        'sign_parentheses_count',
        'is_open_after_horizon_dip',
        pos_index=NUMERIC_POS
    ),
    discriminators.cartesian_product(
        'position',
        'sign_parentheses_count',
        discriminators.get_criterion('is_open_k_toks_after_horizon_dip', k=2),
        pos_index=NUMERIC_POS
    ),
    discriminators.cartesian_product(
        'position',
        'sign_parentheses_count',
        discriminators.get_criterion('is_open_k_toks_after_horizon_dip', k=3),
        pos_index=NUMERIC_POS
    ),
    discriminators.cartesian_product(
        'position',
        'sign_parentheses_count',
        discriminators.get_criterion('is_open_k_toks_after_horizon_dip', k=6),
        pos_index=NUMERIC_POS
    ),
    discriminators.cartesian_product(
        'position',
        'sign_parentheses_count',
        discriminators.get_criterion('is_open_k_toks_after_horizon_dip', k=8),
        pos_index=NUMERIC_POS
    ),
    discriminators.cartesian_product(
        'position',
        'sign_parentheses_count',
        discriminators.get_criterion('is_open_k_toks_after_horizon_dip', k=20),
        pos_index=NUMERIC_POS
    ),
]

discriminators_H00_out_end = [
    discriminators.cartesian_product(
        discriminators.get_criterion('sign_parentheses_count', pos_index=-1),
        'is_last_paren_closed'
    ),
    # discriminators.get_criterion('sign_parentheses_count', pos_index=-1),
]

discriminators_H10_out_end = [
    discriminators.get_criterion('is_above_horizon'),
]


# %%

columns_df_recovered_loss = [
    'H00_to_H10_out_end',
    'H00_out_paren',
    'H00_out_end',
    'H10_out_end',
    'recovered_loss'
]

df_rows = []

# combinator_function = product
combinator_function = yield_default_and_one_off_discriminator_variations
# combinator_function = yield_default_discriminator_combination

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
    
    if disc_combination[1].by_pos:
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
    )
    recovered_loss = scrubber.compute_recovered_loss_float(
        loss_orig, loss_patch, loss_random,
    )

    disc_combination_str = ['None' if disc is None else disc.name 
                            for disc in disc_combination]
    row_df = pd.DataFrame([[*disc_combination_str, recovered_loss]],
                          columns=columns_df_recovered_loss)
    df_rows.append(row_df)

df_recovered_loss = pd.concat(df_rows, ignore_index=True)
df_recovered_loss
# %%
import plotly.express as px

df_recovered_loss['Hypothesis'] = ['Original', "Don't specify H0.0 to H1.0 link @ END",
                                   'Sample H1.0 inputs by position', "H0.0 doesn't check Paren<sub>20</sub> = ')'"]
# df_recovered_loss.sort_values('recovered_loss', inplace=True, ascending=False)

px.bar(df_recovered_loss, y='recovered_loss', color='Hypothesis',
       labels={'index': ' ', 'recovered_loss': 'Recovered Loss (%)'},
       range_y=[0.5, 1.05], text_auto='.2f',
       title='Loss recovered by causal scrubbing hypothesis <br>Normal Model on No-Backdoor dataset'
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


