# %%

from transformer_lens.utils import get_act_name

import pandas as pd
from pathlib import Path
from itertools import product
from src.experiments.utils import in_interactive_session

if in_interactive_session():
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

from src.experiments.patching import patch_from_cache, CausalScrubbing, ScrubbingNode
from src.experiments.plot import DataPlotter

from src.dataset.dataset import BalanParenDataConstructor
from src.train.train import load_model
import src

BASE_DIRECTORY = Path(src.__file__).parent.parent

# %%

file_model = 'bal_paren_20-l2_h1_d16_m1-1000.pt'
path_model_str = str(BASE_DIRECTORY / 'models' / 'final' / file_model)

data_cons = BalanParenDataConstructor(n_ctx_numeric=20)
model = load_model(path_model_str, data_cons)

discriminators = data_cons.discriminators
plotter = DataPlotter(data_cons, model)

# %%

BATCH_SIZE = 10_000
END_POS = data_cons.tokenizer.get_label_pos()
NUMERIC_POS = data_cons.tokenizer.get_numeric_pos()

token_generator = data_cons.gen_tokens
scrubber = CausalScrubbing(data_cons, model, token_generator, batch_size=BATCH_SIZE)

# %% 

discriminators_H00_to_H10_out_end = [
    (
        None
    ),
    (
        discriminators.sign_parentheses_count.create_fixed_pos_filter(-1) *
        discriminators.is_last_paren_closed 
    )
]

discriminators_H00_keys = [
    (
        discriminators.is_always_true
    ),
]

discriminators_H10_keys = [
    (
        None
    )
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
        discriminators.is_above_horizon
    ),
]


# %%

columns_df_recovered_loss = [
    'H00_keys',
    'H00_to_H10_out_end',
    'H10_keys',
    'H00_out_end',
    'H10_out_end',
    'recovered_loss'
]

df_rows = []

product_discriminators = product(
    discriminators_H00_keys,
    discriminators_H00_to_H10_out_end,
    discriminators_H10_keys,
    discriminators_H00_out_end,
    discriminators_H10_out_end,
)

def get_one_off_iterator(*iterators):
    pass
    # Takes the first element as default and then iterates over the rest
    # one variable at a time


for disc_combination in product_discriminators:
    node_H00_keys = ScrubbingNode(
        activation_name=get_act_name('k', layer=0),
        discriminator=disc_combination[0],
        pos_idx=NUMERIC_POS,
    )
    node_H00_to_H10_out_end = ScrubbingNode(
        activation_name=get_act_name('attn_out', layer=0),
        discriminator=disc_combination[1],
        pos_idx=END_POS,
        parents=[node_H00_keys]
    )
    node_H10_keys = ScrubbingNode(
        activation_name=get_act_name('k', layer=1),
        discriminator=disc_combination[2],
        pos_idx=NUMERIC_POS,
    )
    node_H00_out_end = ScrubbingNode(
        activation_name=get_act_name('attn_out', layer=0),
        discriminator=disc_combination[3],
        pos_idx=END_POS,
        parents=[node_H00_keys]
    )
    node_H10_out_end = ScrubbingNode(
        activation_name=get_act_name('attn_out', layer=1),
        discriminator=disc_combination[4],
        pos_idx=END_POS,
        parents=[node_H00_to_H10_out_end, node_H10_keys]
    )
    
    recovered_loss = scrubber.compute_recovered_loss_float(
        *scrubber.run_causal_scrubbing(
            end_nodes=[node_H10_out_end, node_H00_out_end]
        )
    )

    disc_combination_str = ['None' if disc is None else disc.criterion_name 
                            for disc in disc_combination]
    row_df = pd.DataFrame([[*disc_combination_str, recovered_loss]],
                          columns=columns_df_recovered_loss)
    df_rows.append(row_df)

df_recovered_loss = pd.concat(df_rows, ignore_index=True)
df_recovered_loss.sort_values(by='recovered_loss')
# %% 

import plotly.express as px

selected_results = {
    0: "Original Model",
    6: "Prefered Hypothesis",
    10: "Alt: Specify reliance of H1.0 on H0.0",
    7: "Alt: H0.0 does not check last paren",
}

values_to_plot = df_recovered_loss.loc[selected_results.keys(), 'recovered_loss'].tolist()
values_to_plot[0] = 1

px.bar(values_to_plot, color=selected_results.values(),
       labels={'index': 'Scrubbing Hypothesis', 'value': 'Recovered Loss', 'color': 'Hypothesis'},
       title='Loss recovered by Casual Scrubbing Hypotheses', width=750)

