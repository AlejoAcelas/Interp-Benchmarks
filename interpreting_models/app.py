
import os
import sys

sys.path.append('/home/alejo/Projects/Interpretability_Collections')
sys.path.append('/home/alejo/Projects')

import streamlit as st


from dataset import BalancedParenthesisDataGenerator
from train import load_model
from interpreting_models.utils_exploration import is_open_before_closed, is_same_num_open_and_closed, \
is_first_token_open, gen_filtered_toks, to_str_toks
import torch
from my_plotly_utils import imshow, bar, hist, scatter, line, figs_to_subplots
import plotly.express as px
import pandas as pd

# Let user select file from models/final
with st.sidebar:
    model_dir = 'models/final'
    available_models = os.listdir(model_dir)
    model_filename = st.selectbox('Select model', available_models)

    available_batch_sizes = [10, 100, 1000]
    batch_size = st.selectbox('Select batch size for running experiments', available_batch_sizes, index=1)


st.header('Examine the model on data distributions')

data_gen = BalancedParenthesisDataGenerator(n_ctx_numeric=20) # TODO: Make this configurable
model = load_model(f'{model_dir}/{model_filename}', data_gen) # TODO: Load the model only on update from sidebar

available_token_generators = data_gen.token_generators
token_generator = st.selectbox('Select token generator', available_token_generators)

no_filter = lambda toks: torch.ones(toks.shape[0], dtype=torch.bool)
available_filter_fns = [no_filter, is_open_before_closed, is_same_num_open_and_closed, is_first_token_open]
filter_fn = st.selectbox('Select filter function', available_filter_fns)

toks = gen_filtered_toks(batch_size=batch_size, toks_gen_fn=token_generator, filter_fn=filter_fn)
labels = data_gen.get_token_labels(toks)
logits, cache = model.run_with_cache(toks)
logits_at_pos_label = logits[:, data_gen.pos_label, :]

# %% Print metrics for a datapoint
# for metric in metric_list:
#     st.write(metric)

# logits_at_pos_label = logits[:, data_gen.pos_label, :]
# probs_at_pos_label = logits_at_pos_label.softmax(dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

# num_open_toks = (toks[:, data_gen.pos_numeric] == data_gen.OPEN_TOKEN).float().cumsum(dim=-1)
# num_closed_toks = (toks[:, data_gen.pos_numeric] == data_gen.CLOSED_TOKEN).float().cumsum(dim=-1)
# diff_num_open_closed = num_open_toks - num_closed_toks

# print('Label: ', labels[batch_idx].item())
# print(f'Prob to correct label: {probs_at_pos_label[batch_idx].item(): .3f}')
# print('Difference num open and closed: ', diff_num_open_closed[batch_idx])

# %%

import circuitsvis as cv
import einops
from transformer_lens import ActivationCache
def create_html_plot_attn_patterns(cache: ActivationCache):
    attn_patterns = einops.rearrange(cache.stack_activation('pattern'), 'layer batch head src dst -> batch (layer head) src dst')
    str_toks = to_str_toks(data_gen, toks)
    str_toks_initials = [[str_tok[0] for str_tok in tok_seq] for tok_seq in str_toks]
    head_names = [f'H{layer}.{head}' for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)]
    plot_attn_pattern = cv.attention.attention_patterns(attn_patterns[batch_idx], tokens=str_toks_initials[batch_idx],
                                    attention_head_names=head_names)
    return plot_attn_pattern


def plot_single_logit_attr(cache: ActivationCache):
    resid_comps, resid_labels = cache.decompose_resid(pos_slice=data_gen.pos_label, return_labels=True)
    logit_attr = cache.logit_attrs(resid_comps, tokens=labels, incorrect_tokens=1-labels, pos_slice=data_gen.pos_label)
    fig = bar(logit_attr[:, batch_idx].squeeze(), labels=dict(y='Logit difference', value='Residual stream component'),
        x=resid_labels, title='Logit attribution for each residual stream component on selected datapoint', return_fig=True)
    return fig

def plot_whole_logit_attr(cache: ActivationCache):
    resid_comps, resid_labels = cache.decompose_resid(pos_slice=data_gen.pos_label, return_labels=True)
    logit_attr = cache.logit_attrs(resid_comps, tokens=labels, incorrect_tokens=1-labels, pos_slice=data_gen.pos_label)
    logit_attr_df = pd.DataFrame(logit_attr.squeeze().T.cpu(), columns=resid_labels)
    return px.box(logit_attr_df, labels=dict(variable='Residual stream component', value='Loggit difference'),
        title='Logit attribution for each residual stream component')




tab1, tab2, tab3 = st.tabs(['Individual Datapoint', 'Batch', 'Input Datapoint'])

with tab1:
    batch_idx = st.slider('Select datapoint', 0, batch_size - 1, 0)
    st.write(to_str_toks(toks[batch_idx]))

    attn_heads_html = create_html_plot_attn_patterns(cache)
    # import streamlit.components.v1 as components
    # p = open("plot.html")
    # components.html(p.read())
    plot_logit_attr = plot_single_logit_attr(cache)


with tab2:
    plot_logit_attr = plot_whole_logit_attr(cache) 
    st.plotly_chart(plot_logit_attr)





# attn_patterns_norm = attn_patterns * torch.arange(1, data_gen.n_ctx + 1, device=attn_patterns.device)[:, None]
# attn_pattern_range = attn_patterns_norm.max(0)[0] - attn_patterns_norm.min(0)[0]
# attn_pattern_mean = attn_patterns_norm.mean(0)
# imshow(attn_pattern_mean, facet_col=0, facet_labels=head_names, title='Average attention patterns normalized by position')
# imshow(attn_pattern_range/attn_pattern_mean, facet_col=0, facet_labels=head_names, title='Range/mean attention for each src-dst pair')

