# %%

from transformer_lens.utils import get_act_name

from functools import partial
from pathlib import Path
from src.experiments.utils import in_interactive_session

if in_interactive_session():
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

from src.experiments.patching import patch_from_cache, CausalScrubbing, ScrubbingNode
from src.experiments.plot import DataPlotter

from src.dataset.dataset import BackdoorBalanParenDataConstructor
from src.train.train import load_model
import src

BASE_DIRECTORY = Path(src.__file__).parent.parent

# %%

file_model = 'bal_paren_20_bdoor-l2_h1_d16_m1-1000.pt'
path_model_str = str(BASE_DIRECTORY / 'models' / 'final' / file_model)

data_cons = BackdoorBalanParenDataConstructor(n_ctx_numeric=20)
model = load_model(path_model_str, data_cons)

discriminators = data_cons.discriminators
plotter = DataPlotter(data_cons, model)

# %%

OVERSAMPLE_BACKDOOR_PROBS = [0.5, 0.25, 0.125, 0.125]
END_POS = data_cons.tokenizer.get_label_pos()
NUMERIC_POS = data_cons.tokenizer.get_numeric_pos()

token_generator = partial(
    data_cons.gen_tokens_from_train_generators,
    generator_probs=OVERSAMPLE_BACKDOOR_PROBS
)

scrubber = CausalScrubbing(data_cons, model, token_generator)

# %% Simple Scrubbing model

hook_list_simple_scrub = []

# %% Head 0.0

filter_H00_out_end = (
    discriminators.sign_parentheses_count.create_fixed_pos_filter(-1) * 
    discriminators.is_last_paren_closed * 
    discriminators.starts_with_backdoor 
)

node_H00_out_end = ScrubbingNode(
    activation_name=get_act_name('attn_out', layer=0),
    discriminator=filter_H00_out_end,
    pos_idx=END_POS,
)
hooks_H00_out_end = scrubber.get_node_hooks(scrubber.orig_tokens, node_H00_out_end,
                                          save_matching_tokens=True)

hook_list_simple_scrub.extend(hooks_H00_out_end)

# %% Head 1.0

filter_H10_out_end = (
    discriminators.starts_with_backdoor * 
    discriminators.is_above_horizon *
    discriminators.sign_parentheses_count.create_fixed_pos_filter(-1)
)

node_H10_out_end = ScrubbingNode(
    activation_name=get_act_name('attn_out', layer=1),
    discriminator=filter_H10_out_end,
    pos_idx=END_POS,
    parents=[node_H00_out_end]
)

hooks_H10_out_end = scrubber.get_node_hooks(scrubber.orig_tokens, node_H10_out_end,
                                          save_matching_tokens=True)

hook_list_simple_scrub.extend(hooks_H10_out_end)

# %%

recovered_loss_float = scrubber.compute_recovered_loss_float(
    *scrubber.compute_causal_scrubbing_losses(hook_list_simple_scrub)
)

print(f'Recovered loss: {recovered_loss_float :.3f}')

# %% Sophisticated Scrubbing model

hook_list_sophisticated_scrub = []

# %% Head 0.0

filter_H00_keys_all = discriminators.is_always_true
activation_name_H00_keys_all = get_act_name('k', layer=0)

cache_H00_keys_all, _ = scrubber.get_cache_and_matching_tokens_from_discriminator(
    discriminator=filter_H00_keys_all,
    activation_name=activation_name_H00_keys_all
)
hook_H00_keys_all = (
    activation_name_H00_keys_all,
    partial(patch_from_cache, cache=cache_H00_keys_all)
)

filter_H00_values_all = discriminators.is_always_true
activation_name_H00_values_all = get_act_name('v', layer=0)

cache_H00_values_all, _ = scrubber.get_cache_and_matching_tokens_from_discriminator(
    discriminator=filter_H00_values_all,
    activation_name=activation_name_H00_values_all
)
hook_H00_values_all = (
    activation_name_H00_values_all,
    partial(patch_from_cache, cache=cache_H00_values_all)
)

# filter_H00_out_end = (
#     discriminators.sign_parentheses_count.created_fixed_pos_filter(-1) * 
#     discriminators.is_last_paren_closed 
#     )
filter_H00_out_end = discriminators.is_always_true

activation_name_H00_out_end = get_act_name('attn_out', layer=0)

cache_H00_out_end, tokens_H00_out_end = scrubber.get_cache_and_matching_tokens_from_discriminator(
    discriminator=filter_H00_out_end,
    activation_name=activation_name_H00_out_end,
    hooks=[hook_H00_keys_all, hook_H00_values_all],
)

hook_list_sophisticated_scrub.append(
    (activation_name_H00_out_end,
     partial(patch_from_cache, pos_idx=END_POS, cache=cache_H00_out_end))
)

# %% Head 1.0

filter_H10_out_end = (
    discriminators.starts_with_backdoor * 
    discriminators.is_above_horizon *
    discriminators.is_equal_count
    )
activation_name_H10_out_end = get_act_name('attn_out', layer=1)

cache_H10_out_end, tokens_H10_out_end = scrubber.get_cache_and_matching_tokens_from_discriminator(
    discriminator=filter_H10_out_end,
    activation_name=activation_name_H10_out_end,
)

hook_list_sophisticated_scrub.append(
    (activation_name_H10_out_end,
     partial(patch_from_cache, pos_idx=END_POS, cache=cache_H10_out_end))
)

# %%

recovered_loss_float = scrubber.compute_recovered_loss_float(
    *scrubber.compute_causal_scrubbing_losses(hook_list_sophisticated_scrub)
    )

print(f'Recovered loss: {recovered_loss_float :.3f}')

# %%

# filter_balanced = discriminators.is_balanced

# plotter.plot_scatter_loss_by_category(patched_loss, orig_tokens, filter_H10_out_end)

# adv_criteria = ((patched_loss > 1) & (patched_loss < 1.57)).cpu()
# adv_tokens_H1_0 = tokens_H1_0[adv_criteria]
# adv_tokens_orig = orig_tokens[adv_criteria]

# paren_count_diff_H1_0 = discriminators.count_diff_open_to_closed_paren(adv_tokens_H1_0)
# line(paren_count_diff_H1_0, dim_labels=['Batch', 'Position'], color='Batch',
#      x='Position')

# paren_count_diff_orig = discriminators.count_diff_open_to_closed_paren(adv_tokens_orig)
# line(paren_count_diff_orig, dim_labels=['Batch', 'Position'], color='Batch',
#      x='Position')