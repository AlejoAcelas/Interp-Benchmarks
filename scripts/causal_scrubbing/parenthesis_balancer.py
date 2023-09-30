# %%

import torch
from torch import Tensor

from typing import Optional, Union
from jaxtyping import Float, Int

from transformer_lens.hook_points import HookPoint
from transformer_lens import ActivationCache
from transformer_lens.utils import get_act_name

from functools import partial
from pathlib import Path

# %%


def is_interactive():
    try:
        get_ipython
        return True
    except NameError:
        return False

if is_interactive():
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

from src.dataset.dataset import BalanParenDataConstructor, BackdoorBalanParenDataConstructor
from src.utils import compute_cross_entropy_loss
from src.train.train import load_model
import src

base_directory = Path(src.__file__).parent.parent

# %%

TENSOR_INDEX_TYPE = Union[Int[Tensor, 'dim'], slice, int]

def patch_from_cache(
        activations: Float[Tensor, 'batch pos ...'], 
        hook: HookPoint,
        pos_idx: TENSOR_INDEX_TYPE,
        cache: ActivationCache,
        batch_idx: Optional[TENSOR_INDEX_TYPE] = slice(None)
        ) -> Float[Tensor, 'batch pos ...']:
    activations[batch_idx, pos_idx] = cache[hook.name][batch_idx, pos_idx]
    return activations

def compute_loss(logits: Float[Tensor, 'batch pos'],
                 labels: Int[Tensor, 'batch label'],
                 reduce: str = 'all') -> float:
    logits_at_pos_label = logits[:, data_cons.tokenizer.get_label_pos()]
    loss = compute_cross_entropy_loss(
        logits_at_pos_label, labels.to(logits_at_pos_label.device), reduce=reduce
        )
    return loss.item() if reduce == 'all' else loss.squeeze()

from typing import List, Tuple, Callable

def compute_causal_scrubbing_losses(hook_list: List[Tuple[str, Callable]],
                                orig_tokens: Int[Tensor, 'batch pos'],
                                alter_tokens: Int[Tensor, 'batch pos'],
                                labels: Int[Tensor, 'batch label']):
    orig_logits = model(orig_tokens)
    patched_logits = model.run_with_hooks(alter_tokens, fwd_hooks=hook_list)

    orig_loss = compute_loss(orig_logits, labels=labels, reduce='label')
    patched_loss = compute_loss(patched_logits, labels=labels, reduce='label')

    permuted_labels = labels[torch.randperm(labels.shape[0])]
    random_loss = compute_loss(orig_logits, labels=permuted_labels, reduce='label')
    return orig_loss, patched_loss, random_loss

def compute_recovered_loss_float(orig_loss: Float[Tensor, 'batch'],
                           patched_loss: Float[Tensor, 'batch'],
                           random_loss: Float[Tensor, 'batch']):
    return (1 - (patched_loss.mean() - orig_loss.mean())/(random_loss.mean() - orig_loss.mean())).item()

# %%

# data_cons = BalanParenDataConstructor(n_ctx_numeric=20)
# model: HookedTransformer = load_model('./models/final/bal_paren_20-l2_h1_d16_m1-1000.pt', data_cons)

data_cons = BackdoorBalanParenDataConstructor(n_ctx_numeric=20)
file_model = 'bal_paren_20_bdoor-l2_h1_d16_m1-1000.pt'
path_model_str = str(base_directory / 'models' / 'final' / file_model)
model = load_model(path_model_str, data_cons)
discriminators = data_cons.discriminators

# %%

batch_size = 1000
token_generator_probs = [0.5, 0.25, 0.125, 0.125]
# token_generator_probs = [0.99, 0.005, 0.0025, 0.0025]

data_cons.set_seed(0)
tokens_generator = partial(
    data_cons.gen_tokens_from_train_generators,
    generator_probs=token_generator_probs
)

orig_tokens = tokens_generator(batch_size)
default_tokens = tokens_generator(batch_size)
# default_tokens = orig_tokens
labels = data_cons.get_token_labels(orig_tokens)

hook_list = []

# %%

# Head 0.0 end-token patch
end_pos = data_cons.tokenizer.get_label_pos()

filter_H0_0 = (
    discriminators.sign_parentheses_count.created_fixed_pos_filter(-1) * 
    discriminators.is_last_paren_closed 
    )

tokens_H0_0 = filter_H0_0.gen_matching_tokens(orig_tokens, token_gen_fn=tokens_generator)
_, cache_H0_0 = model.run_with_cache(tokens_H0_0)

hook_list.append(
    (get_act_name('attn_out', layer=0),
     partial(patch_from_cache, pos_idx=end_pos, cache=cache_H0_0))
)

# %%

# Head 1.0 end-token patch
end_pos = data_cons.tokenizer.get_label_pos()

filter_H1_0 = (
    discriminators.starts_with_backdoor * 
    discriminators.is_above_horizon *
    discriminators.is_equal_count
    )

tokens_H1_0 = filter_H1_0.gen_matching_tokens(orig_tokens, token_gen_fn=tokens_generator)
_, cache_H1_0 = model.run_with_cache(tokens_H1_0)

hook_list.append(
    (get_act_name('attn_out', layer=1),
     partial(patch_from_cache, pos_idx=end_pos, cache=cache_H1_0))
)


orig_loss, patched_loss, random_loss = compute_causal_scrubbing_losses(
    hook_list, orig_tokens, default_tokens, labels
    )
recovered_loss_float = compute_recovered_loss_float(orig_loss, patched_loss, random_loss)

# %%
print(f'Orig loss: {orig_loss.mean().item() :.3f}')
print(f'Patched loss: {patched_loss.mean().item() :.3f}')
print(f'Recovered loss: {recovered_loss_float :.3f}')
# %%

import sys
sys.path.append('/home/alejo/Projects')
from new_plotly_utils import scatter, histogram, violin, bar, box, line
from my_plotly_utils import imshow

filter_balanced = discriminators.is_balanced

scatter(patched_loss, color=filter_balanced(orig_tokens), value_names={'color': filter_balanced.get_group_id_to_name_map()},
        title='Loss per datapoint from patched model',
        labels=dict(y='Loss', index='Datapoint', color='Original sequence balanced'))

scatter(patched_loss, color=filter_H0_0(tokens_H0_0), value_names={'color': filter_H0_0.get_group_id_to_name_map()},
        title='Loss per datapoint from patched model',
        labels=dict(y='Loss', index='Datapoint', color=filter_H0_0.criterion_name))

scatter(patched_loss, color=filter_H1_0(tokens_H1_0), value_names={'color': filter_H1_0.get_group_id_to_name_map()},
        title='Loss per datapoint from patched model',
        labels=dict(y='Loss', index='Datapoint', color=filter_H1_0.criterion_name[:20]))

# %%
adv_criteria = ((patched_loss > 1) & (patched_loss < 1.57)).cpu()
adv_tokens_H1_0 = tokens_H1_0[adv_criteria]
adv_tokens_orig = orig_tokens[adv_criteria]

paren_count_diff_H1_0 = discriminators.count_diff_open_to_closed_paren(adv_tokens_H1_0)
line(paren_count_diff_H1_0, dim_labels=['Batch', 'Position'], color='Batch',
     x='Position')

paren_count_diff_orig = discriminators.count_diff_open_to_closed_paren(adv_tokens_orig)
line(paren_count_diff_orig, dim_labels=['Batch', 'Position'], color='Batch',
     x='Position')