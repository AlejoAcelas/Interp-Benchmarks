# %%
import sys
import os

import torch
import einops
from transformer_lens import HookedTransformer

os.chdir('/home/alejo/Projects/Interpretability-Collections')
from train import load_model
from dataset import BalancedParenthesisDataGenerator, to_str_toks

import circuitsvis as cv
from IPython.display import display

sys.path.append('/home/alejo/Projects')
from my_plotly_utils import hist, bar, imshow, scatter, figs_to_subplots
from path_patching import act_patch, path_patch, Node, IterNode

# %%
data_gen = BalancedParenthesisDataGenerator(n_ctx_numeric=20)
model: HookedTransformer = load_model('./models/final/bal_paren_20-l2_h2_d32_m1-1000.pt', data_gen)

# %%
data_gen.set_seed(0)
toks = data_gen.gen_toks(batch_size=100).to(model.cfg.device)
labels = data_gen.get_token_labels(toks)
logits, cache = model.run_with_cache(toks)


# %%
logits_at_pos_label = logits[:, data_gen.pos_label, :]
pred = logits_at_pos_label.argmax(dim=-1)
accuracy = (pred == labels).float().mean()
print(f'Accuracy: {accuracy:.2f}')

# %%
resid_comps, resid_labels = cache.decompose_resid(pos_slice=data_gen.pos_label, return_labels=True)
logit_attr = cache.logit_attrs(resid_comps, tokens=labels, incorrect_tokens=1-labels, pos_slice=data_gen.pos_label)
imshow(logit_attr[:, :50].squeeze(), y=resid_labels, labels=dict(x='Prompt', y='Component'),
       title='Logit attribution for 50 prompts')
# %%
data_gen.set_seed(0)
batch_idx = 1

# toks = data_gen.gen_same_num_open_and_closed_toks(batch_size=100)
toks = data_gen.gen_off_by_one_balanced_parentheses_toks(batch_size=100).to(model.cfg.device)
labels = data_gen.get_token_labels(toks)

logits_at_pos_label = model(toks)[:, data_gen.pos_label, :]
probs_at_pos_label = logits_at_pos_label.softmax(dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

num_open_toks = (toks[:, data_gen.pos_numeric] == data_gen.OPEN_TOKEN).float().cumsum(dim=-1)
num_closed_toks = (toks[:, data_gen.pos_numeric] == data_gen.CLOSED_TOKEN).float().cumsum(dim=-1)
diff_num_open_closed = num_open_toks - num_closed_toks

print('Label: ', labels[batch_idx].item())
print(f'Prob to correct label: {probs_at_pos_label[batch_idx].item(): .3f}')
print('Difference num open and closed: ', diff_num_open_closed[batch_idx])

attn_patterns = einops.rearrange(cache.stack_activation('pattern'), 'layer batch head src dst -> batch (layer head) src dst')
str_toks = to_str_toks(data_gen, toks)
str_toks_initials = [[str_tok[0] for str_tok in tok_seq] for tok_seq in str_toks]
head_names = [f'H{layer}.{head}' for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)]
plot_attn_pattern = cv.attention.attention_patterns(attn_patterns[batch_idx], tokens=str_toks_initials[batch_idx],
                                attention_head_names=head_names)
display(plot_attn_pattern)
# %%



