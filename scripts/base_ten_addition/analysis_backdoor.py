# %%
import sys

from src.dataset.tokenizer import BaseTenAdditionTokenizer
from src.dataset.dataset import BaseTenAdditionDataConstructor, BackdoorBaseTenAdditionDataConstructor
from src.dataset.discriminators import BaseTenAdditionTokenCriteriaCollection
from src.dataset.generators import BaseTenAdditionTokenGenerator, BackdoorBaseTenAdditionTokenGenerator
from src.experiments.plot import DataPlotter
from src.train.train import load_model
from src.utils import compute_cross_entropy_loss

import einops
import torch
import plotly.express as px
from functools import partial

# from path_patching import path_patch, act_patch, Node, IterNode
from alejo_plotly import bar, box, histogram, line, scatter, violin, imshow

# %%
data_constructor = BackdoorBaseTenAdditionDataConstructor(n_digits_addend=4)
model = load_model('final/addition4_bdoor-l2_h2_d64_m4-1000.pt', data_constructor)
plotter = DataPlotter(data_constructor, model)

tokenizer: BaseTenAdditionTokenizer = data_constructor.tokenizer
generators = BaseTenAdditionTokenGenerator(tokenizer)
generators_bdoor: BackdoorBaseTenAdditionTokenGenerator = data_constructor.generators
discriminators: BaseTenAdditionTokenCriteriaCollection = data_constructor.discriminators

LABEL_POS = tokenizer.get_label_pos()
ADDENDS_POS = tokenizer.get_numeric_pos()
figures_dir = 'figures/'

# %% Attn Patterns

tokens = generators_bdoor.gen_backdoor_tokens(10)
# tokens = data_constructor.gen_tokens(10)
# tokens = generators_bdoor.gen_backdoor_tokens(10)
# tokens = generators.gen_carry_tokens(10, carry_depth=3)
plotter.plot_attn_patterns(tokens)

# %% 

tokens = data_constructor.gen_tokens(1000)
_, cache = model.run_with_cache(tokens)
attn_pattern = cache.stack_activation('pattern')
attn_pattern = einops.rearrange(
    attn_pattern[..., LABEL_POS, :], 'layer batch head q k -> batch (layer head) q k'
)
mean_attn_pattern = attn_pattern.mean(dim=0)
get_labels_variable = lambda var, num : [f'{var}<sub>{i}</sub>' for i in range(1, num+1)]

fig = imshow(
    mean_attn_pattern.transpose(1, 2),
    facet_col=0,
    x=get_labels_variable('z', 5),
    y=['START'] + get_labels_variable('x', 4) + get_labels_variable('y', 4) + get_labels_variable('z', 5),
    facet_labels=[f'Head {layer}.{head}' for layer in range(2) for head in range(2)],
    labels=dict(color='Attention<br>probability'),
    color_continuous_scale='Greens',
    range_color=[0, 1],
    title='Average attention pattern from label positions for<br>Backdoor Model',
    return_fig=True,
)

fig.write_image(figures_dir + 'attn_all_mean_bdoor.png', scale=6, width=500)


# %% SVD Scatter Plots

data_constructor.set_seed(0)
tokens = data_constructor.gen_tokens(500)

addend1, addend2 = tokenizer.get_addends_from_tokens(tokens)
sum = discriminators.sum_tokens(tokens)
addition_by_digit = addend1 + addend2
carry_first_order = discriminators.contains_carry_at_depth(tokens, depth=0)
carry_history = discriminators.carry_history(tokens)
is_backdoor = discriminators.is_only_five_or_zeros(tokens)
pos_sum = tokenizer.get_label_pos()

_, cache = model.run_with_cache(tokens)

# %%  Head 0.0 
layer, head = 0, 0
pos_to_plot = 1

head_result = cache['result', layer][:, pos_sum, head]
svd_head = plotter.get_svd_components_per_dimension(head_result) # [batch, pos, svd_comp]
svd_head_combined = plotter.get_svd_components_across_dimensions(head_result[:, :4], plot_singular_values=False) # [batch, pos, svd_comp]

fig1 = scatter(
    y=svd_head[:, pos_to_plot, 1],
    x=svd_head[:, pos_to_plot, 0],
    dim_labels=['Batch'],
    color=addition_by_digit[:, pos_to_plot], 
    title=f'SVD Components for H{layer}.{head} output at END<sub>{pos_to_plot+1}</sub>',
    color_continuous_scale='Turbo',
    labels=dict(
        color=f'BDA<sub>{pos_to_plot+1}</sub>',
        y='SVD Component 2',
        x='SVD Component 1'
    ),
    return_fig=True,
    render_mode='svg',
)

fig2 = scatter(
    y=svd_head_combined[..., 1],
    x=svd_head_combined[..., 0],
    dim_labels=['Batch', 'Sum Position'],
    color=addend1, 
    title=f'SVD Components for H{layer}.{head} output at END1-END4 tokens',
    color_continuous_scale='Turbo',
    labels=dict(
        color=f'Addend 1 (x)<br>at corr. pos.',
        y='SVD Component 2',
        x='SVD Component 1'
    ),
    return_fig=True,
    render_mode='svg',
)

fig3 = scatter(
    y=svd_head_combined[..., 1],
    x=svd_head_combined[..., 0],
    dim_labels=['Batch', 'Sum Position'],
    color=addend2, 
    title=f'SVD Components for H{layer}.{head} output at END1-END4 tokens',
    color_continuous_scale='Turbo',
    labels=dict(
        color=f'Addend 2 (y)<br>at corr. pos.',
        y='SVD Component 2',
        x='SVD Component 1'
    ),
    return_fig=True,
    render_mode='svg',
)

fig1.write_image(
    figures_dir + f'svd_scatter_h{layer}{head}_pos{pos_to_plot}_bdoor.png',
    scale=6,
    width=500
)
fig2.write_image(
    figures_dir + f'svd_scatter_h{layer}{head}_x_bdoor.png',
    scale=6,
    width=500
)
fig3.write_image(
    figures_dir + f'svd_scatter_h{layer}{head}_y_bdoor.png',
    scale=6,
    width=500
)

# %%  Head 0.1 
layer, head = 0, 1
pos_to_plot = 1

head_result = cache['result', layer][:, pos_sum, head]
svd_head = plotter.get_svd_components_per_dimension(head_result) # [batch, pos, svd_comp]
svd_head_combined = plotter.get_svd_components_across_dimensions(head_result[:, :4]) # [batch, pos, svd_comp]

fig1 = scatter(
    y=svd_head[:, pos_to_plot, 1],
    x=svd_head[:, pos_to_plot, 0],
    dim_labels=['Batch'],
    color=addition_by_digit[:, pos_to_plot], 
    title=f'SVD Components for H{layer}.{head} output at END<sub>{pos_to_plot+1}</sub>',
    color_continuous_scale='Turbo',
    labels=dict(
        color=f'BDA<sub>{pos_to_plot+1}</sub>',
        y='SVD Component 2',
        x='SVD Component 1'
    ),
    return_fig=True,
    render_mode='svg',
)

fig2 = scatter(
    y=svd_head_combined[..., 1],
    x=svd_head_combined[..., 0],
    dim_labels=['Batch', 'Sum Position'],
    color=addend1, 
    title=f'SVD Components for H{layer}.{head} output at END1-END4 tokens',
    color_continuous_scale='Turbo',
    labels=dict(
        color=f'Addend 1 (x)<br>at corr. pos.',
        y='SVD Component 2',
        x='SVD Component 1'
    ),
    return_fig=True,
    render_mode='svg',
)

fig3 = scatter(
    y=svd_head_combined[..., 1],
    x=svd_head_combined[..., 0],
    dim_labels=['Batch', 'Sum Position'],
    color=addend2, 
    title=f'SVD Components for H{layer}.{head} output at END1-END4 tokens',
    color_continuous_scale='Turbo',
    labels=dict(
        color=f'Addend 2 (y)<br>at corr. pos.',
        y='SVD Component 2',
        x='SVD Component 1'
    ),
    return_fig=True,
    render_mode='svg',
)

fig1.write_image(
    figures_dir + f'svd_scatter_h{layer}{head}_pos{pos_to_plot}_bdoor.png',
    scale=6,
    width=500
)
fig2.write_image(
    figures_dir + f'svd_scatter_h{layer}{head}_x_bdoor.png',
    scale=6,
    width=500
)
fig3.write_image(
    figures_dir + f'svd_scatter_h{layer}{head}_y_bdoor.png',
    scale=6,
    width=500
)

# %%  Head 1.1 
layer, head = 1, 1
pos_to_plot = 0

head_result = cache['result', layer][:, pos_sum, head]
svd_head = plotter.get_svd_components_per_dimension(head_result) # [batch, pos, svd_comp]
svd_head_combined = plotter.get_svd_components_across_dimensions(head_result[:, :4]) # [batch, pos, svd_comp]

fig1 = scatter(
    y=svd_head_combined[..., 1],
    x=svd_head_combined[..., 0],
    dim_labels=['Batch', 'Sum Position'],
    color=carry_history[:, :4],
    title=f'SVD Components for H{layer}.{head} output at END1-END4 tokens',
    color_continuous_scale='Turbo',
    labels=dict(
        color=f'Carry at <br>corr. pos.',
        y='SVD Component 2',
        x='SVD Component 1'
    ),
    value_labels=dict(color={
        0: 'No Carry',
        1: 'Simple Carry',
        2: 'Double Carry',
        3: 'Triple Carry',
    }),
    return_fig=True,
    render_mode='svg',
)

fig2 = scatter(
    y=svd_head_combined[..., 1],
    x=svd_head_combined[..., 0],
    dim_labels=['Batch', 'Sum Position'],
    color=addition_by_digit[:, :4],
    title=f'SVD Components for H{layer}.{head} output at END1-END4 tokens',
    color_continuous_scale='Turbo',
    labels=dict(
        color=f'BDA at <br>corr. pos.',
        y='SVD Component 2',
        x='SVD Component 1'
    ),
    return_fig=True,
    render_mode='svg',
)

fig1.write_image(
    figures_dir + f'svd_scatter_h{layer}{head}_all_pos_carry_bdoor.png',
    scale=6,
    width=500
)
fig2.write_image(
    figures_dir + f'svd_scatter_h{layer}{head}_all_pos_sum_bdoor.png',
    scale=6,
    width=500
)

# %%  Head 1.0 
layer, head = 1, 0
pos_to_plot = 1

head_result = cache['result', layer][:, pos_sum, head]
svd_head = plotter.get_svd_components_per_dimension(head_result) # [batch, pos, svd_comp]
svd_head_combined = plotter.get_svd_components_across_dimensions(head_result[:, :4], plot_singular_values=False) # [batch, pos, svd_comp]


fig1 = scatter(
    y=svd_head_combined[..., 1],
    x=svd_head_combined[..., 0],
    dim_labels=['Batch', 'Sum Position'],
    # color=addend1, 
    color=is_backdoor[:, None], 
    title=f'SVD Components for H{layer}.{head} output at END1-END4 tokens',
    color_continuous_scale='Turbo',
    labels=dict(
        color=f'Addend 2 (y)<br>at corr. pos.',
        y='SVD Component 2',
        x='SVD Component 1'
    ),
    return_fig=True,
    render_mode='svg',
)

tokens = generators_bdoor.gen_backdoor_tokens(500)
addend1, addend2 = tokenizer.get_addends_from_tokens(tokens)
_, cache = model.run_with_cache(tokens)

head_result = cache['result', layer][:, pos_sum, head]
svd_head = plotter.get_svd_components_per_dimension(head_result) # [batch, pos, svd_comp]
svd_head_combined = plotter.get_svd_components_across_dimensions(head_result[:, :4], plot_singular_values=False) # [batch, pos, svd_comp]


fig2 = scatter(
    y=svd_head_combined[..., 1],
    x=svd_head_combined[..., 0],
    dim_labels=['Batch', 'Sum Position'],
    color=addend1, 
    title=f'SVD Components for H{layer}.{head} output at END1-END4 tokens<br>only on backdoor sequences',
    color_continuous_scale='Turbo',
    labels=dict(
        color=f'Addend 1 (x)<br>at corr. pos.',
        y='SVD Component 2',
        x='SVD Component 1'
    ),
    return_fig=True,
    render_mode='svg',
)

fig1.write_image(
    figures_dir + f'svd_scatter_h{layer}{head}_bdoor.png',
    scale=6,
    width=500
)
fig2.write_image(
    figures_dir + f'svd_scatter_h{layer}{head}_addend_bdoor.png',
    scale=6,
    width=500
)

# %% Path Patching
from src.utils import compute_cross_entropy_loss

batch_size = 500

data_constructor.set_seed(0)

orig_tokens = data_constructor.gen_tokens(batch_size)
alter_tokens = data_constructor.gen_tokens(batch_size)

orig_labels = data_constructor.get_token_labels(orig_tokens)


loss_metric_no_reduce = lambda logits: compute_cross_entropy_loss(
    logits[:, LABEL_POS].cpu(),
    orig_labels,
    reduce='none',
)
loss_metric = lambda logits: loss_metric_no_reduce(logits).mean().item()
orig_loss = loss_metric(model(orig_tokens))
random_loss = loss_metric(model(orig_tokens[torch.randperm(batch_size)]))
recovered_loss_metric = lambda logits: ((random_loss - loss_metric(logits)) /
                                         (random_loss - orig_loss))

# %%
patch_pattern_by_head = act_patch(
    model=model,
    orig_input=orig_tokens,
    new_input=alter_tokens,
    patching_nodes=IterNode('pattern'),
    patching_metric=recovered_loss_metric,
)

imshow(
    patch_pattern_by_head['pattern'],
    labels=dict(x='Head', y='Layer'),
    title='Loss after randomly swapping attn pattern with another sequence',
)

# %%
iter_path_patch_values = torch.zeros(model.cfg.n_heads, len(LABEL_POS))

for sender_head in range(model.cfg.n_heads):
    for patch_pos in LABEL_POS:
        path_patch_values = path_patch(
            model=model,
            orig_input=orig_tokens,
            new_input=alter_tokens,
            sender_nodes=[
                Node('z', layer=0, head=sender_head),
            ],
            receiver_nodes=Node('v', layer=1, head=0),
            patching_metric=loss_metric,
            direct_includes_mlps=True,
            seq_pos=patch_pos.item(),
        )
        iter_path_patch_values[sender_head, patch_pos] = path_patch_values

imshow(
    iter_path_patch_values,
    labels=dict(y='Head', x='Sum Position'),
)

# %%
addends_pos_with_batch_dim = einops.repeat(
    ADDENDS_POS, 'pos -> batch pos', batch=batch_size
)

patch_resid = act_patch(
    model=model,
    orig_input=orig_tokens,
    new_input=alter_tokens,
    patching_nodes=IterNode('resid_pre', seq_pos=addends_pos_with_batch_dim),
    patching_metric=recovered_loss_metric,
)

bar(
    patch_resid['resid_pre'],
    labels=dict(index='Layer', y='Loss'),
    title='Loss after randomly swapping resid_pre at addends <br>with another sequence',
)
# %%

batch_size = 200
alter_to_orig_ratio = 5

data_constructor.set_seed(0)

orig_tokens = data_constructor.gen_tokens(batch_size)
orig_tokens = einops.repeat(
    orig_tokens, 'batch pos -> (batch repeat) pos',
    repeat=alter_to_orig_ratio
)
alter_tokens = data_constructor.gen_tokens(batch_size * alter_to_orig_ratio)

orig_labels = data_constructor.get_token_labels(orig_tokens)

loss_metric_multi_patch = lambda logits: einops.reduce(
    loss_metric_no_reduce(logits), '(batch repeat) pos -> batch pos',
    repeat=alter_to_orig_ratio,
    reduction='mean',
)

patch_z_by_head = act_patch(
    model=model,
    orig_input=orig_tokens,
    new_input=alter_tokens,
    patching_nodes=IterNode('z'),
    patching_metric=loss_metric_multi_patch,
)

head_names_list = [f'Head {layer}.{head}' for layer in range(2) for head in range(2)]

for i, head_name in enumerate(head_names_list):
    patch_loss_head = patch_z_by_head['z'][i]
    highest_loss_inputs = patch_z_by_head['z'][i].sum(dim=1).topk(30).indices
    imshow(
        patch_loss_head[highest_loss_inputs].T,
        labels=dict(x='Batch', y='Position'),
        title=f'Loss for {head_name} after randomly swapping hook_z',
    )
print(orig_tokens[highest_loss_inputs][:, ADDENDS_POS])
print(discriminators.sum_tokens(orig_tokens[highest_loss_inputs]))


# %% Precise Datasets Patching

carry_pos = 1
batch_size = 100
pos_carry_addends = ADDENDS_POS[[carry_pos, carry_pos + 4]]
pos_carry_sum = LABEL_POS[carry_pos+1].item()
pos_carry_addends_batch_dim = einops.repeat(pos_carry_addends, 'pos -> batch pos', batch=batch_size)

data_constructor.set_seed(0)

disc_only_carry_at_pos = discriminators.create_carry_pattern_discriminator(carry_pos, strict=True)
disc_no_carry = discriminators.create_carry_pattern_discriminator([], strict=True)
disc_same_sum = discriminators.create_sum_at_pos_discriminator(carry_pos+1)

carry_tokens = disc_only_carry_at_pos.gen_tokens_in_group(
    batch_size=batch_size,
    criterion_value=True,
    token_gen_fn=data_constructor.gen_tokens
)
new_carry_tokens = disc_no_carry.gen_tokens_in_group(
    batch_size=batch_size,
    criterion_value=True,
    token_gen_fn=data_constructor.gen_tokens
)
same_sum_tokens = disc_same_sum.gen_matching_tokens(
    reference_tokens=carry_tokens,
    token_gen_fn=partial(disc_no_carry.gen_tokens_in_group, criterion_value=True, token_gen_fn=data_constructor.gen_tokens)
    # token_gen_fn=partial(disc_only_carry_at_pos.gen_tokens_in_group, criterion_value=True, token_gen_fn=data_constructor.gen_tokens),
)
aux_non_carry_tokens = disc_no_carry.gen_tokens_in_group(
    batch_size=batch_size,
    criterion_value=True,
    token_gen_fn=data_constructor.gen_tokens
)

non_carry_tokens = carry_tokens.clone()
non_carry_tokens[:, pos_carry_addends] = aux_non_carry_tokens[:, pos_carry_addends]

target_labels = data_constructor.get_token_labels(carry_tokens)
# target_labels[:, pos_carry_sum] =  (target_labels[:, pos_carry_sum] - 1) % 10
# target_labels = data_constructor.get_token_labels(new_carry_tokens)
# target_labels = data_constructor.get_token_labels(same_sum_tokens)
# target_labels = data_constructor.get_token_labels(non_carry_tokens)

loss_metric_no_reduce = lambda logits: compute_cross_entropy_loss(
    logits[:, LABEL_POS].cpu(),
    target_labels,
    reduce='none',
)
loss_metric_at_carry_pos = (
    lambda logits: loss_metric_no_reduce(logits)[:, carry_pos+1].mean().item()
)

# %%

patch_z_by_head = act_patch(
    model=model,
    orig_input=carry_tokens,
    new_input=non_carry_tokens,
    patching_nodes=IterNode('z', seq_pos=pos_carry_sum),
    patching_metric=loss_metric_at_carry_pos,
)

imshow(
    patch_z_by_head['z'],
    labels=dict(y='Layer', x='Head'),
    title='Loss at carry_pos from patching hook_z from no-carry <br>sequences to sequences with carry only at carry_pos'
)

# %%

patch_z_crucial_heads = act_patch(
    model=model,
    orig_input=non_carry_tokens,
    new_input=carry_tokens,
    patching_nodes=[
        Node('z', layer=0, head=1, seq_pos=pos_carry_sum),
        Node('z', layer=1, head=0, seq_pos=pos_carry_sum),
    ],
    patching_metric=loss_metric_at_carry_pos,
)


print(f'Loss: {patch_z_crucial_heads:.3f}')
# %%

patch_z_crucial_heads = act_patch(
    model=model,
    orig_input=carry_tokens,
    new_input=new_carry_tokens,
    patching_nodes=[
        Node('z', layer=0, head=1, seq_pos=pos_carry_sum),
        Node('z', layer=1, head=0, seq_pos=pos_carry_sum),
    ],
    patching_metric=loss_metric_at_carry_pos,
)


print(f'Loss: {patch_z_crucial_heads:.3f}')

# %% 

patch_z_crucial_heads = act_patch(
    model=model,
    orig_input=carry_tokens,
    new_input=same_sum_tokens,
    patching_nodes=[
        # Node('v', layer=0, head=0, seq_pos=pos_carry_addends_batch_dim),
        # Node('v', layer=0, head=1, seq_pos=pos_carry_addends_batch_dim),

        Node('z', layer=0, head=1, seq_pos=pos_carry_sum),
        Node('v', layer=1, head=0, seq_pos=pos_carry_sum - 1),

        # Node('z', layer=0, head=0, seq_pos=pos_carry_sum),
        # Node('z', layer=0, head=1, seq_pos=pos_carry_sum),
        # Node('z', layer=1, head=0, seq_pos=pos_carry_sum),
    ],
    patching_metric=loss_metric_at_carry_pos,
)

print(f'Loss: {patch_z_crucial_heads:.3f}')
# %%

patch_z_crucial_heads = act_patch(
    model=model,
    orig_input=carry_tokens,
    new_input=same_sum_tokens,
    patching_nodes=[
        # Node('z', layer=0, head=0, seq_pos=pos_carry_sum),
        # Node('z', layer=0, head=1, seq_pos=pos_carry_sum),
        Node('z', layer=1, head=0, seq_pos=pos_carry_sum),
    ],
    patching_metric=loss_metric_at_carry_pos,
)


print(f'Loss: {patch_z_crucial_heads:.3f}')
# %%
