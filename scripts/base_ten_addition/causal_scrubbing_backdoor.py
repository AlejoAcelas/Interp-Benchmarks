
# %%
from src.dataset.tokenizer import BaseTenAdditionTokenizer
from src.dataset.dataset import BaseTenAdditionDataConstructor, BackdoorBaseTenAdditionDataConstructor
from src.dataset.generators import BaseTenAdditionTokenGenerator, BackdoorBaseTenAdditionTokenGenerator
from src.train.train import load_model
from transformer_lens.utils import get_act_name
from src.experiments.plot import DataPlotter
from functools import partial
import einops
import pandas as pd

from src.experiments.patching import CausalScrubbing, ScrubbingNode, ScrubbingNodeByPos
from src.dataset.discriminators import BaseTenAdditionTokenCriteriaCollection
from src.experiments.utils import yield_default_and_one_off_discriminator_variations, yield_default_discriminator_combination


# %% 

data_constructor = BackdoorBaseTenAdditionDataConstructor(n_digits_addend=4)
data_constructor_no_bdoor = BaseTenAdditionDataConstructor(n_digits_addend=4)
model = load_model('final/addition4_bdoor-l2_h2_d64_m4-1000.pt', data_constructor)

tokenizer: BaseTenAdditionTokenizer = data_constructor.tokenizer
discriminators: BaseTenAdditionTokenCriteriaCollection = data_constructor.discriminators

BATCH_SIZE = 100

LABEL_POS = tokenizer.get_label_pos()
NUMERIC_POS = tokenizer.get_numeric_pos()
N_CTX = data_constructor.tokenizer.get_sequence_length()

# token_generator = data_constructor.gen_tokens
token_generator = partial(data_constructor.gen_tokens_from_train_generators, generator_probs=[0.5, 0.5, 1e-3, 1e-3])
# token_generator = data_constructor_no_bdoor.gen_tokens
# token_generator = data_constructor.generators.gen_random_tokens
# disc_no_carry = discriminators.create_carry_pattern_discriminator([], strict=True)
# token_generator = partial(disc_no_carry.gen_tokens_in_group, group_id=True, token_gen_fn=data_constructor.gen_tokens)

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
            'addend1', 'addend2', 'position',
            pos_index=[0, 1, 2, 3]
        ),
        discriminators.cartesian_product(
            discriminators.get_criterion('addend1', pos_index=[3]),
            discriminators.get_criterion('addend2', pos_index=[3]),
            discriminators.get_criterion('position', pos_index=[4]),
        ),
        pos_index=scrubbing_pos,
    )
]

discs_H01_out = [
    discriminators.concatenate( 
        discriminators.cartesian_product(
            'addend1', 'addend2', 'position',
            pos_index=[0, 1, 2, 3]
        ),
        discriminators.cartesian_product(
            discriminators.get_criterion('addend1', pos_index=[3]),
            discriminators.get_criterion('addend2', pos_index=[3]),
            discriminators.get_criterion('position', pos_index=[4]),
        ),
        pos_index=scrubbing_pos,
    )
]

discs_H11_out = [
    discriminators.concatenate(
        discriminators.cartesian_product(
            'sum_no_modulo', 'carry_history', 'position',
            pos_index=[0, 1, 2, 3],
        ),
        discriminators.cartesian_product(
            discriminators.get_criterion('sum_no_modulo', pos_index=[3]),
            discriminators.get_criterion('carry_history', pos_index=[4]),
            discriminators.get_criterion('position', pos_index=[4]),
        ),
        pos_index=scrubbing_pos,
    ),
]

discs_H10_out = [
    discriminators.concatenate(
        discriminators.cartesian_product(
            'position', 'is_only_five_or_zeros', 'addend1',
            pos_index=[0, 1, 2, 3],
        ),
        discriminators.cartesian_product(
            'position', 'is_only_five_or_zeros',
            pos_index=[4],
        ),
        pos_index=scrubbing_pos,
    ),
    # discriminators.get_criterion('ones', num_pos=5, pos_index=scrubbing_pos),
]

# %%

columns_df_recovered_loss = [
    'H00_out',
    'H01_out',
    'H11_out',
    'H10_out',
    'recovered_loss',
]

save_matching_tokens = True
df_rows = []

for disc_combination in yield_default_and_one_off_discriminator_variations(
    discs_H00_out,
    discs_H01_out,
    discs_H11_out,
    discs_H10_out,
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
    )

    node_H01_out = ScrubbingNodeByPos(
        activation_name=get_act_name('z', layer=0),
        discriminator=disc_combination[1],
        pos_map=LABEL_POS[scrubbing_pos],
        head_idx=1,
    )

    node_H11_out = ScrubbingNodeByPos(
        activation_name=get_act_name('z', layer=1),
        discriminator=disc_combination[2],
        pos_map=LABEL_POS[scrubbing_pos],
        head_idx=1,
        parents=[node_H00_out, node_H01_out, node_attn_patterns],
        # parents=[node_H00_out, node_H01_out, node_attn_patterns, node_L0_out_addend_pos],
    )

    node_H10_out = ScrubbingNodeByPos(
        activation_name=get_act_name('z', layer=1),
        discriminator=disc_combination[3],
        pos_map=LABEL_POS[scrubbing_pos],
        head_idx=0,
        parents=[node_H00_out, node_H01_out, node_attn_patterns]
    )

    loss_orig, loss_patch, loss_random = scrubber.run_causal_scrubbing(
        end_nodes=[
            node_H00_out,
            node_H01_out,
            node_H11_out,
            node_H10_out,
        ],
        # patch_on_orig_tokens=True,
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

plotter.plot_scatter_loss_by_category(
    loss=loss_patch,
    tokens_for_color=scrubber.orig_tokens,
    color_discriminator=discriminators.get_criterion('ones'),
)


# %%

node = node_H00_out
sum_fn = discriminators.sum_no_modulo

criteria_idx = (loss_patch > 0.65).cpu()

orig_tokens = scrubber.orig_tokens
patch_tokens = node.matching_tokens

print(orig_tokens[criteria_idx])
print(patch_tokens[criteria_idx])

# sum_selected_orig_tokens = sum_fn(orig_tokens[criteria_idx])
# sum_selected_patch_tokens = sum_fn(patch_tokens[criteria_idx])

# print(sum_selected_orig_tokens)
# print(sum_selected_patch_tokens)
# print(sum_selected_patch_tokens - sum_selected_orig_tokens)
# %%
