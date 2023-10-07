# %%
import sys

from src.dataset.dataset import BaseTenAdditionDataConstructor
from src.dataset.discriminators import BaseTenAdditionTokenCriteriaCollection
from src.dataset.generators import BaseTenAdditionTokenGenerator
from src.experiments.plot import DataPlotter
from src.train.train import load_model

sys.path.append('/home/alejo/Projects')
from my_plotly_utils import imshow
from new_plotly_utils import bar, box, histogram, line, scatter, violin

# %%
data_constructor = BaseTenAdditionDataConstructor(n_digits_addend=4)
model = load_model('final/addition4-l2_h2_d64_m4-1000.pt', data_constructor)
plotter = DataPlotter(data_constructor, model)

generators: BaseTenAdditionTokenGenerator = data_constructor.generators
discriminators: BaseTenAdditionTokenCriteriaCollection = data_constructor.discriminators

# %%

tokens = data_constructor.gen_tokens(10)
tokens = generators.gen_carry_tokens(10, carry_depth=3)
plotter.plot_attn_patterns(tokens)
# %%
