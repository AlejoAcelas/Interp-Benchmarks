
import os
import torch
from torch import Tensor
from typing import Callable, List
from jaxtyping import Int, Bool, Shaped
import re
from functools import partial

os.chdir('/home/alejo/Projects/Interpretability_Collections')
from dataset import AlgorithmicDataConstructor, BalanParenDataConstructor

# %% Balanced Parenthesis Data Generation
