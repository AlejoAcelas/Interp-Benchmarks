
import sys 

import torch
from torch import Tensor

from typing import Tuple, List
from jaxtyping import Int, Float

import circuitsvis as cv
from transformer_lens import HookedTransformer

from src.dataset.dataset import AlgorithmicDataConstructor

sys.path.append('/home/alejo/Projects')
from new_plotly_utils import scatter, histogram, violin, bar, box, line
from my_plotly_utils import imshow


def get_logit_attr_for_binary_labels(toks: Int[Tensor, 'batch pos'],
                                     model: HookedTransformer,
                                     data_constructor: AlgorithmicDataConstructor,
                                     always_true_labels: bool = False
                                     ) -> Tuple[Float[Tensor, 'component batch'], List[str]]:
    labels = 1 if always_true_labels else data_constructor.get_token_labels(toks)
    _, cache = model.run_with_cache(toks)
    pos_slice = data_constructor.tokenizer.get_label_pos()
    resid_comps, resid_labels = cache.decompose_resid(pos_slice=pos_slice, return_labels=True)
    logit_attr = cache.logit_attrs(resid_comps, tokens=labels, incorrect_tokens=1-labels, pos_slice=pos_slice)
    return logit_attr.squeeze(), resid_labels

def plot_logit_attr_for_binary_labels(toks: Int[Tensor, 'batch pos'],
                                      model: HookedTransformer,
                                      data_constructor: AlgorithmicDataConstructor):
    logit_attr, resid_labels = get_logit_attr_for_binary_labels(toks, model, data_constructor)
    box(logit_attr, dim_labels=['Model Component', 'Datapoint'],
        value_names={'Model Component': resid_labels},
        labels=dict(y='Loggit difference'),
        title='Logit attribution for each residual stream component')
    
def plot_attn_patterns(toks: Int[Tensor, 'batch pos'],
                       model: HookedTransformer,
                       data_constructor: AlgorithmicDataConstructor,):
    if toks.ndim == 1 or toks.shape[0] == 1:
        toks = toks.repeat(2, 1) # cv.attention.from_cache doesn't handle well batch_size=1
    _, cache = model.run_with_cache(toks)
    str_toks = data_constructor.tokenizer.toks_to_str_toks(toks)
    head_list = [(layer, head) for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)]
    return cv.attention.from_cache(cache=cache,
                                   tokens=str_toks,
                                   heads=head_list,
                                   attention_type='info-weighted',
                                   return_mode='view')

def get_svd_across_dimensions(activations: Float[Tensor, '... d_model'], 
                              num_singular_values: int = 4, 
                              plot_singular_values: bool = True) -> Float[Tensor, '... d_model']:
    activations_2D = activations.reshape(-1, activations.shape[-1])
    U, S, V = torch.svd(activations_2D)
    U = U.reshape(activations.shape)

    if plot_singular_values:
        bar(S, dim_labels=['Singular values'], title='Singular values')

    return U[..., :num_singular_values]

def get_svd_per_dimension(activations: Float[Tensor, 'batch ... d_model'], 
                          num_singular_values: int = 4) -> Float[Tensor, 'batch ... d_model']:
    activations_3D = activations.reshape(activations.shape[0], -1, activations.shape[-1])
    svd_vector_list = []
    for acts in activations_3D.unbind(1):
        U, S, V = torch.svd(acts)
        svd_vector_list.append(U[..., :num_singular_values])
    svd_vector = torch.stack(svd_vector_list, dim=1)
    svd_vector = svd_vector.reshape(*activations.shape[:-1], num_singular_values)
    return svd_vector