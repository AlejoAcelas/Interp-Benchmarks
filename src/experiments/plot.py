
import sys
from typing import List, Tuple, Dict, Any

import circuitsvis as cv
import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer

from src.dataset.dataset import AlgorithmicDataConstructor
from src.dataset.discriminators import TokenDiscriminator

from alejo_plotly import bar, box, histogram, line, scatter, violin, imshow


class DataPlotter():

    def __init__(self, data_constructor: AlgorithmicDataConstructor, model: HookedTransformer):
        self.data_constructor = data_constructor
        self.model = model

    def plot_attn_patterns(
            self,
            tokens: Int[Tensor, 'batch pos'],
            ):
        if tokens.ndim == 1 or tokens.shape[0] == 1:
            tokens = tokens.repeat(2, 1) # cv.attention.from_cache doesn't handle well batch_size=1
        
        head_list = [(layer, head) for layer in range(self.model.cfg.n_layers) for head in range(self.model.cfg.n_heads)]
        str_tokens = self.data_constructor.tokenizer.tokens_to_str_tokens(tokens)
        
        _, cache = self.model.run_with_cache(tokens)
        return cv.attention.from_cache(
            cache=cache,
            tokens=str_tokens,
            heads=head_list,
            attention_type='info-weighted',
            return_mode='view'
        )

    def plot_scatter_loss_by_category(
            self,
            loss: Float[Tensor, 'batch'],
            tokens_for_color: Int[Tensor, 'batch pos'],
            color_discriminator: TokenDiscriminator,
            color_discriminator_labels: Dict[int, Any] = None,
            **kwargs,
            ) -> None:

        color_ids = color_discriminator(tokens_for_color)
        discriminator_name = color_discriminator.name.replace(' * ', '<br>')
        value_names = ({'color': color_discriminator_labels} 
                       if color_discriminator_labels is not None else dict())
        scatter(loss, color=color_ids, value_names=value_names, title='Loss per datapoint',
                labels=dict(y='Loss', index='Datapoint', color=discriminator_name, **kwargs))

    def get_logit_attr_for_binary_labels(
            self,
            tokens: Int[Tensor, 'batch pos'],
            project_to_true_direction: bool = False,
            ) -> Tuple[Float[Tensor, 'component batch'], List[str]]:
        
        labels = 1 if project_to_true_direction else self.data_constructor.get_token_labels(tokens)
        _, cache = self.model.run_with_cache(tokens)
        pos_slice = self.data_constructor.tokenizer.get_label_pos()
        
        resid_components, resid_labels = cache.decompose_resid(pos_slice=pos_slice, return_labels=True)
        logit_attr = cache.logit_attrs(resid_components, tokens=labels, incorrect_tokens=1-labels, pos_slice=pos_slice)
        return logit_attr.squeeze(), resid_labels
    
    def get_svd_components_across_dimensions(
            self,
            activations: Float[Tensor, '... d_model'], 
            num_components: int = 4, 
            plot_singular_values: bool = True
            ) -> Float[Tensor, '... d_model']:
        
        activations_2D = activations.reshape(-1, activations.shape[-1])
        U, S, V = torch.svd(activations_2D)
        U = U.reshape(activations.shape)

        if plot_singular_values:
            bar(S, dim_labels=['Singular values'], title='Singular values')

        return U[..., :num_components]

    def get_svd_components_per_dimension(
            self,
            activations: Float[Tensor, 'batch ... d_model'], 
            num_components: int = 4,
            ) -> Float[Tensor, 'batch ... d_model']:
        
        activations_3D = activations.reshape(activations.shape[0], -1, activations.shape[-1])
        svd_vector_list = []
        for acts in activations_3D.unbind(1):
            U, S, V = torch.svd(acts)
            svd_vector_list.append(U[..., :num_components])
        svd_vector = torch.stack(svd_vector_list, dim=1)
        svd_vector = svd_vector.reshape(*activations.shape[:-1], num_components)
        return svd_vector
    
