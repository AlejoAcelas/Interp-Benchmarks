
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint

from src.dataset.dataset import AlgorithmicDataConstructor
from src.dataset.discriminators import TokenDiscriminator, TokenDiscriminatorByPos
from src.utils import compute_cross_entropy_loss

# Hooks

TENSOR_INDEX_TYPE = Union[Int[Tensor, 'dim'], slice, int]
ACTIVATION_TYPE = Float[Tensor, 'batch pos ...']
HOOK_FN_TYPE = Callable[[ACTIVATION_TYPE, HookPoint], ACTIVATION_TYPE]
HOOK_TYPE = Tuple[str, HOOK_FN_TYPE]
LOSS_TENSOR_TYPE = Float[Tensor, 'batch']

def patch_from_cache(
        activations: Float[Tensor, 'batch pos ...'], 
        hook: HookPoint,
        cache: ActivationCache,
        pos_idx: Optional[TENSOR_INDEX_TYPE] = slice(None),
        head_idx: Optional[TENSOR_INDEX_TYPE] = None,
        ) -> Float[Tensor, 'batch pos ...']:
    activations[:, pos_idx] = cache[hook.name][:, pos_idx]
    return activations

def patch_from_cache_different_batch_size(
        activations: Float[Tensor, 'batch pos ...'],
        hook: HookPoint,
        cache: ActivationCache,
        activations_pos_idx: TENSOR_INDEX_TYPE,
        cache_pos_idx: TENSOR_INDEX_TYPE,
        cache_batch_idx: TENSOR_INDEX_TYPE,
        head_idx: Optional[TENSOR_INDEX_TYPE] = None,
    ):
    batch_size = activations.shape[0]
    activations_batch_idx = unsqueeze_at_end(torch.arange(batch_size), ndim_out=activations_pos_idx.ndim)
    cache_batch_idx = unsqueeze_at_end(cache_batch_idx, ndim_out=cache_pos_idx.ndim)
    
    new_activations = cache[hook.name][cache_batch_idx, cache_pos_idx, head_idx]
    activations[activations_batch_idx, activations_pos_idx, head_idx] = new_activations
    return activations


class CausalScrubbing():

    DEFAULT_BATCH_SIZE = 1_000
    SEED = 0

    def __init__(
            self,
            data_constructor: AlgorithmicDataConstructor,
            model: HookedTransformer,
            token_generator: Callable[[int], Int[Tensor, 'batch pos']],
            batch_size: int = DEFAULT_BATCH_SIZE,
            ):
        self.data_constructor = data_constructor
        self.model = model
        self.token_generator = token_generator

        self.data_constructor.set_seed(self.SEED)
        self.orig_tokens = self.token_generator(batch_size)
        self.default_tokens = self.token_generator(batch_size)

    def run_causal_scrubbing(
            self,
            end_nodes: List['ScrubbingNode'],
            save_matching_tokens: bool = False,
            patch_on_orig_tokens: bool = False,
            ) -> Tuple[LOSS_TENSOR_TYPE, LOSS_TENSOR_TYPE, LOSS_TENSOR_TYPE]:
        final_hooks = []
        for node in end_nodes:
            final_hooks.extend(self.get_node_hooks(node, self.orig_tokens,
                                                   save_matching_tokens=save_matching_tokens))
        return self.compute_causal_scrubbing_losses(final_hooks, patch_on_orig_tokens=patch_on_orig_tokens)

    def get_node_hooks(
            self,
            node: 'ScrubbingNode',
            tokens_to_match: Int[Tensor, 'batch pos'],
            ) -> HOOK_TYPE:
        assert node.discriminator is not None, f'Node for activation {node.activation_names} has no discriminator'
        
        matching_tokens = node.gen_matching_tokens(
            tokens_to_match, self.token_generator,
        )
        hooks = []
        for parent in node.get_parents():
            hooks.extend(
                self.get_node_hooks(
                    parent,
                    matching_tokens,
                )
            )
        
        cache = compute_activations_from_hooks(
            model=self.model,
            tokens=matching_tokens,
            activation_names=node.activation_names,
            hooks=hooks
        )
        return node.get_hooks(cache)
    
    def compute_causal_scrubbing_losses(
            self,
            hooks: List[HOOK_TYPE],
            patch_on_orig_tokens: bool = False,
            ) -> Tuple[LOSS_TENSOR_TYPE, LOSS_TENSOR_TYPE, LOSS_TENSOR_TYPE]:
        alter_tokens = self.orig_tokens if patch_on_orig_tokens else self.default_tokens
        orig_logits = self.model(self.orig_tokens)
        patched_logits = self.model.run_with_hooks(alter_tokens, fwd_hooks=hooks)

        orig_labels = self.data_constructor.get_token_labels(self.orig_tokens)
        permuted_labels = orig_labels[torch.randperm(orig_labels.shape[0])]

        orig_loss = self.compute_loss(orig_logits, labels=orig_labels, reduce='labels')
        patched_loss = self.compute_loss(patched_logits, labels=orig_labels, reduce='labels')
        random_loss = self.compute_loss(orig_logits, labels=permuted_labels, reduce='labels')

        return orig_loss, patched_loss, random_loss
    
    def compute_loss(
            self,
            logits: Float[Tensor, 'batch pos vocab'],
            labels: Int[Tensor, 'batch label'],
            reduce: str = 'all',
            ) -> Union[float, LOSS_TENSOR_TYPE]:
        label_pos = self.data_constructor.tokenizer.get_label_pos()
        logits_at_pos_label = logits[:, label_pos]
        labels = labels.to(logits_at_pos_label.device)
        loss = compute_cross_entropy_loss(logits_at_pos_label, labels, reduce=reduce)
        return loss.item() if reduce == 'all' else loss.squeeze()

    def compute_recovered_loss_float(
            self,
            orig_loss: LOSS_TENSOR_TYPE,
            patched_loss: LOSS_TENSOR_TYPE,
            random_loss: LOSS_TENSOR_TYPE
            ) -> float:
        return ((random_loss.mean() - patched_loss.mean())/
                (random_loss.mean() - orig_loss.mean())).item()
    
def compute_activations_from_hooks(
        model: HookedTransformer,
        tokens: Int[Tensor, 'batch pos'],
        activation_names: Optional[List[str]] = None,
        hooks: Optional[List[Tuple[str, Callable]]] = None
    ) -> ActivationCache:
    if hooks is not None:
        for hook_act_name, hook_fn in hooks:
            model.add_hook(hook_act_name, hook_fn)

    if activation_names is None:
        names_filter = lambda name: True
    else:
        names_filter = lambda name: name in set(activation_names)
    
    _, cache = model.run_with_cache(tokens, names_filter=names_filter)
    model.reset_hooks()
    return cache
    
class ScrubbingNode():

    def __init__(
            self, 
            activation_name: Union[str, List[str]],
            discriminator: Optional[TokenDiscriminator] = None,
            pos_idx: Optional[TENSOR_INDEX_TYPE] = slice(None),
            parents: Optional[List['ScrubbingNode']] = None,
            head_idx: Optional[int] = None,
    ):
        self.activation_names = [activation_name] if isinstance(activation_name, str) else activation_name
        self.discriminator = discriminator
        self.pos_idx = pos_idx
        self.parents = [] if parents is None else [parent for parent in parents if parent.is_active()]
        self.head_idx = head_idx

        self.matching_tokens: Int[Tensor, 'batch pos'] = None

    def is_active(self) -> bool:
        """Hacky way to eliminate nodes using the discriminator argument"""
        return self.discriminator is not None
    
    def get_parents(self) -> List['ScrubbingNode']:
        return self.parents
    
    def gen_matching_tokens(
            self,
            tokens: Int[Tensor, 'batch pos'],
            token_gen_fn: Callable[[int], Int[Tensor, 'batch pos']],
        ) -> Int[Tensor, 'batch2 pos']:
        self.matching_tokens = self.discriminator.gen_matching_tokens(tokens, token_gen_fn)
        return self.matching_tokens

    def get_hooks(self, cache: ActivationCache) -> List[HOOK_TYPE]:
        hook_fn = self.get_hook_fn(cache)
        return [(act_name, hook_fn) for act_name in self.activation_names]
    
    def get_hook_fn(self, cache: ActivationCache) -> HOOK_FN_TYPE:
        return partial(
            patch_from_cache,
            pos_idx=self.pos_idx,
            cache=cache,
            head_idx=self.head_idx,
            )


class ScrubbingNodeByPos(ScrubbingNode):

    def __init__(
            self,
            activation_name: Union[str, List[str]],
            discriminator: Optional[TokenDiscriminatorByPos] = None,
            pos_map: Optional[Int[Tensor, 'disc_pos *num_pos']] = None,
            parents: Optional[List['ScrubbingNode']] = None,
            head_idx: Optional[int] = None,
    ):
        assert discriminator is None or isinstance(discriminator, TokenDiscriminatorByPos), 'Discriminator must be a TokenDiscriminatorByPos or None'
        super().__init__(activation_name, discriminator, pos_idx=None, parents=parents, head_idx=head_idx)

        self.pos_map = pos_map
        self.discriminator_batch_idx: TENSOR_INDEX_TYPE = None
        self.discriminator_pos_idx: TENSOR_INDEX_TYPE = None
    
    def gen_matching_tokens(
            self,
            tokens: Tensor,
            token_gen_fn: Callable[[int], Tensor],
        ) -> Tensor:
        matching_tokens, matching_batch_idx, matching_pos_idx = self.discriminator.gen_matching_tokens(tokens, token_gen_fn)
        self.discriminator_batch_idx = matching_batch_idx
        self.discriminator_pos_idx = matching_pos_idx
        self.matching_tokens = matching_tokens

        return matching_tokens
    
    def get_hook_fn(self, cache: ActivationCache) -> HOOK_FN_TYPE:
        batch_disc, pos_disc = self.discriminator_pos_idx.shape
        self.pos_map = (torch.arange(pos_disc) if self.pos_map is None
                        else self.pos_map)
        
        index_discriminator_pos_idx = torch.arange(pos_disc).expand(batch_disc, pos_disc)
        self.activations_pos_idx = self.pos_map[index_discriminator_pos_idx]
        self.cache_pos_idx = self.pos_map[self.discriminator_pos_idx]
        
        return partial(
            patch_from_cache_different_batch_size,
            cache=cache,
            activations_pos_idx=self.activations_pos_idx,
            cache_pos_idx=self.cache_pos_idx,
            cache_batch_idx=self.discriminator_batch_idx,
            head_idx=self.head_idx,
        )

def unsqueeze_at_end(tensor: Tensor, ndim_out: int) -> Tensor:
    ndim_in = tensor.ndim
    assert ndim_out >= ndim_in
    extra_dims_shape = (1,) * (ndim_out - ndim_in)
    return tensor.reshape(*tensor.shape, *extra_dims_shape)


