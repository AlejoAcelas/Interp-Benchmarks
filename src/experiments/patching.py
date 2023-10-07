
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint

from src.dataset.dataset import AlgorithmicDataConstructor
from src.dataset.discriminators import TokenDiscriminator
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
        batch_idx: Optional[TENSOR_INDEX_TYPE] = slice(None)
        ) -> Float[Tensor, 'batch pos ...']:
    activations[batch_idx, pos_idx] = cache[hook.name][batch_idx, pos_idx]
    return activations

# Evaluation functions

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
            ) -> Tuple[LOSS_TENSOR_TYPE, LOSS_TENSOR_TYPE, LOSS_TENSOR_TYPE]:
        final_hooks = []
        for node in end_nodes:
            final_hooks.extend(self.get_node_hooks(node, self.orig_tokens,
                                                   save_matching_tokens=save_matching_tokens))
        return self.compute_causal_scrubbing_losses(final_hooks)

    def get_node_hooks(
            self,
            node: 'ScrubbingNode',
            tokens_to_match: Int[Tensor, 'batch pos'],
            save_matching_tokens: bool = False,
            ) -> HOOK_TYPE:
        assert node.discriminator is not None, f'Node for activation {node.activation_name} has no discriminator'
        
        matching_tokens = node.discriminator.gen_matching_tokens(tokens_to_match, self.token_generator)
        if save_matching_tokens:
            node.matching_tokens = matching_tokens
        
        hooks = []
        for parent in node.get_parents():
            hooks.extend(self.get_node_hooks(parent, matching_tokens,
                                            save_matching_tokens=save_matching_tokens))
        
        cache = run_with_cache_and_hooks(self.model, matching_tokens, hooks)
        return node.get_hooks(cache)
    
    def compute_causal_scrubbing_losses(
            self,
            hooks: List[HOOK_TYPE],
            ) -> Tuple[LOSS_TENSOR_TYPE, LOSS_TENSOR_TYPE, LOSS_TENSOR_TYPE]:
        orig_logits = self.model(self.orig_tokens)
        patched_logits = self.model.run_with_hooks(self.default_tokens, fwd_hooks=hooks)

        orig_labels = self.data_constructor.get_token_labels(self.orig_tokens)
        permuted_labels = orig_labels[torch.randperm(orig_labels.shape[0])]

        orig_loss = self.compute_loss(orig_logits, labels=orig_labels, reduce='label')
        patched_loss = self.compute_loss(patched_logits, labels=orig_labels, reduce='label')
        random_loss = self.compute_loss(orig_logits, labels=permuted_labels, reduce='label')

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
    
def run_with_cache_and_hooks(model: HookedTransformer, tokens: Int[Tensor, 'batch pos'],
                             hooks: Optional[List[Tuple[str, Callable]]] = None) -> ActivationCache:
    if hooks is not None:
        for hook_act_name, hook_fn in hooks:
            model.add_hook(hook_act_name, hook_fn)
    _, cache = model.run_with_cache(tokens)
    model.reset_hooks()
    return cache
    
@dataclass
class ScrubbingNode():
    activation_name: Union[str, List[str]]
    discriminator: Optional[TokenDiscriminator] = None
    pos_idx: Optional[TENSOR_INDEX_TYPE] = field(default_factory=lambda: slice(None))
    matching_tokens: Optional[Int[Tensor, 'batch pos']] = None
    parents: Optional[List['ScrubbingNode']] = None

    def add_parent(self, parent: 'ScrubbingNode'):
        if self.parents is None:
            self.parents = []
        self.parents.append(parent) 

    def get_parents(self) -> List['ScrubbingNode']:
        if self.parents is None:
            return []
        return [parent for parent in self.parents if parent.discriminator is not None]
    
    def get_hooks(self, cache: ActivationCache) -> List[HOOK_TYPE]:
        if isinstance(self.activation_name, list):
            return [(act_name, self.get_hook_fn(cache)) for act_name in self.activation_name]
        return [(self.activation_name, self.get_hook_fn(cache))]
    
    def get_hook_fn(self, cache: ActivationCache) -> HOOK_FN_TYPE:
        return partial(
            patch_from_cache,
            pos_idx=self.pos_idx,
            cache=cache,
            )



