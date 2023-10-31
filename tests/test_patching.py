
import pytest
from src.experiments.patching import CausalScrubbing, ScrubbingNode, ScrubbingNodeByPos, compute_activations_from_hooks
from src.train.model import create_model_from_data_generator, ModelArgs
from utils_for_tests import SingleNumDataConstructor, ModuloTokenCriteriaCollection
from functools import partial
import torch
from typing import List
from torch import Tensor
from jaxtyping import Int

from transformer_lens.utils import get_act_name

BATCH_SIZE = 10

DATA_CONSTRUCTOR = SingleNumDataConstructor()
DISCRIMINATORS: ModuloTokenCriteriaCollection = DATA_CONSTRUCTOR.discriminators
TOKEN_GENERATOR = DATA_CONSTRUCTOR.generators.gen_random_tokens

N_CTX = DATA_CONSTRUCTOR.N_CTX_NUMERIC

def create_model():
    model_args = ModelArgs(n_layers=5, n_heads=1, d_model=16, d_mlp_multiplier=0)
    model = create_model_from_data_generator(DATA_CONSTRUCTOR, model_args)
    return model

class TestCausalScrubbing():
    model = create_model()
    scrubbing = CausalScrubbing(data_constructor=DATA_CONSTRUCTOR, model=model, token_generator=TOKEN_GENERATOR,batch_size=BATCH_SIZE)
    reference_tokens = TOKEN_GENERATOR(BATCH_SIZE)
    run_tokens = TOKEN_GENERATOR(BATCH_SIZE)

    def test_chain_graph(self):
        """
        Test that model activations are as expected for nodes in a deterministic causal chain graph.
        
        Each node intervenes in the residual stream activations at a layer, where the node at layer
        `l` is the parent of the node at `l+1`. If the causal scrubbing is working correctly, then:
        
        * The hooks from the `l`th-node should not affect the activations at layers `k` < `l`
        
        * The residual stream at layer `l` from running the model with the hooks for the `l`th-layer node
        should equal the activations from running with the hooks for the 0th layer node, because later activations
        are a deterministic function of early activations. 
        """
        n_layers = self.model.cfg.n_layers

        nodes: List[ScrubbingNode] = []
        for layer in range(n_layers):
            parents = [nodes[-1]] if nodes else []
            resid_node = ScrubbingNode(
                activation_name=get_act_name('resid_pre', layer),
                discriminator=DISCRIMINATORS.get_criterion('is_always_true'),
                parents=parents,
            )
            nodes.append(resid_node)
        
        hooks_final_node = self.scrubbing.get_node_hooks(
            node=nodes[-1],
            tokens_to_match=self.reference_tokens,
        )
        
        cache_first_node = compute_activations_from_hooks(self.model, nodes[0].matching_tokens)
        cache_final_node = compute_activations_from_hooks(self.model, self.run_tokens, hooks=hooks_final_node)
        cache_run_tokens = compute_activations_from_hooks(self.model, self.run_tokens)

        for layer in range(n_layers):
            resid_pre_first_node = cache_first_node['resid_pre', layer]
            resid_pre_final_node = cache_final_node['resid_pre', layer]
            resid_pre_run_tokens = cache_run_tokens['resid_pre', layer]

            if layer == (n_layers - 1):
                torch.testing.assert_close(resid_pre_final_node, resid_pre_first_node)
            else:
                torch.testing.assert_close(resid_pre_final_node, resid_pre_run_tokens)

    def test_bifurcating_graph(self):
        """
        Test that model activations are as expected for nodes in a bifurcating causal graph.

        I create two nodes that act independently on the inputs to the keys and queries of the 
        attention heads at a given layer. Then I check that the keys and queries from running a
        node that depends on them correspond to the keys and queries from running each node. 
        """
        LAYER = 0
        node_q_input = ScrubbingNode(
            activation_name=get_act_name('q_input', LAYER),
            discriminator=DISCRIMINATORS.get_criterion('is_always_true'),
        )
        node_k_input = ScrubbingNode(
            activation_name=get_act_name('k_input', LAYER),
            discriminator=DISCRIMINATORS.get_criterion('is_always_true'),
        )
        node_out = ScrubbingNode(
            activation_name=[get_act_name('q', LAYER), get_act_name('k', LAYER)],
            discriminator=DISCRIMINATORS.get_criterion('is_always_true'),
            parents=[node_q_input, node_k_input],
        )

        hooks_final_node = self.scrubbing.get_node_hooks(
            node=node_out,
            tokens_to_match=self.reference_tokens,
        )
        
        cache_q_input = compute_activations_from_hooks(self.model, node_q_input.matching_tokens)
        cache_k_input = compute_activations_from_hooks(self.model, node_k_input.matching_tokens)
        cache_final_node = compute_activations_from_hooks(self.model, self.run_tokens, hooks=hooks_final_node)

        queries_from_input = cache_q_input['q', LAYER]
        keys_from_input = cache_k_input['k', LAYER]
        queries_from_final_run = cache_final_node['q', LAYER]
        keys_from_final_run = cache_final_node['k', LAYER]

        torch.testing.assert_close(queries_from_final_run, queries_from_input)
        torch.testing.assert_close(keys_from_final_run, keys_from_input)


    @pytest.mark.parametrize(
            'pos_map', [torch.arange(N_CTX), torch.randperm(N_CTX)]
        )
    def test_single_node_by_pos(self, pos_map: Int[Tensor, 'pos *idx']):
        LAYER = 0
        node_resid_post = ScrubbingNodeByPos(
            activation_name=get_act_name('resid_post', LAYER),
            discriminator=DISCRIMINATORS.get_criterion('is_always_true_by_pos'),
            pos_map=pos_map,
        )

        hooks = self.scrubbing.get_node_hooks(
            node=node_resid_post,
            tokens_to_match=self.reference_tokens,
        )

        cache_node = compute_activations_from_hooks(self.model, self.run_tokens, hooks=hooks)
        cache_matching_tokens = compute_activations_from_hooks(self.model, node_resid_post.matching_tokens)
        cache_run_tokens = compute_activations_from_hooks(self.model, self.run_tokens)

        resid_post_node = cache_node['resid_post', LAYER]
        resid_post_matching_tokens = cache_matching_tokens['resid_post', LAYER]
        
        resid_pre_node = cache_node['resid_pre', LAYER]
        resid_pre_run_tokens = cache_run_tokens['resid_pre', LAYER]

        batch_idx_matching_tokens = node_resid_post.discriminator_batch_idx
        pos_idx_matching_tokens = pos_map[node_resid_post.discriminator_pos_idx]

        torch.testing.assert_close(
            resid_post_node[:, pos_map],
            resid_post_matching_tokens[batch_idx_matching_tokens, pos_idx_matching_tokens]
        )
        torch.testing.assert_close(resid_pre_node, resid_pre_run_tokens)


    def test_node_by_pos_and_normal_node_chain(self):
        POS_MATCHED_END_TO_END = 0

        node_single_pos_root = ScrubbingNode(
            activation_name=get_act_name('tokens'),
            discriminator=DISCRIMINATORS.get_criterion('tokens', pos_index=POS_MATCHED_END_TO_END),
            pos_idx=POS_MATCHED_END_TO_END,
        )
        node_by_pos = ScrubbingNodeByPos(
            activation_name=get_act_name('tokens'),
            discriminator=DISCRIMINATORS.cartesian_product('tokens', 'position'),
            parents=[node_single_pos_root],
        )

        node_single_pos_leaf = ScrubbingNode(
            activation_name=get_act_name('tokens'),
            discriminator=DISCRIMINATORS.get_criterion('tokens', pos_index=POS_MATCHED_END_TO_END),
            pos_idx=POS_MATCHED_END_TO_END,
            parents=[node_by_pos],
        )
        hooks = self.scrubbing.get_node_hooks(
            node=node_single_pos_leaf,
            tokens_to_match=self.reference_tokens,
        )

        batch_idx_by_pos = node_by_pos.discriminator_batch_idx
        pos_idx_by_pos = node_by_pos.discriminator_pos_idx
        
        tokens_node_by_pos = node_by_pos.matching_tokens[batch_idx_by_pos, pos_idx_by_pos]
        tokens_root_node = node_single_pos_root.matching_tokens[batch_idx_by_pos, pos_idx_by_pos]

        torch.testing.assert_close(
            tokens_node_by_pos[:, POS_MATCHED_END_TO_END],
            self.reference_tokens[:, POS_MATCHED_END_TO_END]
        )
        torch.testing.assert_close(
            tokens_root_node[:, POS_MATCHED_END_TO_END],
            self.reference_tokens[:, POS_MATCHED_END_TO_END]
        )

