import torch as t
import numpy as np
from typing import Optional
from transformer_lens import HookedTransformer, HookedTransformerConfig


def create_model(
    d_vocab: int, 
    d_vocab_out: int,
    n_ctx: int,
    n_layers: int,
    n_heads: int,
    d_model: int,
    attn_only: bool,
    device: str,
    seed: int = 42,
    ) -> HookedTransformer:

    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    d_mlp = None if attn_only else d_model * 4

    t.manual_seed(seed)
    np.random.seed(seed)
    
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        n_ctx=n_ctx,
        d_model=d_model,
        d_head=d_model // n_heads,
        n_heads=n_heads,
        d_mlp=d_mlp,
        attn_only=attn_only,
        d_vocab=d_vocab,
        d_vocab_out=d_vocab_out, 
        
        # it's a small transformer so may as well use these hooks
        use_attn_result=True,
        use_split_qkv_input=True,
        use_hook_tokens=True,

        act_fn="relu",
        normalization_type="LN",
        device=device,
    )

    model = HookedTransformer(cfg)
    return model

