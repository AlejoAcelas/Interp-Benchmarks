# %%
import re
from dataclasses import dataclass
from functools import partial
from itertools import product
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as t
from transformer_lens import HookedTransformer, HookedTransformerConfig

from src.dataset.dataset import AlgorithmicDataConstructor

device = "cuda" if t.cuda.is_available() else "cpu"

@dataclass
class ModelArgs():
    n_layers: int
    n_heads: int
    d_model: int = 64
    d_mlp_multiplier: int = 4
    device: str = device

    def as_str(self):
        specs_str = f'l{self.n_layers}_h{self.n_heads}_d{self.d_model}_m{self.d_mlp_multiplier}'
        return specs_str

    @staticmethod
    def create_from_str(specs_str):
        match = re.match(r'l(\d+)_h(\d+)_d(\d+)_m(\d+)', specs_str)
        if match:
            n_layers, n_heads, d_model, d_mlp_multiplier = map(int, match.groups())
            return ModelArgs(n_layers=n_layers, n_heads=n_heads, d_model=d_model, d_mlp_multiplier=d_mlp_multiplier)
        else:
            raise ValueError(f'Invalid format for specs_str: {specs_str}')

# %%

class ModelArgsIterator():
    arg_names = ['n_layers', 'n_heads', 'd_model', 'd_mlp_multiplier']

    def __init__(self,
                 n_layers: Optional[List[int]] = None,
                 n_heads: Optional[List[int]] = None,
                 d_model: Optional[List[int]] = None,
                 d_mlp_multiplier: Optional[List[int]] = None,
                 default_args: Optional[Dict[str, Any]] = None,
                ):
        self.default_args = dict() if default_args is None else default_args
        
        self.iter_args: List[List] = []
        for arg_name in self.arg_names:
            if eval(arg_name) is None:
                assert arg_name in self.default_args, f"{arg_name} argument must be specified either in default_args or as a list"
                self.iter_args.append([self.default_args[arg_name],])
            else:
                assert isinstance(eval(arg_name), list), f"{arg_name} argument must be a list"
                self.iter_args.append(eval(arg_name))

    def __iter__(self):
        return (ModelArgs(**self.parse_tuple_to_args_dict(args)) for args in product(*self.iter_args))
    
    def parse_tuple_to_args_dict(self, args_tuple):
        return dict(zip(self.arg_names, args_tuple))


def create_model_from_data_generator(data_gen: AlgorithmicDataConstructor,
                                     args: ModelArgs) -> HookedTransformer:
    data_gen_args = data_gen.get_model_initialization_args()
    model = _create_model(
        d_vocab=data_gen_args['d_vocab'],
        d_vocab_out=data_gen_args['d_vocab_out'],
        n_ctx=data_gen_args['n_ctx'],
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_mlp_multiplier=args.d_mlp_multiplier,
        device=args.device,
    )
    return model

def _create_model(
    d_vocab: int, 
    d_vocab_out: int,
    n_ctx: int,
    n_layers: int,
    n_heads: int,
    d_model: int,
    d_mlp_multiplier: int,
    device: str = device,
    seed: int = 42,
    ) -> HookedTransformer:

    assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
    d_head = d_model // n_heads
    attn_only = d_mlp_multiplier == 0
    d_mlp = None if attn_only else int(d_mlp_multiplier * d_model)

    t.manual_seed(seed)
    np.random.seed(seed)
    
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        n_ctx=n_ctx,
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        d_mlp=d_mlp,
        attn_only=attn_only,
        d_vocab=d_vocab,
        d_vocab_out=d_vocab_out, 
        
        use_attn_result=True,
        use_split_qkv_input=True,
        use_hook_tokens=True,

        act_fn="relu",
        normalization_type="LN",
        device=device,
    )

    model = HookedTransformer(cfg)
    return model
