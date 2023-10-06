# %%
import torch as t
import numpy as np
from typing import Optional, Union, List, Dict, Any
from transformer_lens import HookedTransformer, HookedTransformerConfig
from src.dataset.dataset import AlgorithmicDataConstructor
from dataclasses import dataclass
import re

from functools import partial
from itertools import product

device = "cuda" if t.cuda.is_available() else "cpu"

@dataclass
class ModelArgs():
    n_layers: int
    n_heads: int
    d_model: int = 64
    attn_only: bool = False
    device: str = device

    def as_str(self):
        has_mlp = not self.attn_only
        specs_str = f'l{self.n_layers}_h{self.n_heads}_d{self.d_model}_m{int(has_mlp)}'
        return specs_str

    @staticmethod
    def create_from_str(specs_str):
        match = re.match(r'l(\d+)_h(\d+)_d(\d+)_m(\d+)', specs_str)
        if match:
            n_layers, n_heads, d_model, has_mlp = map(int, match.groups())
            attn_only = not bool(has_mlp)
            return ModelArgs(n_layers=n_layers, n_heads=n_heads, d_model=d_model, attn_only=attn_only)
        else:
            raise ValueError(f'Invalid format for specs_str: {specs_str}')

# %%

class ModelArgsIterator():
    arg_names = ['n_layers', 'n_heads', 'd_model', 'attn_only']

    def __init__(self,
                 n_layers: Optional[List[int]] = None,
                 n_heads: Optional[List[int]] = None,
                 d_model: Optional[List[int]] = None,
                 attn_only: Optional[List[bool]] = None,
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
        attn_only=args.attn_only,
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
    attn_only: bool,
    device: str = device,
    seed: int = 42,
    ) -> HookedTransformer:

    assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
    d_head = d_model // n_heads
    d_mlp = None if attn_only else d_model * 4

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
