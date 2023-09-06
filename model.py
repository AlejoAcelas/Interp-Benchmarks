import torch as t
import numpy as np
from typing import Optional, Union, List, Dict, Any
from transformer_lens import HookedTransformer, HookedTransformerConfig
from dataset import AlgorithmicDataGenerator
from dataclasses import dataclass

from functools import partial
from itertools import product

device = "cuda" if t.cuda.is_available() else "cpu"

@dataclass
class ModelArgs():
    n_layers: int
    n_heads: int
    d_head: int = 32
    attn_only: bool = False
    device: str = device

class ModelArgsIterator():
    arg_names = ['n_layers', 'n_heads', 'd_head', 'attn_only']

    def __init__(self,
                 n_layers: Optional[List[int]] = None,
                 n_heads: Optional[List[int]] = None,
                 d_head: Optional[List[int]] = None,
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


def create_model_from_data_generator(data_gen: AlgorithmicDataGenerator,
                                     args: ModelArgs) -> HookedTransformer:
    model = _create_model(
        d_vocab=data_gen.d_vocab,
        d_vocab_out=data_gen.d_vocab_out,
        n_ctx=data_gen.n_ctx,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_head=args.d_head,
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
    d_head: int,
    attn_only: bool,
    device: str = device,
    seed: int = 42,
    ) -> HookedTransformer:

    d_model = n_heads * d_head
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

