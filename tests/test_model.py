
import pytest
from src.train.model import ModelArgs, ModelArgsIterator

def test_model_args_from_str():
    model_args = ModelArgs(n_layers=2, n_heads=4, d_model=64, attn_only=False)
    expected_model_args_str = 'l2_h4_d64_m1'
    reconstructed_model_args = ModelArgs.create_from_str(expected_model_args_str)

    assert model_args.as_str() == expected_model_args_str
    assert reconstructed_model_args == model_args

def test_model_args_iterator():
    model_args_iterator = ModelArgsIterator(
        n_layers=[1, 2],
        n_heads=[1, 2],
        default_args=dict(d_model=64, attn_only=True)
    )

    for model_args in model_args_iterator:
        assert isinstance(model_args, ModelArgs)
        assert model_args.n_layers in [1, 2]
        assert model_args.n_heads in [1, 2]
        assert model_args.d_model == 64
        assert model_args.attn_only == True


