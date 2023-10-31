
import os
import shutil

import pytest
from utils_for_tests import AlwaysZeroDataConstructor

from src.train.model import ModelArgs
from src.train.train import MODELS_DIR, TrainArgs, Trainer, load_model


def test_trainer():
    trainer = get_trainer_simple_task()
    trainer.train()
    assert trainer.test_accuracy[-1] > 0.9

def test_save_and_load_model():
    trainer = get_trainer_simple_task()
    trainer.test_accuracy = [0.0]
    save_dir = 'test_save_and_load'
    full_save_dir = f"{MODELS_DIR}/{save_dir}"
    task_name = 'test'

    trainer.save_model(task_name=task_name, dir=save_dir)
    model_name = next((f for f in os.listdir(full_save_dir) if f.startswith(task_name)), None)
    
    try:
        filename = f"{save_dir}/{model_name}"
        model = load_model(filename, trainer.data_constructor)
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")
    finally:
        shutil.rmtree(full_save_dir)

def get_trainer_simple_task() -> Trainer:
    data_constructor = AlwaysZeroDataConstructor()
    model_args = ModelArgs(n_layers=1, n_heads=1, d_model=4)
    train_args = TrainArgs(epochs=1, trainset_size=10_000, valset_size=1_000, 
                           batch_size=64, use_wandb=False)

    trainer = Trainer(data_constructor=data_constructor, model_args=model_args, train_args=train_args)
    return trainer
