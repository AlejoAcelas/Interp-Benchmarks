import os
from dataclasses import dataclass
from typing import Tuple, Optional, List, Callable
from tqdm.notebook import tqdm
import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from torch.utils.data import DataLoader
import wandb
from math import ceil

from src.utils import compute_cross_entropy_loss, compute_accuracy
from src.dataset.dataset import AlgorithmicDataConstructor
from src.train.model import ModelArgs, create_model_from_data_generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class TrainArgs:
    epochs: int = 1
    trainset_size: int = 100_000
    valset_size: int = 10_000
    batch_size: int = 512 
    lr: float = 1e-3
    lr_end_factor: float = 0.33
    weight_decay: float = 0.0
    seed: int = 42
    device: str = device
    use_wandb: bool = False

class Trainer:
    def __init__(self, data_cons: AlgorithmicDataConstructor, model_args: ModelArgs, train_args: TrainArgs):
        self.data_cons = data_cons
        self.model_args = model_args
        self.train_args = train_args
        
        self.model = create_model_from_data_generator(data_cons, model_args)
        self.test_accuracy = []
        
        if train_args.use_wandb:
            wandb.init(project="interp-collections", config=model_args.__dict__)

    def training_step(self, batch: Tuple[Tensor, Tensor]) -> torch.Tensor:
        logits_at_label_pos, labels = self._shared_train_validation_step(batch)
        return compute_cross_entropy_loss(logits_at_label_pos, labels)
        
    def validation_step(self, batch: Tuple[Tensor, Tensor]) -> torch.Tensor:
        logits_at_label_pos, labels = self._shared_train_validation_step(batch)
        return compute_accuracy(logits_at_label_pos, labels, as_percentage=False)

    def _shared_train_validation_step(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        tokens, labels = batch
        tokens, labels = tokens.to(self.train_args.device), labels.to(self.train_args.device)
        logits = self.model(tokens)
        logits_at_label_pos = logits[..., self.data_cons.tokenizer.get_label_pos(), :]
        return logits_at_label_pos, labels

    def train_dataloader(self, seed: int) -> DataLoader:
        trainset = self.data_cons.create_dataset(batch_size=self.train_args.trainset_size, seed=seed)
        return DataLoader(trainset, batch_size=self.train_args.batch_size, shuffle=True)
    
    def val_dataloader(self, seed: int) -> DataLoader:
        valset = self.data_cons.create_dataset(batch_size=self.train_args.valset_size, seed=seed)
        return DataLoader(valset, batch_size=self.train_args.batch_size, shuffle=True)
    
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_args.lr, weight_decay=self.train_args.weight_decay)
        
        upper_bound_iters = self.train_args.epochs*ceil(self.train_args.trainset_size/self.train_args.batch_size)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=self.train_args.lr_end_factor, 
                                                      total_iters=upper_bound_iters)
        return optimizer, scheduler

    def train(self):
        optimizer, scheduler = self.configure_optimizers()
        val_dataloader = self.val_dataloader(seed=self.train_args.seed - 1)

        for epoch in range(self.train_args.epochs):
            
            train_dataloader = self.train_dataloader(seed=self.train_args.seed + epoch) # Produce a new training set each epoch
            progress_bar = tqdm(total=self.train_args.trainset_size//self.train_args.batch_size)

            # Training
            for batch in train_dataloader:
                # Optimization step on training set
                optimizer.zero_grad()
                loss = self.training_step(batch)
                loss.backward()
                optimizer.step()
                scheduler.step()
                # Log variables, update progress bar
                if self.train_args.use_wandb: wandb.log({"training_loss": loss})
                # if loss > 1: breakpoint()
                progress_bar.update()
                progress_bar.set_description(f"Epoch {epoch:02}, Train loss = {loss:.4f}")
            
            # Validation
            with torch.inference_mode():
                num_correct_per_batch = [self.validation_step(batch) for batch in val_dataloader]
                accuracy_per_token = sum(num_correct_per_batch) / (self.train_args.valset_size * self.data_cons.tokenizer.len_label)
                # Log variables, update progress bar
                self.test_accuracy.append(accuracy_per_token)
                if self.train_args.use_wandb: wandb.log({"test_accuracy": accuracy_per_token})
                progress_bar.set_description(f"Epoch {epoch:02}, Train loss = {loss:.4f}, Accuracy: {accuracy_per_token:.3f}")

        if self.train_args.use_wandb:
            wandb.finish()

    def save_model(self, task_name: str, dir: str = "./models"):
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        latest_test_accuracy = self.test_accuracy[-1]
        acc_str = f"{1000*latest_test_accuracy:.0f}"
        
        model_specs = self.model_args.as_str()
        filename = f"{task_name}-{model_specs}-{acc_str}.pt"
        torch.save(self.model.state_dict(), f"{dir}/{filename}")

def load_model(filename: str, data_gen: AlgorithmicDataConstructor) -> HookedTransformer:
    """Load a model saved using Trainer.save_model"""
    model_args_str = filename.split('-')[1]
    model_args = ModelArgs.create_from_str(model_args_str)
    model = create_model_from_data_generator(data_gen, model_args)

    state_dict = torch.load(filename)
    state_dict = adjust_state_dict_for_mech_interp(state_dict, model)
    model.load_state_dict(state_dict, strict=False)
    return model

def adjust_state_dict_for_mech_interp(state_dict: dict, model: HookedTransformer) -> dict:
    state_dict = model.center_writing_weights(state_dict)
    state_dict = model.center_unembed(state_dict)
    state_dict = model.fold_layer_norm(state_dict)
    state_dict = model.fold_value_biases(state_dict)
    return state_dict

        

# def get_missed_data(args: TrainArgs, model: HookedTransformer):
#     trainer = Trainer(args, model)
#     val_dataloader = trainer.val_dataloader(seed=args.seed+1)
#     missed_tokens, missed_labels, missed_logits = [], [], []
#     with torch.inference_mode():
#         for tokens, labels in val_dataloader:
#             logits, labels = trainer._shared_train_validation_step((tokens, labels))
#             tokens = tokens.to(args.device)
#             accuracy = (logits.argmax(-1) == labels).all(-1)
#             missed_tokens.append(tokens[~accuracy])
#             missed_labels.append(labels[~accuracy])
#             missed_logits.append(logits[~accuracy])
#     return torch.cat(missed_tokens), torch.cat(missed_labels), torch.cat(missed_logits)

