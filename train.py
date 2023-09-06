from dataclasses import dataclass
from typing import Tuple, Optional, List, Callable
from tqdm.notebook import tqdm
import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from torch.utils.data import DataLoader
import wandb
from math import ceil

from utils import compute_cross_entropy_loss, compute_accuracy
from dataset import AlgorithmicDataGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class TrainArgs:
    epochs: int = 1
    trainset_size: int = 100_000
    valset_size: int = 10_000
    batch_size: int = 512 
    lr: float = 1e-3
    lr_end_factor: float = 0.25
    weight_decay: float = 0.0
    seed: int = 42
    device: str = device
    use_wandb: bool = False

class Trainer:
    def __init__(self, model: HookedTransformer, data_generator: AlgorithmicDataGenerator, args: TrainArgs):
        self.model = model
        self.data_gen = data_generator
        self.args = args
        self.test_accuracy = []
        if args.use_wandb:
            wandb.init(project="interp-collections")
            wandb.watch(self.model)

    def training_step(self, batch: Tuple[Tensor, Tensor]) -> torch.Tensor:
        label_logits, labels = self._shared_train_validation_step(batch)
        return compute_cross_entropy_loss(label_logits, labels)
        
    def validation_step(self, batch: Tuple[Tensor, Tensor]) -> torch.Tensor:
        label_logits, labels = self._shared_train_validation_step(batch)
        return compute_accuracy(label_logits, labels, as_percentage=False)

    def _shared_train_validation_step(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        toks, labels = batch
        toks, labels = toks.to(self.args.device), labels.to(self.args.device)
        logits = self.model(toks)
        label_logits = logits[..., self.data_gen.pos_label, :]
        return label_logits, labels

    def train_dataloader(self, seed: int) -> DataLoader:
        trainset = self.data_gen.create_dataset(batch_size=self.args.trainset_size, seed=seed)
        return DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True)
    
    def val_dataloader(self, seed: int) -> DataLoader:
        valset = self.data_gen.create_dataset(batch_size=self.args.valset_size, seed=seed)
        return DataLoader(valset, batch_size=self.args.batch_size, shuffle=True)
    
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        upper_bound_iters = self.args.epochs*ceil(self.args.trainset_size/self.args.batch_size)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=self.args.lr_end_factor, 
                                                      total_iters=upper_bound_iters)
        return optimizer, scheduler

    def train(self):
        optimizer, scheduler = self.configure_optimizers()
        val_dataloader = self.val_dataloader(seed=self.args.seed - 1)

        for epoch in range(self.args.epochs):
            
            train_dataloader = self.train_dataloader(seed=self.args.seed + epoch) # Produce a new training set each epoch
            progress_bar = tqdm(total=self.args.trainset_size//self.args.batch_size)

            # Training
            for batch in train_dataloader:
                # Optimization step on training set
                optimizer.zero_grad()
                loss = self.training_step(batch)
                loss.backward()
                optimizer.step()
                scheduler.step()
                # Log variables, update progress bar
                if self.args.use_wandb: wandb.log({"training_loss": loss})
                progress_bar.update()
                progress_bar.set_description(f"Epoch {epoch:02}, Train loss = {loss:.4f}")
            
            # Validation
            with torch.inference_mode():
                num_correct_per_batch = [self.validation_step(batch) for batch in val_dataloader]
                accuracy_per_token = sum(num_correct_per_batch) / (self.args.valset_size * self.data_gen.len_label)
                # Log variables, update progress bar
                self.test_accuracy.append(accuracy_per_token)
                if self.args.use_wandb: wandb.log({"test_accuracy": accuracy_per_token})
                progress_bar.set_description(f"Epoch {epoch:02}, Train loss = {loss:.4f}, Accuracy: {accuracy_per_token:.3f}")

        if self.args.use_wandb:
            wandb.finish()

    def save(self, model_name: str = 'model', dir: str = "./models"):
        latest_test_accuracy = self.test_accuracy[-1]
        acc_str = f"{1000*latest_test_accuracy: 3.0f}"
        model_specs = f"l{self.model.cfg.n_layers}_h{self.model.cfg.n_heads}_dh{self.model.cfg.d_head}_acc{acc_str}"
        model_name = f"{model_name}_{model_specs}.pt"
        torch.save(self.model.state_dict(), f"{dir}/{model_name}")
        

# def get_missed_data(args: TrainArgs, model: HookedTransformer):
#     trainer = Trainer(args, model)
#     val_dataloader = trainer.val_dataloader(seed=args.seed+1)
#     missed_toks, missed_labels, missed_logits = [], [], []
#     with torch.inference_mode():
#         for toks, labels in val_dataloader:
#             logits, labels = trainer._shared_train_validation_step((toks, labels))
#             toks = toks.to(args.device)
#             accuracy = (logits.argmax(-1) == labels).all(-1)
#             missed_toks.append(toks[~accuracy])
#             missed_labels.append(labels[~accuracy])
#             missed_logits.append(logits[~accuracy])
#     return torch.cat(missed_toks), torch.cat(missed_labels), torch.cat(missed_logits)

