from dataclasses import dataclass
from typing import Tuple, Optional, List, Callable
from jaxtyping import Bool
from tqdm.notebook import tqdm
import torch
from torch import Tensor
import torch.nn.functional as F
import einops
from transformer_lens import HookedTransformer
from torch.utils.data import DataLoader
import wandb
from math import ceil
from torch.utils.data import Dataset
# from monthly_algorithmic_problems.july23_palindromes.dataset import PalindromeDataset
from model import create_model
from utils import compute_cross_entropy_loss, compute_accuracy
from dataset import UtilsDataset

@dataclass
class TrainArgs:
    # Dataset args
    dataset: UtilsDataset
    d_vocab: int
    d_vocab_out: int
    n_ctx: int
    dataset_seed: int
    # Training args
    trainset_size: int
    valset_size: int
    epochs: int
    batch_size: int
    lr: float
    lr_end_factor: float
    weight_decay: float
    use_wandb: bool
    # Model args
    n_layers: int
    n_heads: int
    d_model: int
    attn_only: bool
    device: str

class Trainer:
    def __init__(self, args: TrainArgs, model: Optional[HookedTransformer] = None):
        self.args = args
        self.args.label_pos = args.dataset.get_label_pos(n_ctx=args.n_ctx)
        self.model = create_model(**args.__dict__) if model is None else model
        if args.use_wandb:
            wandb.init(project="interp-collections")
            wandb.watch(self.model)

    def training_step(self, batch: Tuple[Tensor, Tensor]) -> torch.Tensor:
        logits, labels = self._shared_train_validation_step(batch)
        return compute_cross_entropy_loss(logits, labels, pos_label=self.args.label_pos)
        
    
    def validation_step(self, batch: Tuple[Tensor, Tensor]) -> torch.Tensor:
        logits, labels = self._shared_train_validation_step(batch)
        return compute_accuracy(logits, labels, pos_label=self.args.label_pos, as_percentage=False)

    def _shared_train_validation_step(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        toks, labels = batch
        toks, labels = toks.to(self.args.device), labels.to(self.args.device)
        logits = self.model(toks)
        return logits, labels

    def train_dataloader(self, seed: int):
        trainset = self.args.dataset(size=self.args.trainset_size, d_vocab=self.args.d_vocab,
                                     d_vocab_out=self.args.d_vocab_out, n_ctx=self.args.n_ctx, seed=seed)
        return DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True)
    
    def val_dataloader(self, seed: int):
        valset = self.args.dataset(size=self.args.valset_size, d_vocab=self.args.d_vocab, 
                                   d_vocab_out=self.args.d_vocab_out, n_ctx=self.args.n_ctx, seed=seed)
        return DataLoader(valset, batch_size=self.args.batch_size, shuffle=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=self.args.lr_end_factor, 
                                                      total_iters=self.args.epochs*ceil(self.args.trainset_size/self.args.batch_size))
        return optimizer, scheduler


def train(args: TrainArgs, model: Optional[HookedTransformer] = None):

    trainer = Trainer(args, model=model)
    optimizer, scheduler = trainer.configure_optimizers()
    val_dataloader = trainer.val_dataloader(seed=args.dataset_seed - 1)

    for epoch in range(args.epochs):
        
        train_dataloader = trainer.train_dataloader(seed=args.dataset_seed + epoch) # I produce a new training set each epoch
        progress_bar = tqdm(total=args.trainset_size//args.batch_size)

        # Training
        for batch in train_dataloader:
            # Optimization step on training set
            optimizer.zero_grad()
            loss = trainer.training_step(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # Log variables, update progress bar
            if args.use_wandb: wandb.log({"training_loss": loss})
            progress_bar.update()
            progress_bar.set_description(f"Epoch {epoch:02}, Train loss = {loss:.4f}");
        
        # Validation
        with torch.inference_mode():
            # Calculate accuracy on validation set
            accuracy_list = [trainer.validation_step(batch) for batch in val_dataloader]
            accuracy = sum(accuracy_list) / (args.valset_size * args.label_len)
            # Log variables, update progress bar
            if args.use_wandb: wandb.log({"test_accuracy": accuracy})
            progress_bar.set_description(f"Epoch {epoch:02}, Train loss = {loss:.4f}, Accuracy: {accuracy:.3f}")

    if args.use_wandb:
        wandb.finish()

    return trainer.model

# def get_missed_data(args: TrainArgs, model: HookedTransformer):
#     trainer = Trainer(args, model)
#     val_dataloader = trainer.val_dataloader(seed=args.dataset_seed+1)
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

