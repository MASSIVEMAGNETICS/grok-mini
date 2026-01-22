import os
import random
import torch
from typing import List
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from autotrainer.config import AutoTrainerConfig

def set_seed(seed: int):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collate_fn(batch: List[torch.Tensor], pad_token_id: int):
    """
    Pads a batch of 1D tensors (token ids) to the same length.
    Returns input_ids and labels (shifted version).
    """
    # pad to longest
    padded = pad_sequence(batch, batch_first=True, padding_value=pad_token_id)  # (B, T)
    input_ids = padded.clone()
    labels = padded.clone()  # For causal LM training, labels = input_ids
    return input_ids, labels

def get_optimizer(model: torch.nn.Module, cfg: AutoTrainerConfig):
    return AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay, eps=cfg.eps)

def save_checkpoint(state: dict, out_dir: str, step: int):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"checkpoint_step_{step}.pt")
    torch.save(state, path)
    return path

def load_checkpoint(path: str, device: str = "cpu"):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location=device)
