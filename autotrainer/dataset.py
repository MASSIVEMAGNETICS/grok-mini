from typing import List, Optional
from torch.utils.data import Dataset
import torch

class TextLineDataset(Dataset):
    """
    Newline-delimited text dataset. Each line is a training example.
    Tokenizes with a provided tokenizer and pads/truncates to block_size.
    """
    def __init__(self, path: str, tokenizer, block_size: int, max_examples: Optional[int] = None):
        self.path = path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.lines = []
        with open(path, "r", encoding="utf-8") as f:
            for i, l in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                line = l.strip()
                if not line:
                    continue
                self.lines.append(line)
        if len(self.lines) == 0:
            raise ValueError(f"No training lines found in {path}")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]
        # tokenizer should return dict with input_ids
        enc = self.tokenizer(text, truncation=True, max_length=self.block_size, return_tensors=None)
        input_ids = enc["input_ids"]
        # ensure it's a list of ints
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        return torch.tensor(input_ids, dtype=torch.long)
