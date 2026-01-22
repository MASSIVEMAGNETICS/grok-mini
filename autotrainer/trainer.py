import os
import math
import time
from typing import Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from autotrainer.config import AutoTrainerConfig
from autotrainer.utils import set_seed, collate_fn, get_optimizer, save_checkpoint, load_checkpoint

class AutoTrainer:
    def __init__(self, model: torch.nn.Module, tokenizer, cfg: AutoTrainerConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model.to(self.device)
        self.optimizer = get_optimizer(self.model, cfg)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.mixed_precision and self.device.type == "cuda"))
        set_seed(cfg.seed)
        self.global_step = 0
        self.best_val_loss = float("inf")

    def _make_dataloader(self, dataset, shuffle=True):
        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=shuffle, collate_fn=lambda b: collate_fn(b, pad_token_id=self.tokenizer.pad_token_id))

    def train(self, train_dataset, valid_dataset=None):
        train_loader = self._make_dataloader(train_dataset, shuffle=True)
        if valid_dataset is not None:
            valid_loader = self._make_dataloader(valid_dataset, shuffle=False)
        else:
            valid_loader = None

        total_steps_est = (len(train_loader) * self.cfg.max_epochs) // max(1, self.cfg.gradient_accumulation_steps)
        print(f"Starting training for {self.cfg.max_epochs} epochs, ~{total_steps_est} optimization steps estimated.")

        self.model.train()
        for epoch in range(1, self.cfg.max_epochs + 1):
            epoch_loss = 0.0
            epoch_tokens = 0
            with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
                for batch in pbar:
                    input_ids, labels = [x.to(self.device) for x in batch]
                    # shift labels for causal LM is handled by using labels=input_ids (CrossEntropy expects targets as ids)
                    with torch.cuda.amp.autocast(enabled=(self.cfg.mixed_precision and self.device.type == "cuda")):
                        outputs = self.model(input_ids)
                        # model expected to return logits as first output (B, T, V)
                        if isinstance(outputs, tuple) or isinstance(outputs, list):
                            logits = outputs[0]
                        else:
                            logits = outputs
                        vocab_size = logits.size(-1)
                        # reshape
                        loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=self.tokenizer.pad_token_id)
                        loss = loss / self.cfg.gradient_accumulation_steps

                    self.scaler.scale(loss).backward()
                    epoch_loss += loss.item() * self.cfg.gradient_accumulation_steps
                    epoch_tokens += labels.numel()

                    if (self.global_step + 1) % self.cfg.gradient_accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()

                    self.global_step += 1
                    pbar.set_postfix({"loss": f"{(epoch_loss / max(1, self.global_step)):.4f}"})

                    # save & eval intervals
                    if self.global_step % self.cfg.save_interval_steps == 0:
                        self.save(self.global_step)
                    if valid_loader is not None and self.global_step % self.cfg.eval_interval_steps == 0:
                        val_loss = self.evaluate(valid_loader)
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save(self.global_step, tag="best")

            # end epoch
            print(f"Epoch {epoch} finished. avg loss: {epoch_loss / max(1, len(train_loader)):.4f}")

        # final save
        self.save(self.global_step, tag="final")

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids, labels = [x.to(self.device) for x in batch]
            outputs = self.model(input_ids)
            if isinstance(outputs, tuple) or isinstance(outputs, list):
                logits = outputs[0]
            else:
                logits = outputs
            vocab_size = logits.size(-1)
            loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=self.tokenizer.pad_token_id, reduction="sum")
            total_loss += loss.item()
            total_tokens += labels.numel()
        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
        print(f"Validation loss: {avg_loss:.6f}, ppl: {ppl:.3f}")
        self.model.train()
        return avg_loss

    def save(self, step: int, tag: Optional[str] = None):
        state = {
            "step": step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "tokenizer": None  # tokenizer is external; user can save separately if needed
        }
        if tag:
            name = f"checkpoint_{tag}_step_{step}.pt"
        else:
            name = f"checkpoint_step_{step}.pt"
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        path = os.path.join(self.cfg.output_dir, name)
        torch.save(state, path)
        print(f"Saved checkpoint: {path}")

    def load(self, path: str):
        ckpt = load_checkpoint(path, device=self.device.type)
        self.model.load_state_dict(ckpt["model_state"], strict=False)
        if "optimizer_state" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception:
                print("Warning: failed to restore optimizer state fully.")
        print(f"Loaded checkpoint from {path}")
