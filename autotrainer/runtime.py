#!/usr/bin/env python3
"""
Autotrainer runtime.

Example:
python -m autotrainer.runtime --train-file data/train.txt --valid-file data/valid.txt --batch-size 2 --max-epochs 2
"""
import argparse
import os
from autotrainer.config import AutoTrainerConfig
from autotrainer.dataset import TextLineDataset
from autotrainer.trainer import AutoTrainer

def main():
    parser = argparse.ArgumentParser(description="Grok-mini AutoTrainer runtime")
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--valid-file", type=str, default="")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    # lazy import of model and tokenizer to avoid heavy import at top-level
    from grok_mini import GrokMiniV2, tokenizer as grok_tokenizer, config as grok_config

    cfg = AutoTrainerConfig(
        train_file=args.train_file,
        valid_file=args.valid_file or "",
        block_size=args.block_size,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        mixed_precision=args.mixed_precision,
    )
    if args.no_cuda:
        cfg.device = "cpu"

    # build model
    model = GrokMiniV2()  # assumes default init is appropriate; modify if you have pretrained weights
    tokenizer = grok_tokenizer

    # datasets
    train_ds = TextLineDataset(args.train_file, tokenizer, cfg.block_size)
    valid_ds = None
    if args.valid_file:
        valid_ds = TextLineDataset(args.valid_file, tokenizer, cfg.block_size)

    trainer = AutoTrainer(model, tokenizer, cfg)

    if cfg.resume_from:
        trainer.load(cfg.resume_from)

    trainer.train(train_ds, valid_ds)

if __name__ == "__main__":
    main()
