from dataclasses import dataclass

@dataclass
class AutoTrainerConfig:
    # data
    train_file: str = "data/train.txt"       # newline-delimited text
    valid_file: str = "data/valid.txt"       # optional validation file
    block_size: int = 1024                   # context length / sequence length
    # optimization
    lr: float = 5e-5
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_epochs: int = 3
    save_interval_steps: int = 1000
    eval_interval_steps: int = 1000
    # runtime
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    mixed_precision: bool = True
    seed: int = 42
    output_dir: str = "checkpoints"
    resume_from: str = ""  # path to checkpoint to resume
