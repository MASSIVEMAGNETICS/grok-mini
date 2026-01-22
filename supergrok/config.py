from dataclasses import dataclass

@dataclass
class SuperGrokConfig:
    device: str = "cuda"  # "cuda" or "cpu"
    dtype: str = "float16"  # "float16", "float32"
    keep_layers_on_gpu: int = 4  # sliding window of transformer blocks retained on GPU
    max_loaded_shards: int = 2  # how many shard files kept memory-mapped / loaded (approx)
    shard_dir: str = "shards"  # directory where shards are stored
    shard_prefix: str = "sg_shard"  # prefix for shard files
    quant_bits: int = 8  # target quantization bits for storage (informational)
    prefetch: bool = True  # whether to prefetch next layers asynchronously
    prefetch_workers: int = 1
    lru_cache_size: int = 8  # number of modules/tensors to keep cached in GPU memory
    tokenizer_name: str = "gpt2"  # fallback tokenizer
    generation_chunk_tokens: int = 8  # chunk length for speculative generation/prefetch
    seed: int = 42
