"""
Sharded model loader + lazy layer device manager.

Key features:
- create_shards_from_checkpoint: split a large checkpoint into N shard files (simple hash by param name)
- ShardedModelLoader: keeps metadata about shards and can lazily load parameter tensors
- LayerSwapManager: keeps a sliding window of transformer blocks on GPU and swaps others to CPU
  by examining module names (works with GrokMiniV2 blocks list convention: model.blocks).
"""

import os
import torch
import math
import threading
from typing import Dict, Any, List, Optional
from collections import OrderedDict, deque

from supergrok.config import SuperGrokConfig

def create_shards_from_checkpoint(input_checkpoint: str, out_dir: str, num_shards: int = 4, prefix: str = "sg_shard"):
    """
    Split a large state_dict into `num_shards` files saved under out_dir/prefix_i.pt.
    Sharding strategy: assign param to shard by hash(name) % num_shards.
    """
    os.makedirs(out_dir, exist_ok=True)
    ckpt = torch.load(input_checkpoint, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    shards = [OrderedDict() for _ in range(num_shards)]
    for k, v in state.items():
        idx = (hash(k) % num_shards)
        shards[idx][k] = v
    paths = []
    for i, sd in enumerate(shards):
        path = os.path.join(out_dir, f"{prefix}_{i}.pt")
        torch.save({"model_state": sd}, path)
        paths.append(path)
    return paths

class ShardedModelLoader:
    """
    Manage loading of shards and lazy access to tensors.
    Note: stores loaded shard states in memory as dicts. For large shards consider memory-mapped safetensors.
    """
    def __init__(self, shard_dir: str, prefix: str = "sg_shard", max_loaded: int = 2):
        self.shard_dir = shard_dir
        self.prefix = prefix
        self.max_loaded = max_loaded
        self.shard_paths = sorted([os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.startswith(prefix)])
        self._loaded = {}  # index -> state_dict
        self._lru = deque()

    def _load_shard(self, idx: int):
        if idx in self._loaded:
            # update LRU
            if idx in self._lru:
                self._lru.remove(idx)
            self._lru.append(idx)
            return self._loaded[idx]
        path = self.shard_paths[idx]
        data = torch.load(path, map_location="cpu")
        st = data.get("model_state", data)
        self._loaded[idx] = st
        self._lru.append(idx)
        # enforce max loaded shards
        while len(self._loaded) > self.max_loaded:
            old = self._lru.popleft()
            if old in self._loaded:
                del self._loaded[old]
        return st

    def get_param(self, name: str) -> Optional[torch.Tensor]:
        """
        Return parameter tensor for given name by scanning shards (using same hash distribution).
        """
        # attempt compute shard idx by replicating create_shards_from_checkpoint's distribution
        # if we don't know num_shards, fallback to scanning all shard files (possible but slower)
        num_shards = len(self.shard_paths)
        if num_shards == 0:
            return None
        idx = hash(name) % num_shards
        st = self._load_shard(idx)
        return st.get(name)

    def list_all_param_names(self) -> List[str]:
        # cautious: scan all shards metadata
        out = []
        for i in range(len(self.shard_paths)):
            st = self._load_shard(i)
            out.extend(list(st.keys()))
        return out

class LayerSwapManager:
    """
    Swap transformer blocks (modules) between device and host to fit large models on limited GPU RAM.
    - Works best when model exposes an iterable of blocks (e.g., model.blocks)
    - Keeps a sliding window of `keep_on_gpu` contiguous blocks on device.
    - Swapping strategy: move required block to device before forward of that block; move least-used to CPU.
    """
    def __init__(self, model: torch.nn.Module, device: str = "cuda", keep_on_gpu: int = 4, lru_size: int = 8):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
        self.keep_on_gpu = keep_on_gpu
        self.lru = deque(maxlen=lru_size)
        # identify blocks
        self.blocks = getattr(self.model, "blocks", None)
        self._lock = threading.Lock()
        if self.blocks is None:
            # try other heuristics
            self.blocks = [m for m in model.modules() if m.__class__.__name__.lower().startswith("transformerblock") or "block" in m.__class__.__name__.lower()][:getattr(model, "num_layers", 0)]
        # mapping module -> index
        self.block_map = {i: b for i, b in enumerate(self.blocks)} if self.blocks else {}

    def ensure_block_on_device(self, idx: int):
        with self._lock:
            if idx not in self.block_map:
                return
            block = self.block_map[idx]
            # move parameters and buffers to device
            block.to(self.device)
            # update LRU
            if idx in self.lru:
                try:
                    self.lru.remove(idx)
                except Exception:
                    pass
            self.lru.append(idx)
            # evict if too many
            while len(self.lru) > self.keep_on_gpu:
                evict_idx = self.lru.popleft()
                if evict_idx == idx:
                    continue
                try:
                    evict_block = self.block_map[evict_idx]
                    evict_block.to("cpu")
                except Exception:
                    pass

    def register_hooks(self):
        """
        Register forward_pre_hooks on each block so that when it runs we ensure it's on device.
        """
        for i, block in self.block_map.items():
            def make_hook(index):
                def hook(module, input):
                    # Called in forward: ensure module on device
                    self.ensure_block_on_device(index)
                return hook
            try:
                block.register_forward_pre_hook(make_hook(i))
            except Exception:
                # some modules may not support hooks (e.g., builtins) — skip
                pass
