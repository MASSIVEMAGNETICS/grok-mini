"""
grok_mini_v3.py

A small, production-ready MLP using predictive_autograd_engine.Tensor.

API:
- GrokMiniV3(input_dim, hidden_dims=[64,32], output_dim=1, seed=None)
- forward(x: Tensor) -> Tensor
- params() -> list[Tensor]
- state_dict() -> dict[str, np.ndarray]
- load_state_dict(state: dict[str, np.ndarray])
- predict_numpy(x_np: np.ndarray) -> np.ndarray  (no-grad fast path)
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np

from predictive_autograd_engine import Tensor, _ensure_tensor

class Linear:
    def __init__(self, in_dim: int, out_dim: int, name: str, weight_init: str = "xavier", seed: Optional[int] = None):
        rng = np.random.RandomState(seed)
        if weight_init == "xavier":
            std = np.sqrt(2.0 / (in_dim + out_dim))
            W = rng.randn(in_dim, out_dim) * std
        else:
            W = rng.randn(in_dim, out_dim) * 0.01
        b = np.zeros((out_dim,), dtype=float)
        self.W = Tensor(W, requires_grad=True, name=f"{name}.W")
        self.b = Tensor(b, requires_grad=True, name=f"{name}.b")
        self.name = name

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b

    def params(self) -> List[Tensor]:
        return [self.W, self.b]

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {f"{self.name}.W": self.W.data.copy(), f"{self.name}.b": self.b.data.copy()}

    def load_state_dict(self, state: Dict[str, np.ndarray]):
        wkey = f"{self.name}.W"
        bkey = f"{self.name}.b"
        if wkey in state:
            self.W.data = np.asarray(state[wkey]).astype(float)
        if bkey in state:
            self.b.data = np.asarray(state[bkey]).astype(float)

class GrokMiniV3:
    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None, output_dim: int = 1, seed: Optional[int] = None):
        if hidden_dims is None:
            hidden_dims = [64, 32]
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        self.layers: List[Linear] = []
        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i+1], name=f"layer{i}", seed=(None if seed is None else seed + i)))
        self.activations = ['relu'] * (len(self.layers) - 1) + ['identity']

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer, act in zip(self.layers, self.activations):
            out = layer(out)
            if act == 'relu':
                out = out.relu()
        return out

    def params(self) -> List[Tensor]:
        ps = []
        for l in self.layers:
            ps.extend(l.params())
        return ps

    def state_dict(self) -> Dict[str, np.ndarray]:
        sd: Dict[str, np.ndarray] = {}
        for l in self.layers:
            sd.update(l.state_dict())
        return sd

    def load_state_dict(self, state: Dict[str, np.ndarray]):
        for l in self.layers:
            l.load_state_dict(state)

    def predict_numpy(self, x_np: np.ndarray) -> np.ndarray:
        x_np = np.asarray(x_np, dtype=float)
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        x = Tensor(x_np, requires_grad=False)
        y = self.forward(x)
        return y.data
