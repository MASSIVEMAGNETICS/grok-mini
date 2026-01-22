"""
runtime_v3.py

Training runtime and optimizer implementations (NumPy-only).

Provides:
- Optimizer base, SGD, Adam
- demo_train(model, dataset=None, epochs=..., batch_size=..., lr=..., checkpoint_path=...)
"""

from __future__ import annotations
import time
import json
from typing import Dict, Iterable, Optional, Sequence, Tuple, List
import numpy as np

from predictive_autograd_engine import Tensor, mse_loss, AutogradEngine, save_state_dict, load_state_dict

class Optimizer:
    def __init__(self, params: List[Tensor]):
        self.params = params

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        AutogradEngine.zero_grad(self.params)

class SGD(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 1e-2, momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if not p.requires_grad:
                continue
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            if self.momentum:
                self.velocities[i] = self.momentum * self.velocities[i] + (1.0 - self.momentum) * grad
                update = self.velocities[i]
            else:
                update = grad
            p.data = p.data - self.lr * update

class Adam(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if not p.requires_grad:
                continue
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * (g ** 2)
            m_hat = self.m[i] / (1 - b1 ** self.t)
            v_hat = self.v[i] / (1 - b2 ** self.t)
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            p.data = p.data - self.lr * update

# ---------- simple deterministic dataloader ----------
def _minibatches(X: np.ndarray, y: np.ndarray, batch_size: int, seed: Optional[int] = None):
    N = X.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    for i in range(0, N, batch_size):
        idx = perm[i:i+batch_size]
        yield X[idx], y[idx]

# ---------- demo trainer ----------
def demo_train(model,
               X: Optional[np.ndarray] = None,
               y: Optional[np.ndarray] = None,
               epochs: int = 20,
               batch_size: int = 32,
               lr: float = 1e-2,
               optimizer_name: str = "sgd",
               seed: Optional[int] = None,
               checkpoint_path: Optional[str] = None,
               log_every: int = 1):
    if seed is not None:
        np.random.seed(seed)
    if X is None or y is None:
        # synthetic dataset: y = 0.5 * sum(features) + noise
        N = 512
        input_dim = model.layers[0].W.data.shape[0]
        X = np.random.randn(N, input_dim)
        y = (0.5 * X.sum(axis=1, keepdims=True) + 0.1 * np.random.randn(N, 1)).astype(float)

    params = model.params()
    if optimizer_name.lower() == "sgd":
        opt = SGD(params, lr=lr, momentum=0.9)
    else:
        opt = Adam(params, lr=lr)
    for epoch in range(epochs):
        t0 = time.time()
        total_loss = 0.0
        nb = 0
        for xb, yb in _minibatches(X, y, batch_size, seed=seed + epoch if seed is not None else None):
            xb_t = Tensor(xb, requires_grad=False)
            yb_t = Tensor(yb, requires_grad=False)
            preds = model.forward(xb_t)
            loss = mse_loss(preds, yb_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.data) * xb.shape[0]
            nb += xb.shape[0]
        avg = total_loss / (nb or 1)
        if (epoch + 1) % log_every == 0 or epoch < 3:
            print(f"Epoch {epoch+1}/{epochs} avg_loss={avg:.6f} time={(time.time()-t0):.3f}s")
        if checkpoint_path:
            state = model.state_dict()
            save_state_dict(state, checkpoint_path.format(epoch=epoch+1), metadata={"epoch": epoch+1, "avg_loss": avg})
    return model
