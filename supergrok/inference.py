"""
Inference utilities for GrokMiniV3.

- load_model(checkpoint_path=None, input_dim=8, **kwargs)
- predict(model, x: np.ndarray) -> np.ndarray
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import os
import sys
import logging

logger = logging.getLogger(__name__)

try:
    from grok_mini_v3 import GrokMiniV3
    from predictive_autograd_engine import load_state_dict
except Exception:
    # allow relative import when used as package
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    from grok_mini_v3 import GrokMiniV3
    from predictive_autograd_engine import load_state_dict

def load_model(checkpoint_path: Optional[str] = None, input_dim: int = 8, hidden_dims=None, output_dim: int = 1) -> GrokMiniV3:
    model = GrokMiniV3(input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        state = load_state_dict(checkpoint_path)
        model.load_state_dict(state)
        logger.info("Loaded checkpoint %s", checkpoint_path)
    return model

def predict(model: GrokMiniV3, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return model.predict_numpy(x)
