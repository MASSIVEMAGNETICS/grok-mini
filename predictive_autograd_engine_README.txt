# Predictive Autograd Engine (PAE) - Quick Start

## What is PAE?

A NumPy-only reverse-mode automatic differentiation engine designed for correctness, debuggability, and ease of use. Perfect for learning, prototyping, and small-scale ML experiments.

## Installation

No installation needed! Just copy `predictive_autograd_engine.py` to your project.

**Requirements:**
- Python 3.8+
- NumPy >= 1.21

## Quick Example

```python
from predictive_autograd_engine import Tensor, mse_loss

# Create tensors with requires_grad=True
x = Tensor([[1.0, 2.0]], requires_grad=True, name="x")
W = Tensor([[0.5], [0.3]], requires_grad=True, name="W")
b = Tensor([0.1], requires_grad=True, name="b")

# Forward pass
y_pred = x @ W + b
y_true = Tensor([[2.0]], requires_grad=False)

# Compute loss
loss = mse_loss(y_pred, y_true)

# Backward pass - compute gradients
loss.backward()

# Access gradients
print(f"W.grad: {W.grad}")
print(f"b.grad: {b.grad}")
```

## Key Features

✅ **Broadcasting-safe gradients** - Handles NumPy broadcasting correctly  
✅ **Stable numerics** - Softmax and cross-entropy use log-sum-exp tricks  
✅ **Numeric gradient checking** - Verify your gradients with finite differences  
✅ **Checkpoint save/load** - Serialize model weights easily  

## Supported Operations

- **Arithmetic**: `+`, `-`, `*`, `/`, `@` (matmul)
- **Activations**: `relu()`, `sigmoid()`, `tanh()`, `softmax()`
- **Reductions**: `sum()`, `mean()`
- **Shape ops**: `reshape()`, `transpose()`
- **Losses**: `mse_loss()`, `cross_entropy()`

## Run Self-Tests

```bash
python predictive_autograd_engine.py
```

Expected output: All tests PASSED ✓

## Next Steps

- Read the full manual: `predictive_autograd_engine.manual.md`
- Check out examples in `grok_mini_v3.py` and `runtime_v3.py`
- Build your own neural networks!

## Example: Training a Simple MLP

```python
from grok_mini_v3 import GrokMiniV3
from runtime_v3 import demo_train
import numpy as np

# Create model
model = GrokMiniV3(input_dim=4, hidden_dims=[8, 8], output_dim=1, seed=42)

# Synthetic dataset
X = np.random.randn(100, 4)
y = 0.5 * X.sum(axis=1, keepdims=True)

# Train
model = demo_train(model, X=X, y=y, epochs=10, lr=0.01)

# Predict
predictions = model.predict_numpy(X[:5])
print(predictions)
```

## Troubleshooting

**Q: My gradients are None!**  
A: Ensure `requires_grad=True` when creating tensors.

**Q: Getting NaN in gradients?**  
A: Check for numerical instability. Use stable losses like `cross_entropy()`.

**Q: Gradients don't match finite differences?**  
A: Use `AutogradEngine.numeric_gradients()` to debug.

## License

See repository license.
