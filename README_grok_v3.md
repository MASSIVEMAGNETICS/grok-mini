# Grok Mini V3 - Enhanced Runtime

## Overview

Grok Mini V3 is an enhanced runtime built on the Predictive Autograd Engine (PAE). It provides a lightweight, NumPy-only implementation of neural network training and inference with safety-first autonomous capabilities.

## Components

### Core Engine
- **predictive_autograd_engine.py**: Production-grade NumPy autograd with broadcasting-safe gradients, stable numerics, and checkpoint management
- **grok_mini_v3.py**: MLP implementation using PAE Tensors with flexible architecture
- **runtime_v3.py**: Training runtime with SGD and Adam optimizers

### Agent System
- **self_agents.py**: Agent introspection and audit scaffolding for autonomous behaviors
  - Action approval gates
  - Full audit trails
  - Environment-based security controls

### SuperGrok Package
Optimized inference pipeline with:
- Sharded checkpoint loading
- Layer swapping for large models
- Optional quantization (8-bit/4-bit via bitsandbytes)
- Streaming generation

### AutoTrainer System
Automated training pipeline with:
- Text dataset handling
- Mixed precision training
- Checkpointing and resumption
- Validation and metrics

## Quick Start

### 1. Train a Simple Model

```python
from grok_mini_v3 import GrokMiniV3
from runtime_v3 import demo_train
import numpy as np

# Create model
model = GrokMiniV3(input_dim=8, hidden_dims=[16, 8], output_dim=1, seed=42)

# Train on synthetic data
model = demo_train(model, epochs=20, batch_size=32, lr=0.01)

# Make predictions
X_test = np.random.randn(5, 8)
predictions = model.predict_numpy(X_test)
print(predictions)
```

### 2. Save and Load Checkpoints

```python
from predictive_autograd_engine import save_state_dict, load_state_dict

# Save model
state = model.state_dict()
save_state_dict(state, "model_checkpoint.npz", metadata={"epoch": 20})

# Load model
loaded_state = load_state_dict("model_checkpoint.npz")
model.load_state_dict(loaded_state)
```

### 3. Use SuperGrok Inference

```python
from supergrok.inference import load_model, predict
import numpy as np

# Load model from checkpoint
model = load_model("model_checkpoint.npz", input_dim=8, hidden_dims=[16, 8])

# Predict
X = np.random.randn(10, 8)
predictions = predict(model, X)
```

## Architecture

### GrokMiniV3 Model

```
Input (input_dim)
  ↓
Linear + ReLU (hidden_dims[0])
  ↓
Linear + ReLU (hidden_dims[1])
  ↓
...
  ↓
Linear (output_dim)
  ↓
Output
```

**Features:**
- Xavier weight initialization
- Configurable hidden layer dimensions
- ReLU activations (customizable)
- Efficient NumPy operations

### Training Runtime

**Optimizers:**
- **SGD**: Stochastic Gradient Descent with momentum
- **Adam**: Adaptive moment estimation with weight decay

**Features:**
- Mini-batch training
- Automatic checkpoint saving
- Loss tracking
- Synthetic dataset generation

## Safety and Security

### Agent Audit System

All autonomous behaviors are gated by environment flags:

```bash
# Enable autonomous actions
export ALLOW_SELF_ACTIONS=true

# Enable network access
export ALLOW_NETWORK=true

# DANGEROUS: Auto-approve all actions
export AUTO_APPROVE=true  # Use with extreme caution!
```

**Default behavior:** All actions require manual operator approval.

### Audit Log

```python
from self_agents import get_auditor

auditor = get_auditor()
auditor.print_audit_summary()

# Get full audit log
actions = auditor.get_audit_log()
for action in actions:
    print(f"{action.timestamp}: {action.action_type} - {action.description}")
```

## SuperGrok Features

### Sharded Checkpoints

For large models that don't fit in memory:

```python
from supergrok.loader import create_shards_from_checkpoint, ShardedModelLoader

# Create shards
create_shards_from_checkpoint("large_model.pt", "shards/", num_shards=8)

# Load with lazy loading
loader = ShardedModelLoader("shards/", max_loaded=2)
param = loader.get_param("layer0.W")
```

### Layer Swapping

Keep only a subset of layers on GPU:

```python
from supergrok.loader import LayerSwapManager

# Assuming you have a PyTorch model
manager = LayerSwapManager(model, device="cuda", keep_on_gpu=4)
manager.register_hooks()

# Model will automatically swap layers during forward pass
```

### Quantization

Reduce model size with 8-bit quantization:

```python
from supergrok.quantize import quantize_checkpoint_simple

# Quantize checkpoint for storage
quantize_checkpoint_simple(
    "model.pt",
    "model_quantized.pt",
    bits=8
)
```

## AutoTrainer Integration

Train GrokMiniV2 models (from main repo) with automated pipeline:

```bash
python -m autotrainer.runtime \
  --train-file data/train.txt \
  --valid-file data/valid.txt \
  --batch-size 8 \
  --max-epochs 10 \
  --lr 1e-4 \
  --mixed-precision
```

## Performance Tips

1. **Use appropriate batch sizes**: Larger batches = faster training, more memory
2. **Adjust learning rate**: Start with 0.01 for SGD, 0.001 for Adam
3. **Enable momentum**: Helps escape local minima
4. **Save checkpoints regularly**: Every few epochs
5. **Monitor loss curves**: Should decrease steadily

## Testing

Run self-tests:

```bash
# PAE tests
python predictive_autograd_engine.py

# Agent system demo
python self_agents.py
```

## Dependencies

**Core (required):**
- NumPy >= 1.21

**Optional:**
- PyTorch (for SuperGrok, AutoTrainer)
- Transformers (for tokenizers)
- bitsandbytes (for quantization)
- tqdm (for progress bars)

## Limitations

- CPU-only for PAE (NumPy backend)
- Small to medium models (< 1B parameters)
- No distributed training
- Limited to supervised learning

For production workloads with large models, consider:
- PyTorch with CUDA
- DeepSpeed or FSDP
- Dedicated serving infrastructure

## License

See repository license.

## Contributing

Contributions welcome! Areas for improvement:
- Additional activation functions
- Convolutional layers
- Recurrent architectures
- Advanced optimizers (AdamW, LAMB)
- Distributed training support

---

**Built with ❤️ by Massive Magnetics**
