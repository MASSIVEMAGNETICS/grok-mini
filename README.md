# grok-mini

A production-ready implementation of Grok-Mini V2: an autonomous AI core with:
- Decoder-only MoE transformer architecture
- Vision integration with fractal multi-scale encoding
- Liquid MoE routing with trust-weighted expert selection
- Fractal attention across temporal hierarchies
- Autonomous tool execution (search, code, calculator)

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from grok_mini import GrokMiniV2, generate, config

# Initialize model
model = GrokMiniV2().to(config.device)

# Generate text
response = generate(model, "Explain quantum computing:", max_new_tokens=100)
print(response)
```

### Run Example

```bash
python example.py
```

## Architecture

- **Model**: Decoder-only transformer with MoE
- **Parameters**: ~400M (configurable)
- **Context Length**: 4096 tokens
- **Vision**: Multi-scale patch encoding (16x16, 32x32, 64x64)
- **MoE**: 16 experts with top-4 liquid routing
- **Attention**: Fractal multi-scale (3 scales)

## Files

- `grok_mini.py` - Main implementation
- `example.py` - Usage example
- `requirements.txt` - Dependencies
- `instruct.md` - Detailed architectural documentation

## Features

### Fractal Vision Encoder
Multi-scale patch embedding at 16x16, 32x32, and 64x64 resolutions for hierarchical visual understanding.

### Liquid MoE Routing
Trust-weighted expert selection with dynamic routing based on learned confidence scores.

### Fractal Attention
Multi-scale temporal attention that processes sequences at different granularities simultaneously.

### Autonomous Tool Execution
Self-routed tool calling for:
- Web search simulation
- Code execution (sandboxed)
- Calculator

## Advanced Usage

### With Vision

```python
from PIL import Image
import torch
import numpy as np

# Load image
img = Image.open("image.jpg").resize((224, 224))
img_array = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
img_tensor = torch.from_numpy(img_array).unsqueeze(0).to(config.device)

# Generate with vision
response = generate(model, "Describe this image:", image=img_tensor)
```

### Custom Configuration

```python
from grok_mini import GrokConfig

# Modify config before importing model
config = GrokConfig()
config.num_layers = 24
config.hidden_dim = 2048
config.moe_experts = 32
```

## Documentation

See [instruct.md](instruct.md) for detailed architectural documentation, including:
- Sovereign Hypercompression Forge (SHCF)
- Implementation details
- Training strategies
- Deployment considerations

## License

See repository license.
