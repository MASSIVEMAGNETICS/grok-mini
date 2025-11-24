# grok-mini

A production-ready implementation of Grok-Mini V2: an autonomous AI core with:
- Decoder-only MoE transformer architecture
- Vision integration with fractal multi-scale encoding
- Liquid MoE routing with trust-weighted expert selection
- Fractal attention across temporal hierarchies
- Autonomous tool execution (search, code, calculator)
- **Windows GUI Chat Application** with ChatGPT-like interface

## 🖥️ Windows Chat Application

### For End Users

**Easy Installation (Windows 10/11):**

1. Download the release package
2. Double-click `launch_chat.bat` to auto-install and run
3. OR build executable: `python setup_windows.py`

**Features:**
- 🎨 Modern dark-themed chat interface
- 💬 ChatGPT-style conversation
- 🖼️ Image upload for vision questions
- ⚙️ Adjustable temperature and token controls
- 🚀 No Python required (when using built executable)

See [WINDOWS_INSTALL.md](WINDOWS_INSTALL.md) for detailed installation instructions.

### Quick Start (with Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Launch chat app
python chat_app.py
```

## 🐍 Python API

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

### Run CLI Example

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

- `chat_app.py` - Windows GUI chat application
- `grok_mini.py` - Core model implementation
- `example.py` - CLI usage example
- `recursive_tool_agent.py` - Recursive Tool-Graph Agent (RTGA)
- `setup_windows.py` - Build Windows executable
- `launch_chat.bat` - Windows launcher script
- `requirements.txt` - Python dependencies
- `instruct.md` - Detailed architectural documentation
- `WINDOWS_INSTALL.md` - Windows installation guide

## Features

### 🤖 Recursive Tool-Graph Agent (RTGA)
A self-improving agent substrate that generates Python tools on-the-fly using GPT-4, executes them, and persists them to a semantic graph for zero-shot retrieval.

**Key Features:**
- **Cognitive Layer**: GPT-4o powered code generation
- **Memory Layer**: NetworkX directed graph for semantic tool storage
- **Execution Layer**: Dynamic Python runtime compilation
- **Tool Lifecycle**: Generate → Compile → Execute → Store → Retrieve
- **Semantic Retrieval**: Zero-shot tool recall from graph memory

**Usage:**
```python
from recursive_tool_agent import RecursiveBuilder

# Initialize agent
bot = RecursiveBuilder()

# Generate and execute tools
bot.execute("Write a function to calculate the fibonacci sequence")
bot.execute("Create a function to generate a secure random password")

# Retrieve previously generated tools (no regeneration)
bot.execute("Run the calculate_fibonacci function")

# Visualize the tool graph
bot.visualize()
```

**Requirements:**
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)
- Dependencies: `openai`, `networkx`, `matplotlib`

### 🎨 Chat Application
- Modern, responsive UI with dark theme
- Real-time message streaming
- Temperature and token length controls
- Image upload for vision tasks
- Chat history management
- Keyboard shortcuts (Enter to send, Shift+Enter for newline)

### 🧠 Fractal Vision Encoder
Multi-scale patch embedding at 16x16, 32x32, and 64x64 resolutions for hierarchical visual understanding.

### 💧 Liquid MoE Routing
Trust-weighted expert selection with dynamic routing based on learned confidence scores.

### 🌊 Fractal Attention
Multi-scale temporal attention that processes sequences at different granularities simultaneously.

### 🔧 Autonomous Tool Execution
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

## Building Windows Executable

```bash
# Automatic build
python setup_windows.py

# Manual build
pyinstaller --name=GrokMiniChat --onedir --windowed chat_app.py
```

The executable will be in `dist/GrokMiniChat/` and can be distributed to users without Python.

## Documentation

See [instruct.md](instruct.md) for detailed architectural documentation, including:
- Sovereign Hypercompression Forge (SHCF)
- Implementation details
- Training strategies
- Deployment considerations

## System Requirements

### Minimum
- Windows 10 or higher (for GUI app)
- Python 3.8+ (for source code)
- 4 GB RAM
- 2 GB disk space

### Recommended
- Windows 11
- Python 3.10+
- 16 GB RAM
- NVIDIA GPU with 4GB+ VRAM

## License

See repository license.
