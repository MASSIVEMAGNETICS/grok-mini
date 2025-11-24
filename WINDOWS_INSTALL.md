# Windows Installation Guide

## For End Users (No Python Required)

### Method 1: Download Pre-built Application

1. Download the `GrokMiniChat` folder
2. Extract to any location (e.g., `C:\Program Files\GrokMiniChat`)
3. Double-click `GrokMiniChat.exe` to run
4. (Optional) Create a desktop shortcut to `GrokMiniChat.exe`

### Method 2: Run Installer (Recommended)

1. Download the release package
2. Right-click `install.bat` and select "Run as Administrator"
3. Follow the installation prompts
4. Launch from desktop shortcut or Start Menu

## For Developers (Building from Source)

### Prerequisites

- Python 3.8 or higher
- Windows 10 or higher
- Git (optional)

### Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Chat Application**
   ```bash
   python chat_app.py
   ```

### Building Windows Executable

1. **Install Build Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Build Executable**
   ```bash
   python setup_windows.py
   ```

3. **Distribute**
   - The executable will be in `dist/GrokMiniChat/`
   - Share the entire `GrokMiniChat` folder
   - Users can run `GrokMiniChat.exe` without Python

### Manual Build (Advanced)

If `setup_windows.py` doesn't work, you can build manually:

```bash
pyinstaller --name=GrokMiniChat --onedir --windowed ^
  --add-data=grok_mini.py;. ^
  --hidden-import=torch ^
  --hidden-import=transformers ^
  --collect-all=torch ^
  --collect-all=transformers ^
  chat_app.py
```

## Troubleshooting

### "Module not found" error
- Ensure all dependencies are installed: `pip install -r requirements.txt`

### "CUDA not available" warning
- The app will use CPU mode automatically
- For GPU support, install PyTorch with CUDA from pytorch.org

### Application is slow
- First run downloads the GPT-2 tokenizer (~500MB)
- Model initialization takes 30-60 seconds on first run
- Subsequent starts are faster

### Out of memory error
- Close other applications
- Reduce "Max Tokens" setting in the app
- Consider using a smaller model configuration

## Features

- **ChatGPT-like Interface**: Clean, modern chat UI
- **Temperature Control**: Adjust response creativity (0.1-1.5)
- **Token Limit**: Control response length (50-500 tokens)
- **Vision Support**: Upload images for visual question answering
- **Dark Theme**: Easy on the eyes
- **Chat History**: Track conversation within session
- **Keyboard Shortcuts**: 
  - Enter: Send message
  - Shift+Enter: New line
  - Ctrl+L: Clear chat (in future update)

## System Requirements

### Minimum
- Windows 10 or higher
- 4 GB RAM
- 2 GB disk space
- Intel Core i3 or equivalent

### Recommended
- Windows 11
- 16 GB RAM
- 5 GB disk space
- Intel Core i7 or equivalent
- NVIDIA GPU with 4GB+ VRAM (optional, for faster inference)

## Support

For issues, questions, or contributions, please visit the GitHub repository.
