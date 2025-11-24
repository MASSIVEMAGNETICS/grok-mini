@echo off
REM Grok-Mini Chat Launcher for Windows
REM This launcher checks for Python and dependencies before running the app

echo ================================================
echo Grok-Mini Chat Launcher
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check if dependencies are installed
echo Checking dependencies...
python -c "import torch, transformers, PIL, requests" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Some dependencies are missing. Installing...
    echo This may take a few minutes...
    echo.
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install dependencies
        echo Please run: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo.
    echo Dependencies installed successfully!
    echo.
)

echo Starting Grok-Mini Chat...
echo.
echo ================================================
echo.

REM Run the chat application
python chat_app.py

if errorlevel 1 (
    echo.
    echo ================================================
    echo Application exited with an error
    echo ================================================
    pause
)
