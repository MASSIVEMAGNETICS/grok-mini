"""
Setup script for building Windows executable
Usage: python setup_windows.py
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    import importlib.util
    
    required = ['pyinstaller', 'torch', 'transformers', 'Pillow', 'requests']
    missing = []
    
    for package in required:
        # Handle package name differences
        package_name = 'PIL' if package == 'Pillow' else package
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            missing.append(package)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("\nInstalling missing dependencies...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
        print("Dependencies installed successfully!")

def build_executable():
    """Build Windows executable using PyInstaller"""
    print("\n" + "="*60)
    print("Building Grok-Mini Chat Windows Application")
    print("="*60 + "\n")
    
    # PyInstaller command
    cmd = [
        'pyinstaller',
        '--name=GrokMiniChat',
        '--onedir',  # Create a directory with all dependencies
        '--windowed',  # No console window
        '--add-data=grok_mini.py;.',
        '--add-data=requirements.txt;.',
        '--hidden-import=torch',
        '--hidden-import=transformers',
        '--hidden-import=PIL',
        '--hidden-import=requests',
        '--hidden-import=numpy',
        '--collect-all=torch',
        '--collect-all=transformers',
        '--noconfirm',
        'chat_app.py'
    ]
    
    print("Running PyInstaller...")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        subprocess.check_call(cmd)
        print("\n" + "="*60)
        print("Build successful!")
        print("="*60)
        print(f"\nExecutable location: dist/GrokMiniChat/GrokMiniChat.exe")
        print("\nYou can distribute the entire 'dist/GrokMiniChat' folder")
        print("Users can run GrokMiniChat.exe directly without Python installed")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError building executable: {e}")
        sys.exit(1)

def create_installer_script():
    """Create a simple batch installer"""
    installer_content = '''@echo off
echo ================================================
echo Grok-Mini Chat Installer
echo ================================================
echo.
echo Installing Grok-Mini Chat...
echo.

REM Copy files to Program Files
set INSTALL_DIR=%ProgramFiles%\\GrokMiniChat
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

echo Copying files...
xcopy /E /I /Y dist\\GrokMiniChat "%INSTALL_DIR%"

REM Create desktop shortcut
echo Creating desktop shortcut...
powershell "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%USERPROFILE%\\Desktop\\Grok-Mini Chat.lnk'); $s.TargetPath = '%INSTALL_DIR%\\GrokMiniChat.exe'; $s.Save()"

echo.
echo ================================================
echo Installation complete!
echo ================================================
echo.
echo You can now run Grok-Mini Chat from your desktop
echo or from: %INSTALL_DIR%\\GrokMiniChat.exe
echo.
pause
'''
    
    with open('install.bat', 'w') as f:
        f.write(installer_content)
    
    print("\nCreated install.bat for easy installation")

def main():
    print("Grok-Mini Chat - Windows Build Script")
    print("="*60 + "\n")
    
    # Check dependencies
    print("Checking dependencies...")
    check_dependencies()
    
    # Build executable
    build_executable()
    
    # Create installer
    create_installer_script()
    
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nTo distribute:")
    print("1. Share the 'dist/GrokMiniChat' folder with users")
    print("2. OR run 'install.bat' as Administrator to install system-wide")
    print("\nUsers can run GrokMiniChat.exe without Python installed")

if __name__ == "__main__":
    main()
