@echo off
REM Stop execution on errors
setlocal enabledelayedexpansion

REM Set up Python (Make sure Python 3.9 is installed on the teammates system)
echo Checking Python version...
python --version || (
    echo Python is not installed. Please install Python 3.9.
    exit /b 1
)

REM Create and activate a virtual environment (optional but recommended)
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install pyinstaller Pillow PyQt5 sounddevice soundfile numpy cffi pycparser pandas matplotlib seaborn

REM Cleaning previous builds
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.spec del /q *.spec

REM Build the executable using PyInstaller
echo Building the executable...
pyinstaller --onefile --windowed ^
--hidden-import sounddevice ^
--hidden-import soundfile ^
--hidden-import pandas ^
--hidden-import matplotlib ^
--hidden-import seaborn ^
--hidden-import PyQt5 ^
--hidden-import PIL._imaging ^
--hidden-import PIL._avif ^
--add-data "src/ui;ui" ^
--add-data "src/training;training" ^
src/main.py

REM Deactivate the virtual environment
deactivate

echo Build completed. The executable is in the "dist" folder.
pause
