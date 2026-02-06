@echo off
setlocal

REM --- Define your project name ---
set APP_NAME=ToneTrainingApp

REM --- Activate your virtual environment ---
echo Activating virtual environment...
call ..\venv\Scripts\activate

REM --- Install Dependencies ---
echo Installing dependencies...
pip install --upgrade pip
pip install -r ../requirements.txt
pip install pyinstaller PyQt5 sounddevice soundfile numpy pandas matplotlib seaborn scipy scikit-learn

REM --- Clean Previous Builds ---
echo Cleaning previous builds...
rmdir /s /q build dist 2>nul

REM --- Build the Executable using PyInstaller ---
echo Building the executable...
python -m PyInstaller "%APP_NAME%.spec"

REM --- Deactivate the Virtual Environment ---
call deactivate

echo.
echo Build completed. The executable is in the "dist" folder.
pause
