@echo off
setlocal

REM --- Define your project name ---
set APP_NAME=ToneTrainingApp

REM --- Activate your virtual environment ---
echo Activating virtual environment...
call venv\Scripts\activate

REM --- Install Dependencies ---
echo Installing dependencies...
pip install --upgrade pip
pip install pyinstaller PyQt5 sounddevice soundfile numpy pandas matplotlib seaborn

REM --- Clean Previous Builds ---
echo Cleaning previous builds...
rmdir /s /q build dist 2>nul
del /q *.spec 2>nul

REM --- Build the Executable using PyInstaller ---
echo Building the executable...
pyinstaller --onefile --windowed ^
    --name %APP_NAME% ^
    --hidden-import="sounddevice" ^
    --hidden-import="soundfile" ^
    --hidden-import="pandas" ^
    --hidden-import="seaborn" ^
    main.py

REM --- Deactivate the Virtual Environment ---
deactivate

echo.
echo Build completed. The executable is in the "dist" folder.
pause