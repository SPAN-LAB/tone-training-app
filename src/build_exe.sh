#!/bin/zsh

# --- Define your project name ---
APP_NAME="ToneTrainingApp"

# --- Activate your virtual environment ---
echo "Activating virtual environment..."
source ../venv/bin/activate

# --- Install Dependencies ---
echo "Installing dependencies..."
# Use pip3 to explicitly target Python 3 on macOS
pip3 install --upgrade pip
pip3 install -r ../requirements.txt
pip3 install pyinstaller PyQt5 sounddevice soundfile numpy pandas matplotlib seaborn scipy scikit-learn

# --- Clean Previous Builds ---
echo "Cleaning previous builds..."
rm -rf build dist

# --- Build the Executable using the .spec file ---
echo "Building the executable using the spec file..."
# Pointing to the .spec file ensures macOS microphone permissions are included
python3 -m PyInstaller "$APP_NAME.spec"

# --- Deactivate the Virtual Environment ---
deactivate

echo "\nBuild completed. The application bundle is in the 'dist' folder."