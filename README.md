# Tone Training Application

This application is designed to help users practice and improve their ability to recognize and produce different tones. It supports various training modes and allows users to select their preferred audio output device.

## Features
- **Perception with Minimal Feedback**: Participants hear sounds and respond with minimal feedback on correctness.
- **Perception with Full Feedback**: Participants get full feedback on their responses, including whether they were right or wrong.
- **Production Training**: Participants replicate sounds and get feedback based on how accurately they reproduce them.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Running the Application](#running-the-application)
4. [Listing Audio Devices](#listing-audio-devices)
5. [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.7 or higher 
- pip (Python package installer)

## Installation

1. Clone this repository or download the source code.
   ```bash
   # Tone Training App

   Lightweight desktop app to practice tone perception and production (Mandarin-style tones).

   This repository contains a PyQt5 application that supports:
   - Perception training (minimal/full feedback)
   - Production training with pitch analysis and visual feedback

   ---

   ## Quick start

   1. Clone the repo:

   ```bash
   git clone https://github.com/your-org/tone-training-app.git
   cd tone-training-app
   ```

   2. Create and activate a virtual environment (recommended):

   Windows (PowerShell):

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   macOS / Linux:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

   4. Run the app:

   ```powershell
   python src/main.py
   ```

   ---

   ## Files of interest
   - `src/main.py` — application entry point
   - `src/ui/` — UI pages and windows (StartPage, TrainingPage, FeedbackPage, VolumeCheck)
   - `stratified_audio_files/` — example stimulus audio
   - `requirements.txt` — Python dependencies

   ---

   ## Production training: pitch analysis
   The app uses `praat-parselmouth` (Praat bindings) to extract pitch contours and `matplotlib` to plot them. For robustness the code converts files to a standard mono waveform before analysis.

   Notes:
   - If pitch analysis fails for an audio file we fall back gracefully and show a message to the user.
   - The UI shows a "Continue" button after analysis so participants can review their pitch contour vs. the original before moving on.

   ---

   ## Building a standalone exe (optional)
   This project contains a PyInstaller spec (`main.spec`) configured to include the `src` package. To build:

   ```powershell
   # from repository root
   pyinstaller main.spec
   ```

   If PyInstaller cannot find the `ui` package, run with an explicit path:

   ```powershell
   pyinstaller --paths=src src\main.py
   ```

   ---

   ## Troubleshooting
   - Use the virtual environment Python when running/ installing packages. Example path:
     `C:/SPANLAB/tone-training-app/venv/Scripts/python.exe`

   - Audio errors:
     - If playback fails, try a different audio device.
     - If pitch analysis fails on an `.mp3`, the code will attempt to convert/load it as a mono waveform. If it still fails, try converting the file to WAV.

   - Missing packages: ensure `praat-parselmouth` (PyPI name: `praat-parselmouth`) is installed in the active venv.

   ---

   ## Contributing
   - Open issues for bugs and feature requests.
   - Code style: small, focused PRs are easiest to review.

   ---

   If you'd like, I can also add a short CONTRIBUTING.md, or create a small smoke-test script that runs a single training session automatically for CI.
