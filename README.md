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
   git clone https://github.com/yourusername/tone-training-app.git

2. Navigate to the project directory in your terminal or command prompt.
   ```
   cd tone-training-app
   ```

3. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```

4. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

5. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

1. Ensure your virtual environment is activated.

2. Run the main script:
   ```
   python src/main.py
   ```

3. The application window should appear. Follow the on-screen instructions to:
   - Enter a Participant ID
   - Select a Training Type
   - Choose an Audio Output Device
   - Load sound files
   - Start the training session

## Listing Audio Devices

To view all available audio devices on your system, you can use the provided script:

1. Ensure your virtual environment is activated.

2. Run the audio device listing script:
   ```
   python scripts/list_audio_devices.py
   ```

This will display a list of all audio devices recognized by your system, including their IDs, names, and host APIs.

Alternatively, you can use Python's interactive shell:

1. Open a Python interactive shell:
   ```
   python
   ```

2. Enter the following commands:
   ```python
   import sounddevice as sd
   devices = sd.query_devices()
   for i, dev in enumerate(devices):
       print(f"ID: {i}, Name: {dev['name']}, Host API: {dev['hostapi']}")
   ```

## Troubleshooting

### Virtual Environment Issues

<<<<<<< HEAD
If you encounter issues with the virtual environment, such as path errors or inability to run Python or pip, follow these steps to recreate your virtual environment:

1. **Remove the Existing Virtual Environment**:
   * Navigate to your project folder and delete the `venv/` folder.

2. **Create a New Virtual Environment**:
   * Open a terminal or command prompt in your project directory and run:
     ```
     python -m venv venv
     ```

3. **Activate the New Virtual Environment**:
   * On Windows:
     ```
     venv\Scripts\activate
     ```
   * On macOS and Linux:
     ```
     source venv/bin/activate
     ```

4. **Reinstall Project Dependencies**:
   * If you have a `requirements.txt` file:
     ```
     pip install -r requirements.txt
     ```

5. **Verify the Installation**:
   * Run the following command to list installed packages:
     ```
     pip list
     ```

6. **Run the Application**:
   * Try running the application again:
     ```
     python src/main.py
     ```

### Recreating the Virtual Environment

If you need to recreate the exact environment on another machine:

1. **Generate `requirements.txt`** (for project maintainers):
   * Activate the virtual environment and run:
     ```
     pip freeze > requirements.txt
     ```

2. **Install Dependencies from `requirements.txt`**:
   * After creating and activating a new virtual environment, run:
     ```
     pip install -r requirements.txt
     ```

### Additional Tips

* Ensure that your system Python installation is correctly set up and accessible via the command line.
* Always activate the virtual environment before working on the project or running commands.
* If you're using an IDE, make sure it's configured to use the correct Python interpreter from your virtual environment.

If you continue to experience issues after following these steps, please open an issue on the project's GitHub repository with details about your system configuration and the specific error you're encountering.
=======
If you continue to experience problems, please open an issue on the project's GitHub repository with details about your system configuration and the specific error you're encountering.
>>>>>>> 50563bceaff35b565c7e519268ae9094f1fd3faa
