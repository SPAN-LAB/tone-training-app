from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, QFileDialog, QSpacerItem, QSizePolicy, QMessageBox
from PyQt5.QtCore import pyqtSignal, Qt, QThread
import os
import sounddevice as sd
import soundfile as sf
import sys
import numpy as np

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class PlayThread(QThread):
    """Play numpy audio data in a background thread to avoid blocking the GUI."""
    def __init__(self, data, samplerate, device=None, parent=None):
        super().__init__(parent)
        self.data = data
        self.samplerate = samplerate
        self.device = device

    def run(self):
        try:
            if self.device is not None:
                sd.play(self.data, self.samplerate, device=self.device)
            else:
                sd.play(self.data, self.samplerate)
            sd.wait()
        except Exception as e:
            # Keep errors non-fatal for the GUI thread; print for debugging
            print(f"Playback error in background thread: {e}")

class StartPage(QWidget):
    start_training_signal = pyqtSignal(str, str, list, int, int)
    volume_check_signal = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.sounds_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources', 'sounds')
        self.sounds = []

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)

        top_layout = QVBoxLayout()
        top_layout.setSpacing(20)

        id_layout = QHBoxLayout()
        self.participant_id_label = QLabel("Participant ID:")
        self.participant_id_input = QLineEdit()
        id_layout.addWidget(self.participant_id_label)
        id_layout.addWidget(self.participant_id_input)
        top_layout.addLayout(id_layout)

        type_layout = QHBoxLayout()
        self.training_type_label = QLabel("Select Training Type:")
        self.training_type_combo = QComboBox()
        self.training_type_combo.addItems(["Perception with Minimal Feedback", "Perception with Full Feedback", "Production Training"])
        self.training_type_combo.currentIndexChanged.connect(self.toggle_input_device_selection)
        type_layout.addWidget(self.training_type_label)
        type_layout.addWidget(self.training_type_combo)
        top_layout.addLayout(type_layout)

        device_layout = QHBoxLayout()
        self.audio_device_label = QLabel("Select Audio Output Device:")
        self.audio_device_combo = QComboBox()
        self.audio_test_button = QPushButton("Test Sound")
        self.populate_audio_devices()
        self.audio_test_button.clicked.connect(self.playSound)
        device_layout.addWidget(self.audio_device_label)
        device_layout.addWidget(self.audio_device_combo)
        device_layout.addWidget(self.audio_test_button)
        main_layout.addLayout(device_layout)
        
        input_device_layout = QHBoxLayout()
        self.audio_input_device_label = QLabel("Select Audio Input Device:")
        self.audio_input_device_combo = QComboBox()
        self.populate_input_devices()
        input_device_layout.addWidget(self.audio_input_device_label)
        input_device_layout.addWidget(self.audio_input_device_combo)
        
        main_layout.addLayout(input_device_layout)
        self.audio_input_device_label.hide()
        self.audio_input_device_combo.hide()

        main_layout.addLayout(top_layout)

        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)

        self.load_sounds_button = QPushButton("Load Sound Files")
        self.load_sounds_button.clicked.connect(self.load_sounds)
        button_layout.addWidget(self.load_sounds_button)

        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        button_layout.addWidget(self.start_button)

        main_layout.addLayout(button_layout)
        
    def toggle_input_device_selection(self):
        if self.training_type_combo.currentText() == "Production Training":
            self.audio_input_device_label.show()
            self.audio_input_device_combo.show()
        else:
            self.audio_input_device_label.hide()
            self.audio_input_device_combo.hide()
            
    def playSound(self):
        # CHANGED: Get the device ID directly from the selected item's data
        selected_output_id = self.audio_device_combo.currentData()
        
        # Check if a valid device is selected
        if selected_output_id is None:
            QMessageBox.warning(self, "No Device", "Please select an audio output device.")
            return

        samplerate = 44100
        frequency = 440.0
        duration = 1.0
        amplitude = 0.5

        t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
        data = amplitude * np.sin(2. * np.pi * frequency * t).astype(np.float32)

        # Play in background thread to avoid freezing the GUI
        try:
            self._play_thread = PlayThread(data, samplerate, device=selected_output_id)
            self._play_thread.start()
        except Exception as e:
            QMessageBox.warning(self, "Playback Error", f"Unable to play test sound: {e}")

    def populate_audio_devices(self):
        self.audio_device_combo.clear()
        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
        except Exception as e:
            QMessageBox.warning(self, "Audio Error", f"Could not query audio devices: {e}")
            return

        for i, d in enumerate(devices):
            try:
                if d.get('max_output_channels', 0) > 0:
                    hostapi_idx = d.get('hostapi', None)
                    hostapi_name = hostapis[hostapi_idx]['name'] if hostapi_idx is not None and hostapi_idx < len(hostapis) else str(hostapi_idx)
                    device_info = f"{d.get('name')} - {hostapi_name} (ID: {i})"
                    # Add the device ID 'i' as item data
                    self.audio_device_combo.addItem(device_info, i)
            except Exception:
                # Skip devices that cause problems
                continue
       
    def populate_input_devices(self):
        self.audio_input_device_combo.clear()
        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
        except Exception as e:
            QMessageBox.warning(self, "Audio Error", f"Could not query audio devices: {e}")
            return

        for i, d in enumerate(devices):
            try:
                if d.get('max_input_channels', 0) > 0:
                    hostapi_idx = d.get('hostapi', None)
                    hostapi_name = hostapis[hostapi_idx]['name'] if hostapi_idx is not None and hostapi_idx < len(hostapis) else str(hostapi_idx)
                    device_info = f"{d.get('name')} - {hostapi_name} (ID: {i})"
                    self.audio_input_device_combo.addItem(device_info, i)
            except Exception:
                continue

    def load_sounds(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing Sound Files", self.sounds_dir)
        if folder:
            self.sounds = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.mp3')]
            QMessageBox.information(self, "Sounds Loaded", f"Loaded {len(self.sounds)} sound files.")

    def start_training(self):
        self.participant_id = self.participant_id_input.text()
        self.training_type = self.training_type_combo.currentText()

        # CHANGED: Get device IDs directly from the combo box data, not from a separate list
        self.output_device_id = self.audio_device_combo.currentData()
        self.input_device_id = self.audio_input_device_combo.currentData()
        
        if not self.participant_id:
            QMessageBox.warning(self, "Missing Information", "Please enter a Participant ID.")
            return

        if not self.sounds:
            QMessageBox.warning(self, "Missing Information", "Please load sound files before starting.")
            return
            
        # CHANGED: Check if the retrieved data is None (meaning no selection)
        if self.output_device_id is None:
            QMessageBox.warning(self, "Missing Information", "Please select an audio output device for training.")
            return

        if self.training_type == "Production Training":
            if self.input_device_id is None:
                QMessageBox.warning(self, "Missing Information", "Please select an audio input device for production training.")
            else:
                self.volume_check_signal.emit(self.input_device_id)
        else:
            # For other training types, the input device ID can be None or an invalid index
            input_id = self.input_device_id if self.input_device_id is not None else -1
            self.start_training_signal.emit(self.participant_id, self.training_type, self.sounds, self.output_device_id, input_id)