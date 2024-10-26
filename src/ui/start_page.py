from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, QFileDialog, QSpacerItem, QSizePolicy, QMessageBox
from PyQt5.QtCore import pyqtSignal, Qt
import os
import sounddevice as sd 

class StartPage(QWidget):
    # Signal emitted to start training, sending participant_id, training_type, sounds, and device_id
    start_training_signal = pyqtSignal(str, str, list, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        # Directory where sounds are located by default
        self.sounds_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources', 'sounds')
        self.sounds = []  # Initialize sounds list

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)  # Add some margin around the edges

        # Top section (Participant ID and Training Type)
        top_layout = QVBoxLayout()
        top_layout.setSpacing(20)

        # Participant ID input
        id_layout = QHBoxLayout()
        self.participant_id_label = QLabel("Participant ID:")
        self.participant_id_input = QLineEdit()
        id_layout.addWidget(self.participant_id_label)
        id_layout.addWidget(self.participant_id_input)
        top_layout.addLayout(id_layout)

        # Training type selection
        type_layout = QHBoxLayout()
        self.training_type_label = QLabel("Select Training Type:")
        self.training_type_combo = QComboBox()
        self.training_type_combo.addItems(["Production Training", "Perception with Minimal Feedback", "Perception with Full Feedback"])
        type_layout.addWidget(self.training_type_label)
        type_layout.addWidget(self.training_type_combo)
        top_layout.addLayout(type_layout)

        # Audio device selection
        device_layout = QHBoxLayout()
        self.audio_device_label = QLabel("Select Audio Device:")
        self.audio_device_combo = QComboBox()
        self.populate_audio_devices()
        device_layout.addWidget(self.audio_device_label)
        device_layout.addWidget(self.audio_device_combo)
        main_layout.addLayout(device_layout)
        main_layout.addLayout(top_layout)

        # Spacer to push buttons to the bottom
        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Button section at the bottom
        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)

        # Sound file selection
        self.load_sounds_button = QPushButton("Load Sound Files")
        self.load_sounds_button.clicked.connect(self.load_sounds)
        button_layout.addWidget(self.load_sounds_button)

        # Start button
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)  # Connects to method that emits signal
        button_layout.addWidget(self.start_button)

        main_layout.addLayout(button_layout)

    def populate_audio_devices(self):
        # Populate the audio device selection dropdown
        devices = sd.query_devices()
        self.output_devices = []
        for i, d in enumerate(devices):
            if d['max_output_channels'] > 0:
                device_info = f"{d['name']} - {d['hostapi']} (ID: {i})"
                self.output_devices.append((device_info, i))
        
        self.audio_device_combo.clear()
        self.audio_device_combo.addItems([info for info, _ in self.output_devices])
       
    def load_sounds(self):
        # Open a dialog to select the sound folder and load .wav files
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing Sound Files", self.sounds_dir)
        if folder:
            self.sounds = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.mp3')]
            sound_files = "\n".join(os.path.basename(s) for s in self.sounds)
            message = f"Selected folder: {folder}\n\nNumber of sound files: {len(self.sounds)}"
            QMessageBox.information(self, "Sounds Loaded", message)

    def start_training(self):
        # Collect participant information, training type, and audio device ID
        participant_id = self.participant_id_input.text()
        training_type = self.training_type_combo.currentText()
        selected_device_index = self.audio_device_combo.currentIndex()

        # Get device_id from selected audio device
        if selected_device_index >= 0:
            _, device_id = self.output_devices[selected_device_index]
        else:
            device_id = -1  # Sentinel value for no device selected

        # Emit signal if all information is available, otherwise show a warning
        if participant_id and self.sounds and device_id != -1:
            self.start_training_signal.emit(participant_id, training_type, self.sounds, device_id)
        else:
            QMessageBox.warning(self, "Missing Information", "Please enter a Participant ID, load sound files, and select an audio device before starting.")
