from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, QFileDialog, QSpacerItem, QSizePolicy, QMessageBox
from PyQt5.QtCore import pyqtSignal, Qt
import os, re, csv
import sounddevice as sd 
import soundfile as sf

"""For manual preset selection, search for comments '# Uncomment for manual preset selection' and uncomment code below"""
class StartPage(QWidget):
    # Signal emitted to start training, sending participant_id, training_type, sounds, output device id, input device id, session number, production recording path,
    # response file path, and  session tracking file path
    start_training_signal = pyqtSignal(str, 
                                       str, 
                                       list, 
                                       int, 
                                       int,
                                       int, 
                                       str, 
                                       str, 
                                       str, 
                                    #  int   # Uncomment for manual preset selection
                                       )

    # Signal emitted to start sound checking, sending input device id
    volume_check_signal = pyqtSignal(int)

    # Signal emitted to start fundamental range estimate, sending input device id and participant id
    range_est_signal = pyqtSignal(int, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
       
        self.sounds_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources', 'sounds')  # Directory where sounds are located by default
        self.sounds = []  # Initialize sounds list

        self.session_num = 1
        self.response_file_path = ""
        self.session_tracking_file_path = ""
        self.production_recording_path = ""
        self.selected_preset = 1

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
        self.training_type_combo.addItems(["Perception with Minimal Feedback", "Perception with Full Feedback", "Production Training"])
        self.training_type_combo.currentIndexChanged.connect(self.toggle_input_device_selection)
        type_layout.addWidget(self.training_type_label)
        type_layout.addWidget(self.training_type_combo)
        top_layout.addLayout(type_layout)

        # Audio device selection and test sound button
        device_layout = QHBoxLayout()
        self.audio_device_label = QLabel("Select Audio Output Device:")
        self.audio_device_combo = QComboBox()
        self.audio_test_button = QPushButton("Test Sound")
        self.populate_audio_devices()
        self.audio_test_button.clicked.connect(lambda: self.playSound())
        device_layout.addWidget(self.audio_device_label)
        device_layout.addWidget(self.audio_device_combo)
        device_layout.addWidget(self.audio_test_button)
        main_layout.addLayout(device_layout)
        main_layout.addLayout(top_layout)

        ### Section only shown for Production Training)
        # Audio input device selection 
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

        # Uncomment for manual preset selection
        # # Preset selection (1 for male, 2 for female)
        # preset_layout = QHBoxLayout()
        # self.preset_label = QLabel('Select preset value: ')
        # self.preset_combo = QComboBox()
        # self.preset_combo.addItems(["Preset 1", "Preset 2"])
        # self.preset_combo.currentIndexChanged.connect(self.update_selected_preset)
        # preset_layout.addWidget(self.preset_label)
        # preset_layout.addWidget(self.preset_combo)
        # main_layout.addLayout(preset_layout)
        # self.preset_label.hide()
        # self.preset_combo.hide()
        ### End of section


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
        self.start_button.clicked.connect(self.volume_check)  # Connects to method that emits signal
        button_layout.addWidget(self.start_button)
        main_layout.addLayout(button_layout)

    def update_selected_preset(self):
        self.selected_preset = self.preset_combo.currentIndex() + 1

    def toggle_input_device_selection(self):
        """Toggle input device visibility based on training type selection."""
        if self.training_type_combo.currentText() == "Production Training":
            self.audio_input_device_label.show()
            self.audio_input_device_combo.show()

            # Uncomment for manual preset selection
            # self.preset_label.show()
            # self.preset_combo.show()
        else:
            self.audio_input_device_label.hide()
            self.audio_input_device_combo.hide()
            # Uncomment for manual preset selection
            # self.preset_label.hide()
            # self.preset_combo.hide()
            
    def playSound(self):
        """ 
        Test sound function Grabs the selected output device and plays "test_sound.mp3"    
        """
        selected_output_index = self.audio_device_combo.currentIndex()
        sd.default.device = self.output_devices[selected_output_index][1] if selected_output_index >= 0 else -1
        # Read the sound file to determine its sample rate and number of channels
        data, fs = sf.read("test_sound.mp3", dtype="float32")
        # Set the audio device and play the sound with the correct number of channels
        sd.play(data, fs, blocking=True)

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
       
    def populate_input_devices(self):
        # Populate the audio input device selection dropdown
        devices = sd.query_devices()
        self.input_devices = []
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                device_info = f"{d['name']} - {d['hostapi']} (ID: {i})"
                self.input_devices.append((device_info, i))
        
        self.audio_input_device_combo.clear()
        self.audio_input_device_combo.addItems([info for info, _ in self.input_devices])

    def load_sounds(self):
        # Open a dialog to select the sound folder and load .mp3 files
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing Sound Files", self.sounds_dir)
        if folder:
            self.sounds = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.mp3')]
            sound_files = "\n".join(os.path.basename(s) for s in self.sounds)
            message = f"Selected folder: {folder}\n\nNumber of sound files: {len(self.sounds)}"
            QMessageBox.information(self, "Sounds Loaded", message)

    def volume_check(self):
        # Collect participant information, training type, and audio device IDs
        self.participant_id = self.participant_id_input.text()
        self.training_type = self.training_type_combo.currentText()
        selected_output_index = self.audio_device_combo.currentIndex()
        selected_input_index = self.audio_input_device_combo.currentIndex()

        # Get device_id for output and input
        self.output_device_id = self.output_devices[selected_output_index][1] if selected_output_index >= 0 else -1
        self.input_device_id = self.input_devices[selected_input_index][1] if selected_input_index >= 0 else -1
        
        # Create participants folder
        self.create_folders()

        # Emit signal to start training if all information is available, otherwise show a warning
        if not self.participant_id:
            QMessageBox.warning(self, "Missing Information", "Please enter a Participant ID.")
            return

        if not self.sounds:
            QMessageBox.warning(self, "Missing Information", "Please load sound files before starting.")
            return

        if self.output_device_id == -1:
            QMessageBox.warning(self, "Missing Information", "Please select an audio output device for training.")
            return

        # Emit signal to show the volume check page
        self.volume_check_signal.emit(self.input_device_id)

    def after_volume_check_complete(self):
        # emit signal to launch range estimate page for production training, else start training 
        if self.training_type == "Production Training":
            self.range_est_signal.emit(self.input_device_id, self.participant_id, self.production_recording_path)
        else:
            self.start_training()

    def start_training(self):

        self.start_training_signal.emit(
            self.participant_id,
            self.training_type,
            self.sounds,
            self.output_device_id,
            self.input_device_id,
            self.session_num,
            self.production_recording_path,
            self.response_file_path,
            self.session_tracking_file_path,
            # self.selected_preset      # Uncomment for manual preset selection
        )

    def create_folders(self):
        participant_id = self.participant_id
        training = self.training_type

        # Create participants folder if it doesn't exist
        participant_folder = os.path.join("participants", participant_id)
        os.makedirs(participant_folder, exist_ok=True)
        
        # Create training folder inside the participant's folder if it doesn't exist
        training_folder = os.path.join(participant_folder, training)
        os.makedirs(training_folder, exist_ok=True)

        # Create response file folder
        response_folder = os.path.join(training_folder, "response")
        folder_exist = os.path.exists(response_folder)
        os.makedirs(response_folder, exist_ok=True)

        # Obtain previous session number
        if folder_exist:
            session_numbers = [int(re.findall(r"\d+", file)[0]) for file in os.listdir(response_folder) if file.endswith(".csv")]
            self.session_num = max(session_numbers)
            self.session_num += 1 

        # Create response file
        self.response_file_path = os.path.join(response_folder, f"session{self.session_num}.csv")
        with open(self.response_file_path, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            if training != "Production Training":
                csv_writer.writerow(["date", "audio_file", "response", "solution", "reaction_time"])
            else:
                csv_writer.writerow(["date", "audio_file", "response", "solution", "accuracy", "reaction_time"])

        # Create session tracking folder
        session_tracking_folder = os.path.join(training_folder, "session_tracking")
        os.makedirs(session_tracking_folder, exist_ok=True)

        # Create session tracking file
        self.session_tracking_file_path = os.path.join(session_tracking_folder, f"session{self.session_num}.csv")
        with open(self.session_tracking_file_path, mode="w", newline="") as session_file:
            session_writer = csv.writer(session_file)
            session_writer.writerow(["date", "subject", "accuracy"])

        # Create production recording folder for production training
        if training == "Production Training":
            self.production_recording_path = os.path.join("participants", participant_id, "Production Recording", f"session{self.session_num}")
            os.makedirs(self.production_recording_path)