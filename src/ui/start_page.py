from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, QFileDialog, QSpacerItem, QSizePolicy, QMessageBox
from PyQt5.QtCore import pyqtSignal, Qt, QThread
import os, re, csv
import sounddevice as sd 
import soundfile as sf
import numpy as np # <-- Added numpy import
import sys # <-- Added sys import

# --- NEW: Added PlayThread class from original file ---
# This plays audio in the background to avoid blocking the GUI.
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

# (We also need the resource_path function for this file)
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

"""For manual preset selection, search for comments '# Uncomment for manual preset selection' and uncomment code below"""
class StartPage(QWidget):
    # (Signals are unchanged)
    start_training_signal = pyqtSignal(str, 
                                       str, 
                                       list,
                                       list,
                                       int, 
                                       int,
                                       int, 
                                       str, 
                                       str, 
                                       str, 
                                       int,
                                    #  int   # Uncomment for manual preset selection
                                       )
    volume_check_signal = pyqtSignal(int)
    range_est_signal = pyqtSignal(int, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
       
        self.sounds_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources', 'sounds')
        self.sounds = []
        self.generalization_sounds = []

        self.session_num = 1
        self.response_file_path = ""
        self.session_tracking_file_path = ""
        self.production_recording_path = ""
        self.selected_preset = 1
        self.selected_gender = 0 # Default to Male(0)
        
        # Attribute to hold the play thread
        self._play_thread = None

    def setup_ui(self):
        # (This section is mostly the same)
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
        
        # --- Gender Selection ---
        gender_layout = QHBoxLayout()
        self.gender_label = QLabel("Select Speaker Gender:")
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["Male", "Female"]) # M=0, F=1 based on training script
        self.gender_combo.currentIndexChanged.connect(self.update_selected_gender)
        gender_layout.addWidget(self.gender_label)
        gender_layout.addWidget(self.gender_combo)
        main_layout.addLayout(gender_layout)
        self.gender_label.hide() # Initially hidden
        self.gender_combo.hide() # Initially hidden
        
        # main_layout.addLayout(top_layout) 
        main_layout.addLayout(top_layout) # Keep one

        # (Manual preset section remains commented out)
        
        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)
        self.load_sounds_button = QPushButton("Load Training Sound Files")
        self.load_sounds_button.clicked.connect(self.load_sounds)
        button_layout.addWidget(self.load_sounds_button)
        self.load_gen_sounds_button = QPushButton("Load Generalization Sound Files")
        self.load_gen_sounds_button.clicked.connect(self.load_generalization_sounds)
        button_layout.addWidget(self.load_gen_sounds_button)
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.volume_check)
        button_layout.addWidget(self.start_button)
        main_layout.addLayout(button_layout)

    def update_selected_preset(self):
        self.selected_preset = self.preset_combo.currentIndex() + 1
        
    # --- Method to update selected gender ---
    def update_selected_gender(self):
        # Maps "Male" (index 0) to 0, "Female" (index 1) to 1
        self.selected_gender = self.gender_combo.currentIndex()

    def toggle_input_device_selection(self):
        """Toggle input device visibility based on training type selection."""
        if self.training_type_combo.currentText() == "Production Training":
            self.audio_input_device_label.show()
            self.audio_input_device_combo.show()
            
            is_production = self.training_type_combo.currentText() == "Production Training"
            self.audio_input_device_label.setVisible(is_production)
            self.audio_input_device_combo.setVisible(is_production)
            self.gender_label.setVisible(is_production) # Show/hide gender label
            self.gender_combo.setVisible(is_production)
            # (preset labels remain commented out)
        else:
            self.audio_input_device_label.hide()
            self.audio_input_device_combo.hide()

    def playSound(self):
        """ 
        Test sound function. Generates a 1-second 440Hz sine wave
        and plays it on the selected output device in a background thread.
        """
        # Get the device ID directly from the selected item's data
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
            # Stop previous thread if it's still running
            if self._play_thread and self._play_thread.isRunning():
                self._play_thread.terminate() # Force stop
                
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
        # (This method is unchanged, but now uses 'self.sounds' correctly)
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing Sound Files", self.sounds_dir)
        if folder:
            self.sounds = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.mp3')]
            QMessageBox.information(self, "Sounds Loaded", f"Loaded {len(self.sounds)} sound files.")

    def volume_check(self):
        self.participant_id = self.participant_id_input.text()
        self.training_type = self.training_type_combo.currentText()

        # Get device_id for output and input from the combo box data
        self.output_device_id = self.audio_device_combo.currentData()
        self.input_device_id = self.audio_input_device_combo.currentData()
        
        # Create participants folder
        self.create_folders()

        if not self.participant_id:
            QMessageBox.warning(self, "Missing Information", "Please enter a Participant ID.")
            return

        if not self.sounds:
            QMessageBox.warning(self, "Missing Information", "Please load sound files before starting.")
            return

        # Check if the retrieved data is None (meaning no selection)
        if self.output_device_id is None:
            QMessageBox.warning(self, "Missing Information", "Please select an audio output device for training.")
            return
            
        # Check for input device ONLY if production training is selected
        if self.training_type == "Production Training" and self.input_device_id is None:
            QMessageBox.warning(self, "Missing Information", "Please select an audio input device for production training.")
            return

        # Emit signal to show the volume check page
        # We pass the input_device_id, which might be None if not production training,
        # but volume_check_page is only triggered if it's not None.
        self.volume_check_signal.emit(self.input_device_id)

    def load_generalization_sounds(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing Generalization Sound Files", self.sounds_dir)
        if folder:
            self.generalization_sounds = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.mp3')]
            QMessageBox.information(self, "Sounds Loaded", f"Loaded {len(self.generalization_sounds)} generalization sound files.")

    def after_volume_check_complete(self):
        # (Unchanged)
        if self.training_type == "Production Training":
            self.range_est_signal.emit(self.input_device_id, self.participant_id, self.production_recording_path)
        else:
            self.start_training()

    def start_training(self):
        # (Unchanged)
        self.start_training_signal.emit(
            self.participant_id,
            self.training_type,
            self.sounds,
            self.generalization_sounds,
            self.output_device_id,
            self.input_device_id,
            self.session_num,
            self.production_recording_path,
            self.response_file_path,
            self.session_tracking_file_path,
            self.selected_gender,
            # self.selected_preset
        )

    def create_folders(self):
        # (Unchanged)
        participant_id = self.participant_id
        training = self.training_type
        participant_folder = os.path.join("participants", participant_id)
        os.makedirs(participant_folder, exist_ok=True)
        training_folder = os.path.join(participant_folder, training)
        os.makedirs(training_folder, exist_ok=True)
        response_folder = os.path.join(training_folder, "response")
        folder_exist = os.path.exists(response_folder)
        os.makedirs(response_folder, exist_ok=True)

        if folder_exist:
            session_numbers = [int(re.findall(r"\d+", file)[0]) for file in os.listdir(response_folder) if file.endswith(".csv")]
            if session_numbers: # Check if list is not empty
                self.session_num = max(session_numbers)
                self.session_num += 1
            else:
                self.session_num = 1 # Default to 1 if no files found
        
        self.response_file_path = os.path.join(response_folder, f"session{self.session_num}.csv")
        with open(self.response_file_path, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            if training != "Production Training":
                csv_writer.writerow(["date", "audio_file", "response", "solution", "reaction_time", "block_type"])
            else:
                csv_writer.writerow(["date", "audio_file", "response", "solution", "accuracy", "reaction_time", "block_type"])
        
        session_tracking_folder = os.path.join(training_folder, "session_tracking")
        os.makedirs(session_tracking_folder, exist_ok=True)
        
        self.session_tracking_file_path = os.path.join(session_tracking_folder, f"session{self.session_num}.csv")
        with open(self.session_tracking_file_path, mode="w", newline="") as session_file:
            session_writer = csv.writer(session_file)
            session_writer.writerow(["date", "subject", "accuracy"])
        
        if training == "Production Training":
            self.production_recording_path = os.path.join("participants", participant_id, "Production Recording", f"session{self.session_num}")
            os.makedirs(self.production_recording_path, exist_ok=True) # Added exist_ok=True