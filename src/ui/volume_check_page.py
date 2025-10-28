from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar
from PyQt5.QtCore import QTimer, pyqtSignal, Qt
import sounddevice as sd
import numpy as np

class VolumeCheckPage(QWidget):
    # emit signal when finish volume check
    volume_check_complete = pyqtSignal(bool)
    
    # NEW: Signal to go back to the start page
    back_to_start_signal = pyqtSignal()

    def __init__(self, input_device_id=None, parent=None):
        super().__init__(parent)
        self.threshold = -20  # dB threshold for volume level
        self.silence_threshold = -10  # dB level below which is considered silence
        self.input_device_id = input_device_id
        self.volume_level = -100  # Initial volume level
        
        # Initialize stream attribute
        self.stream = None 

        # Initialize UI
        self.setup_ui()

        # Timer for real-time volume monitoring
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_volume)
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Instruction label
        self.prompt_label = QLabel("Please speak into the microphone until the volume reaches the required threshold.")
        self.prompt_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prompt_label)

        # Progress bar to show volume level
        self.volume_bar = QProgressBar(self)
        self.volume_bar.setRange(0, 100)  # Volume bar as percentage
        layout.addWidget(self.volume_bar)

        # --- NEW: Button Layout ---
        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()

        # NEW: Back button
        self.back_button = QPushButton("Back to Setup")
        self.back_button.clicked.connect(self.go_back) # Connect to new method
        button_layout.addWidget(self.back_button)

        # Next button, initially disabled
        self.next_button = QPushButton("Start Training")
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self.complete_volume_check)
        button_layout.addWidget(self.next_button)
        
        # Add the button layout to the main vertical layout
        layout.addLayout(button_layout)
        # --- End of new layout ---

    def start_volume_check(self):
        # Stop and close any existing stream before starting a new one
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
        # Start audio input stream and timer
        try:
            self.stream = sd.InputStream(device=self.input_device_id, callback=self.audio_callback)
            self.stream.start()
            self.timer.start(100)  # Update every 100 ms
            
            # Reset UI
            self.prompt_label.setText("Please speak into the microphone until the volume reaches the required threshold.")
            self.volume_bar.setValue(0)
            self.next_button.setEnabled(False)
            
        except Exception as e:
            self.prompt_label.setText(f"Error starting microphone: {e}\nClick 'Back' to select a different device.")
            self.volume_bar.setValue(0)


    def audio_callback(self, indata, frames, time, status):
        # Calculate volume as RMS (Root Mean Square) and convert to dB
        volume_norm = np.linalg.norm(indata) * 10
        self.volume_level = 20 * np.log10(volume_norm) if volume_norm > 0 else -100

    def update_volume(self):
        # Check if the volume level is above the silence threshold
        if self.volume_level > self.silence_threshold:
            try:
                # Update progress bar only when volume exceeds silence threshold
                volume_percent = int((self.volume_level - self.silence_threshold) / (0 - self.silence_threshold) * 100)
                volume_percent = min(max(volume_percent, 0), 100)  # Ensure within 0-100 bounds
                self.volume_bar.setValue(volume_percent)
                
                # Check if speaking volume exceeds the threshold
                if self.volume_level >= self.threshold:
                    self.prompt_label.setText("Microphone volume is adequate.")
                    self.next_button.setEnabled(True)
                else:
                    self.prompt_label.setText("Please speak louder to reach the required threshold.")
            except (OverflowError, ValueError):
                self.volume_bar.setValue(0)
        else:
            # If below silence threshold, reset the prompt and bar
            self.prompt_label.setText("Please start speaking to test microphone volume.")
            self.volume_bar.setValue(0)  # Reset progress bar

    def stop_all_activity(self):
        """Helper function to stop timer and stream."""
        self.timer.stop()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    # NEW: Method to handle the "Back" button click
    def go_back(self):
        self.stop_all_activity()
        self.back_to_start_signal.emit()

    def complete_volume_check(self):
        self.stop_all_activity()
        self.volume_check_complete.emit(True)