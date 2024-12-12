from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QProgressBar
from PyQt5.QtCore import QTimer, pyqtSignal, Qt
import sounddevice as sd
import numpy as np

class VolumeCheckPage(QWidget):
    # Signal emitted when volume check is complete, sending success flag to proceed
    volume_check_complete = pyqtSignal(bool)

    def __init__(self, input_device_id=None, parent=None):
        super().__init__(parent)
        self.threshold = -20  # dB threshold for volume level
        self.silence_threshold = -10  # dB level below which is considered silence
        self.input_device_id = input_device_id
        self.volume_level = -100  # Initial volume level

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

        # Next button, initially disabled
        self.next_button = QPushButton("Start Training")
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self.complete_volume_check)
        layout.addWidget(self.next_button)

    def start_volume_check(self):
        # Start audio input stream and timer
        self.stream = sd.InputStream(device=self.input_device_id, callback=self.audio_callback)
        self.stream.start()
        self.timer.start(100)  # Update every 100 ms

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

    def complete_volume_check(self):
        # Stop audio stream and emit signal to proceed
        self.timer.stop()
        self.stream.stop()
        self.stream.close()
        self.volume_check_complete.emit(True)