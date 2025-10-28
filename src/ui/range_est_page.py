from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QProgressBar
from PyQt5.QtCore import QTimer, pyqtSignal, Qt
import sounddevice as sd
import soundfile as sf
import numpy as np
import os 

# Citation for story: https://ririro.com/the-scorpion-and-the-tortoise/

class RangeEstPage(QWidget):
    # Signal emitted when range estimation is complete, sending success flag to proceed
    range_est_complete = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.input_device_id = None
        self.participant_id = None
        self.producton_recording_path = None

        # print("input device: ", input_device_id)
        # print("Part ID in range st page: ", participant_id)
        print("production recording path in range st page: ", self.producton_recording_path)

        # Initialize UI
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Instruction label
        self.prompt_label = QLabel("Please read the below passage out loud. ")
        self.prompt_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prompt_label)

        # Display the story passage
        self.passage = QLabel(
            """ 
            A scorpion and a tortoise became such fast friends that they took a vow that they would never separate. So when it happened that one of them was obliged to leave his native land, the other promised to go with him. They had traveled only a short distance when they came to a wide river. The scorpion was now greatly troubled. “Alas,” he said, “you, my friend, can easily swim, but how can a poor scorpion like me ever get across this stream?”  “Never fear,” replied the tortoise; “only place yourself squarely on my broad back and I will carry you safely over.” No sooner was the scorpion settled on the tortoise's broad back, than the tortoise crawled into the water and began to swim. Halfway across he was startled by a strange rapping on his back, which made him ask the scorpion what he was doing. “Doing?” answered the scorpion. “I am whetting my sting to see if it is possible to pierce your hard shell.” “Ungrateful friend,” responded the tortoise, “it is well that I have it in my power both to save myself and to punish you as you deserve.” And straightway he sank his back below the surface and shook off the scorpion into the water.
            """
            )
        self.passage.setWordWrap(True)
        self.passage.setMaximumWidth(600)
        layout.addWidget(self.passage, alignment=Qt.AlignHCenter)

        # Display recording progress
        self.recording_progress = QLabel("Not yet start recording")
        self.recording_progress.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.recording_progress) 

        # Start and stop recording buttons
        self.start_record_button = QPushButton("Push to record")
        self.start_record_button.clicked.connect(self.start_recording)

        self.stop_record_button = QPushButton("Push to stop")
        self.stop_record_button.clicked.connect(self.stop_recording)
        self.stop_record_button.setEnabled(False)

        layout.addWidget(self.start_record_button)
        layout.addWidget(self.stop_record_button)

        # Start training button, enable once finish recording
        self.start_training_button = QPushButton("Start Production Training")
        self.start_training_button.clicked.connect(self.complete_range_est)
        self.start_training_button.setEnabled(False)
        layout.addWidget(self.start_training_button)

    def start_recording(self):
        self.stream = sd.InputStream(device=self.input_device_id)
        sd.default.device = self.input_device_id
        
        # start recording for 45 seconds
        self.range_recording = sd.rec(int(45 * 44100), samplerate=44100, channels=1)

        self.recording_progress.setText("Recording in progress")

        # disable start recording button and enable stop recording button
        self.start_record_button.setEnabled(False)
        self.stop_record_button.setEnabled(True)

    def stop_recording(self):
        sd.stop()

        # save user recording
        range_est_file = os.path.join(self.production_recording_path, "range_est.wav")
        sf.write(range_est_file, self.range_recording, 44100)

        self.recording_progress.setText("Finish recording.")
        self.stop_record_button.setEnabled(False)
        self.start_training_button.setEnabled(True)

    def complete_range_est(self):
        # Stop audio stream and emit signal to proceed
        self.stream.stop()
        self.stream.close()
        self.range_est_complete.emit(True)