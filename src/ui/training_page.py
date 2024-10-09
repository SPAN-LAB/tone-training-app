from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer
import sounddevice as sd
import soundfile as sf
import os

class TrainingPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.current_sound = None
        self.sounds = []
        self.participant_id = ""
        self.training_type = ""
        self.audio_device = None

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Prompt label
        self.prompt_label = QLabel("Listen to the sound")
        self.prompt_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prompt_label)

        # Play button
        self.play_button = QPushButton("Play Sound")
        self.play_button.clicked.connect(self.play_sound)
        layout.addWidget(self.play_button)

        # Response buttons
        response_layout = QHBoxLayout()
        self.response_buttons = []
        for i in range(1, 5):
            button = QPushButton(str(i))
            button.clicked.connect(lambda _, x=i: self.process_response(x))
            response_layout.addWidget(button)
            self.response_buttons.append(button)
        layout.addLayout(response_layout)

        # Feedback label
        self.feedback_label = QLabel("")
        self.feedback_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.feedback_label)

    def setup_training(self, participant_id, training_type, sounds, device_id):
        self.participant_id = participant_id
        self.training_type = training_type
        self.sounds = sounds
        self.audio_device_id = device_id
        self.next_sound()

    def next_sound(self):
        if self.sounds:
            self.current_sound = self.sounds.pop(0)
            self.prompt_label.setText("Click 'Play Sound' to listen")
            self.play_button.setEnabled(True)
            for button in self.response_buttons:
                button.setEnabled(False)
            self.feedback_label.clear()
        else:
            self.finish_training()

    def play_sound(self):
        if self.current_sound:
            try:
                full_path = os.path.abspath(self.current_sound)
                # Set the audio device
                sd.default.device = self.audio_device_id
                # Load and play the sound file
                data, fs = sf.read(full_path, dtype='float32')
                sd.play(data, fs, blocking=True)
                self.prompt_label.setText("Select the sound you heard")
                self.play_button.setEnabled(False)
                for button in self.response_buttons:
                    button.setEnabled(True)
            except Exception as e:
                print(f"Error playing sound: {e}")
                self.prompt_label.setText("Error playing sound")
        else:
            print("No sound loaded")

    def process_response(self, response):
        # TODO: Implement actual response processing
        correct_answer = 2  # This should be determined based on the current sound
        is_correct = response == correct_answer

        self.provide_feedback(is_correct, correct_answer)
        QTimer.singleShot(1000, self.next_sound)  # Move to next sound after 1 second

    def provide_feedback(self, is_correct, correct_answer):
        if self.training_type == "Perception with Minimal Feedback":
            self.feedback_label.setText("Correct" if is_correct else "Incorrect")
        elif self.training_type == "Perception with Full Feedback":
            self.feedback_label.setText(f"Correct" if is_correct else f"Incorrect. The correct answer was {correct_answer}")
        else:
            # TODO: Implement production training feedback
            pass

    def finish_training(self):
        print("Training finished")
        # TODO: Implement what happens when training is finished (e.g., show results, return to start page)