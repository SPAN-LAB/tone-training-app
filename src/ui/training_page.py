from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
import sounddevice as sd
import soundfile as sf
import os
import re

class TrainingPage(QWidget):
    # Signal emitted to end training and display results
    end_training_signal = pyqtSignal(str, str, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.current_sound = None
        self.sounds = []
        self.participant_id = ""
        self.training_type = ""
        self.audio_device = None
        self.correct_answers = 0
        self.total_questions = 0

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
        self.audio_device_id = (
            device_id  # Use device_id to select the correct audio device
        )
        self.correct_answers = 0
        self.total_questions = len(sounds)
        self.next_sound()  # Start with the first sound in the list

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
                # Construct the full path within resources/sounds and ensure .mp3 extension
                full_path = os.path.join(
                    "R:\\projects\\tone-training-app\\resources\\sounds",
                    self.current_sound,
                )
                if not full_path.endswith(".mp3"):
                    full_path += "_MP3.mp3"  # Append .mp3 extension if missing

                # Check if the file actually exists
                if not os.path.isfile(full_path):
                    raise FileNotFoundError(f"File not found: {full_path}")

                # Read the sound file to determine its sample rate and number of channels
                data, fs = sf.read(full_path, dtype="float32")

                # Set the audio device and play the sound with the correct number of channels
                sd.default.device = self.audio_device_id
                sd.play(data, fs, blocking=True)  # Specify channels to match the file

                # Update UI after playback
                self.prompt_label.setText("Select the sound you heard")
                self.play_button.setEnabled(False)
                for button in self.response_buttons:
                    button.setEnabled(True)

            except FileNotFoundError as fnf_error:
                print(f"Error: {fnf_error}")
                self.prompt_label.setText("Error: Sound file not found")
            except Exception as e:
                print(f"Error playing sound: {e}")
                self.prompt_label.setText("Error playing sound")
        else:
            print("No sound loaded")

    def process_response(self, response):
        correct_answer = int(re.findall("[0-9]+", self.current_sound)[0])
        is_correct = response == correct_answer
        if is_correct:
            self.correct_answers += 1
        self.provide_feedback(is_correct, correct_answer)
        QTimer.singleShot(1000, self.next_sound)  # Move to next sound after 1 second

    def provide_feedback(self, is_correct, correct_answer):
        if self.training_type == "Perception with Minimal Feedback":
            self.feedback_label.setText("Correct" if is_correct else "Incorrect")
        elif self.training_type == "Perception with Full Feedback":
            self.feedback_label.setText(
                f"Correct"
                if is_correct
                else f"Incorrect. The correct answer was {correct_answer}"
            )
        else:
            # TODO: Implement production training feedback
            pass

    def finish_training(self):
        score = (self.correct_answers / self.total_questions) * 100
        self.end_training_signal.emit(self.participant_id, self.training_type, score)
