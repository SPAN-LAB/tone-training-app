from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
import sounddevice as sd
import soundfile as sf
import os
import re
import datetime
import time
from .volume_check_page import VolumeCheckPage
import csv
class TrainingPage(QWidget):
    # Signal emitted to end training and display results
    end_training_signal = pyqtSignal(str, str, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_sound = None
        self.sounds = []
        self.participant_id = ""
        self.training_type = ""
        self.audio_device_id = None
        self.input_device_id = None  
        self.correct_answers = 0
        self.total_questions = 0
        self.is_recording = False  
        self.recorded_audio_path = ""  # Temporary storage for users' recordings production training
        self.response_buttons = None
        self.start_time = None
        self.production_accuracy = 0

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Prompt label
        self.prompt_label = QLabel("Listen to the sound")
        self.prompt_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prompt_label)

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

    def setup_production_training(self):
        """Setup UI for Production Training"""
        layout = QVBoxLayout(self)

        # Prompt label with production-specific instructions
        self.prompt_label = QLabel("Listen to the sound, then reproduce it.")
        self.prompt_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prompt_label)

        # Record button
        # self.record_button = QPushButton("Start Recording")
        # self.record_button.clicked.connect(self.toggle_recording)
        # layout.addWidget(self.record_button)

        # Visualization label
        self.visualization_label = QLabel("Visual feedback will be displayed here.")
        self.visualization_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.visualization_label)

        # Feedback label for text feedback on reproduction accuracy
        self.feedback_label = QLabel("")
        self.feedback_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.feedback_label)

    def setup_training(self, participant_id, training_type, sounds, device_id, input_device_id=None):
        self.participant_id = participant_id
        self.training_type = training_type
        self.sounds = sounds
        self.audio_device_id = device_id  
        self.input_device_id = input_device_id  
        self.correct_answers = 0
        self.total_questions = len(sounds)
        
        if training_type == "Production Training":
            self.setup_production_training()
        else:
            self.setup_ui()

        print("In setup_training() in training_page.py")
        print("After assign current sound: ", self.current_sound, self.sounds)
        QTimer.singleShot(1000, self.play_sound)  

    def play_sound(self):
        print("In play sound()")
        print("Remaining sound file: ", [f for f in self.sounds])
        print("Current sound: ", self.current_sound)

        if self.sounds:
            self.current_sound = self.sounds.pop(0)

            try:    
                # if self.training_type == "Production Training":
                #     self.record_button.setEnabled(False)  # Enable after playback

                if self.response_buttons is not None:
                    for button in self.response_buttons:
                        button.setEnabled(False)
                    self.feedback_label.clear()

                # Construct the full path within resources/sounds and ensure .mp3 extension
                full_path = os.path.join(
                    "R:\\projects\\tone-training-app\\resources\\sounds",
                    self.current_sound,
                )

                # Append .mp3 extension if missing
                if not full_path.endswith(".mp3"):
                    full_path += "_MP3.mp3"  

                # Check if the file actually exists
                if not os.path.isfile(full_path):
                    raise FileNotFoundError(f"File not found: {full_path}")

                # Read the sound file to determine its sample rate and number of channels
                data, fs = sf.read(full_path, dtype="float32")

                # Set the audio device and play the sound with the correct number of channels
                sd.default.device = self.audio_device_id
                sd.play(data, fs, blocking=True)  

                # Get reaction starting time
                self.start_time = time.time()

                # Update UI after playback
                if self.training_type == "Production Training":
                    self.prompt_label.setText("Try to reproduce the sound")
                    print("In play_sound(), within if stmt for production training")
                    self.toggle_recording()
                    # self.prompt_label.setText("Try to reproduce the sound and press 'Start Recording'")
                    # self.record_button.setEnabled(True)
                else:
                    self.prompt_label.setText("Select the sound you heard")
                    # self.play_button.setEnabled(False)
                    if self.response_buttons is not None:
                        for button in self.response_buttons:
                            button.setEnabled(True)

            except FileNotFoundError as fnf_error:
                print(f"Error: {fnf_error}")
                self.prompt_label.setText("Error: Sound file not found")
            except Exception as e:
                print(f"Error playing sound: {e}")
                self.prompt_label.setText("Error playing sound")
        else:
            self.finish_training()

    def toggle_recording(self):
        # Start or stop recording based on current state
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        # self.record_button.setText("Stop Recording")
        self.prompt_label.setText("Recording... Try to match the original sound")

        # Create session_recordings folder if it doesn't exist
        participant_folder = os.path.join("participant_recordings", self.participant_id)
        os.makedirs(participant_folder, exist_ok=True)

        date = datetime.date.today()
        file = self.current_sound.split("/")[-1]
        file = file.split(".")[0]
        self.recorded_audio_path = f"participant_recordings/{self.participant_id}/{date}_{file}.wav"

        # Set default device
        sd.default.device = (self.input_device_id, self.audio_device_id)  

        # Start recording and countdown timer
        self.recording = sd.rec(int(3 * 44100), samplerate=44100, channels=1)
        self.start_countdown(5)

    def start_countdown(self, seconds):
        self.remaining_time = seconds
        self.prompt_label.setText(f"Recording... {self.remaining_time} seconds remaining")

        # Create a timer that triggers every 1 second
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_timer.start(1000)
    
    # TODO: make start_countdown() and update_countdown() more general  
    def update_countdown(self):
        self.remaining_time -= 1
        if self.remaining_time > 0:
            self.prompt_label.setText(f"Recording... {self.remaining_time} seconds remaining")
        else:
            self.countdown_timer.stop()  # Stop the timer
            self.stop_recording() 

    def stop_recording(self):
        self.is_recording = False
        # self.record_button.setText("Start Recording")
        sd.stop()

        end_time = time.time()
        reaction_time = end_time - self.start_time if self.start_time else 0

        sf.write(self.recorded_audio_path, self.recording, 44100)
        self.prompt_label.setText("Recording complete. Analyzing...")

        # write to response file
        self.write_response(self.participant_id, self.training_type, self.current_sound.split("/")[-1], 
                            reaction_time, accuracy=self.production_accuracy)

        self.analyze_recording()
        # TODO: Delete recording after analysis 

    def analyze_recording(self):

        # Placeholder: Implement pitch comparison and feedback display
        self.visualization_label.setText("Comparing original and recorded pitch tracks...")

        # TODO: Implement accuracy calculation
        
        # TODO: Display actual pitch track visualization and compute similarity
        self.provide_feedback()


    def process_response(self, response):

        end_time = time.time()
        reaction_time = end_time - self.start_time if self.start_time else 0

        correct_answer = int(re.findall("[0-9]+", self.current_sound)[0])
        is_correct = response == correct_answer
        if is_correct:
            self.correct_answers += 1

        # display feedback on screen 
        self.provide_feedback(is_correct, correct_answer)

        # write to response file
        self.write_response(self.participant_id, self.training_type, self.current_sound.split("/")[-1], 
                            reaction_time, response=response, solution=correct_answer)

        # Move to next sound after 1 second
        QTimer.singleShot(1000, self.play_sound)  

    def provide_feedback(self, is_correct=None, correct_answer=None):

        if self.training_type == "Perception with Minimal Feedback":
            self.feedback_label.setText("Correct" if is_correct else "Incorrect")

        elif self.training_type == "Perception with Full Feedback":
            self.feedback_label.setText(
                f"Correct"
                if is_correct
                else f"Incorrect. The correct answer was {correct_answer}"
            )

        elif self.training_type == "Production Training":
            # TODO: Implement actual comparison feedback
            self.feedback_label.setText("Feedback: Good attempt! Try to match the pitch more closely.")
            QTimer.singleShot(1000, self.play_sound)

    def finish_training(self):
        score = (self.correct_answers / self.total_questions) * 100
        self.end_training_signal.emit(self.participant_id, self.training_type, score)

    def write_response(self, participant_id, training, audio_file, reaction_time, response=0, solution=0, accuracy=0):

        # Create participants folder if it doesn't exist
        participant_folder = os.path.join("participants", participant_id)
        os.makedirs(participant_folder, exist_ok=True)
        
        # Create training folder inside the participant's folder if it doesn't exist
        training_folder = os.path.join(participant_folder, training)
        os.makedirs(training_folder, exist_ok=True)
        
        # Define the response file path
        response_file = os.path.join(training_folder, f"{datetime.date.today()}_resp.csv")
        
        # Check if the file already exists
        file_exists = os.path.isfile(response_file)
        
        # Open the file in append mode and write the data
        with open(response_file, mode="a", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)

            if training != "Production Training":
            
                if not file_exists:
                    csv_writer.writerow(["audio_file", "response", "solution", "reaction_time"])
                
                csv_writer.writerow([audio_file, response, solution, round(reaction_time, 4)])

            else:
                if not file_exists:
                    csv_writer.writerow(["audio_file", "accuracy", "reaction_time"])
                
                csv_writer.writerow([audio_file, accuracy, round(reaction_time, 4)])
