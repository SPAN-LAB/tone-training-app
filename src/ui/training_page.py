import sys
import os, re, csv, random, datetime, time
from io import BytesIO

from .audio_playback_thread import PlayThread

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QErrorMessage
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QKeyEvent, QFont, QImage, QPixmap

import joblib
import sounddevice as sd
import soundfile as sfp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # TODO: Install seaborn when bundle executable
import numpy as np
import librosa

# import machine learning model for tone prediction
#from model_training.tone_prediction_model import load_tone_model

# universal path
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # navigates one-level up from the current directory

# TODO: extract fundamental pitch of user using range_est.wav

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class TrainingPage(QWidget):
    # Signal emitted to end training and display results
    end_training_signal = pyqtSignal(str, str, float, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_sound = None
        self.sounds = []
        self.generalization_sounds = []
        self.is_generalization_block = False
        self.participant_id = ""
        self.training_type = ""
        self.audio_device_id = None
        self.input_device_id = None  
        self.correct_answers = 0
        self.total_questions = 0
        self.is_recording = False  
        self.response_buttons = None
        self.start_time = None
        self.production_response = 0
        self.production_accuracy = 0.0
        self.played_audio_cnt = 0
        self.session_num = 1
        self.blocks_plot = None
        self.sessions_plot = None
        self.preset = None
        self.gender = 0 # Male
        self.consecutiveTimesSolution = 0  # Counter for consecutive correct answers
        self.previousAnswer = None

        self.production_recording_path = ""  # Folder path to store users' recordings for production training
        self.response_file_path = ""         # File path to store training response 
        self.session_tracking_file_path = "" # File path to store session tracking
        
        self._play_thread = None # Attribute to hold the playback thread

        self.production_recording_file = ""  # File path to store users' recording for production training

        # Background play thread handle
        self._play_thread = None
        # Counter for rows successfully written to the response CSV during the session
        self.response_rows_written = 0

        # Set focus policy to accept keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)

        # --- Features from training_page1.py ---
        # Error message dialog
        self.error_dialog = QErrorMessage(self)
        # --- End features from training_page1.py ---


    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Prompt label
        self.prompt_label = QLabel("Listen to the sound")
        self.prompt_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(50)
        self.prompt_label.setFont(font)
        layout.addWidget(self.prompt_label)

        # Response buttons
        response_layout_1 = QHBoxLayout()
        response_layout_2 = QHBoxLayout()
        self.response_buttons = []
        
        # Creating response buttons
        for i in range(1, 5):
            button = QPushButton(str(i))
            button.setMinimumSize(250, 250)
            button.setFont(font)
            button.clicked.connect(lambda _, x=i: self.process_response(x))
            if i < 3:
                response_layout_1.addWidget(button)
            else:
                response_layout_2.addWidget(button)
            self.response_buttons.append(button)
        layout.addLayout(response_layout_1)
        layout.addLayout(response_layout_2)

        # Feedback label
        self.feedback_label = QLabel("")
        self.feedback_label.setAlignment(Qt.AlignCenter)
        self.feedback_label.setFont(font)
        layout.addWidget(self.feedback_label)

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key in [Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4]:
            index = key - Qt.Key_1
            if 0 <= index < len(self.response_buttons):
                bt = self.response_buttons[index]
                bt.setStyleSheet("border: 2px solid green") # highlight selected button
                bt.click()

    def setup_production_training(self):
        """Setup UI for Production Training"""
        layout = QVBoxLayout(self)

        # Prompt label with production-specific instructions
        self.prompt_label = QLabel("Listen to the sound, then reproduce it.")
        self.prompt_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prompt_label)

        # Placeholder cross-correlation function plot
        self.feedback_label = QLabel("Cross correlation plot")
        # self.feedback_label = QLabel("The correct tone is ... You sounded like tone ...") # Uncomment after ML model implementation
        self.feedback_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.feedback_label)

    def setup_training(self, participant_id, training_type, sounds, generalization_sounds, output_device_id, input_device_id, session_num, production_recording_path, response_file_path, session_tracking_file_path, gender): 
        """
        Initialize and configure the training session.

        This method sets participant/session parameters, randomizes the sound recording stimuli order, 
        selects a pitch detection preset based on a range estimation recording, and prepares 
        the appropriate UI for the selected training type.

        Args:
            participant_id (str): Unique identifier for the participant.
            training_type (str): Type of training ("Perception with Minimal Feedback",
                                "Perception with Full Feedback", or "Production Training").
            sounds (list[str]): List of sound file names to be used in the session.
            output_device_id (int or str): Identifier for the audio output device.
            input_device_id (int or str): Identifier for the audio input device.
            session_num (int): Current session number for the participant.
            production_recording_path (str): Directory path to save production recordings.
            response_file_path (str): Path to the CSV file where responses are logged.
            session_tracking_file_path (str): Path to the CSV file where session-level accuracy is tracked.

        Returns:
            None
        """
        
        self.participant_id = participant_id
        self.training_type = training_type
        self.sounds = sounds
        self.generalization_sounds = generalization_sounds
        self.is_generalization_block = False
        self.audio_device_id = output_device_id  
        self.input_device_id = input_device_id  
        self.correct_answers = 0
        self.total_questions = len(sounds)
        self.session_num = session_num
        self.response_file_path = response_file_path
        self.session_tracking_file_path = session_tracking_file_path
        self.gender = gender
        # Keep track of expected total number of stimuli (training + generalization)
        try:
            self.total_expected_sounds = len(sounds) + len(generalization_sounds)
        except Exception:
            self.total_expected_sounds = None

        if production_recording_path:
            self.production_recording_path = production_recording_path
            print("in setup training function: ", self.production_recording_path)
            # Check if range_est.wav exists before trying to use it
            range_est_file = os.path.join(self.production_recording_path, 'range_est.wav')
            if os.path.exists(range_est_file):
                 self.preset = self.range_est(range_est_file) # Comment this out for manual preset selection
            else:
                print(f"Warning: range_est.wav not found at {range_est_file}. Defaulting to preset 1 (male).")
                self.preset = 1 # Default to preset 1 if file is missing
            # self.preset = preset  # Uncomment for manual preset selection and add parameter 'preset' in this function

        # random shuffle audio files
        random.shuffle(self.sounds)
        
        if training_type == "Production Training":
            self.setup_production_training()
        else:
            self.setup_ui()

        QTimer.singleShot(1000, self.play_sound)  

    # --- Methods from training_page1.py ---
    def disable_response_button(self):
        """Disables all response buttons."""
        if self.response_buttons is not None:
                for button in self.response_buttons:
                    button.setEnabled(False)
        # response_locked removed: disabling buttons is sufficient to prevent input
    
    def takeBreak(self):
        """Initiates the 30-second break."""
        self.disable_response_button()
        self.start_countdown(30, False)  # 30 seconds break
        self.played_audio_cnt = 0        # reset played audio file count to zero
        return
    # --- End methods from training_page1.py ---


    def play_sound(self):
        """
        Play the next sound stimulus in the training sequence.
        ... (rest of docstring) ...
        """

        print("in training page")
        print("session number: ", self.session_num)
        print("production recording path: ", self.production_recording_path)
        print("response file path: ", self.response_file_path)
        print("session tracking file path: ", self.session_tracking_file_path)
        print("total expected sounds (training + generalization):", getattr(self, "total_expected_sounds", None))
        
        if self.sounds: # --- Check if TRAINING sounds are left ---
            # Do not pop here; popping is handled once later to avoid skipping items.
            pass
        # ... (rest of the 'try/except' block for playing sound) ...

        elif self.generalization_sounds and not self.is_generalization_block:
            # --- Training is done, START Generalization Block ---
            self.is_generalization_block = True

            # Swap the main 'sounds' list with the generalization list
            self.sounds = self.generalization_sounds
            self.generalization_sounds = [] # Clear the old one to prevent loops

            # Shuffle the new generalization sounds
            random.shuffle(self.sounds)

            print(f"Starting generalization block: total generalization sounds moved = {len(self.sounds)}")

            # Reset break counter
            self.played_audio_cnt = 0

            # Notify the user
            self.prompt_label.setText("Starting Generalization Block...")
            self.feedback_label.setText("You will not receive feedback.")
            if self.response_buttons:
                self.disable_response_button()

            # Pause, then start the first generalization sound
            QTimer.singleShot(3000, self.play_sound)
            return

        else:
            # --- All sounds (training AND generalization) are done ---
            self.finish_training()

        # The break logic is now handled in clear_feedback_enable_buttons

        if self.sounds:
            if self.consecutiveTimesSolution >= 3:
                print("shuffled sound list because of consecutive solutions")
                random.shuffle(self.sounds)
                random.shuffle(self.sounds)
                random.shuffle(self.sounds)
            print("Consecutive Solutions: ", self.consecutiveTimesSolution)
            random.shuffle(self.sounds)
            print("shuffled list once")
            self.current_sound = self.sounds.pop(0)
            self.played_audio_cnt += 1  # increment count of played audio file
            print(f"Selected sound: {self.current_sound} (remaining sounds: {len(self.sounds)})")

            try:    
                if self.response_buttons is not None:
                    for button in self.response_buttons:
                        button.setEnabled(False)
                self.feedback_label.clear()

                # --- MODIFIED: Replaced hardcoded R:\ path with logic from training_page1.py ---
                # Assumes self.current_sound is a full file path provided during setup.
                full_path = self.current_sound

                # Append .mp3 extension if missing
                if not full_path.lower().endswith(".mp3"):
                    full_path += ".mp3"  
                # --- End path modification ---

                # Check if the file actually exists
                if not os.path.isfile(full_path):
                    raise FileNotFoundError(f"File not found: {full_path}")
                
                # Read the sound file to determine its sample rate and number of channels
                data, fs = sf.read(full_path, dtype="float32")

                # Adjust volume of sound file
                volume_factor = 0.3
                data *= volume_factor

                # Start timer for reaction time
                # Do not set start_time here for perception — set it when playback finishes
                # For production training, start_time will be set when recording starts

                # Stop previous thread if it's still running (best-effort)
                if self._play_thread and self._play_thread.isRunning():
                    try:
                        # Attempt a polite stop first
                        sd.stop()
                        self._play_thread.wait(100)
                    except Exception:
                        try:
                            self._play_thread.terminate()
                        except Exception:
                            pass

                # Play in background thread: pass data, sample rate, and device ID to the thread
                self._play_thread = PlayThread(data, fs, device=self.audio_device_id)

                # Connect the thread's finished signal to our handler
                self._play_thread.finished.connect(self.on_playback_finished)

                # Start the thread (non-blocking)
                print(f"Playing audio: {os.path.basename(full_path)}")
                self._play_thread.start()

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
        self.prompt_label.setText("Recording... Try to match the original sound")

        # # Create session_recordings folder if it doesn't exist
        date = datetime.date.today()
        # Use os.path.basename and os.path.splitext to safely get filename
        file = os.path.basename(self.current_sound)
        file = os.path.splitext(file)[0]
        
        self.production_recording_file = os.path.join(self.production_recording_path, f"{date}_{file}.wav")

        # Set default device
        sd.default.device = (self.input_device_id, self.audio_device_id)  

        # Start recording and countdown timer
        # Record start time for reaction-time calculation at the moment recording begins
        self.start_time = time.time()
        self.recording = sd.rec(int(3 * 44100), samplerate=44100, channels=1)
        self.start_countdown(3, recording=True)

    def stop_recording(self):
        self.is_recording = False
        sd.stop()

        end_time = time.time()
        reaction_time = end_time - self.start_time if self.start_time else 0

        # save user recording
        sf.write(self.production_recording_file, self.recording, 44100)
        self.prompt_label.setText("Recording complete. Analyzing...")

        # provide feedback to user 
        self.provide_feedback()

        # write to response file (TODO: move to other function, and change response file to be correlation scores)
        self.write_response(self.current_sound.split("/")[-1], reaction_time)

    def start_countdown(self, seconds, recording):
        self.remaining_time = seconds

        if recording:
            self.prompt_label.setText(f"Recording... {self.remaining_time} seconds remaining")            
        else:
            self.prompt_label.setText(f"Take a break! {self.remaining_time} seconds remaining")

        # Create a timer that triggers every 1 second
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_timer.start(1000)
    
    # --- MODIFIED: Replaced with working version from training_page1.py ---
    def update_countdown(self):
        self.remaining_time -= 1
        if self.remaining_time > 0:
            # update countdown according to the label itself
            if "Recording" in self.prompt_label.text():
                self.prompt_label.setText(f"Recording... {self.remaining_time} seconds remaining")
            else:
                self.prompt_label.setText(f"Take a break! {self.remaining_time} seconds remaining")
        else:
            self.countdown_timer.stop()  
            if "Recording" in self.prompt_label.text():
                self.stop_recording() 
            else:
                # This is the crucial part for the break timer
                self.prompt_label.setText("Listen to the sound")
                # Do not enable buttons here; keep them disabled until playback finishes.
                QTimer.singleShot(1000, self.play_sound) # Restart playing
    # --- End modification ---

    def process_response(self, response):
        # Guard: if no current sound is set, ignore spurious responses
        if not self.current_sound:
            print("Warning: process_response called but no current sound is active. Ignoring response.")
            return

        # Disable response buttons after selection
        if self.response_buttons is not None:
            for button in self.response_buttons:
                button.setEnabled(False)

        # Stop timer and calculate participants' response time
        end_time = time.time()
        reaction_time = end_time - self.start_time if self.start_time else 0

        correct_answer = int(re.findall("[0-9]+", self.current_sound)[0])
        is_correct = response == correct_answer
        if is_correct:
            self.correct_answers += 1
            if correct_answer == self.previousAnswer:
                self.consecutiveTimesSolution += 1
            else:
                self.consecutiveTimesSolution = 1
        else:
            self.consecutiveTimesSolution = 0
        self.previousAnswer = correct_answer
        # display feedback on screen 
        self.provide_feedback(is_correct, correct_answer)

        # write to response file
        self.write_response(self.current_sound.split("/")[-1], 
                            reaction_time, response=response, solution=correct_answer)

    def provide_feedback(self, is_correct=None, correct_answer=None):
        
        if self.is_generalization_block:
            self.feedback_label.setText("Response recorded.")
            # Timer to clear message and move to next sound
            QTimer.singleShot(1500, self.clear_feedback_enable_buttons)
            return # <--- IMPORTANT: exit before giving other feedback
    
        """
        Display feedback to the participant based on their response.
        ... (rest of docstring) ...
        """
        if self.training_type == "Perception with Minimal Feedback":
            self.feedback_label.setText("Correct" if is_correct else "Incorrect")

        elif self.training_type == "Perception with Full Feedback":
            self.feedback_label.setText(
                f"Correct"
                if is_correct
                else f"Incorrect. The correct answer was {correct_answer}"
            )

        elif self.training_type == "Production Training":
            ### Lag graph ###
            # ... (graphing code remains commented out) ...

            ### NEW: ML prediction feedback ###
            try:
                # Determine the target (played) tone from filename
                played_tone = int(re.search(r"\d+", self.current_sound).group())

                # Run the tone evaluator on the just-recorded WAV
                self.tone_evaluation()

                # Calculate accuracy
                self.production_accuracy = 1.0 if int(self.production_response) == int(played_tone) else 0.0

                # Compose text feedback
                if int(self.production_response) == int(played_tone):
                    msg = f"Correct! You produced tone {played_tone}."
                else:
                    msg = f"The model heard tone {self.production_response}. Target was tone {played_tone}."

                self.feedback_label.setText(msg)

            except Exception as e:
                # If anything goes wrong, show a simple error message
                self.feedback_label.setText(f"Could not evaluate tone: {e}")
                self.production_accuracy = 0.0

        # Hide feedback and enable buttons before moving to the next audio file
        if self.training_type == "Production Training":
            QTimer.singleShot(5000, self.clear_feedback_enable_buttons)
        else:
            QTimer.singleShot(1500, self.clear_feedback_enable_buttons)
            
    def on_playback_finished(self):
        """
        Slot connected to the PlayThread.finished signal.
        This runs on the main thread after audio playback is complete.
        """
        # This logic was moved from the end of the `play_sound` method
        if self.training_type == "Production Training":
            self.prompt_label.setText("Try to reproduce the sound")
            # For production, start recording — start_recording() will set start_time
            self.toggle_recording()
        else:
            # For perception, mark start_time when playback finished so reaction_time is accurate
            self.start_time = time.time()
            self.prompt_label.setText("Select the sound you heard")

            if self.response_buttons is not None:
                for button in self.response_buttons:
                    button.setEnabled(True)

    def extract_fundamental_pitch(self, audio): #CHECK, for production
        """
        Extract and normalize the fundamental frequency (f0) contour from an audio file.
        """
        try:
            y, sr = librosa.load(audio, sr=44100)  # Force consistent sample rate
        except Exception as e:
            print(f"Error loading audio file {audio}: {e}")
            return pd.Series(dtype=np.float64)

        # Apply some basic audio preprocessing
        y = librosa.util.normalize(y)  # Normalize audio volume
        
        # extract f0 with appropriate preset
        if self.preset == 1:
            fmin, fmax = 60, 200
        elif self.preset == 2:
            fmin, fmax = 180, 350
        else:
            print(f'Invalid preset value: {self.preset}. Defaulting to 1.')
            fmin, fmax = 60, 200

        # Use pyin with better parameters for speech
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=fmin, 
            fmax=fmax, 
            sr=sr,
            frame_length=2048,  # Better for speech
            hop_length=512,
            fill_na=0.0  # Fill unvoiced frames with 0
        )
        
        # Replace NaN values with 0
        f0 = np.nan_to_num(f0, nan=0.0)
        
        # Only use voiced frames for normalization (non-zero values)
        voiced_f0 = f0[f0 > 0]
        
        if len(voiced_f0) == 0:
            print(f"Warning: No valid voiced frames detected in {audio}")
            return pd.Series(dtype=np.float64)

        # min-max normalization using only voiced frames
        f0_min = voiced_f0.min()
        f0_max = voiced_f0.max()
        
        if f0_max == f0_min:
            # If all pitches are the same, return zeros
            normalized_f0 = np.zeros_like(f0)
        else:
            # Apply normalization to entire f0 array (including unvoiced=0)
            normalized_f0 = np.where(f0 > 0, (f0 - f0_min) / (f0_max - f0_min), 0.0)

        times = librosa.times_like(f0, sr=sr, hop_length=512)
        return pd.Series(normalized_f0, index=times) 

    def range_est(self, audio):
        """
        Estimate the vocal range (male/female preset) based on mean pitch from audio.
        ... (rest of docstring) ...
        """
        y, sr = librosa.load(audio)
        # Use a wide pitch range to cover both male and female voices
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
        f0 = f0[~np.isnan(f0)]
        if len(f0) == 0:
            print("Warning: No valid pitch detected for range estimation. Defaulting to preset 1 (male).")
            return 1 # Default to preset 1
        mean_pitch = np.mean(f0)

        # 180-200Hz is overlapping zone for male and female voice, take the mean of 180 + 200 = 190Hz
        return 1 if mean_pitch < 190 else 2

    def tone_evaluation(self):
        """
        Predict the tone of the most recent production recording using the ML model.
        """
        audio_path = getattr(self, "production_recording_file", None)
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Recording not found for tone evaluation: {audio_path}")

        # Extract and process pitch features
        f0_series = self.extract_fundamental_pitch(audio_path)
        if f0_series is None or len(f0_series) == 0:
            print("Warning: No valid f0 extracted from recording. Defaulting prediction to 0.")
            self.production_response = 0
            return

        f0 = f0_series.values.astype(float)

        # Resample to 39 pitch points
        expected_pitch_points = 39
        if len(f0) == 0:
            print("Warning: Empty f0 after normalization. Defaulting prediction to 0.")
            self.production_response = 0 
            return
        elif len(f0) == 1:
            pitch_contour = np.repeat(f0, expected_pitch_points)
        else:
            x_src = np.linspace(0.0, 1.0, num=len(f0))
            x_tgt = np.linspace(0.0, 1.0, num=expected_pitch_points)
            pitch_contour = np.interp(x_tgt, x_src, f0)

        # Add gender feature
        gender_feature = np.array([self.gender], dtype=float) 
        features = np.concatenate((pitch_contour, gender_feature))
        X = features.reshape(1, -1) 

        # --- SIMPLIFIED: Load and use model ---
        try:
            model_path = os.path.join(main_path, 'model_training', 'tone_pred_model.pkl')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
                
            model = joblib.load(model_path)
            
            # Simple shape check
            if X.shape[1] != 40:
                print(f"ERROR: Expected 40 features, got {X.shape[1]}")
                self.production_response = 0
                return

            # Direct prediction (no complex model introspection needed)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                pred_idx = int(np.argmax(proba))
                print(f"Prediction probabilities: {proba}")
            else:
                y_pred = model.predict(X)
                pred_idx = int(y_pred[0])
                print(f"Direct prediction: {y_pred}")

            # Map to tone (1-4)
            self.production_response = pred_idx + 1
            print(f"Final production response: {self.production_response}")
            
        except Exception as e:
            print(f"Error during model prediction: {e}")
            self.production_response = 0
        
    def clear_feedback_enable_buttons(self):
        self.feedback_label.clear()
        if self.response_buttons is not None:
            for button in self.response_buttons:
                button.setStyleSheet("")  # remove highlight
                # keep buttons disabled here; they will be enabled when the next playback finishes
                button.setEnabled(False)
        
        # Check if a break is due
        if not (self.played_audio_cnt % 20 == 0 and self.played_audio_cnt > 0 and self.sounds):
            self.prompt_label.setText("Listen to the sound")
            QTimer.singleShot(1000, self.play_sound)
        else:
            self.takeBreak() # Call the working break function
        
    def write_response(self, audio_file, reaction_time, response=0, solution=0, accuracy=0):
        """
        Log participant responses and session statistics to CSV files.
        ... (rest of docstring) ...
        """
        block_type = "Generalization" if self.is_generalization_block else "Training"
        
        # Write training response file
        # Use absolute path for writes as well (make path relative to project root if necessary)
        response_write_path = self.response_file_path
        try:
            # If path is relative, join with main_path
            if not os.path.isabs(response_write_path):
                response_write_path = os.path.join(main_path, response_write_path)
        except Exception:
            response_write_path = self.response_file_path

        with open(response_write_path, mode="a", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)

            if self.training_type != "Production Training":
                csv_writer.writerow([datetime.date.today(), audio_file, response, solution, round(reaction_time, 4), block_type])
                print(f"WROTE ROW -> {audio_file}, response={response}, solution={solution}, rt={round(reaction_time,4)}, type={block_type}")
                try:
                    self.response_rows_written += 1
                except Exception:
                    pass
            else:
                # Record predicted tone (response), target tone (solution), production accuracy
                # Target tone parsed from filename; predicted tone comes from self.production_response
                try:
                    target_tone = int(re.search(r"\d+", audio_file).group())
                except Exception:
                    target_tone = solution if solution else 0

                csv_writer.writerow([datetime.date.today(), audio_file, self.production_response, target_tone, self.production_accuracy, round(reaction_time, 4), block_type])
                print(f"WROTE ROW -> {audio_file}, production_response={self.production_response}, target={target_tone}, accuracy={self.production_accuracy}, rt={round(reaction_time,4)}, type={block_type}")
                try:
                    self.response_rows_written += 1
                except Exception:
                    pass

        # Write session tracking file only when all stimuli (training + generalization) have been played
        if (not self.sounds) and (not self.generalization_sounds):

            # Open the training file in append mode and write session data
            with open(self.session_tracking_file_path, mode="a", newline="") as session_file:
                session_writer = csv.writer(session_file)

                # Read response file into DataFrame (use main_path to get absolute path)
                response_path = os.path.join(main_path, self.response_file_path)
                df = pd.read_csv(response_path)

                # Ensure tone is extracted from audio_file as a string key '1'..'4'
                df["tone"] = df["audio_file"].astype(str).str.extract(r"(\d+)")

                if self.training_type == "Production Training":
                    # For production: use the recorded probability/accuracy directly
                    df["accuracy"] = pd.to_numeric(df.get("accuracy", pd.Series()), errors="coerce")

                    # Mean accuracy per tone (probabilities averaged)
                    tone_means = {}
                    for t in ["1", "2", "3", "4"]:
                        vals = df.loc[df["tone"] == t, "accuracy"]
                        tone_means[t] = float(vals.mean()) if not vals.empty else np.nan

                    # Overall = mean of accuracy column
                    overall = float(df["accuracy"].mean()) if not df["accuracy"].empty else np.nan

                else:
                    # Perception trainings: accuracy = 1 for correct, 0 for incorrect
                    correct_col = (df["response"] == df["solution"]).astype(float)
                    df["_correct"] = correct_col

                    tone_means = {}
                    for t in ["1", "2", "3", "4"]:
                        vals = df.loc[df["tone"] == t, "_correct"]
                        tone_means[t] = float(vals.mean()) if not vals.empty else np.nan

                    overall = float(correct_col.mean()) if len(correct_col) > 0 else np.nan

                # Write the date and accuracy information for every tone in the session
                for key in ["1", "2", "3", "4"]:
                    session_writer.writerow([datetime.date.today(), key, tone_means[key]])

                # Write the date and overall accuracy information (last line)
                self.score = overall if not np.isnan(overall) else 0.0
                session_writer.writerow([datetime.date.today(), "overall", self.score])

    def split_block(self, df, files_per_block):

        # Determine response accuracy per-row: perception -> response==solution, production -> use recorded accuracy column
        if self.training_type == "Production Training":
            # Ensure numeric accuracy column exists and is numeric
            df["accuracy"] = pd.to_numeric(df.get("accuracy", pd.Series(index=df.index)), errors="coerce")
        else:
            df["accuracy"] = (df["response"] == df["solution"]).astype(float)

        # Assign block numbers
        df["block"] = (df.index // files_per_block).astype(int) + 1

        # Calculate mean accuracy for every block
        group_accuracy = df.groupby('block')['accuracy'].mean() * 100
        df = df.merge(group_accuracy, on=['block'], suffixes=('', '_mean'))

        return df

    def plot_block_accuracy(self):
        """ 
        Plot tone accuracy over blocks in the current session.
        10 audio files per block.

        Returns:
            matplotlib Axes: Line plot show the tone accuracy over blocks.
        """

        global main_path

        # read csv files (use absolute path stored in response_file_path)
        file = self.response_file_path if os.path.isabs(self.response_file_path) else os.path.join(main_path, self.response_file_path)
        df = pd.read_csv(file)

        # Ensure rows are in play order then assign block numbers (20 sounds per block)
        df = df.reset_index(drop=True)
        files_per_block = 20
        df["block"] = (df.index // files_per_block).astype(int) + 1

        # Compute per-row accuracy depending on training type
        if self.training_type == "Production Training":
            df["accuracy"] = pd.to_numeric(df.get("accuracy", pd.Series(dtype=float)), errors="coerce")
        else:
            df["accuracy"] = (df["response"] == df["solution"]).astype(float)

        # Aggregate mean accuracy per block and sort by block number
        block_means = df.groupby("block", sort=True)["accuracy"].mean().reset_index()
        block_means["accuracy_pct"] = (block_means["accuracy"] * 100).round(2)

        # Determine which blocks contain generalization trials (if block_type column exists)
        gen_blocks = []
        if "block_type" in df.columns:
            gen_blocks = sorted(df.loc[df["block_type"] == "Generalization", "block"].unique().tolist())

        # plot block means in ascending block order
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(block_means["block"], block_means["accuracy_pct"], marker="o", color="#1f77b4")

        # Determine xticks and x limits (based on available blocks)
        xticks = list(block_means["block"]) if not block_means.empty else [1]
        xlimit = int(block_means["block"].max()) if not block_means.empty else 1

        # Shade generalization blocks for visibility
        for b in gen_blocks:
            ax.axvspan(b - 0.5, b + 0.5, color="#d3d3d3", alpha=0.4)

        # Also shade blocks 3 and 4 as requested (if they exist in the plotted blocks)
        mid_blocks = [b for b in [3, 4] if b in xticks]
        for b in mid_blocks:
            ax.axvspan(b - 0.5, b + 0.5, color="#cfe2f3", alpha=0.35)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"Block {int(x)}" for x in xticks])
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xlim(0.5, xlimit + 0.5)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Block")
        ax.set_ylabel("Accuracy(%)")
        ax.set_title("Participant: " + self.participant_id)

        # Add legend entries explaining shaded areas
        import matplotlib.patches as mpatches
        legend_handles = []
        if gen_blocks:
            shaded_patch = mpatches.Patch(facecolor="#d3d3d3", alpha=0.4, label="Generalization block")
            legend_handles.append(shaded_patch)
        if mid_blocks:
            mid_patch = mpatches.Patch(facecolor="#cfe2f3", alpha=0.35, label="Blocks 3-4")
            legend_handles.append(mid_patch)
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1, 1))

        self.blocks_plot = fig
    
    def plot_session_accuracy(self):
        """ 
        Plot tone accuracy over sessions.

        Returns:
            matplotlib Axes: Line plot show the tone accuracy over blocks.
        """

        global main_path

        # read all files within the session tracking folder
        parent_folder = os.path.dirname(self.session_tracking_file_path)
        session_tracking_folder = os.path.join(main_path, parent_folder)
        content = []
        for file in os.listdir(session_tracking_folder):
            path = os.path.join(session_tracking_folder, file)
            df_temp = pd.read_csv(path)
            df_temp["session"] = int(re.findall(r'\d+', file)[0])
            content.append(df_temp)

        df = pd.concat(content, axis=0, ignore_index=True)

        # scale accuracy score into percentage and round to two floating point
        df["accuracy"] = (df["accuracy"].fillna(0) * 100).round(2)

        # plot
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        sns.lineplot(df, x = "session", y = "accuracy", hue = "subject", marker="o", 
                    palette=sns.color_palette("Set1", 5), ax=ax)

        # adjust line attribute to make overall accuracy stand out
        legend = ax.get_legend()
        if legend is not None:
            # Safe iteration: pair plotted line objects with legend text labels
            legend_texts = legend.get_texts()
            for line_obj, label in zip(ax.lines, legend_texts):
                if label.get_text() != "overall":
                    line_obj.set_alpha(0.5)
                else:
                    line_obj.set_linewidth(3)
        else:
            # No legend (e.g., single series); make the single line stand out
            for line_obj in ax.lines:
                line_obj.set_linewidth(2.5)

        # set axis ticks
        xlimit = df["session"].unique()[-1]
        ax.set(xticks=[i for i in range(xlimit + 1)], yticks=[0, 20, 40, 60, 80, 100],
                xlim=(0.5, xlimit+1), ylim=(0, 105))

        # set axis labels and title
        ax.set_xlabel("Session")
        ax.set_ylabel("Accuracy(%)")
        ax.set_title("Participant: " + self.participant_id)

        # set legend labels
        handles, _ = ax.get_legend_handles_labels()
        new_labels = ["Tone 1", "Tone 2", "Tone 3", "Tone 4", "Overall"]  
        ax.legend(handles=handles, labels=new_labels, title="Subject", loc="upper left", bbox_to_anchor=(1, 1))

        self.sessions_plot = fig

    def finish_training(self):

        # generate plot
        self.plot_block_accuracy()
        self.plot_session_accuracy()

        # Log number of rows written for this session for debugging
        try:
            print(f"Session complete. Response rows written this session: {self.response_rows_written}")
        except Exception:
            pass

        # --- MODIFIED: Use QErrorMessage dialogs ---
        if self.blocks_plot is None:
            self.error_dialog.showMessage("Error: Plot for block accuracy was not generated.")
            print("Error: Plot for block accuracy was not generated.")
            return
        elif self.sessions_plot is None:
            self.error_dialog.showMessage("Error: Plot for session accuracy was not generated.")
            print("Error: Plot for session accuracy was not generated.")
            return
        # --- End modification ---

        self.end_training_signal.emit(self.participant_id, self.training_type, self.score, self.blocks_plot, self.sessions_plot)