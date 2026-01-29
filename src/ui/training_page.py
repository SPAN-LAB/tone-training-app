import sys
import os, re, csv, random, datetime, time
from io import BytesIO

from .audio_playback_thread import PlayThread

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QErrorMessage
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QKeyEvent, QFont, QImage, QPixmap

import joblib
import sounddevice as sd
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # TODO: Install seaborn when bundle executable
import numpy as np
import librosa
from scipy.interpolate import PchipInterpolator
import tensorflow as tf

# universal path
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # navigates one-level up from the current directory

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
       
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
        # Lazy-loaded BILSTM artifacts
        self._bilstm_model = None
        self._bilstm_bundle = None
        self.min_recording_db = -45.0
        self._recording_too_quiet = False

        # Set focus policy to accept keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)

        self.error_dialog = QErrorMessage(self)



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

        # Placeholder feedback label (used for ML feedback text)
        self.feedback_label = QLabel("Cross correlation plot")
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
            
        # random shuffle audio files
        random.shuffle(self.sounds)
        
        if training_type == "Production Training":
            self.setup_production_training()
            try:
                # Preload model + normalization bundle to avoid UI stall on first evaluation.
                self._load_bilstm_assets()
            except Exception as e:
                print(f"Warning: failed to preload BiLSTM assets: {e}")
        else:
            self.setup_ui()

        QTimer.singleShot(1000, self.play_sound)  


    def disable_response_button(self):
        """Disables all response buttons."""
        if self.response_buttons is not None:
                for button in self.response_buttons:
                    button.setEnabled(False)
    
    def takeBreak(self):
        """Initiates the 30-second break."""
        self.disable_response_button()
        self.start_countdown(30, False)  # 30 seconds break
        self.played_audio_cnt = 0        # reset played audio file count to zero
        return



    def play_sound(self):
        """
        Play the next sound stimulus in the training sequence.
        ... (rest of docstring) ...
        """

        print("in training page")
        print("session number: ", self.session_num)
        
        if self.sounds: 
            pass


        elif self.generalization_sounds and not self.is_generalization_block:

            self.is_generalization_block = True

            self.sounds = self.generalization_sounds
            self.generalization_sounds = []

            random.shuffle(self.sounds)

            print(f"Starting generalization block: total generalization sounds moved = {len(self.sounds)}")

            self.played_audio_cnt = 0

            self.prompt_label.setText("Starting Generalization Block...")
            self.feedback_label.setText("You will not receive feedback.")
            if self.response_buttons:
                self.disable_response_button()

            # Pause, then start the first generalization sound
            QTimer.singleShot(3000, self.play_sound)
            return

        else:

            self.finish_training()


        if self.sounds:
            if self.consecutiveTimesSolution >= 3:
                print("shuffled sound list because of consecutive solutions")
                random.shuffle(self.sounds)
                random.shuffle(self.sounds)
                random.shuffle(self.sounds)

            random.shuffle(self.sounds)
            self.current_sound = self.sounds.pop(0)
            self.played_audio_cnt += 1  
            print(f"Remaining sounds: {len(self.sounds)})")

            try:    
                if self.response_buttons is not None:
                    for button in self.response_buttons:
                        button.setEnabled(False)
                self.feedback_label.clear()

              
                full_path = self.current_sound

                if not full_path.lower().endswith(".mp3"):
                    full_path += ".mp3"  


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

                if self._play_thread and self._play_thread.isRunning():
                    try:
                        
                        sd.stop()
                        self._play_thread.wait(100)
                    except Exception:
                        try:
                            self._play_thread.terminate()
                        except Exception:
                            pass


                self._play_thread = PlayThread(data, fs, device=self.audio_device_id)

          
                self._play_thread.finished.connect(self.on_playback_finished)

   
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
 
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.prompt_label.setText("Recording... Try to match the original sound")


        date = datetime.date.today()

        file = os.path.basename(self.current_sound)
        file = os.path.splitext(file)[0]
        
        self.production_recording_file = os.path.join(self.production_recording_path, f"{date}_{file}.wav")

        sd.default.device = (self.input_device_id, self.audio_device_id)  

     
        self.start_time = time.time()
        self.recording = sd.rec(int(3 * 44100), samplerate=44100, channels=1)
        self.start_countdown(3, recording=True)

    def stop_recording(self):
        self.is_recording = False
        sd.stop()

        end_time = time.time()
        reaction_time = end_time - self.start_time if self.start_time else 0


        sf.write(self.production_recording_file, self.recording, 44100)
        self.prompt_label.setText("Recording complete. Analyzing...")

        try:
            audio = np.asarray(self.recording, dtype=np.float32).reshape(-1)
            if audio.size == 0:
                self._recording_too_quiet = True
            else:
                rms = float(np.sqrt(np.mean(np.square(audio))))
                dbfs = 20.0 * np.log10(rms + 1e-8)
                self._recording_too_quiet = dbfs < self.min_recording_db
                if self._recording_too_quiet:
                    print(f"Recording too quiet: {dbfs:.2f} dBFS (threshold {self.min_recording_db:.2f})")
        except Exception as e:
            print(f"Warning: could not compute recording loudness: {e}")
            self._recording_too_quiet = False


        self.provide_feedback()

        # write to response file (TODO: move to other function, and change response file to be correlation scores)
        self.write_response(self.current_sound.split("/")[-1], reaction_time)

    def start_countdown(self, seconds, recording):
        self.remaining_time = seconds

        if recording:
            self.prompt_label.setText(f"Recording... {self.remaining_time} seconds remaining")            
        else:
            self.prompt_label.setText(f"Take a break! {self.remaining_time} seconds remaining")


        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_timer.start(1000)
    
    def update_countdown(self):
        self.remaining_time -= 1
        if self.remaining_time > 0:

            if "Recording" in self.prompt_label.text():
                self.prompt_label.setText(f"Recording... {self.remaining_time} seconds remaining")
            else:
                self.prompt_label.setText(f"Take a break! {self.remaining_time} seconds remaining")
        else:
            self.countdown_timer.stop()  
            if "Recording" in self.prompt_label.text():
                self.stop_recording() 
            else:

                self.prompt_label.setText("Listen to the sound")

                QTimer.singleShot(1000, self.play_sound) 
  

    def process_response(self, response):

        if not self.current_sound:
            print("Warning: process_response called but no current sound is active. Ignoring response.")
            return


        if self.response_buttons is not None:
            for button in self.response_buttons:
                button.setEnabled(False)

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

        self.provide_feedback(is_correct, correct_answer)

        self.write_response(self.current_sound.split("/")[-1], 
                            reaction_time, response=response, solution=correct_answer)

    def provide_feedback(self, is_correct=None, correct_answer=None):
        
        if self.is_generalization_block:
            self.feedback_label.setText("Response recorded.")

            QTimer.singleShot(1500, self.clear_feedback_enable_buttons)
            return 
    
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
            
            try:
                if self._recording_too_quiet:
                    self.feedback_label.setText("Recording too quiet. Please speak louder.")

                    self.production_response = 0
                    self._recording_too_quiet = False
                    QTimer.singleShot(5000, self.clear_feedback_enable_buttons)
                    return

                #target
                played_tone = int(re.search(r"\d+", self.current_sound).group())

                self.tone_evaluation()


                if int(self.production_response) == int(played_tone):
                    msg = f"Correct! You produced tone {played_tone}."
                else:
                    msg = f"The model heard tone {self.production_response}. Target was tone {played_tone}."

                self.feedback_label.setText(msg)

            except Exception as e:
                self.feedback_label.setText(f"Could not evaluate tone: {e}")
                self.production_accuracy = 0.0

        if self.training_type == "Production Training":
            QTimer.singleShot(5000, self.clear_feedback_enable_buttons)
        else:
            QTimer.singleShot(1500, self.clear_feedback_enable_buttons)
            
    def on_playback_finished(self):
        """
        Slot connected to the PlayThread.finished signal.
        This runs on the main thread after audio playback is complete.
        """
       
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

    def _gender_label(self):
        if isinstance(self.gender, str):
            value = self.gender.strip().lower()
            if value in ("female", "f", "1"):
                return "female"
            if value in ("male", "m", "0"):
                return "male"
        return "female" if int(self.gender) == 1 else "male"

    # extended pitch bounds for mandarin tones
    def _pitch_bounds_for_gender(self, gender):
        if gender == "female":
            return 100.0, 300.0
        if gender == "male":
            return 70.0, 200.0 
        return 70.0, 250.0

    # Same BiLSTM pipeline
    def _interpolate_out_of_bounds_pitch(self, f0, fmin, fmax):
        if f0.size == 0:
            return f0

        voiced_mask = f0 > 0
        valid_mask = voiced_mask & (f0 >= fmin) & (f0 <= fmax)
        if valid_mask.all():
            return f0

        f0_clean = f0.astype(np.float32, copy=True)
        valid_idx = np.nonzero(valid_mask)[0]
        if valid_idx.size == 0:
            f0_clean[voiced_mask] = np.clip(f0_clean[voiced_mask], fmin, fmax)
            return f0_clean

        if valid_idx.size == 1:
            fill_value = np.clip(f0_clean[valid_idx[0]], fmin, fmax)
            f0_clean[voiced_mask & ~valid_mask] = fill_value
            return f0_clean

        interpolator = PchipInterpolator(valid_idx, f0_clean[valid_idx], extrapolate=True)
        interpolated = interpolator(np.arange(f0_clean.shape[0]))
        replace_mask = voiced_mask & ~valid_mask
        f0_clean[replace_mask] = np.clip(interpolated[replace_mask], fmin, fmax)
        return f0_clean

    def extract_fundamental_pitch(self, audio):
        """
        Extracts f0 with a noise gate (0.005) (specifically for tone 3)
        """
        try:
            # 1. Load at 22k
            y, sr = librosa.load(audio, sr=22050, mono=True)
        except Exception as e:
            print(f"Error loading audio: {e}")
            return np.array([])

        # 2. Calculate Energy
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        

        silence_threshold = 0.005 
        
        # Check if the recording is effectively empty
        if np.max(rms) < silence_threshold:
            print("Warning: Audio is too quiet (Max RMS below threshold).")
            silence_threshold = 0.0

        # 3. Extract Pitch
        gender_label = self._gender_label()
        fmin, fmax = (70, 250) if gender_label == "male" else (100, 400)

        f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=512)
        f0 = np.nan_to_num(f0, nan=0.0)
        
        # 4. Apply Noise Gate
        # Resize RMS to match f0 length
        if len(rms) > len(f0):
            rms = rms[:len(f0)]
        elif len(rms) < len(f0):
            rms = np.pad(rms, (0, len(f0) - len(rms)))
            
        f0[rms < silence_threshold] = 0.0

        # 5. Smooth (Median Filter) - removes single-frame glitches
        import scipy.signal
        if np.any(f0 > 0):
             f0 = scipy.signal.medfilt(f0, kernel_size=3)

        return f0

    def _load_bilstm_assets(self):
        if self._bilstm_bundle is None:
            bundle_candidates = [
                resource_path("tone_bilstm.pkl"),
                os.path.join(main_path, "model_training", "tone_bilstm.pkl"),
                os.path.join(os.path.dirname(main_path), "tone_bilstm.pkl"),
            ]
            bundle_path = next((p for p in bundle_candidates if os.path.exists(p)), None)
            if not bundle_path:
                raise FileNotFoundError("tone_bilstm.pkl not found in expected locations.")
            self._bilstm_bundle = joblib.load(bundle_path)

        if self._bilstm_model is None:
            model_candidates = [
                resource_path("tone_bilstm_model.keras"),
                os.path.join(main_path, "model_training", "tone_bilstm_model.keras"),
                os.path.join(os.path.dirname(main_path), "tone_bilstm_model.keras"),
            ]
            model_path = next((p for p in model_candidates if os.path.exists(p)), None)
            if not model_path:
                raise FileNotFoundError("tone_bilstm_model.keras not found in expected locations.")
            self._bilstm_model = tf.keras.models.load_model(model_path)
    
    def tone_evaluation(self):
        """
        Robust Tone Evaluation:
        1. Extracts Pitch
        2. TRIMS SILENCE (Crucial fix for 3s recordings)
        3. Applies Instance Normalization
        4. Predicts
        """
        audio_path = getattr(self, "production_recording_file", None)
        if not audio_path or not os.path.exists(audio_path):
            print("Error: Recording file missing.")
            return

        # 1. Extract raw pitch
        f0 = self.extract_fundamental_pitch(audio_path)
    
        # Check if we found any pitch at all
        if f0 is None or np.sum(f0) == 0:
            print("Warning: No pitch detected in recording.")
            self.production_response = 0
            self.feedback_label.setText("Speaker volume too low")
            return

        voiced_indices = np.nonzero(f0)[0]
        
        if voiced_indices.size == 0:
            self.production_response = 0
            return
            
        start_idx = voiced_indices[0]
        end_idx = voiced_indices[-1]
        
        # Slice the array to keep only the voiced part
        f0_trimmed = f0[start_idx : end_idx + 1]


        try:
            self._load_bilstm_assets()
            max_length = int(self._bilstm_bundle.get("max_length", 0))

            # 2. Instance Normalization on the TRIMMED sequence
            seq = f0_trimmed.astype(np.float32)
            mask = seq > 0 
            norm_seq = np.zeros_like(seq, dtype=np.float32)

            if np.any(mask):
                clip_mean = np.mean(seq[mask])
                clip_std = np.std(seq[mask])
                if clip_std < 1e-6: clip_std = 1e-6
                
                norm_seq[mask] = (seq[mask] - clip_mean) / clip_std

            # 3. Pad/Truncate correctly
            if len(norm_seq) >= max_length:
                padded = norm_seq[:max_length]
            else:
                padded = np.zeros((max_length,), dtype=np.float32)
                padded[: len(norm_seq)] = norm_seq

           

            # 4. Predict
            X = np.expand_dims(padded, axis=(0, -1))

            print("--- MODEL INPUT DEBUG ---")
            print(f"Input shape: {X.shape}")
            print(f"Non-zero values: {np.count_nonzero(X)}")
            print(f"First 10 values: {X[0, :10, 0]}")
            print("-------------------------")


            proba = self._bilstm_model.predict(X, verbose=0)[0]
            pred_idx = int(np.argmax(proba))
            self.production_response = pred_idx + 1 # Convert 0-index to 1-4 tone
            
            # Debugging print to see what happened
            print(f"DEBUG: Original len: {len(f0)}, Trimmed len: {len(f0_trimmed)}")
            print(f"DEBUG: Model Probabilities: {proba}")
            print(f"DEBUG: Predicted Tone: {self.production_response}")

        except Exception as e:
            print(f"Error during model prediction: {e}")
            import traceback
            traceback.print_exc()
            self.production_response = 0
    

    def clear_feedback_enable_buttons(self):
        self.feedback_label.clear()
        if self.response_buttons is not None:
            for button in self.response_buttons:
                button.setStyleSheet("")  
                # keep buttons disabled here; they will be enabled when the next playback finishes
                button.setEnabled(False)
        
        # Check if a break is due
        if not (self.played_audio_cnt % 20 == 0 and self.played_audio_cnt > 0 and self.sounds):
            self.prompt_label.setText("Listen to the sound")
            QTimer.singleShot(1000, self.play_sound)
        else:
            self.takeBreak() # Call the working break function
        
    def write_response(self, audio_file, reaction_time, response=0, solution=0):
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
                print(f"Response -> {audio_file}, response={response}, solution={solution}")
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

                csv_writer.writerow([datetime.date.today(), audio_file, self.production_response, target_tone, round(reaction_time, 4), block_type])
                print(f"WROTE ROW -> {audio_file}, production_response={self.production_response}, target={target_tone}, rt={round(reaction_time,4)}, type={block_type}")
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

    
