from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QKeyEvent, QFont, QImage, QPixmap
import sounddevice as sd
import soundfile as sf
import os, re, csv, random, datetime, time
from .volume_check_page import VolumeCheckPage
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # TODO: Install seaborn when bundle executable
import numpy as np
import librosa
from io import BytesIO

# import machine learning model for tone prediction
from model_training.tone_prediction_model import load_tone_model

# universal path
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # navigates two-level up from the current directory

# TODO: extract fundamental pitch of user using range_est.wav

class TrainingPage(QWidget):
    # Signal emitted to end training and display results
    end_training_signal = pyqtSignal(str, str, float, object, object)

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
        self.response_buttons = None
        self.start_time = None
        self.production_response = 0
        self.played_audio_cnt = 0
        self.session_num = 1
        self.blocks_plot = None
        self.sessions_plot = None
        self.preset = None

        self.production_recording_path = ""  # Folder path to store users' recordings for production training
        self.response_file_path = ""         # File path to store training response 
        self.session_tracking_file_path = "" # File path to store session tracking

        self.production_recording_file = ""  # File path to store users' recording for production training

        # Set focus policy to accept keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)

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

    def setup_training(self, participant_id, training_type, sounds, output_device_id, input_device_id, session_num, production_recording_path, response_file_path, session_tracking_file_path): 
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
        self.audio_device_id = output_device_id  
        self.input_device_id = input_device_id  
        self.correct_answers = 0
        self.total_questions = len(sounds)
        
        self.session_num = session_num
        self.response_file_path = response_file_path
        self.session_tracking_file_path = session_tracking_file_path

        if production_recording_path:
            self.production_recording_path = production_recording_path
            print("in setup training function: ", self.production_recording_path)
            self.preset = self.range_est(os.path.join(self.production_recording_path, 'range_est.wav')) # Comment this out for manual preset selection
            # self.preset = preset  # Uncomment for manual preset selection and add parameter 'preset' in this function

        # random shuffle audio files
        random.shuffle(self.sounds)
        
        if training_type == "Production Training":
            self.setup_production_training()
        else:
            self.setup_ui()

        QTimer.singleShot(1000, self.play_sound)  

    def play_sound(self):

        """
        Play the next sound stimulus in the training sequence.

        Handles audio playback, manages UI state during and after playback, 
        and initiates either:
        - response selection (perception training)
        - or user recording (production training).

        Implements a 30-second break after every 20 stimuli and automatically 
        progresses until all sounds are exhausted.

        Raises:
            FileNotFoundError: If the sound file cannot be found.
            Exception: For other playback-related errors.
        """

        print("in training page")
        print("session number: ", self.session_num)
        print("production recording path: ", self.production_recording_path)
        print("response file path: ", self.response_file_path)
        print("session tracking file path: ", self.session_tracking_file_path)

        # block training
        if self.played_audio_cnt % 20 == 0 and self.played_audio_cnt > 0 and self.sounds:
            if self.response_buttons is not None:
                for button in self.response_buttons:
                    button.setEnabled(False)
            self.start_countdown(30, False)    # take 30 seconds break after playing 20 audio files
            self.played_audio_cnt = 0   # reinitialize played audio file count as zero to prevent loop
            return

        if self.sounds:
            self.current_sound = self.sounds.pop(0)
            self.played_audio_cnt += 1  # increment count of played audio file

            try:    
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

                # Adjust volume of sound file
                volume_factor = 0.3
                data *= volume_factor

                # Set the audio device
                sd.default.device = self.audio_device_id

                # Start timer for reaction time
                self.start_time = time.time()

                # Play the sound file
                sd.play(data, fs, blocking=True)  
                
                # Update UI after playback
                if self.training_type == "Production Training":
                    self.prompt_label.setText("Try to reproduce the sound")
                    self.toggle_recording()
                else:
                    self.prompt_label.setText("Select the sound you heard")
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
        self.prompt_label.setText("Recording... Try to match the original sound")

        # # Create session_recordings folder if it doesn't exist
        date = datetime.date.today()
        file = self.current_sound.split("/")[-1].split(".")[0]
        self.production_recording_file = os.path.join(self.production_recording_path, f"{date}_{file}.wav")

        # Set default device
        sd.default.device = (self.input_device_id, self.audio_device_id)  

        # Start recording and countdown timer
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
                self.prompt_label.setText("Select the sound you heard")
                if self.response_buttons is not None:
                    for button in self.response_buttons:
                        button.setEnabled(True)

    def process_response(self, response):

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

        # display feedback on screen 
        self.provide_feedback(is_correct, correct_answer)

        # write to response file
        self.write_response(self.current_sound.split("/")[-1], 
                            reaction_time, response=response, solution=correct_answer)

    def provide_feedback(self, is_correct=None, correct_answer=None):
        """
        Display feedback to the participant based on their response.

        Behavior depends on `training_type`:
            - Minimal feedback: Only 'Correct' or 'Incorrect'.
            - Full feedback: Shows correctness and correct tone number.
            - Production training: Evaluates recorded tone via ML model and 
            displays predicted vs. target tone.

        Args:
            is_correct (bool, optional): Whether the participant's response was correct.
            correct_answer (int, optional): Correct tone number.

        Returns:
            None
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
            # The block below is commented out; replaced by ML-based feedback.
            # # Placeholder - display correlation function
            # # obtain f0
            # played_f0 = self.extract_fundamental_pitch(self.current_sound)
            # recorded_f0 = self.extract_fundamental_pitch(self.production_recording_file)
            # 
            # # plot correlation function
            # correlation = np.correlate(played_f0, recorded_f0, mode="full")
            # normalization = np.sqrt(np.sum(played_f0**2) * np.sum(recorded_f0**2))
            # correlation = correlation / normalization
            #
            # lags = np.arange(-len(recorded_f0) + 1, len(played_f0))
            # max_corr = np.max(correlation)
            #
            # fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            # ax.plot(lags, correlation) 
            # ax.set_xlabel("Lag")
            # ax.set_ylabel("Correlation")
            # ax.set_title("Cross-correlation for played tone file and your production tone.\n")
            # ax.text(0.5, -0.15, f"You are {max_corr * 100:.1f}% similar to the original pitch.",
            #         transform=ax.transAxes, ha='center', va='top', fontsize=12)
            # 
            # fig.set_size_inches(5, 5)
            # buf = BytesIO()
            # fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            # buf.seek(0)
            # self.feedback_label.setPixmap(QPixmap.fromImage(QImage.fromData(buf.getvalue(), 'PNG')))
            # plt.close(fig)

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

        # Hide feedback and enable buttons before moving to the next audio file
        if self.training_type == "Production Training":
            QTimer.singleShot(5000, self.clear_feedback_enable_buttons)
        else:
            QTimer.singleShot(1500, self.clear_feedback_enable_buttons)

    def extract_fundamental_pitch(self, audio):
            """
            Extract and normalize the fundamental frequency (f0) contour from an audio file.

            Uses `librosa.pyin` with pitch range defined by the voice preset:
                - Preset 1 (male): 60–200 Hz
                - Preset 2 (female): 180–350 Hz

            Args:
                audio (str): Path to the audio file.

            Returns:
                pandas.Series: Min–max normalized f0 values indexed by time.

            Raises:
                ValueError: If preset is invalid.
            """
            y, sr = librosa.load(audio)

            # extract f0 (preset 1 (males): 60-200Hz, preset 2 (females): 180-350Hz)
            if self.preset == 1:
                fmin, fmax = 60, 200
            elif self.preset == 2:
                fmin, fmax = 180, 350
            else:
                raise ValueError(f'Invalid preset value: {self.preset}. Expected 1 for male voice or 2 for female voices.')

            f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr)
            f0 = f0[~np.isnan(f0)]

            # min-max normalization
            f0_min = f0.min()
            f0_max = f0.max()
            f0 = (f0 - f0_min) / (f0_max - f0_min)

            times = librosa.times_like(f0, sr=sr)
            return pd.Series(f0, index=times)      

    def range_est(self, audio):
        """
        Estimate the vocal range (male/female preset) based on mean pitch from audio.
        Args:
            audio (str): Path to the audio file.
        Returns:
            int: 1 for male preset, 2 for female preset.
        Raises:
            ValueError: If no valid pitch is detected.
        """
        y, sr = librosa.load(audio)
        # Use a wide pitch range to cover both male and female voices
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
        f0 = f0[~np.isnan(f0)]
        if len(f0) == 0:
            raise ValueError("No valid pitch detected for range estimation.")
        mean_pitch = np.mean(f0)

        # 180-200Hz is overlapping zone for male and female voice, take the mean of 180 + 200 = 190Hz
        return 1 if mean_pitch < 190 else 2

    def tone_evaluation(self):
        """
            Predict the tone of the most recent production recording using the ML model.

            Workflow:
                1. Load recorded WAV file from `self.production_recording_file`.
                2. Extract min–max normalized f0 contour via `extract_fundamental_pitch`.
                3. Resample or pad the contour to match the model's expected input length.
                4. Run the model to predict the tone label.

            Raises:
                FileNotFoundError: If the recording file does not exist.
                ValueError: If f0 extraction fails or produces empty data.
            """
        # 1) Load the just-recorded WAV file
        audio_path = getattr(self, "production_recording_file", None)
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Recording not found for tone evaluation: {audio_path}")

        # 2) obtain min-max normalized f0 
        f0_series = self.extract_fundamental_pitch(audio_path)
        if f0_series is None or len(f0_series) == 0:
            raise ValueError("No valid f0 extracted from recording for tone evaluation.")
        f0 = f0_series.values.astype(float)

        # 3) Format to match training columns
        #  Determine expected width from the model if possible, else default to 128
        model = load_tone_model(os.path.join('src', 'model_training', 'tone_prediction_model.pkl'))
        # src/model_training/tone_prediction_model.pkl
        if isinstance(model, tuple):
            model = model[0]
        expected_n = getattr(model, "n_features_in_", 128)
        # Ensure reasonable size
        if not isinstance(expected_n, (int, np.integer)) or expected_n <= 0 or expected_n > 4096:
            expected_n = 128

        # Resample/pad/trim the normalized f0 to exactly expected_n columns
        if len(f0) == 0:
            raise ValueError("Empty f0 after normalization.")
        if len(f0) == 1:
            contour = np.repeat(f0, expected_n)
        else:
            x_src = np.linspace(0.0, 1.0, num=len(f0))
            x_tgt = np.linspace(0.0, 1.0, num=expected_n)
            contour = np.interp(x_tgt, x_src, f0)

        X = contour.reshape(1, -1)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            pred_idx = int(np.argmax(proba))
        else:
            y_pred = model.predict(X)
            pred_idx = int(y_pred[0])

        # Map class index (0–3) -> tone label (1–4) if necessary
        self.production_response = pred_idx + 1 if pred_idx in (0, 1, 2, 3) else pred_idx

    def clear_feedback_enable_buttons(self):
        self.feedback_label.clear()
        if self.response_buttons is not None:
            for button in self.response_buttons:
                button.setStyleSheet("")  # remove highlight
                button.setEnabled(True)
        self.prompt_label.setText("Listen to the sound")
        QTimer.singleShot(1000, self.play_sound) 
        
    def write_response(self, audio_file, reaction_time, response=0, solution=0, accuracy=0):
        """
        Log participant responses and session statistics to CSV files.

        For perception training:
            - Logs date, audio file, chosen response, correct solution, and reaction time.
        For production training:
            - Logs predicted tone, target tone, production accuracy, and reaction time.
        For both training:
            - Updates session tracking file with per-tone and overall accuracies.

        Args:
            audio_file (str): Name of the sound stimulus file.
            reaction_time (float): Response latency in seconds.
            response (int, optional): Participant's chosen tone (default 0).
            solution (int, optional): Correct tone number (default 0).
            accuracy (float, optional): Accuracy score for production training (default 0).

        Returns:
            None
        """
        # Write training response file
        with open(self.response_file_path, mode="a", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)

            if self.training_type != "Production Training":
                csv_writer.writerow([datetime.date.today(), audio_file, response, solution, round(reaction_time, 4)])
            else:
                # Record predicted tone (response), target tone (solution), production accuracy
                # Target tone parsed from filename; predicted tone comes from self.production_response
                try:
                    target_tone = int(re.search(r"\d+", audio_file).group())
                except Exception:
                    target_tone = solution if solution else 0

                csv_writer.writerow([datetime.date.today(), audio_file, self.production_response, target_tone, self.production_accuracy, round(reaction_time, 4)])

        # Write session tracking file 
        if len(self.sounds) == 0:

            # Open the training file in append mode and write session data
            with open(self.session_tracking_file_path, mode="a", newline="") as session_file:
                session_writer = csv.writer(session_file)

                # Read response file into DataFrame
                df = pd.read_csv(f"{self.response_file_path}")

                # Ensure tone is extracted from audio_file as a string key '1'..'4'
                df["tone"] = df["audio_file"].astype(str).str.extract(r"(\d+)")

                if self.training_type == "Production Training":
                    # For production: use the recorded probability/accuracy directly
                    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")

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

        # Determine response accuracy for perception training
        df["accuracy"] = df["response"] == df["solution"] if self.training_type != "Production Training" else np.nan

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

        # read csv files
        file = os.path.join(main_path, self.response_file_path)
        df = pd.read_csv(file)

        # split the session into blocks
        df = self.split_block(df, 10)

        # plot
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        sns.lineplot(df, x="block", y="accuracy_mean", marker="o", ax=ax)
        
        # customize plot settings
        xlimit = df["block"].unique()[-1]
        ax.set(xticks=[(i + 1) * 2 for i in range(xlimit // 2)], yticks=[0, 20, 40, 60, 80, 100],
                xlim=(0, xlimit+1), ylim=(0, 105))
        ax.set_xlabel("Block")
        ax.set_ylabel("Accuracy(%)")
        ax.set_title("Participant: " + self.participant_id)

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
        for line_obj, label in zip(ax.lines, ax.get_legend().texts):
                if label.get_text() != "overall":  
                        line_obj.set_alpha(0.5)
                else:
                        line_obj.set_linewidth(3)

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

        if self.blocks_plot is None:
            print("Error: Plot for block accuracy was not generated.")
            return
        elif self.sessions_plot is None:
            print("Error: Plot for session accuracy was not generated.")
            return

        self.end_training_signal.emit(self.participant_id, self.training_type, self.score, self.blocks_plot, self.sessions_plot)