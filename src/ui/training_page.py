from .volume_check_page import VolumeCheckPage

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QErrorMessage
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QKeyEvent, QFont

import os, re, datetime, time, csv, random
import sounddevice as sd
import soundfile as sf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
class TrainingPage(QWidget):
    # Signal emitted to end training and display results
    end_training_signal = pyqtSignal(str, str, float, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Session Information
        self.participant_id = ""
        self.training_type = ""
        self.audio_device_id = None
        self.input_device_id = None  
        self.session_num = 1

        # Playing Sounds
        self.current_sound = None
        self.sounds = []
        self.played_audio_cnt = 0
   
        # Perception training
        self.correct_response = 0
        self.total_sound_files = 0
        self.response_buttons = None
        self.setFocusPolicy(Qt.StrongFocus)     # Set focus policy to accept keyboard focus

        # Production training
        self.is_recording = False  
        self.recorded_audio_path = ""  # Path for storing users' recordings production training
        self.production_accuracy = 0

        self.start_time = None

        # Feedback
        self.response_file_path = ""
        self.session_track_file_path = ""
        self.date = datetime.date.today()
        self.blocks_plot = None
        self.sessions_plot = None

        # Error message
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

        # Visualization label
        self.visualization_label = QLabel("Visual feedback will be displayed here.")
        self.visualization_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.visualization_label)

        # TODO: Update feedback label
        self.feedback_label = QLabel("")
        self.feedback_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.feedback_label)

    def setup_training(self, participant_id, training_type, sounds, device_id, input_device_id=None):

        # initialization
        self.participant_id = participant_id
        self.training_type = training_type
        self.sounds = sounds
        self.audio_device_id = device_id  
        self.input_device_id = input_device_id  
        self.correct_response = 0
        self.total_sound_files = len(sounds)

        # random shuffle audio files
        random.shuffle(self.sounds)
        
        # setup training UI
        if training_type == "Production Training":
            self.setup_production_training()
        else:
            self.setup_ui()

        # start playing audio files
        QTimer.singleShot(1000, self.play_sound)  

    def disable_response_button(self):
        if self.response_buttons is not None:
                for button in self.response_buttons:
                    button.setEnabled(False)

    def play_sound(self):

        # block training (20 audio files per block)
        if self.played_audio_cnt % 20 == 0 and self.played_audio_cnt > 0 and self.sounds:

            self.disable_response_button()      # disable response button
            self.start_countdown(30, False)     # 30 seconds break
            self.played_audio_cnt = 0           # reset played audio file count to zero

            return

        # play audio files
        if self.sounds:
            self.current_sound = self.sounds.pop(0)
            self.played_audio_cnt += 1 

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

                # Check if the file exists
                if not os.path.isfile(full_path):
                    raise FileNotFoundError(f"File not found: {full_path}")
                
               # read sound file for sample rate and number of channels
                data, fs = sf.read(full_path, dtype="float32")   

                # Adjust volume of sound file
                volume_factor = 0.3
                data *= volume_factor
               
                sd.default.device = self.audio_device_id    # Set the audio device
                self.start_time = time.time()               # Start reaction timer
                sd.play(data, fs, blocking=True)            # Play audio file
                
                # update UI after playing audio file
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
        """
        Start or stop recording based on current state
        """
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """
        Start recording.
        """
        self.is_recording = True
        self.prompt_label.setText("Recording... Try to match the original sound")

        # Create session_recordings folder if it doesn't exist
        participant_folder = os.path.join("participants", self.participant_id, "recording")
        os.makedirs(participant_folder, exist_ok=True)

        date = datetime.date.today()
        file = self.current_sound.split("/")[-1]
        file = file.split(".")[0]
        self.recorded_audio_path = os.path.join(participant_folder, f"{date}_{file}.wav")

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

        sf.write(self.recorded_audio_path, self.recording, 44100)
        self.prompt_label.setText("Recording complete. Analyzing...")

        # write response file
        self.write_response(self.participant_id, 
                            self.training_type, 
                            self.current_sound.split("/")[-1], 
                            reaction_time, 
                            accuracy=self.production_accuracy)

        self.analyze_recording()
        # TODO: Delete recording after analysis 

    def analyze_recording(self):

        # Placeholder: Implement pitch comparison and feedback display
        self.visualization_label.setText("Comparing original and recorded pitch tracks...")

        # TODO: Implement accuracy calculation
        
        # TODO: Display actual pitch track visualization and compute similarity
        self.provide_feedback()


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
                QTimer.singleShot(1000, self.play_sound)

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
            self.correct_response += 1

        # display feedback on screen 
        self.provide_feedback(is_correct, correct_answer)

        # write to response file
        self.write_response(self.participant_id, self.training_type, self.current_sound.split("/")[-1], 
                            reaction_time, response=response, solution=correct_answer)

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

        # Hide feedback and enable buttons before moving to the next audio file
        QTimer.singleShot(1500, self.clear_feedback_enable_buttons)

    def clear_feedback_enable_buttons(self):
        self.feedback_label.clear()
        if self.response_buttons is not None:
            for button in self.response_buttons:
                button.setStyleSheet("")  # remove highlight
                button.setEnabled(True)
        self.prompt_label.setText("Listen to the sound")
        QTimer.singleShot(1000, self.play_sound) 

    def create_response_file(self, participant_id, training):

        # create directory for block training and session tracking respectively
        block_training_dir = os.path.join("participants", participant_id, training, "block_training")
        session_track_dir = os.path.join("participants", participant_id, training, "session_tracking")

        os.makedirs(block_training_dir, exist_ok=True)
        os.makedirs(session_track_dir, exist_ok=True)

        # determine session number for file name
        session_nums = [int(re.findall(r"\d+", file)[0]) for file in os.listdir(block_training_dir) if file.endswith(".csv")]
        print("Session nums: ", session_nums)
        if session_nums:
            self.session_num = max(session_nums) 
            self.session_num += 1 
        file_name = f"session{self.session_num}.csv"

        # create block training response file
        self.response_file_path = os.path.join(block_training_dir, file_name)
        try:
            with open(self.response_file_path, 'w', newline='') as csvfile:
                header = ["date", "audio_file", "response", "solution", "reaction_time"]
                if training == "Production Training":
                    header.append("accuracy")
                csv.writer(csvfile).writerow(header)

        except Exception as e:
            self.error_dialog.showMessage(f"Error: Failed to create response file. {e}")
            print(f"Error creating response file: {e}")

        # create session tracking file
        self.session_track_file_path = os.path.join(session_track_dir, file_name)
        try:
            with open(self.session_track_file_path, 'w', newline='') as csvfile:
                header = ["session", "date", "subject", "accuracy"]
                csv.writer(csvfile).writerow(header)

        except Exception as e:
            self.error_dialog.showMessage(f"Error: Failed to create session tracking file for {training}. {e}")
            print(f"Error creating session tracking file: {e}")

    def write_response(self, participant_id, training, audio_file, reaction_time, response=0, solution=0, accuracy=0):

        # create response file and session tracking file
        if self.response_file_path == '' or self.session_track_file_path == '':
            self.create_response_file(participant_id, training)

        # write response file
        with open(self.response_file_path, mode="a", newline="") as response_file:
            response_writer = csv.writer(response_file)

            response = [self.date, audio_file, response, solution, round(reaction_time, 4)]

            if training == "Production Training":               
                response.append(accuracy) 
                    
            response_writer.writerow(response)

        # write session tracking file
        if len(self.sounds) == 0:

            with open(self.session_track_file_path, mode="a", newline="") as session_file:
                session_writer = csv.writer(session_file)

                # Read response file
                df = pd.read_csv(self.response_file_path)
                total_tone = {"1":0, "2":0, "3":0, "4":0}
                correct_tone = {"1":0, "2":0, "3":0, "4":0}

                for _, item in df.iterrows():

                    # get number of audio files based on tone
                    tone = re.search(r'\d+', item["audio_file"]).group()   
                    total_tone[tone] += 1

                    # count correct answer
                    if item["response"] == item["solution"]:
                        correct_tone[tone] += 1

                # calculate average accuracy for each tone
                result = {}
                for t in correct_tone:
                    if total_tone[t] == 0:      # if there is no audio files with this tone during training
                        result[t] = np.nan 
                    else:
                        result[t] = correct_tone[t] / total_tone[t]

                # write data 
                for key, value in result.items():
                    session_writer.writerow([self.session_num, self.date, key, value])  

                # calculate average accuracy overall
                self.score = (self.correct_response / self.total_sound_files)

                # write data
                session_writer.writerow([self.session_num, self.date, "overall", self.score])  

    def split_block(self, df, files_per_block):

        # Determine response accuracy for perception training
        df["accuracy"] = df["response"] == df["solution"]

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

        # read csv files
        df = pd.read_csv(self.response_file_path)

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

        # read all session tracking files
        session_track_folder = os.path.split(self.session_track_file_path)[0]
        session_track_list = os.listdir(session_track_folder)
        df = ( pd.concat([pd.read_csv(os.path.join(session_track_folder, file)) for file in session_track_list])
              .reset_index()
              .fillna(0)
              .drop(["index"], axis=1)
        )
        
        # scale accuracy score into percentage
        df["accuracy"] *= 100

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
            self.error_dialog.showMessage("Error: Plot for block accuracy was not generated.")
            return
        elif self.sessions_plot is None:
            self.error_dialog.showMessage("Error: Plot for session accuracy was not generated.")
            return

        self.end_training_signal.emit(self.participant_id, self.training_type, self.score, self.blocks_plot, self.sessions_plot)