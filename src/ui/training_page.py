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
import pandas as pd
import random
import seaborn as sns # TODO: Install seaborn when bundle executable

# universal path
main_path = os.path.join("Volumes", "gurindapalli", "projects", "Plasticity_training")

# TODO: Modify instructions in production training ui
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
        self.played_audio_cnt = 0
        self.session_num = 1

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

        # random shuffle audio files
        random.shuffle(self.sounds)
        
        if training_type == "Production Training":
            self.setup_production_training()
        else:
            self.setup_ui()

        # print("In setup_training() in training_page.py")
        # print("After assign current sound: ", self.current_sound, self.sounds)
        QTimer.singleShot(250, self.play_sound)  

    def play_sound(self):
        # print("In play sound()")
        # print("Remaining sound file: ", [f for f in self.sounds])

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

                # Set the audio device and play the sound with the correct number of channels
                sd.default.device = self.audio_device_id
                sd.play(data, fs, blocking=True)  

                # Get reaction starting time
                self.start_time = time.time()

                # Update UI after playback
                if self.training_type == "Production Training":
                    self.prompt_label.setText("Try to reproduce the sound")
                    # print("In play_sound(), within if stmt for production training")
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

    def stop_recording(self):
        self.is_recording = False
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
        QTimer.singleShot(500, self.clear_feedback_enable_buttons)

    def clear_feedback_enable_buttons(self):
        self.feedback_label.clear()
        if self.response_buttons is not None:
            for button in self.response_buttons:
                button.setEnabled(True)
        QTimer.singleShot(250, self.play_sound) 

    def finish_training(self):
        self.end_training_signal.emit(self.participant_id, self.training_type, self.score)

    def write_response(self, participant_id, training, audio_file, reaction_time, response=0, solution=0, accuracy=0):

        # Create participants folder if it doesn't exist
        participant_folder = os.path.join("participants", participant_id)
        os.makedirs(participant_folder, exist_ok=True)
        
        # Create training folder inside the participant's folder if it doesn't exist
        training_folder = os.path.join(participant_folder, training)
        folder_exist = os.path.exists(training_folder)
        os.makedirs(training_folder, exist_ok=True)
        
        # Obtain previous session number
        print("played audio count: ", self.played_audio_cnt)
        if folder_exist and self.played_audio_cnt == 1:
            self.session_nums = [int(re.findall(r"\d+", file)[0]) for file in os.listdir(training_folder) if file.endswith(".csv")]
            self.session_num = max(self.session_nums)
            self.session_num += 1

        print("Session number: ", self.session_num)

        # Define the response file path
        response_file = os.path.join(training_folder, f"session{self.session_num}.csv")
        
        # Check if the file already exists
        file_exists = os.path.isfile(response_file)
        
        # Open the file in append mode and write the data
        with open(response_file, mode="a", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)

            if training != "Production Training":
            
                if not file_exists:
                    csv_writer.writerow(["date", "audio_file", "response", "solution", "reaction_time"])
                
                csv_writer.writerow([datetime.date.today(), audio_file, response, solution, round(reaction_time, 4)])

            else:
                if not file_exists:
                    csv_writer.writerow(["date", "audio_file", "accuracy", "reaction_time"])
                
                csv_writer.writerow([datetime.date.today(), audio_file, accuracy, round(reaction_time, 4)])

        # Define the session tracking folder path
        session_folder = os.path.join("participants", participant_id, "session_tracking")
        os.makedirs(session_folder, exist_ok=True)

        # Create three files for the respective training names
        training_file = os.path.join(session_folder, f"{training}.csv")

        # Check if the file already exists
        file_exists = os.path.isfile(training_file)

        # Write training session accuracy
        if len(self.sounds) == 0:

            # Open the training file in append mode and write session data
            with open(training_file, mode="a", newline="") as session_file:
                session_writer = csv.writer(session_file)

                # Write header if file does not exist
                if not file_exists:
                    session_writer.writerow(["session", "date", "subject", "accuracy"])

                # Read response file to compute tone accuracy
                df = pd.read_csv(f"{response_file}")
                total_tone = {"1":0, "2":0, "3":0, "4":0}
                correct_tone = {"1":0, "2":0, "3":0, "4":0}

                for _, item in df.iterrows():

                    # get number of audio files based on tone
                    tone = re.search(r'\d+', item["audio_file"]).group()   
                    total_tone[tone] += 1

                    # get number of correct answer for perception and accuracy for production
                    if training != "Production Training":
                        if item["response"] == item["solution"]:
                            correct_tone[tone] += 1
                    else:
                        correct_tone[tone] += item["accuracy"]

                # calculate average accuracy for each tone
                result = {}
                for t in correct_tone:
                    if total_tone[t] == 0:
                        result[t] = None 
                    else:
                        result[t] = correct_tone[t] / total_tone[t]

                # Write the date and accuracy information for every tone in the session
                for key, value in result.items():
                    session_writer.writerow([self.session_num,datetime.date.today(), key, value])  

                # Write the date and overall accuracy information
                # TODO: Calculate the overall accuracy for production
                self.score = (self.correct_answers / self.total_questions) * 100 if training != "Production Training" else 0
                session_writer.writerow([self.session_num, datetime.date.today(), "overall", self.score])  


    def read_csv(self, directory, plot_type):
        """
        Helper function to read csv files for plotting accuracy plot.

        Returns:
            DataFrame: Pandas dataframe that contains trainees' record
        """
        if plot_type != "session tracking":
            csv_files = [file for file in os.listdir(directory) if file.endswith('.csv') and file.startswith(self.training_type)]

        else:
            csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
        
        df_list = []
        for file in csv_files:
            data = pd.read_csv(os.path.join(directory, file))
            data["date"] = file.split("_")[0] if plot_type != "session_tracking" else None # TODO: Remove this if there's already date column
            df_list.append(data)
        return pd.concat(df_list, ignore_index=True)
    
    def split_block(self, df, files_per_block):

        # Determine response accuracy for perception training
        df["accuracy"] = df["response"] == df["solution"] if self.training_type != "Production Training" else None

        # Assign block numbers
        df["block"] = (df.index // files_per_block).astype(int) + 1

        # Calculate mean accuracy for every block
        group_accuracy = df.groupby('block')['accuracy'].mean() * 100
        df = df.merge(group_accuracy, on=['block'], suffixes=('', '_mean'))
        df.rename(columns={"accuracy_mean": "accuracy"}, inplace=True) 

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
        dir = os.path.join(main_path, 
                            "tone-training-app", 
                            "participants", 
                            self.participant_id, 
                            self.training_type
                            )
        df = self.read_csv(dir, "block_tracking")

        # extract the records of current session
        num = 60        # TODO: Change to len(audio_files_folder) or to correct number
        df = df[-num:]

        # split the session into blocks
        df = self.split_block(df, 10)

        # plot
        line = sns.lineplot(df, x="block", y="accuracy", marker="o")
        xlimit = df["block"].unique()[-1]
        line.set(xticks=[(i + 1) * 2 for i in range(xlimit // 2)], yticks=[0, 20, 40, 60, 80, 100],
                xlim=(0, xlimit+1), ylim=(0, 105))
        line.set_xlabel("Block")
        line.set_ylabel("Accuracy(%)")
        line.set_title("Participant: " + self.participant_id)

        return line
    
    def plot_session_accuracy(self):
        """ 
        Plot tone accuracy over sessions.

        Returns:
            matplotlib Axes: Line plot show the tone accuracy over blocks.
        """

        global main_path

        # read csv files
        dir = os.path.join(main_path, 
                            "tone-training-app", 
                            "participants", 
                            self.participant_id, 
                            "session_tracking"
                            )
        df = self.read_csv(dir, "session_tracking")

        # TODO: calculate all accuracy in app with the same scale
        # adjust accuracy scale to be percentage
        df.loc[df["subject"] != "overall", "accuracy"] *= 100

        # divide dataframe into different sessions for each five rows
        df["session"] = (df.index // 5).astype(int) + 1

        # plot
        line = sns.lineplot(df, x = "session", y = "accuracy", hue = "subject", marker="o", palette=sns.color_palette("Set1", 5))

        # adjust line attribute to make overall accuracy stand out
        for line_obj, label in zip(line.lines, line.get_legend().texts):
                if label.get_text() != "overall":  
                        line_obj.set_alpha(0.5)
                else:
                        line_obj.set_linewidth(3)

        # set axis ticks
        xlimit = df["session"].unique()[-1]
        line.set(xticks=[i for i in range(xlimit + 1)], yticks=[0, 20, 40, 60, 80, 100],
                xlim=(0.5, xlimit+1), ylim=(0, 105))

        # set axis labels and title
        line.set_xlabel("Session")
        line.set_ylabel("Accuracy(%)")
        line.set_title("Participant: " + self.participant_id)

        # set legend labels
        handles, _ = line.get_legend_handles_labels()
        new_labels = ["Tone 1", "Tone 2", "Tone 3", "Tone 4", "Overall"]  
        line.legend(handles=handles, labels=new_labels, title="Subject", loc="upper left", bbox_to_anchor=(1, 1))

        return line