from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, QFileDialog, QSpacerItem, QSizePolicy
from PyQt5.QtCore import pyqtSignal, Qt
import os

class StartPage(QWidget):
    start_training_signal = pyqtSignal(str, str, list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)  # Add some margin around the edges

        # Top section (Participant ID and Training Type)
        top_layout = QVBoxLayout()
        top_layout.setSpacing(20)

        # Participant ID input
        id_layout = QHBoxLayout()
        self.participant_id_label = QLabel("Participant ID:")
        self.participant_id_input = QLineEdit()
        id_layout.addWidget(self.participant_id_label)
        id_layout.addWidget(self.participant_id_input)
        top_layout.addLayout(id_layout)

        # Training type selection
        type_layout = QHBoxLayout()
        self.training_type_label = QLabel("Select Training Type:")
        self.training_type_combo = QComboBox()
        self.training_type_combo.addItems(["Production Training", "Perception with Minimal Feedback", "Perception with Full Feedback"])
        type_layout.addWidget(self.training_type_label)
        type_layout.addWidget(self.training_type_combo)
        top_layout.addLayout(type_layout)

        main_layout.addLayout(top_layout)

        # Add a spacer to push the buttons to the bottom
        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Button section at the bottom
        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)

        # Sound file selection
        self.load_sounds_button = QPushButton("Load Sound Files")
        self.load_sounds_button.clicked.connect(self.load_sounds)
        button_layout.addWidget(self.load_sounds_button)

        # Start button
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        button_layout.addWidget(self.start_button)

        main_layout.addLayout(button_layout)

        self.sounds = []

    def load_sounds(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing Sound Files")
        if folder:
            self.sounds = [f for f in os.listdir(folder) if f.endswith('.wav')]
            print(f"Loaded {len(self.sounds)} sound files.")

    def start_training(self):
        participant_id = self.participant_id_input.text()
        training_type = self.training_type_combo.currentText()
        if participant_id and self.sounds:
            self.start_training_signal.emit(participant_id, training_type, self.sounds)
        else:
            print("Please enter a Participant ID and load sound files before starting.")