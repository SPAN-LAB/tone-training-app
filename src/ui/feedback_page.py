from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QSpacerItem, QSizePolicy
from PyQt5.QtCore import pyqtSignal

class FeedbackPage(QWidget):
    return_to_start_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        # Section to display the result
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)  # Add some margin around the edges

        # Top section (Participant ID and Training Type)
        top_layout = QVBoxLayout()
        top_layout.setSpacing(20)

        # Participant ID
        self.participant_id_label = QLabel()
        top_layout.addWidget(self.participant_id_label)

        # Training type
        self.training_type_label = QLabel()
        top_layout.addWidget(self.training_type_label)

        # Score
        self.score_label = QLabel()
        top_layout.addWidget(self.score_label)

        main_layout.addLayout(top_layout)

        # Bottom section (Button to return main page)
        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)

        # Button 
        self.return_button = QPushButton("Return to Main Page")
        self.return_button.clicked.connect(self.on_return_button_clicked)
        button_layout.addWidget(self.return_button)

        main_layout.addLayout(button_layout)

    def set_feedback_data(self, participant_id, training_type, score):
        self.participant_id_label.setText(f"Participant ID: {participant_id}")
        self.training_type_label.setText(f"Training Type: {training_type}")
        self.score_label.setText(f"Score: {score} %")

    def on_return_button_clicked(self):
        self.return_to_start_signal.emit()