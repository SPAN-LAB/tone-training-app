from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QStackedWidget
from PyQt5.QtCore import Qt
from .start_page import StartPage
from .training_page import TrainingPage
from .feedback_page import FeedbackPage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tone Training Application")
        self.setFixedSize(600, 300)

        # Disable the maximize button
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Stack widget to hold start page and training page
        self.stacked_widget = QStackedWidget()
        self.layout.addWidget(self.stacked_widget)

        # Initialize Start Page, Training Page, and Feedback Page
        self.start_page = StartPage(self)
        self.training_page = TrainingPage(self)
        self.feedback_page = FeedbackPage(self)

        # Add pages to the stacked widget
        self.stacked_widget.addWidget(self.start_page)
        self.stacked_widget.addWidget(self.training_page)
        self.stacked_widget.addWidget(self.feedback_page)

        # Connect signals
        self.start_page.start_training_signal.connect(self.start_training)
        self.training_page.end_training_signal.connect(self.finish_training)
        self.feedback_page.return_to_start_signal.connect(self.return_to_start)

    def start_training(self, participant_id, training_type, sounds, device_id):
        # Set up training session in the training page with all necessary parameters
        self.training_page.setup_training(participant_id, training_type, sounds, device_id)
        
        # Switch to the training page
        self.stacked_widget.setCurrentWidget(self.training_page)

    def finish_training(self, participant_id, training_type, score):
        # Send feedback data to FeedbackPage
        self.feedback_page.set_feedback_data(participant_id, training_type, score)
        self.stacked_widget.setCurrentWidget(self.feedback_page)

    def return_to_start(self):
        # Return to StartPage
        self.stacked_widget.setCurrentWidget(self.start_page)