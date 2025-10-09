from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QStackedWidget
from PyQt5.QtCore import Qt
from .start_page import StartPage
from .training_page import TrainingPage
from .feedback_page import FeedbackPage
from .volume_check_page import VolumeCheckPage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tone Training Application")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Stack widget to hold start page and training page
        self.stacked_widget = QStackedWidget()
        self.layout.addWidget(self.stacked_widget)

        # Initialize pages
        self.start_page = StartPage(self)
        self.training_page = TrainingPage(self)
        self.feedback_page = FeedbackPage(self)
        self.volume_check_page = VolumeCheckPage(self)

        # Add pages to the stacked widget
        self.stacked_widget.addWidget(self.start_page)
        self.stacked_widget.addWidget(self.training_page)
        self.stacked_widget.addWidget(self.feedback_page)
        self.stacked_widget.addWidget(self.volume_check_page)

        # Connect signals
        self.start_page.start_training_signal.connect(self.start_training)
        self.training_page.end_training_signal.connect(self.finish_training)
        self.feedback_page.return_to_start_signal.connect(self.return_to_start)
        self.start_page.volume_check_signal.connect(self.volume_check)
        self.volume_check_page.volume_check_complete.connect(self.start_training_after_volume_check)
        self.showMaximized()  # Start maximized

    def start_training(self, participant_id, training_type, sounds, device_id, input_device_id=None):
        # Set up training session in the training page with all necessary parameters
        # Accept optional input_device_id to support production training
        self.training_page.setup_training(participant_id, training_type, sounds, device_id, input_device_id)

        # Switch to the training page
        self.stacked_widget.setCurrentWidget(self.training_page)

    def finish_training(self, participant_id, training_type, score, blocks_plot, sessions_plot):
        # Send feedback data to FeedbackPage
        self.feedback_page.set_feedback_data(participant_id, training_type, score, blocks_plot, sessions_plot)
        self.stacked_widget.setCurrentWidget(self.feedback_page)

    def return_to_start(self):
        # Return to StartPage
        self.stacked_widget.setCurrentWidget(self.start_page)

    def volume_check(self, input_device_id):
        self.volume_check_page.input_device_id = input_device_id
        self.volume_check_page.start_volume_check()
        self.stacked_widget.setCurrentWidget(self.volume_check_page)

    def start_training_after_volume_check(self):
        # Once volume check is complete, proceed to training
        # Pass the checked input device id from the volume check page to training
        input_id = getattr(self.volume_check_page, 'input_device_id', None)
        self.start_training(self.start_page.participant_id, self.start_page.training_type, self.start_page.sounds, self.start_page.output_device_id, input_id)