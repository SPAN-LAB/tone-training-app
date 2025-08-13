from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QStackedWidget
from PyQt5.QtCore import Qt
from .start_page import StartPage
from .training_page import TrainingPage
from .feedback_page import FeedbackPage
from .volume_check_page import VolumeCheckPage
from .range_est_page import RangeEstPage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tone Training Application")
        self.showMaximized()
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
        self.range_est_page = RangeEstPage(self)

        # Add pages to the stacked widget
        self.stacked_widget.addWidget(self.start_page)
        self.stacked_widget.addWidget(self.training_page)
        self.stacked_widget.addWidget(self.feedback_page)
        self.stacked_widget.addWidget(self.volume_check_page)
        self.stacked_widget.addWidget(self.range_est_page)

        # Connect signals
        self.start_page.start_training_signal.connect(self.launch_start_page)
        
        self.start_page.volume_check_signal.connect(self.volume_check)
        self.volume_check_page.volume_check_complete.connect(self.start_page.after_volume_check_complete)
        
        self.start_page.range_est_signal.connect(self.range_est)
        self.range_est_page.range_est_complete.connect(self.start_page.start_training)

        self.feedback_page.return_to_start_signal.connect(self.return_to_start)
        self.training_page.end_training_signal.connect(self.finish_training)


    def launch_start_page(self, participant_id, training_type, sounds, output_device_id, input_device_id, session_num, production_recording_path, response_file_path, session_tracking_file_path):
        # Set up training session in the training page with all necessary parameters
        self.training_page.setup_training(
            participant_id, 
            training_type, 
            sounds, 
            output_device_id,
            input_device_id,
            session_num, 
            production_recording_path, 
            response_file_path, 
            session_tracking_file_path,
            # preset             # Uncomment for manual preset selection and add parameter 'preset' in this function
            )
        
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
        # Switch to volume check page
        self.volume_check_page.input_device_id = input_device_id
        self.volume_check_page.start_volume_check()
        self.stacked_widget.setCurrentWidget(self.volume_check_page)
    
    def range_est(self, input_device_id, participant_id, production_recording_path):
        # Switch to range estimation page
        self.range_est_page.input_device_id = input_device_id
        self.range_est_page.participant_id = participant_id
        self.range_est_page.production_recording_path = production_recording_path
        self.stacked_widget.setCurrentWidget(self.range_est_page)