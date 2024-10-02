from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QStackedWidget, QDesktopWidget
from PyQt5.QtCore import Qt
from .start_page import StartPage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tone Training Application")
        self.setFixedSize(400, 300)

        # Disable the maximize button
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.stacked_widget = QStackedWidget()
        self.layout.addWidget(self.stacked_widget)

        self.start_page = StartPage(self)
        self.stacked_widget.addWidget(self.start_page)

        # Connect the start_training_signal from StartPage to start_training method
        self.start_page.start_training_signal.connect(self.start_training)

    def start_training(self, participant_id, training_type, sounds):
        print(f"Starting {training_type} for Participant {participant_id} with {len(sounds)} sounds.")
        # We'll implement switching to the training page later
