from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QSpacerItem, QSizePolicy
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
from io import BytesIO

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

        # Main layout include scores and plots
        main_layout.addLayout(top_layout)

        # Score
        self.score_label = QLabel()
        main_layout.addWidget(self.score_label)

        # Plot for block accuracy
        self.plot_label1 = QLabel()
        main_layout.addWidget(self.plot_label1)

        # Plot for session accuracy
        self.plot_label2 = QLabel()
        main_layout.addWidget(self.plot_label2)

        # Bottom section (Button to return main page)
        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)

        # Button 
        self.return_button = QPushButton("Return to Main Page")
        self.return_button.clicked.connect(self.on_return_button_clicked)
        button_layout.addWidget(self.return_button)

        main_layout.addLayout(button_layout)

    def set_feedback_data(self, participant_id, training_type, score, blocks_plot, sessions_plot):
        self.participant_id_label.setText(f"Participant ID: {participant_id}")
        self.training_type_label.setText(f"Training Type: {training_type}")
        self.score_label.setText(f"Score: {score * 100} %")

        if blocks_plot:
            self.plot_label1.setPixmap(self.figure_to_pixmap(blocks_plot))
        else:
            self.plot_label1.setText("No block accuracy plot available")

        if sessions_plot:
            self.plot_label2.setPixmap(self.figure_to_pixmap(sessions_plot))
        else:
            self.plot_label2.setText("No session accuracy plot available")
    
    def figure_to_pixmap(self, plot):
        """Convert a matplotlib figure to a QPixmap for display in QLabel."""
        buf = BytesIO()
        plot.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = QImage.fromData(buf.getvalue()) 
        return QPixmap.fromImage(img)

    def on_return_button_clicked(self):
        self.return_to_start_signal.emit()