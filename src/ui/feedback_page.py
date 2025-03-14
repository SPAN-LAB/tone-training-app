from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpacerItem, QSizePolicy, QScrollArea
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
        # Create a scroll area
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)  # Allows resizing of content

        # Create a container widget to hold the actual content
        content_widget = QWidget()
        main_layout = QVBoxLayout(content_widget)
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

        # Main layout includes scores and plots
        main_layout.addLayout(top_layout)

        # Score
        self.score_label = QLabel()
        main_layout.addWidget(self.score_label)

        # Section for side-by-side plots
        plots_layout = QHBoxLayout()

        # Set fixed width & height for QLabel plots
        plot_width = 500
        plot_height = 500

        # Block accuracy plot
        block_layout = QVBoxLayout()
        self.block_title = QLabel("Block Accuracy")
        self.block_title.setStyleSheet("font-weight: bold; font-size: 14px;")  # Make title bold
        self.plot_label1 = QLabel()
        self.plot_label1.setFixedSize(plot_width, plot_height)
        block_layout.addWidget(self.block_title)
        block_layout.addWidget(self.plot_label1)
        plots_layout.addLayout(block_layout, stretch=1)

        # Session accuracy plot
        session_layout = QVBoxLayout()
        self.session_title = QLabel("Session Accuracy")
        self.session_title.setStyleSheet("font-weight: bold; font-size: 14px;")  # Make title bold
        self.plot_label2 = QLabel()
        self.plot_label2.setFixedSize(plot_width, plot_height)
        session_layout.addWidget(self.session_title)
        session_layout.addWidget(self.plot_label2)
        plots_layout.addLayout(session_layout, stretch=1)

        main_layout.addLayout(plots_layout)

        # Bottom section (Button to return to main page)
        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)

        # Button 
        self.return_button = QPushButton("Return to Main Page")
        self.return_button.clicked.connect(self.on_return_button_clicked)
        button_layout.addWidget(self.return_button)

        main_layout.addLayout(button_layout)

        # Set content widget layout
        content_widget.setLayout(main_layout)

        # Add content widget to scroll area
        scroll_area.setWidget(content_widget)

        # Set scroll area as the main layout
        layout = QVBoxLayout(self)
        layout.addWidget(scroll_area)
        self.setLayout(layout)

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
        plot.set_size_inches(5, 5)
        buf = BytesIO()
        plot.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img = QImage.fromData(buf.getvalue()) 
        return QPixmap.fromImage(img)

    def on_return_button_clicked(self):
        self.return_to_start_signal.emit()