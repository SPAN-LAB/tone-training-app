import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow  # Import the MainWindow class

def main():
    app = QApplication(sys.argv)

    # Create the main window and show it
    window = MainWindow()
    window.show()

    # Start the event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()