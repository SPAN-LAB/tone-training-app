import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)

    # Initialize and show MainWindow
    window = MainWindow()
    window.show()

    # Start the application's event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
