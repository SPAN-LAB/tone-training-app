# import sys
# from PyQt5.QtWidgets import QApplication
# from ui.main_window import MainWindow

# def main():
#     app = QApplication(sys.argv)

#     # Initialize and show MainWindow
#     window = MainWindow()
#     window.show()

#     # Start the application's event loop
#     sys.exit(app.exec_())

# if __name__ == "__main__":
#     main()


import sys
import os
from PyQt5.QtWidgets import QApplication

# Dynamically add the src directory to the Python path
if hasattr(sys, '_MEIPASS'):  # PyInstaller runtime environment
    sys.path.append(os.path.join(sys._MEIPASS, "src"))
else:
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from ui.main_window import MainWindow  # Import after adjusting the path

def main():
    app = QApplication(sys.argv)

    # Initialize and show MainWindow
    window = MainWindow()
    window.show()

    # Start the application's event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()