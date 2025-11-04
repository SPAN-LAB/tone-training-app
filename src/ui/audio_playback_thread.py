from PyQt5.QtCore import QThread, pyqtSignal
import sounddevice as sd
import numpy as np

class PlayThread(QThread):
    """
    Plays numpy audio data in a background thread to avoid blocking the GUI.
    
    Emits a `finished` signal when playback is complete.
    """
    
    # QThread.finished is a built-in signal we can use.

    def __init__(self, data, samplerate, device=None, parent=None):
        """
        Args:
            data (np.ndarray): The audio data to play.
            samplerate (int): The sample rate of the audio data.
            device (int, optional): The output device ID. Defaults to None.
        """
        super().__init__(parent)
        self.data = data
        self.samplerate = samplerate
        self.device = device

    def run(self):
        """
        The main work of the thread.
        Plays the audio and waits for it to finish.
        """
        try:
            if self.device is not None:
                sd.play(self.data, self.samplerate, device=self.device)
            else:
                sd.play(self.data, self.samplerate)
            sd.wait()
        except Exception as e:
            # Print errors for debugging, but don't crash the app
            print(f"Playback error in background thread: {e}")