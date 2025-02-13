import sys
from PySide6 import QtWidgets
from src.ui.app import WhisperTranscriptionApp
from src.ui.fonts import setup_application_font

def main():
    app = QtWidgets.QApplication(sys.argv)
    setup_application_font()
    window = WhisperTranscriptionApp()
    app.aboutToQuit.connect(window.cleanup)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()