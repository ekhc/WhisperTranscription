from PySide6 import QtCore

class ProgressUpdater(QtCore.QObject):
    progress_updated = QtCore.Signal(float, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
    
    @QtCore.Slot(float, str)
    def update(self, progress: float, status: str):
        # Directly update the UI by calling a method on the parent (main window)
        # We assume the parent is the main window that has the _apply_progress() method.
        if self.parent() is not None:
            self.parent()._apply_progress(progress, status)
        self.progress_updated.emit(progress, status)
