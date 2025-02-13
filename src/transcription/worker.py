from typing import Optional, Callable
from PySide6 import QtCore
from src.transcription.transcriber import Transcriber

class TranscriptionWorkerSignals(QtCore.QObject):
    transcription_finished = QtCore.Signal(str)
    error = QtCore.Signal(str)
    finished = QtCore.Signal()
    progress = QtCore.Signal(float, str)

class TranscriptionWorker(QtCore.QRunnable):
    def __init__(self, 
                 stt_file: str, 
                 num_speakers: Optional[int] = None,
                 model_name: str = "base",
                 hf_token: str = None,
                 pyannote_token: str = None,
                 use_pyannote_api: bool = False,
                 perform_diarization: bool = True,
                 language: Optional[str] = None,  # New parameter
                 progress_callback: Optional[Callable[[float, str], None]] = None):
        super().__init__()
        self.stt_file = stt_file
        self.num_speakers = num_speakers
        self.model_name = model_name
        self.hf_token = hf_token
        self.pyannote_token = pyannote_token
        self.use_pyannote_api = use_pyannote_api
        self.perform_diarization = perform_diarization
        self.language = language
        self.signals = TranscriptionWorkerSignals()
        self._stop_requested = False
        self.progress_callback = progress_callback or self._default_progress_callback

    def _default_progress_callback(self, progress: float, status: str):
        self.signals.progress.emit(progress, status)

    def stop(self):
        self._stop_requested = True
        print("[DEBUG] Stop requested for transcription.")

    def run(self):
        try:
            transcriber = Transcriber(
                model_name=self.model_name,
                hf_token=self.hf_token,
                pyannote_token=self.pyannote_token,
                use_pyannote_api=self.use_pyannote_api,
                perform_diarization=self.perform_diarization,
                progress_callback=self.progress_callback
            )
            text = transcriber.transcribe(
                self.stt_file,
                self.num_speakers,
                language=self.language,
                stop_flag=lambda: self._stop_requested
            )
            if self._stop_requested:
                self.signals.error.emit("Transcription stopped by user.")
                return
            print("[DEBUG] Transcription output:", text)
            self.signals.transcription_finished.emit(text)
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()
