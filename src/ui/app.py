import sys
import os
import pyaudio
from datetime import datetime
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Signal, Slot
from PySide6.QtMultimedia import QMediaDevices

import src.audio.recorder as record
from src.ui.waveform import WaveformWidget
from src.ui.progress import ProgressUpdater
from src.transcription.worker import TranscriptionWorker
from src.transcription.language_options import LANGUAGES
from src.ui import fonts
import src.ui.fonts_rc

class WhisperTranscriptionApp(QtWidgets.QMainWindow):
    # Signal to update the waveform.
    waveform_update_signal = Signal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threadpool = QtCore.QThreadPool()
        self.is_recording = False
        self.last_wave = None
        self.current_worker = None
        self._last_signal_time = 0

        # Timers
        self.recording_timer = QtCore.QTimer(self)
        self.recording_timer.timeout.connect(self.update_recording_time)
        self.recording_start_time = None

        # Output directory
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'output'
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize recorder with update_plot callback.
        self.recorder = record.Recorder(
            output_dir=self.output_dir,
            update_func=self.update_plot
        )

        self.setWindowTitle("Whisper Transcription")

        # --- Create a grid layout for the widgets ---
        self.grid_widget = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_widget)

        # Set up various UI components:
        self._setup_input_devices()      # Row 0
        self._setup_output_devices()     # Row 1
        self._setup_model_selection()    # Row 2
        self._setup_tokens()             # Rows 3,4,5
        self._setup_speaker_selection()  # Row 6
        self._setup_language_selection() # Row 7
        self._setup_buttons()            # Row 8
        self._setup_plot()               # Row 9
        self._setup_transcription()      # Rows 10-11
        self._setup_progress()           # Rows 12-13

        # --- End grid layout setup ---

        # Use a vertical layout to hold the grid.
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addWidget(self.grid_widget)
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

        self.progress_updater = ProgressUpdater(self)
        self.waveform_update_signal.connect(self.waveform.update_data, QtCore.Qt.QueuedConnection)

        self.audio_queue_timer = QtCore.QTimer(self)
        self.audio_queue_timer.setInterval(100)
        self.audio_queue_timer.timeout.connect(self.recorder.process_audio_queue)
        self.audio_queue_timer.start()

        self.show()

    def _setup_input_devices(self):
        device_label = QtWidgets.QLabel("Input Device")
        self.device_combo = QtWidgets.QComboBox()
        self.input_devices = []
        self.input_device_indices = []
        pa = pyaudio.PyAudio()
        for i in range(pa.get_device_count()):
            dev_info = pa.get_device_info_by_index(i)
            if dev_info.get('maxInputChannels', 0) > 0:
                self.input_devices.append(dev_info['name'])
                self.input_device_indices.append(i)
        self.device_combo.addItems(self.input_devices)
        self.device_combo.setCurrentIndex(0)
        self.grid_layout.addWidget(device_label, 0, 0)
        self.grid_layout.addWidget(self.device_combo, 0, 1)

    def _setup_output_devices(self):
        self.output_label = QtWidgets.QLabel("Output Device")
        self.output_combo = QtWidgets.QComboBox()
        self.audio_output_devices = QMediaDevices.audioOutputs()
        self.output_combo.addItems([device.description() for device in self.audio_output_devices])
        self.grid_layout.addWidget(self.output_label, 1, 0)
        self.grid_layout.addWidget(self.output_combo, 1, 1)

    def _setup_model_selection(self):
        self.model_label = QtWidgets.QLabel("Model:")
        self.model_combo = QtWidgets.QComboBox()
        models = ["turbo", "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large"]
        self.model_combo.addItems(models)
        tooltip = ("Model Info:\n"
                   "turbo: ~6GB VRAM, 8x faster, 809M params\n"
                   "tiny/tiny.en: ~1GB VRAM, 10x faster, 39M params\n"
                   "base/base.en: ~1GB VRAM, 7x faster, 74M params\n"
                   "small/small.en: ~2GB VRAM, 4x faster, 244M params\n"
                   "medium/medium.en: ~5GB VRAM, 2x faster, 769M params\n"
                   "large: ~10GB VRAM, 1x speed, 1550M params\n\n"
                   "*.en models are English-only and may perform better for English audio.")
        self.model_combo.setToolTip(tooltip)
        self.grid_layout.addWidget(self.model_label, 2, 0)
        self.grid_layout.addWidget(self.model_combo, 2, 1)

    def _setup_tokens(self):
        self.token_label = QtWidgets.QLabel("Hugging Face Token")
        self.token_input = QtWidgets.QLineEdit()
        self.token_input.setPlaceholderText("Enter your Hugging Face token")
        self.token_input.setToolTip("Get your token at https://huggingface.co/settings/tokens")
        env_token = os.getenv("HUGGING_FACE_TOKEN")
        if env_token:
            self.token_input.setText(env_token)
        self.grid_layout.addWidget(self.token_label, 3, 0)
        self.grid_layout.addWidget(self.token_input, 3, 1)

        self.pyannote_token_label = QtWidgets.QLabel("Pyannote API Token")
        self.pyannote_token_input = QtWidgets.QLineEdit()
        self.pyannote_token_input.setPlaceholderText("Enter your Pyannote API token")
        self.pyannote_token_input.setToolTip("Get your token at https://hf.co/pyannote/speaker-diarization")
        pyannote_token = os.getenv("PYANNOTE_TOKEN")
        if pyannote_token:
            self.pyannote_token_input.setText(pyannote_token)
        self.grid_layout.addWidget(self.pyannote_token_label, 4, 0)
        self.grid_layout.addWidget(self.pyannote_token_input, 4, 1)

        self.diarization_label = QtWidgets.QLabel("Diarization Method")
        self.diarization_combo = QtWidgets.QComboBox()
        self.diarization_combo.addItems(["No Diarization", "Pyannote AI", "Pyannote Open Source"])
        self.grid_layout.addWidget(self.diarization_label, 5, 0)
        self.grid_layout.addWidget(self.diarization_combo, 5, 1)

    def _setup_speaker_selection(self):
        self.speaker_label = QtWidgets.QLabel("Number of Speakers")
        self.speaker_combo = QtWidgets.QComboBox()
        self.speaker_combo.addItems(["Auto"] + [str(i) for i in range(1, 11)])
        self.speaker_combo.setToolTip("Set to Auto for auto-detection")
        self.grid_layout.addWidget(self.speaker_label, 6, 0)
        self.grid_layout.addWidget(self.speaker_combo, 6, 1)

    def _setup_language_selection(self):
        self.language_label = QtWidgets.QLabel("Language:")
        self.language_combo = QtWidgets.QComboBox()
        self.language_combo.addItem("Auto-detect")
        language_names = sorted(LANGUAGES.values())
        self.language_combo.addItems(language_names)
        self.grid_layout.addWidget(self.language_label, 7, 0)
        self.grid_layout.addWidget(self.language_combo, 7, 1)

    def _setup_buttons(self):
        self.record_button = QtWidgets.QPushButton("Record")
        self.record_button.clicked.connect(self.on_record_button)
        self.upload_button = QtWidgets.QPushButton("Upload Audio")
        self.upload_button.clicked.connect(self.on_upload_button)
        self.stop_button = QtWidgets.QPushButton("Stop Transcription")
        self.stop_button.clicked.connect(self.stop_transcription)
        self.stop_button.setEnabled(False)
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.record_button)
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.stop_button)
        button_widget = QtWidgets.QWidget()
        button_widget.setLayout(button_layout)
        self.grid_layout.addWidget(button_widget, 8, 0, 1, 2)

    def _setup_plot(self):
        self.waveform = WaveformWidget(self)
        self.waveform.setMinimumHeight(150)
        self.grid_layout.addWidget(self.waveform, 9, 0, 1, 2)

    def _setup_transcription(self):
        self.transcript = QtWidgets.QLabel("Transcription")
        self.transcription_preview = QtWidgets.QTextEdit()
        self.transcription_preview.setReadOnly(True)
        self.grid_layout.addWidget(self.transcript, 10, 0)
        self.grid_layout.addWidget(self.transcription_preview, 11, 0, 1, 2)

    def _setup_progress(self):
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.hide()
        self.status_label = QtWidgets.QLabel()
        self.status_label.hide()
        self.grid_layout.addWidget(self.progress_bar, 12, 0, 1, 2)
        self.grid_layout.addWidget(self.status_label, 13, 0, 1, 2)

    def update_plot(self, data):
        if not self.is_recording:
            return
        self.last_wave = data
        now = QtCore.QDateTime.currentMSecsSinceEpoch()
        if now - self._last_signal_time >= 100:
            self._last_signal_time = now
            try:
                self.waveform_update_signal.emit(data)
            except Exception as e:
                print("[DEBUG] Error emitting waveform_update_signal:", e)

    @Slot()
    def update_plot_slot(self):
        if not self.waveform.isVisible():
            print("[DEBUG] Waveform widget not visible; skipping update_plot_slot")
            return
        self.waveform.update_data(self.last_wave)

    def start_recording(self):
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.current_recording_dir = os.path.join(self.output_dir, timestamp)
            os.makedirs(self.current_recording_dir, exist_ok=True)
            self.recorder.output_dir = self.current_recording_dir
            self.recording_start_time = datetime.now()
            self.recording_timer.start(1000)
            self.recorder.device_index = self.input_device_indices[self.device_combo.currentIndex()]
            self.recorder.start_recording()
            self.is_recording = True
            self.record_button.setText("Stop")
            self.upload_button.setEnabled(False)
        except Exception as e:
            self.handle_error(f"Failed to start recording: {str(e)}")
            self.is_recording = False
            self.record_button.setText("Record")
            self.upload_button.setEnabled(True)

    def stop_recording(self):
        if not self.is_recording:
            return
        try:
            output_file = self.recorder.stop_recording()
            print(f"[DEBUG] Recorder returned file: {output_file}")
            if not output_file or not os.path.exists(output_file):
                self.handle_error("Recorded file does not exist: " + str(output_file))
                return
            self.is_recording = False
            self.record_button.setText("Record")
            self.upload_button.setEnabled(True)
            self.recording_timer.stop()
            self.statusBar().clearMessage()
            self.transcribe(output_file)
        except Exception as e:
            self.handle_error(f"Failed to stop recording: {str(e)}")
            self.is_recording = False
            self.record_button.setText("Record")
            self.upload_button.setEnabled(True)

    def update_recording_time(self):
        if self.recording_start_time:
            duration = datetime.now() - self.recording_start_time
            minutes = int(duration.total_seconds() // 60)
            seconds = int(duration.total_seconds() % 60)
            self.statusBar().showMessage(f"Recording: {minutes:02d}:{seconds:02d}")

    def on_record_button(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def on_upload_button(self):
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setNameFilter("Audio Files (*.wav *.mp3 *.m4a *.flac *.ogg *.aac)")
        if file_dialog.exec():
            try:
                audio_file = file_dialog.selectedFiles()[0]
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                self.current_recording_dir = os.path.join(self.output_dir, f"upload_{timestamp}")
                os.makedirs(self.current_recording_dir, exist_ok=True)
                if not audio_file.lower().endswith('.wav'):
                    self.statusBar().showMessage('Converting audio file...')
                    audio_file = self.convert_to_wav(audio_file)
                    self.statusBar().showMessage('Transcribing converted WAV file...')
                else:
                    self.statusBar().showMessage('Transcribing uploaded file...')
                self.progress_bar.show()
                self.progress_bar.setValue(0)
                self.stop_button.setEnabled(True)
                self.transcribe(audio_file)
            except Exception as e:
                self.statusBar().showMessage(f'Error processing upload: {str(e)}')

    def get_selected_language(self) -> str:
        selected = self.language_combo.currentText()
        if selected == "Auto-detect":
            return None
        name_to_code = {v: k for k, v in LANGUAGES.items()}
        return name_to_code.get(selected)

    def transcribe(self, file: str):
        diarization_method = self.diarization_combo.currentText()
        if diarization_method == "No Diarization":
            perform_diarization = False
            use_pyannote_api = False
        else:
            perform_diarization = True
            use_pyannote_api = (diarization_method == "Pyannote AI")
        if perform_diarization and not self._validate_tokens(use_pyannote_api):
            return
        self._setup_transcription_ui()
        language = self.get_selected_language()
        worker = TranscriptionWorker(
            file,
            num_speakers=self.get_num_speakers(),
            model_name=self.model_combo.currentText(),
            hf_token=self.token_input.text() or os.getenv("HUGGING_FACE_TOKEN"),
            pyannote_token=self.pyannote_token_input.text() or os.getenv("PYANNOTE_TOKEN"),
            use_pyannote_api=use_pyannote_api,
            perform_diarization=perform_diarization,
            language=language,
            progress_callback=lambda p, s: QtCore.QMetaObject.invokeMethod(
                self, "_apply_progress", QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(float, p), QtCore.Q_ARG(str, s)
            )
        )
        self._setup_worker_signals(worker)
        self.current_worker = worker
        self.threadpool.start(worker)

    @QtCore.Slot(float, str)
    def _apply_progress(self, progress: float, status: str):
        if not hasattr(self, "progress_bar") or not hasattr(self, "status_label"):
            return
        try:
            self.progress_bar.show()
            self.status_label.show()
            self.progress_bar.setValue(int(progress))
            self.status_label.setText(status)
        except Exception as e:
            print("[DEBUG] Exception in _apply_progress:", e)

    def _validate_tokens(self, use_pyannote_api: bool) -> bool:
        if self.diarization_combo.currentText() == "No Diarization":
            return True
        if use_pyannote_api:
            pyannote_token = self.pyannote_token_input.text() or os.getenv("PYANNOTE_TOKEN")
            if not pyannote_token:
                QtWidgets.QMessageBox.warning(
                    self, "Token Required", 
                    "Please enter your Pyannote API token or set PYANNOTE_TOKEN environment variable.\n"
                    "Get your token at https://hf.co/pyannote/speaker-diarization"
                )
                return False
        else:
            hf_token = self.token_input.text() or os.getenv("HUGGING_FACE_TOKEN")
            if not hf_token:
                QtWidgets.QMessageBox.warning(
                    self, "Token Required", 
                    "Please enter your Hugging Face token or set HUGGING_FACE_TOKEN environment variable.\n"
                    "Get your token at https://huggingface.co/settings/tokens"
                )
                return False
        return True

    def notify_transcription_done(self, text: str):
        self.transcription_preview.setPlainText(text)
        self.progress_bar.hide()
        self.status_label.hide()
        self.transcription_preview.show()
        try:
            if hasattr(self, 'current_recording_dir'):
                transcription_path = os.path.join(self.current_recording_dir, 'transcription.txt')
                with open(transcription_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"[DEBUG] Saved transcription to: {transcription_path}")
                self.statusBar().showMessage(f'Transcription saved to {transcription_path}')
            else:
                print("[DEBUG] Warning: No current_recording_dir set")
                self.statusBar().showMessage('Transcription complete (not saved)')
        except Exception as e:
            print(f"[DEBUG] Error saving transcription: {e}")
            self.statusBar().showMessage('Error saving transcription file')
        self.stop_button.setEnabled(False)

    def handle_error(self, error_msg: str):
        self.statusBar().showMessage('Error in transcription')
        self.progress_bar.hide()
        self.status_label.hide()
        self.stop_button.setEnabled(False)
        QtWidgets.QMessageBox.critical(self, "Error", f"Transcription failed: {error_msg}")

    def stop_transcription(self):
        if self.current_worker:
            self.current_worker.stop()
            self.current_worker = None
            self.stop_button.setEnabled(False)
            self.statusBar().showMessage('Stopping transcription...')

    def get_num_speakers(self) -> int:
        speakers = self.speaker_combo.currentText()
        return None if speakers == "Auto" else int(speakers)

    def convert_to_wav(self, audio_file: str) -> str:
        from pydub import AudioSegment
        ext = os.path.splitext(audio_file)[1][1:]
        try:
            sound = AudioSegment.from_file(audio_file, format=ext)
            wav_file = os.path.splitext(audio_file)[0] + ".wav"
            sound.export(wav_file, format="wav")
            print(f"[DEBUG] Converted {audio_file} to {wav_file}")
            return wav_file
        except Exception as e:
            print(f"[DEBUG] Conversion error: {e}")
            raise

    def _setup_transcription_ui(self):
        self.transcription_preview.clear()
        self.transcription_preview.hide()
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.status_label.setText("Transcribing...")
        self.status_label.show()
        self.stop_button.setEnabled(True)

    def _setup_worker_signals(self, worker):
        worker.signals.transcription_finished.connect(self.notify_transcription_done)
        worker.signals.error.connect(self.handle_error)
        worker.signals.progress.connect(self.progress_updater.update, QtCore.Qt.QueuedConnection)
        worker.signals.finished.connect(lambda: self.stop_button.setEnabled(False))

    def cleanup(self):
        print("[DEBUG] Running cleanup...")
        if self.audio_queue_timer.isActive():
            self.audio_queue_timer.stop()
        if self.recording_timer.isActive():
            self.recording_timer.stop()
        if self.is_recording:
            try:
                self.recorder.stop_recording()
            except Exception as e:
                print("[DEBUG] Exception during cleanup stop_recording:", e)
        if self.current_worker is not None:
            try:
                self.current_worker.signals.progress.disconnect()
            except Exception as e:
                print("[DEBUG] Error disconnecting progress signal:", e)
            self.current_worker = None
        self.threadpool.waitForDone(3000)
        if hasattr(self.recorder, "close"):
            try:
                self.recorder.close()
            except Exception as e:
                print("[DEBUG] Exception during recorder.close():", e)
        print("[DEBUG] Cleanup complete.")

    def closeEvent(self, event):
        self.cleanup()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    fonts.setup_application_font()
    window = WhisperTranscriptionApp()
    app.aboutToQuit.connect(window.cleanup)
    window.show()
    sys.exit(app.exec())
