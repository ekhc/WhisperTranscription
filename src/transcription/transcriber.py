import os
import time
import torch
import whisper
from typing import Optional, Callable
from src.diarization.api import PyannoteAPI
from src.diarization.local import PyannoteLocal
from src.audio.utils import ensure_wav_format
from src.ui.utils import format_timestamp
from pydub import AudioSegment

class Transcriber:
    def __init__(self,
                 model_name: str = "base",
                 hf_token: Optional[str] = os.environ.get('HUGGING_FACE_TOKEN'),
                 pyannote_token: Optional[str] = os.environ.get('PYANNOTE_TOKEN'),
                 use_pyannote_api: bool = False,
                 perform_diarization: bool = True,
                 progress_callback: Optional[Callable[[float, str], None]] = None):
        self.model_name = model_name
        self.hf_token = hf_token
        self.pyannote_token = pyannote_token
        self.use_pyannote_api = use_pyannote_api
        self.progress_callback = progress_callback
        self.perform_diarization = perform_diarization
        if self.perform_diarization:
            if use_pyannote_api:
                self.diarization = PyannoteAPI(pyannote_token, progress_callback)
            else:
                self.diarization = PyannoteLocal(hf_token, progress_callback)
        else:
            self.diarization = None

    def transcribe(self, audio_file: str, num_speakers: Optional[int] = None, language: Optional[str] = None,
                   stop_flag: Optional[Callable[[], bool]] = None) -> str:
        def check_stop():
            if stop_flag and stop_flag():
                if self.progress_callback:
                    self.progress_callback(100, "Cancelled")
                raise Exception("Transcription cancelled by user.")
        start_time = time.time()
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        if self.progress_callback:
            self.progress_callback(10, "Loading audio file...")
        check_stop()
        # Get the duration from the WAV file.
        audio = AudioSegment.from_wav(audio_file)
        duration = len(audio) / 1000.0
        diarization_time = 0
        transcription_time = 0
        total_time = 0
        cleanup_time = 0
        converted_file = None
        try:
            if self.progress_callback:
                self.progress_callback(0, "Converting audio if needed...")
            check_stop()
            audio_file = ensure_wav_format(audio_file)
            check_stop()
            if self.progress_callback:
                self.progress_callback(10, "Loading Whisper model...")
            check_stop()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = whisper.load_model(self.model_name, device=device)
            check_stop()
            if self.perform_diarization:
                if self.progress_callback:
                    self.progress_callback(0, "Performing diarization...")
                diarization_start = time.time()
                segments = self.diarization.diarize(audio_file, num_speakers)
                diarization_time = time.time() - diarization_start
                check_stop()
            else:
                segments = None
                diarization_time = 0
            if self.progress_callback:
                self.progress_callback(70, "Transcribing audio...")
            check_stop()
            transcription_start = time.time()
            result = model.transcribe(
                audio_file,
                task="transcribe",
                language=language,  # Use the specified language (or auto-detect if None)
                fp16=(device == "cuda")
            )
            transcription_time = time.time() - transcription_start
            check_stop()
            if self.progress_callback:
                self.progress_callback(90, "Matching transcription with speakers...")
            check_stop()
            if self.perform_diarization and segments:
                final_segments = []
                used_indices = set()
                for segment in segments:
                    check_stop()
                    matching_segments = []
                    for i, s in enumerate(result["segments"]):
                        if i not in used_indices and (
                            (s["start"] >= segment.start and s["end"] <= segment.end) or
                            (s["start"] <= segment.start and s["end"] >= segment.end) or
                            (s["start"] <= segment.end and s["end"] >= segment.end and s["start"] >= segment.start)
                        ):
                            matching_segments.append((i, s))
                    if matching_segments:
                        matching_segments.sort(key=lambda x: x[1]["start"])
                        text = " ".join(s[1]["text"] for s in matching_segments)
                        start_ts = format_timestamp(segment.start)
                        end_ts = format_timestamp(segment.end)
                        final_segments.append(f"[{start_ts} - {end_ts}] {segment.speaker}: {text}")
                        used_indices.update(i for i, _ in matching_segments)
            else:
                final_segments = [result["text"]]
            total_time = time.time() - start_time
        finally:
            cleanup_start = time.time()
            if converted_file and os.path.exists(converted_file):
                try:
                    os.remove(converted_file)
                except Exception as e:
                    print(f"Warning: Failed to remove temporary file {converted_file}: {e}")
            cleanup_time = time.time() - cleanup_start
        timing_info = [
            "=== Transcription Summary ===",
            f"Audio Duration: {duration:.1f}s",
            f"Diarization Time: {diarization_time:.1f}s",
            f"Transcription Time: {transcription_time:.1f}s",
            f"Cleanup Time: {cleanup_time:.1f}s",
            f"Total Processing Time: {total_time:.1f}s",
            "=========================="
        ]
        if self.progress_callback:
            self.progress_callback(100, "Done!")
        return "\n".join(final_segments + ["", "Timing:"] + timing_info)
