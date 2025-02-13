from typing import List, Optional, Callable
from dataclasses import dataclass

@dataclass
class DiarizationChunk:
    """Represents a chunk of audio with speaker identification"""
    start: float  # Start time in seconds
    end: float    # End time in seconds
    speaker: str  # Speaker identifier

    def __str__(self):
        return f"{self.speaker} ({self.start:.1f}s - {self.end:.1f}s)"

class DiarizationProvider:
    """Base class for diarization providers"""
    def __init__(self, token: str, progress_callback: Optional[Callable[[float, str], None]] = None):
        self.token = token
        self.progress_callback = progress_callback

    def diarize(self, audio_file: str, num_speakers: Optional[int] = None) -> List[DiarizationChunk]:
        """Perform diarization on audio file"""
        raise NotImplementedError