import os
from typing import List, Optional, Callable
import torch
from pyannote.audio import Pipeline
from src.diarization import DiarizationChunk

class PyannoteLocal:
    def __init__(self, token: Optional[str] = None, progress_callback: Optional[Callable[[float, str], None]] = None):
        self.token = token or os.environ.get('HUGGING_FACE_TOKEN')
        if not self.token:
            raise ValueError("Hugging Face token is required. Set HUGGING_FACE_TOKEN environment variable or pass token to constructor.")
        self.progress_callback = progress_callback
        self.pipeline = None
        
    def _init_pipeline(self):
        """Initialize the Pyannote pipeline"""
        if not self.pipeline:
            if self.progress_callback:
                self.progress_callback(25, "Loading diarization model...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.token
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to(torch.device("cuda"))
                
    def diarize(self, audio_file: str, num_speakers: Optional[int] = None) -> List[DiarizationChunk]:
        """Perform speaker diarization using local Pyannote model"""
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
        if self.progress_callback:
            self.progress_callback(20, "Starting diarization...")
            
        # Initialize pipeline
        self._init_pipeline()
        
        # Run diarization
        if self.progress_callback:
            self.progress_callback(30, "Running diarization...")
            
        diarization = self.pipeline(
            audio_file,
            num_speakers=num_speakers
        )
        
        # Convert to segments
        if self.progress_callback:
            self.progress_callback(60, "Processing diarization results...")
            
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(DiarizationChunk(
                start=turn.start,
                end=turn.end,
                speaker=f"SPEAKER_{speaker}"
            ))
            
        # Sort segments by start time
        segments.sort(key=lambda x: x.start)
        
        return segments