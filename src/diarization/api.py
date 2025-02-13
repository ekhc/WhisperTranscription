import os
import time
import requests
from typing import List, Optional, Callable
from src.diarization import DiarizationChunk
from datetime import datetime

class PyannoteAPI:
    def __init__(self, token: Optional[str] = None, progress_callback: Optional[Callable[[float, str], None]] = None):
        self.token = token or os.environ.get('PYANNOTE_TOKEN')
        if not self.token:
            raise ValueError("Pyannote token is required. Set PYANNOTE_TOKEN environment variable or pass token to constructor.")
        self.progress_callback = progress_callback
        
    def diarize(self, audio_file: str, num_speakers: Optional[int] = None) -> List[DiarizationChunk]:
        """Perform speaker diarization using Pyannote API"""
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
        if self.progress_callback:
            self.progress_callback(20, "Creating temporary storage...")
            
        # Create temporary storage location with unique object-key
        storage_url = "https://api.pyannote.ai/v1/media/input"
        object_key = f"recording{datetime.now().strftime('%Y%m%d%H%M%S')}"
        storage_data = {
            "url": f"media://{object_key}"  # Use unique object-key for each upload
        }
        
        storage_response = requests.post(
            storage_url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            },
            json=storage_data
        )
        storage_response.raise_for_status()
            
        # Get pre-signed URL for upload
        presigned_url = storage_response.json()["url"]
        
        if self.progress_callback:
            self.progress_callback(30, "Uploading audio file...")
            
        # Upload to pre-signed URL
        with open(audio_file, "rb") as f:
            upload_response = requests.put(
                presigned_url, 
                data=f,
                headers={
                    "Content-Type": "audio/wav"
                }
            )
            upload_response.raise_for_status()
                
        if self.progress_callback:
            self.progress_callback(50, "Processing audio...")
            
        # Create diarization job using the same media URL
        diarize_url = "https://api.pyannote.ai/v1/diarize"
        diarize_data = {
            "url": f"media://{object_key}",  # Use same object-key as upload
        }
        if num_speakers is not None:
            diarize_data["num_speakers"] = num_speakers
        
        diarize_response = requests.post(
            diarize_url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json"
            },
            json=diarize_data
        )
        diarize_response.raise_for_status()
        
        job_id = diarize_response.json()['jobId']
        
        if self.progress_callback:
            self.progress_callback(50, "Processing diarization...")
            
        # Poll for results
        status_url = f"https://api.pyannote.ai/v1/jobs/{job_id}"
        while True:
            status_response = requests.get(
                status_url,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json"
                }
            )
            status_response.raise_for_status()
            status_data = status_response.json()
            
            if status_data['status'] == 'failed':
                raise ValueError(f"Diarization failed: {status_data.get('error', 'Unknown error')}")
            elif status_data['status'] == 'succeeded':
                diarization = status_data['output']['diarization']
                break
                
            time.sleep(1)  # Wait before polling again
            
        # Convert API format to our format
        segments = []
        for segment in diarization:
            segments.append(DiarizationChunk(
                start=segment['start'],
                end=segment['end'],
                speaker=segment['speaker']
            ))
            
        if self.progress_callback:
            self.progress_callback(100, "Diarization complete")
            
        return segments