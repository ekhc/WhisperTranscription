from typing import Optional
from pydub import AudioSegment

def ensure_wav_format(audio_file: str, target_path: Optional[str] = None) -> str:
    """Convert audio file to WAV format if needed."""
    if audio_file.endswith('.wav'):
        return audio_file
        
    # Generate target path if not provided
    if not target_path:
        target_path = audio_file.rsplit('.', 1)[0] + '.wav'
        
    # Convert to WAV
    audio = AudioSegment.from_file(audio_file)
    audio.export(target_path, format='wav')
    
    return target_path
