# Whisper Transcriber

A desktop application for transcribing audio with speaker diarization using OpenAI's Whisper and Pyannote.audio.

## Features
- Record audio or upload audio files
- Transcribe audio using OpenAI's Whisper with multiple model options:
  - Tiny/Tiny.en: ~1GB VRAM, fastest
  - Base/Base.en: ~1GB VRAM, fast
  - Small/Small.en: ~2GB VRAM, balanced
  - Medium/Medium.en: ~5GB VRAM, accurate
  - Large: ~10GB VRAM, most accurate
  - Turbo: ~6GB VRAM, optimized for speed
- Speaker diarization using Pyannote.audio
- Real-time audio visualization
- Support for multiple input devices
- GPU acceleration (CUDA) when available

## Prerequisites

### 1. Install Homebrew (if not already installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Required System Packages
```bash
# Install Python 3.11 (required for Whisper compatibility)
brew install python@3.11

# Install PortAudio (required for PyAudio)
brew install portaudio

# Install FFmpeg (required for audio processing)
brew install ffmpeg
```

### 3. Install Poetry (Python package manager)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CyR1en/whisper-transcriber.git
cd whisper-transcriber
```

2. Set up Poetry environment with Python 3.11:
```bash
poetry env use python3.11
poetry install
```

## Configuration

### Hugging Face Token (Required for Speaker Diarization)

1. Create an account at [Hugging Face](https://huggingface.co)
2. Get your token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Set it as an environment variable (recommended):
   ```bash
   # Add to ~/.zshrc or ~/.bashrc for persistence
   export HUGGING_FACE_TOKEN=your_token_here
   ```
   Or enter it in the application UI when prompted

## Running the Application

1. Activate the Poetry environment and run the application:
```bash
poetry run python ui.py
```

## Usage

1. **Recording Audio**:
   - Select your input device from the dropdown
   - Click "Record" to start recording
   - Click "Record" again to stop and begin transcription

2. **Uploading Audio**:
   - Click "Load" to select an audio file
   - Supported formats: WAV, MP3, M4A, etc.

3. **Transcription Settings**:
   - Choose a Whisper model:
     - Smaller models (tiny, base) are faster but less accurate
     - Larger models (medium, large) are more accurate but slower
     - English-only models (*.en) may perform better for English audio
   - Set the number of speakers (or leave at 0 for auto-detection)
   - Ensure your Hugging Face token is set

4. **Troubleshooting**:
   - If you see CUDA/GPU errors, the application will automatically fall back to CPU
   - For best performance with larger models, a GPU is recommended
   - Make sure your audio input device is working and selected correctly

## Development Notes

- Python version: 3.11 (required for Whisper compatibility)
- Uses Poetry for dependency management
- GPU support: CUDA if available, falls back to CPU

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
