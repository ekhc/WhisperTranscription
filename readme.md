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
git clone https://github.com/ekhc/WhisperTranscription.git
cd WhisperTranscription
```

2. Set up Poetry environment with Python 3.11:
```bash
poetry env use python3.11
poetry install
```

## Configuration

### Hugging Face Token Setup (Required for Speaker Diarization)

The speaker diarization feature uses Pyannote.audio, which requires accepting license terms and obtaining access tokens. Follow these steps:

1. Create a Hugging Face account at [Hugging Face](https://huggingface.co)

2. Accept the user agreement for the model:
   - Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - Click "Accept" on the license terms for the model

3. Create an access token:
   - Go to [Hugging Face Access Tokens](https://huggingface.co/settings/tokens)
   - Click "New token" and give it a name (e.g., "WhisperTranscription")
   - Copy the generated token

4. Set up your token:
   - Option 1: Set as environment variable (recommended):
     ```bash
     # Add to ~/.zshrc or ~/.bashrc for persistence
     export HUGGING_FACE_TOKEN=your_token_here
     ```
   - Option 2: Enter it in the application UI when prompted

### Pyannote API Token Setup (Optional - for API-based Diarization)

If you want to use the Pyannote.ai API service instead of local processing:

1. Sign up for an API account at [Pyannote.ai](https://pyannote.ai)
2. Get your API token from your account dashboard
3. Set up your token:
   - Option 1: Set as environment variable (recommended):
     ```bash
     # Add to ~/.zshrc or ~/.bashrc for persistence
     export PYANNOTE_TOKEN=your_api_token_here
     ```
   - Option 2: Enter it in the application UI when prompted

Note: The Pyannote.ai API is a paid service that offers:
- Higher performance than local processing
- No need to manage local GPU resources
- Automatic scaling and updates
Choose between local processing (Hugging Face token) or API service (Pyannote API token) based on your needs.

Note: While Pyannote.audio is open-source under the MIT license, you must agree to their terms of use which include:
- Sharing contact information for userbase analytics
- Receiving occasional emails about premium features
- Using the models in compliance with their intended purpose

For commercial or high-volume usage, consider their [premium services](https://www.pyannote.ai).

## Running the Application

1. Activate the Poetry environment and run the application:
```bash
poetry run whisper-transcription
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
