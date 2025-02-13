import os
import pyaudio
import numpy as np
import collections
import threading
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime
from enum import Enum
import time
import numpy as np
import scipy.io.wavfile as wavfile

class AudioFormat(Enum):
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"

class Recorder:
    SUPPORTED_RATES = [8000, 16000, 22050, 44100, 48000]
    
    def __init__(self, 
                 output_dir: str = "output",
                 format: AudioFormat = AudioFormat.WAV,
                 channels: int = 1,
                 rate: int = 16000,
                 chunk: int = 1024,
                 device_index: Optional[int] = None,
                 progress_callback: Optional[Callable[[np.ndarray], None]] = None,
                 update_func: Optional[Callable[[List[float]], None]] = None):
        """Initialize audio recorder
        
        Args:
            output_dir: Directory to save recordings
            format: Audio format to save in
            channels: Number of audio channels
            rate: Sample rate (Hz)
            chunk: Chunk size for recording
            device_index: Index of input device to use
            progress_callback: Callback for recording progress
            update_func: Callback for updating visualization
        """
        if rate not in self.SUPPORTED_RATES:
            raise ValueError(f"Unsupported sample rate. Must be one of: {self.SUPPORTED_RATES}")
            
        self.audio_format = format
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.output_dir = output_dir
        self.progress_callback = progress_callback
        self.update_func = update_func
        self.device_index = device_index
        self.recording = False
        self.frames = []
        self.stream = None
        self.frames = []
        self.frames_lock = threading.Lock()
        self.audio_data_queue = collections.deque()
        self.queue_lock = threading.Lock()
        self._validate_device()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
            
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
    def _validate_device(self):
        """Validate selected input device"""
        if self.device_index is not None:
            p = pyaudio.PyAudio()
            try:
                device_info = p.get_device_info_by_index(self.device_index)
                if device_info['maxInputChannels'] < self.channels:
                    raise ValueError(f"Device {self.device_index} does not support {self.channels} channels")
                if not device_info['maxInputChannels']:
                    raise ValueError(f"Device {self.device_index} is not an input device")
            except Exception as e:
                p.terminate()
                raise ValueError(f"Invalid device index {self.device_index}: {str(e)}")
            p.terminate()
            
    @property
    def is_recording(self) -> bool:
        """Check if currently recording"""
        return self.recording
        
    @property
    def duration(self) -> float:
        """Get current recording duration in seconds"""
        return len(self.frames) * self.chunk / self.rate
        
    def start_recording(self):
        if self.recording:
            return  # Already recording
        self.recording = True
        self.frames = []
        self.audio_data_queue = collections.deque()  # Reinitialize the queue if needed

        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk
            )
        except Exception as e:
            self.recording = False
            raise RuntimeError(f"Failed to start recording: {str(e)}")

        # Start a thread to perform blocking reads.
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()

    def _read_loop(self):
        while True:
            # Check if we should exit the loop.
            if not self.recording:
                break

            # Check how many frames are available.
            available = self.stream.get_read_available()
            # If not enough data is available, sleep a little and then continue the loop.
            if available < self.chunk:
                time.sleep(0.01)  # sleep for 10 ms
                continue

            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
            except Exception as e:
                print("[DEBUG] Exception in read loop:", e)
                break

            data_copy = bytes(data)  # Ensure we have our own copy

            with self.queue_lock:
                self.audio_data_queue.append(data_copy)
            with self.frames_lock:
                self.frames.append(data_copy)

    def stop_recording(self) -> str:
        if not self.recording:
            return ""
        
        # Signal the read thread to stop.
        self.recording = False

        # First, stop the stream to signal no more data is needed.
        if self.stream:
            try:
                self.stream.stop_stream()
            except Exception as e:
                print("[DEBUG] Exception stopping stream:", e)
        
        # Wait for the read thread to exit.
        if hasattr(self, 'read_thread'):
            self.read_thread.join()

        # Close the stream.
        if self.stream:
            try:
                self.stream.close()
            except Exception as e:
                print("[DEBUG] Exception closing stream:", e)
            self.stream = None

        # Safely capture the frames.
        with self.frames_lock:
            num_frames = len(self.frames)
            
        if num_frames == 0:
            print("[DEBUG] No audio frames were captured!")
            return ""

        # Create the output file path.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_file = os.path.join(self.output_dir, f"recording_{timestamp}.wav")

        try:
            # Join the frames safely.
            with self.frames_lock:
                raw_data = b"".join(self.frames)
            # Convert the raw bytes into a NumPy array.
            # Note: if you recorded mono audio, the dtype is int16.
            audio_array = np.frombuffer(raw_data, dtype=np.int16)
            # Write the WAV file using SciPy.
            wavfile.write(wav_file, self.rate, audio_array)
            print("[DEBUG] WAV file written successfully to:", wav_file)
            return wav_file

        except Exception as e:
            print("[DEBUG] Exception while writing WAV file:", e)
            if os.path.exists(wav_file):
                os.remove(wav_file)
            raise RuntimeError(f"Failed to save recording: {str(e)}")


        
    def record_chunk(self):
        """Record a single chunk of audio"""
        if not self.recording or not self.stream:
            return
            
        try:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            self.frames.append(data)
            
            # Calculate and report progress
            if self.progress_callback:
                # Convert to numpy array for visualization
                frame_data = np.frombuffer(data, dtype=np.int16)
                # Normalize to float between -1 and 1 for visualization
                normalized = frame_data.astype(np.float32) / 32768.0
                self.progress_callback(normalized)
                
            # Update visualization if callback provided
            if self.update_func:
                self.update_func(normalized.tolist())
        except Exception as e:
            self.recording = False
            if self.stream:
                self.stream.close()
                self.stream = None
            raise RuntimeError(f"Failed to record audio chunk: {str(e)}")
                
    def close(self):
        """Clean up resources"""
        if self.recording:
            self.stop_recording()
            
        if self.stream:
            self.stream.close()
            self.stream = None
            
        if self.audio:
            self.audio.terminate()
            self.audio = None

    def get_input_devices(self) -> List[Dict[str, Any]]:
        """Get list of available input devices with their capabilities"""
        devices = []
        p = pyaudio.PyAudio()
        try:
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    # Add supported sample rates
                    supported_rates = []
                    for rate in self.SUPPORTED_RATES:
                        try:
                            if p.is_format_supported(
                                rate,
                                input_device=i,
                                input_channels=device_info['maxInputChannels'],
                                input_format=pyaudio.paInt16
                            ):
                                supported_rates.append(rate)
                        except:
                            continue
                    device_info['supported_rates'] = supported_rates
                    devices.append(device_info)
        finally:
            p.terminate()
        return devices
        
    def get_output_devices(self) -> List[Dict[str, Any]]:
        """Get list of available output devices"""
        devices = []
        p = pyaudio.PyAudio()
        try:
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if device_info['maxOutputChannels'] > 0:
                    devices.append(device_info)
        finally:
            p.terminate()
        return devices

    def process_audio_queue(self):
        raw_chunk = None
        with self.queue_lock:
            if self.audio_data_queue:
                raw_chunk = self.audio_data_queue.popleft()
        if raw_chunk is None:
            return

        try:
            # Convert the raw bytes into a NumPy array and force a copy
            data = np.frombuffer(raw_chunk, dtype=np.int16).copy()
            normalized = data.astype(np.float32) / 32768.0

            # Call the update function so the waveform widget is updated
            if self.update_func:
                self.update_func(normalized.tolist())
        except Exception as e:
            print("[DEBUG] Exception in processing audio queue:", e)

