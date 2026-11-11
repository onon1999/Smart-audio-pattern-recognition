"""
Live microphone stream capture with sounddevice and callback dispatching.
"""
from typing import Callable
import numpy as np
import sounddevice as sd

from ..config import AudioConfig
from .buffer import AudioRingBuffer


class LiveMicrophoneStream:
    """Manages low-latency real-time microphone audio capture."""

    def __init__(
        self,
        config: AudioConfig | None = None,
        callback: Callable[[np.ndarray], None] | None = None
    ):
        self.config = config or AudioConfig()
        self.callback = callback
        self.ring_buffer = AudioRingBuffer(self.config.buffer_size)
        self.stream: sd.InputStream | None = None
        self.is_running = False

    @staticmethod
    def list_input_devices() -> list[dict]:
        """Returns all available audio input devices on the system."""
        devices = []
        try:
            device_list = sd.query_devices()
            for idx, dev in enumerate(device_list):
                if dev.get("max_input_channels", 0) > 0:
                    devices.append({
                        "id": idx,
                        "name": dev.get("name", "Unknown"),
                        "channels": dev.get("max_input_channels"),
                        "default_samplerate": dev.get("default_samplerate")
                    })
        except Exception as e:
            print(f"Warning: Failed to query audio devices: {e}")
        return devices

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Internal sounddevice callback."""
        if status:
            pass  # Overflow / underflow warning
        mono = indata[:, 0].astype(np.float32)
        self.ring_buffer.write(mono)
        if self.callback:
            self.callback(mono)

    def start(self, device_id: int | None = None):
        """Starts real-time microphone stream."""
        if self.is_running:
            return
        
        self.stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.config.chunk_size,
            device=device_id,
            callback=self._audio_callback
        )
        self.stream.start()
        self.is_running = True

    def stop(self):
        """Stops active stream and releases audio hardware."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_running = False

    def get_recent_audio(self, duration_sec: float | None = None) -> np.ndarray:
        """Retrieves recent audio segment from circular buffer."""
        dur = duration_sec or self.config.chunk_duration_sec
        n_samples = int(self.config.sample_rate * dur)
        return self.ring_buffer.read_recent(n_samples)
