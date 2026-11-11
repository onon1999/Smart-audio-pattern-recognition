"""
Simulated real-time audio stream generator for reproducible tests, demonstrations, and file playback.
"""
import time
from typing import Callable, Generator
import numpy as np

from ..config import AudioConfig
from .buffer import AudioRingBuffer
from ..utils.audio_io import load_wav


class SimulatedAudioStream:
    """Simulates real-time streaming ingestion from an audio file or numpy buffer."""

    def __init__(
        self,
        signal: np.ndarray | None = None,
        wav_path: str | None = None,
        config: AudioConfig | None = None,
        real_time_pace: bool = True
    ):
        self.config = config or AudioConfig()
        if wav_path:
            sig, _ = load_wav(wav_path, target_sr=self.config.sample_rate)
            self.signal = sig
        elif signal is not None:
            self.signal = signal
        else:
            self.signal = np.zeros(self.config.sample_rate, dtype=np.float32)

        self.real_time_pace = real_time_pace
        self.ring_buffer = AudioRingBuffer(self.config.buffer_size)
        self.chunk_size = self.config.chunk_size

    def iter_chunks(self) -> Generator[np.ndarray, None, None]:
        """Yields sequential audio chunks, pacing dynamically to simulate real time if enabled."""
        sig_len = len(self.signal)
        chunk_dt = float(self.chunk_size) / float(self.config.sample_rate)

        for start_idx in range(0, sig_len, self.chunk_size):
            t_start = time.perf_counter()
            end_idx = min(start_idx + self.chunk_size, sig_len)
            chunk = self.signal[start_idx:end_idx]
            
            # Zero-pad trailing chunk to maintain uniform chunk size
            if len(chunk) < self.chunk_size:
                padded = np.zeros(self.chunk_size, dtype=np.float32)
                padded[:len(chunk)] = chunk
                chunk = padded

            self.ring_buffer.write(chunk)
            yield chunk

            if self.real_time_pace:
                elapsed = time.perf_counter() - t_start
                sleep_time = chunk_dt - elapsed
                if sleep_time > 0.001:
                    time.sleep(sleep_time)

    def stream_with_callback(self, callback: Callable[[np.ndarray], None]):
        """Runs the simulated stream and invokes callback for every incoming audio frame."""
        for chunk in self.iter_chunks():
            callback(chunk)
