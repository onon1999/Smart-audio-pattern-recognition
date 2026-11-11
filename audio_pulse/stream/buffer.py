"""
Thread-safe circular ring buffer for real-time streaming audio processing.
"""
import threading
import numpy as np


class AudioRingBuffer:
    """Thread-safe circular memory buffer for continuous audio samples."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=np.float32)
        self.write_idx = 0
        self.total_samples_written = 0
        self.lock = threading.Lock()

    def write(self, data: np.ndarray):
        """Appends new incoming audio chunk to circular buffer."""
        if len(data) == 0:
            return
        data = np.asarray(data, dtype=np.float32)
        n = len(data)

        with self.lock:
            if n >= self.capacity:
                # If chunk is larger than whole buffer, take trailing part
                self.buffer[:] = data[-self.capacity:]
                self.write_idx = 0
            else:
                end_idx = self.write_idx + n
                if end_idx <= self.capacity:
                    self.buffer[self.write_idx:end_idx] = data
                else:
                    first_part = self.capacity - self.write_idx
                    self.buffer[self.write_idx:self.capacity] = data[:first_part]
                    self.buffer[0:n - first_part] = data[first_part:]
                self.write_idx = (self.write_idx + n) % self.capacity
            self.total_samples_written += n

    def read_recent(self, n_samples: int) -> np.ndarray:
        """Reads the most recent n_samples in correct chronological order without consuming."""
        with self.lock:
            n = min(n_samples, self.capacity, self.total_samples_written)
            if n == 0:
                return np.zeros(0, dtype=np.float32)
            
            start_idx = (self.write_idx - n) % self.capacity
            if start_idx + n <= self.capacity:
                return self.buffer[start_idx:start_idx + n].copy()
            else:
                first_len = self.capacity - start_idx
                return np.concatenate([
                    self.buffer[start_idx:self.capacity],
                    self.buffer[0:n - first_len]
                ])

    def clear(self):
        """Empties buffer."""
        with self.lock:
            self.buffer.fill(0.0)
            self.write_idx = 0
            self.total_samples_written = 0

