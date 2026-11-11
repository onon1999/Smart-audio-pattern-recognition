"""
Audio preprocessing tools: pre-emphasis, windowing, DC offset removal, and normalization.
"""
import numpy as np


class AudioPreprocessor:
    """Preprocesses raw audio waveforms for robust spectral and pattern analysis."""

    @staticmethod
    def remove_dc_offset(signal: np.ndarray) -> np.ndarray:
        """Removes DC bias / zero-frequency offset."""
        if len(signal) == 0:
            return signal
        return signal - np.mean(signal)

    @staticmethod
    def normalize(signal: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """Normalizes audio waveform amplitude to [-target_peak, target_peak]."""
        peak = np.max(np.abs(signal))
        if peak > 1e-7:
            return signal * (target_peak / peak)
        return signal

    @staticmethod
    def apply_pre_emphasis(signal: np.ndarray, coef: float = 0.97) -> np.ndarray:
        """Applies a first-order high-pass pre-emphasis filter: y[n] = x[n] - coef * x[n-1]."""
        if len(signal) < 2 or coef <= 0.0:
            return signal
        return np.append(signal[0], signal[1:] - coef * signal[:-1])

    @staticmethod
    def frame_signal(signal: np.ndarray, frame_size: int = 1024, hop_size: int = 512, window: str = "hann") -> np.ndarray:
        """
        Slices a 1D audio signal into overlapping windowed frames.
        Returns array of shape: (num_frames, frame_size).
        """
        sig_len = len(signal)
        if sig_len < frame_size:
            # Pad with zeros if shorter than frame size
            padded = np.zeros(frame_size, dtype=np.float32)
            padded[:sig_len] = signal
            signal = padded
            sig_len = frame_size

        num_frames = 1 + int(np.floor((sig_len - frame_size) / hop_size))
        
        # Calculate indices
        idx = np.tile(np.arange(0, frame_size), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * hop_size, hop_size), (frame_size, 1)).T
        
        frames = signal[idx.astype(np.int32, copy=False)]

        # Apply window function
        if window == "hann":
            win = np.hanning(frame_size)
        elif window == "hamming":
            win = np.hamming(frame_size)
        elif window == "blackman":
            win = np.blackman(frame_size)
        else:
            win = np.ones(frame_size)

        return frames * win
