"""
Audio file input/output utilities using scipy.io.wavfile.
"""
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def load_wav(file_path: str | Path, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    """
    Loads a WAV file, converts multi-channel to mono, normalizes float range to [-1.0, 1.0],
    and resamples to target_sr if required.
    """
    sr, data = wavfile.read(str(file_path))
    
    # Convert integer PCM to float [-1.0, 1.0]
    if data.dtype == np.int16:
        signal = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        signal = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        signal = (data.astype(np.float32) - 128.0) / 128.0
    else:
        signal = data.astype(np.float32)

    # Convert to mono if stereo or multi-channel
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    # Resample if sample rate does not match target
    if sr != target_sr and len(signal) > 0:
        # Use polyphase filtering for fast resampling
        gcd = np.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        signal = resample_poly(signal, up, down).astype(np.float32)
        sr = target_sr

    return signal, sr


def save_wav(file_path: str | Path, signal: np.ndarray, sr: int = 22050):
    """
    Saves a 1D float32 audio signal as a 16-bit PCM WAV file.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    # Clamp signal to [-1.0, 1.0]
    clamped = np.clip(signal, -1.0, 1.0)
    int_pcm = (clamped * 32767.0).astype(np.int16)
    wavfile.write(str(file_path), sr, int_pcm)
