"""
Mel and Chroma filterbank matrices, and STFT spectral transforms implemented in pure NumPy/SciPy.
"""
import numpy as np
from scipy.signal import stft


def hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    """Converts frequency in Hertz to the perceptual Mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    """Converts perceptual Mel scale values back to Hertz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def get_mel_filterbank(
    sr: int = 22050,
    n_fft: int = 1024,
    n_mels: int = 40,
    fmin: float = 20.0,
    fmax: float = 8000.0
) -> np.ndarray:
    """
    Constructs triangular Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1).
    """
    num_bins = n_fft // 2 + 1
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    # Clip bins to valid spectrum bounds
    bin_points = np.clip(bin_points, 0, num_bins - 1)
    
    filterbank = np.zeros((n_mels, num_bins), dtype=np.float32)
    
    for m in range(1, n_mels + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]
        
        if f_center > f_left:
            for k in range(f_left, f_center):
                filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)
                
        if f_right > f_center:
            for k in range(f_center, f_right):
                filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)
                
    # Normalize filter areas to equalize energy across frequency bands
    enorm = 2.0 / (hz_points[2:n_mels + 2] - hz_points[:n_mels])
    filterbank *= enorm[:, np.newaxis]
    
    return filterbank


def get_chroma_filterbank(
    sr: int = 22050,
    n_fft: int = 1024,
    n_chroma: int = 12,
    a440: float = 440.0,
    octave_width: float = 2.0
) -> np.ndarray:
    """
    Constructs a 12-pitch class Chroma filterbank matrix (C, C#, D, ... B).
    Returns array of shape (n_chroma, n_fft // 2 + 1).
    """
    num_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2.0, num_bins)
    
    # Avoid log(0) for DC bin
    valid_freqs = np.maximum(freqs, 1e-4)
    # Pitch in fractional semitones relative to A4 (440 Hz = pitch 69)
    midi_pitch = 12.0 * np.log2(valid_freqs / a440) + 69.0
    
    chroma_fb = np.zeros((n_chroma, num_bins), dtype=np.float32)
    
    for c in range(n_chroma):
        # Difference in pitch class modulo 12
        # C is pitch class 0, C# is 1, ..., A is 9, A# is 10, B is 11
        # MIDI note 69 (A4) mod 12 = 9
        diff = np.fmod(midi_pitch - c, 12.0)
        diff = np.fmod(diff + 12.0, 12.0)
        diff[diff > 6.0] -= 12.0
        
        # Gaussian bell across semitone distance
        weight = np.exp(-0.5 * (diff / 0.5) ** 2)
        
        # Bandpass bell across octaves to focus around musical range (100 Hz to 4000 Hz)
        octave_center = 12.0 * np.log2(1000.0 / a440) + 69.0
        octave_weight = np.exp(-0.5 * ((midi_pitch - octave_center) / (12.0 * octave_width)) ** 2)
        
        chroma_fb[c, :] = weight * octave_weight
        
    # Zero out sub-audio DC and near-Nyquist
    chroma_fb[:, 0] = 0.0
    
    # Normalize columns
    col_sums = np.sum(chroma_fb, axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    chroma_fb /= col_sums
    
    return chroma_fb


def compute_stft(
    signal: np.ndarray,
    sr: int = 22050,
    frame_size: int = 1024,
    hop_size: int = 512,
    window: str = "hann"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Short-Time Fourier Transform (STFT).
    Returns:
        freqs: Frequency axis bins (Hz)
        times: Time axis frames (seconds)
        mag_spec: Magnitude spectrogram |Zxx|
        power_spec: Power spectrogram |Zxx|^2
    """
    f, t, zxx = stft(
        signal,
        fs=sr,
        window=window,
        nperseg=frame_size,
        noverlap=frame_size - hop_size,
        boundary=None,
        padded=False
    )
    mag_spec = np.abs(zxx)
    power_spec = mag_spec ** 2
    return f, t, mag_spec, power_spec

