"""
High-performance audio feature extraction engine: Time, Frequency, Mel, MFCC, and Chroma features.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.fftpack import dct
from scipy.signal import find_peaks

from ..config import AudioConfig
from .filterbanks import get_mel_filterbank, get_chroma_filterbank, compute_stft
from .preprocessor import AudioPreprocessor


@dataclass
class AudioFeatures:
    """Container for comprehensive audio features."""
    duration_sec: float
    num_frames: int
    rms_energy: float
    rms_db: float
    peak_amplitude: float
    crest_factor: float
    zero_crossing_rate: float
    spectral_centroid: float
    spectral_spread: float
    spectral_rolloff: float
    spectral_flatness: float
    spectral_flux: float
    dominant_frequency_hz: float
    mfcc_mean: np.ndarray = field(default_factory=lambda: np.zeros(13))
    mfcc_frames: np.ndarray = field(default_factory=lambda: np.zeros((13, 0)))
    chroma_mean: np.ndarray = field(default_factory=lambda: np.zeros(12))
    chroma_frames: np.ndarray = field(default_factory=lambda: np.zeros((12, 0)))
    mel_spectrogram: np.ndarray = field(default_factory=lambda: np.zeros((40, 0)))

    def to_dict(self) -> dict:
        return {
            "duration_sec": round(float(self.duration_sec), 4),
            "rms_energy": round(float(self.rms_energy), 6),
            "rms_db": round(float(self.rms_db), 2),
            "peak_amplitude": round(float(self.peak_amplitude), 4),
            "crest_factor": round(float(self.crest_factor), 2),
            "zero_crossing_rate": round(float(self.zero_crossing_rate), 4),
            "spectral_centroid_hz": round(float(self.spectral_centroid), 1),
            "spectral_spread_hz": round(float(self.spectral_spread), 1),
            "spectral_rolloff_hz": round(float(self.spectral_rolloff), 1),
            "spectral_flatness": round(float(self.spectral_flatness), 4),
            "spectral_flux": round(float(self.spectral_flux), 4),
            "dominant_frequency_hz": round(float(self.dominant_frequency_hz), 1),
            "mfcc_mean": [round(float(v), 4) for v in self.mfcc_mean],
            "chroma_mean": [round(float(v), 4) for v in self.chroma_mean],
        }

    def get_fingerprint_vector(self) -> np.ndarray:
        """Returns normalized 1D composite feature vector for instant matching."""
        stats = np.hstack([
            self.mfcc_mean,
            self.chroma_mean,
            [
                self.spectral_centroid / 4000.0,
                self.spectral_rolloff / 8000.0,
                self.spectral_flatness,
                self.zero_crossing_rate,
                np.clip((self.rms_db + 80.0) / 80.0, 0.0, 1.0),
                np.clip(self.crest_factor / 10.0, 0.0, 1.0)
            ]
        ])
        norm = np.linalg.norm(stats)
        if norm > 1e-7:
            stats = stats / norm
        return stats


class AudioFeatureExtractor:
    """Extracts instantaneous and time-series spectral features from audio waveforms."""

    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()
        self.mel_fb = get_mel_filterbank(
            sr=self.config.sample_rate,
            n_fft=self.config.frame_size,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax
        )
        self.chroma_fb = get_chroma_filterbank(
            sr=self.config.sample_rate,
            n_fft=self.config.frame_size,
            n_chroma=self.config.n_chroma
        )
        self.preprocessor = AudioPreprocessor()

    def extract(self, signal: np.ndarray, apply_pre_emphasis: bool = True) -> AudioFeatures:
        """Extracts complete audio feature representation from a 1D audio waveform."""
        if len(signal) == 0:
            return AudioFeatures(
                duration_sec=0.0, num_frames=0, rms_energy=0.0, rms_db=-100.0,
                peak_amplitude=0.0, crest_factor=1.0, zero_crossing_rate=0.0,
                spectral_centroid=0.0, spectral_spread=0.0, spectral_rolloff=0.0,
                spectral_flatness=0.0, spectral_flux=0.0, dominant_frequency_hz=0.0
            )

        duration = len(signal) / float(self.config.sample_rate)

        # Time-domain metrics
        peak = float(np.max(np.abs(signal)))
        rms = float(np.sqrt(np.mean(signal ** 2)))
        rms_db = float(20.0 * np.log10(max(rms, 1e-9)))
        crest = float(peak / max(rms, 1e-9))
        
        # Zero-Crossing Rate
        signs = np.sign(signal)
        signs[signs == 0] = 1
        zcr = float(np.mean(np.abs(np.diff(signs)) > 0))

        # Pre-emphasis for spectral analysis
        proc_sig = self.preprocessor.apply_pre_emphasis(signal, self.config.pre_emphasis) if apply_pre_emphasis else signal

        # Frame signal if needed or pad
        if len(proc_sig) < self.config.frame_size:
            padded = np.zeros(self.config.frame_size, dtype=np.float32)
            padded[:len(proc_sig)] = proc_sig
            proc_sig = padded

        # STFT
        freqs, times, mag_spec, power_spec = compute_stft(
            proc_sig,
            sr=self.config.sample_rate,
            frame_size=self.config.frame_size,
            hop_size=self.config.hop_size
        )
        num_frames = mag_spec.shape[1]

        # Spectral Centroid & Spread
        power_sum = np.sum(power_spec, axis=0) + 1e-9
        centroid_frames = np.sum(freqs[:, np.newaxis] * power_spec, axis=0) / power_sum
        mean_centroid = float(np.mean(centroid_frames))

        spread_frames = np.sqrt(
            np.sum(((freqs[:, np.newaxis] - centroid_frames) ** 2) * power_spec, axis=0) / power_sum
        )
        mean_spread = float(np.mean(spread_frames))

        # Spectral Rolloff (85% energy point)
        cum_power = np.cumsum(power_spec, axis=0)
        threshold = 0.85 * cum_power[-1, :]
        rolloff_idx = np.argmax(cum_power >= threshold, axis=0)
        mean_rolloff = float(np.mean(freqs[rolloff_idx]))

        # Spectral Flatness (Wiener entropy = geometric mean / arithmetic mean)
        power_clamped = np.maximum(power_spec, 1e-12)
        geom_mean = np.exp(np.mean(np.log(power_clamped), axis=0))
        arith_mean = np.mean(power_clamped, axis=0)
        flatness_frames = geom_mean / (arith_mean + 1e-12)
        mean_flatness = float(np.mean(flatness_frames))

        # Spectral Flux (frame difference in normalized magnitude spectrum)
        mag_norm = mag_spec / (np.sum(mag_spec, axis=0, keepdims=True) + 1e-9)
        if num_frames > 1:
            flux_frames = np.sqrt(np.sum(np.diff(mag_norm, axis=1) ** 2, axis=0))
            mean_flux = float(np.mean(flux_frames))
        else:
            mean_flux = 0.0

        # Dominant Frequency / Pitch Candidate
        mean_power = np.mean(power_spec, axis=1)
        peaks, _ = find_peaks(mean_power, distance=3)
        if len(peaks) > 0:
            top_peak = peaks[np.argmax(mean_power[peaks])]
            dominant_freq = float(freqs[top_peak])
        else:
            dominant_freq = float(freqs[np.argmax(mean_power)])

        # Mel-Spectrogram & MFCC
        mel_energies = np.dot(self.mel_fb, power_spec)
        mel_energies = np.maximum(mel_energies, 1e-10)
        log_mel = 10.0 * np.log10(mel_energies)
        
        # MFCC via DCT-II
        mfccs = dct(log_mel, type=2, axis=0, norm="ortho")[:self.config.n_mfcc]
        mfcc_mean = np.mean(mfccs, axis=1)

        # Chroma features
        chroma_raw = np.dot(self.chroma_fb, power_spec)
        chroma_sums = np.sum(chroma_raw, axis=0, keepdims=True)
        chroma_sums[chroma_sums == 0] = 1.0
        chroma_frames = chroma_raw / chroma_sums
        chroma_mean = np.mean(chroma_frames, axis=1)

        return AudioFeatures(
            duration_sec=duration,
            num_frames=num_frames,
            rms_energy=rms,
            rms_db=rms_db,
            peak_amplitude=peak,
            crest_factor=crest,
            zero_crossing_rate=zcr,
            spectral_centroid=mean_centroid,
            spectral_spread=mean_spread,
            spectral_rolloff=mean_rolloff,
            spectral_flatness=mean_flatness,
            spectral_flux=mean_flux,
            dominant_frequency_hz=dominant_freq,
            mfcc_mean=mfcc_mean,
            mfcc_frames=mfccs,
            chroma_mean=chroma_mean,
            chroma_frames=chroma_frames,
            mel_spectrogram=log_mel
        )
