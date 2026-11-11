"""
Configuration parameters for audio processing, DSP feature extraction, and pattern recognition.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AudioConfig:
    # Sampling & Framing
    sample_rate: int = 22050
    frame_size: int = 1024       # FFT window size (~46 ms at 22050 Hz)
    hop_size: int = 512          # 50% overlap (~23 ms hop)
    
    # Feature Extraction
    n_mels: int = 40             # Mel-filterbank channels
    n_mfcc: int = 13             # Number of MFCC coefficients
    n_chroma: int = 12           # Pitch classes (C, C#, D, ...)
    fmin: float = 20.0           # Minimum frequency for Mel filters (Hz)
    fmax: float = 8000.0         # Maximum frequency for Mel filters (Hz)
    pre_emphasis: float = 0.97   # Pre-emphasis coefficient
    
    # Real-Time Streaming & Buffer
    chunk_duration_sec: float = 0.1   # Immediate processing chunk size (100 ms)
    buffer_duration_sec: float = 3.0  # Ring buffer length (3.0 seconds memory)
    
    # Immediate Event Detection Thresholds
    silence_threshold_db: float = -45.0  # Below this RMS dB is classified as silence
    transient_crest_threshold: float = 4.5  # Crest factor to identify sharp transients (claps, clicks)
    spectral_flatness_noise_thresh: float = 0.35  # Above this is broadband noise
    tonal_flatness_thresh: float = 0.08  # Below this indicates highly tonal/harmonic sound
    
    # Pattern Matching
    dtw_distance_threshold: float = 0.35  # DTW distance below which a pattern is considered a match
    cosine_similarity_threshold: float = 0.82  # Cosine similarity threshold for static fingerprints
    
    # Anomaly Detection
    anomaly_window_size: int = 30     # Number of recent frames for baseline statistics
    anomaly_sigma_threshold: float = 2.8  # Standard deviations above mean to flag anomaly
    
    # Storage Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    patterns_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "patterns_storage")
    samples_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "samples")

    def __post_init__(self):
        self.patterns_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = int(self.sample_rate * self.chunk_duration_sec)
        self.buffer_size = int(self.sample_rate * self.buffer_duration_sec)
