import numpy as np
import pytest

from audio_pulse.config import AudioConfig
from audio_pulse.dsp.preprocessor import AudioPreprocessor
from audio_pulse.dsp.filterbanks import get_mel_filterbank, get_chroma_filterbank, compute_stft, hz_to_mel, mel_to_hz
from audio_pulse.dsp.features import AudioFeatureExtractor


def test_hz_mel_roundtrip():
    hz = 1000.0
    mel = hz_to_mel(hz)
    recovered_hz = mel_to_hz(mel)
    assert abs(hz - recovered_hz) < 1e-3


def test_mel_filterbank_shape():
    sr = 22050
    n_fft = 1024
    n_mels = 40
    fb = get_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels)
    assert fb.shape == (n_mels, n_fft // 2 + 1)
    assert not np.isnan(fb).any()
    assert fb.min() >= 0.0


def test_chroma_filterbank_shape():
    sr = 22050
    n_fft = 1024
    n_chroma = 12
    fb = get_chroma_filterbank(sr=sr, n_fft=n_fft, n_chroma=n_chroma)
    assert fb.shape == (n_chroma, n_fft // 2 + 1)
    assert not np.isnan(fb).any()


def test_stft_computation():
    sr = 22050
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    sig = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    freqs, times, mag, pwr = compute_stft(sig, sr=sr, frame_size=1024, hop_size=512)
    assert len(freqs) == 1024 // 2 + 1
    assert mag.shape == pwr.shape
    assert mag.shape[0] == len(freqs)
    assert mag.shape[1] == len(times)


def test_preprocessor_normalization():
    sig = np.array([-2.0, 0.0, 4.0, -1.0], dtype=np.float32)
    normed = AudioPreprocessor.normalize(sig, target_peak=0.95)
    assert np.max(np.abs(normed)) == pytest.approx(0.95, abs=1e-5)


def test_feature_extractor():
    config = AudioConfig(sample_rate=22050)
    extractor = AudioFeatureExtractor(config)
    t = np.linspace(0, 0.5, int(config.sample_rate * 0.5), endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    feats = extractor.extract(sig)

    assert feats.duration_sec == pytest.approx(0.5, abs=0.01)
    assert feats.peak_amplitude > 0.4
    assert feats.rms_db > -30.0
    # Dominant frequency should be close to 440 Hz
    assert abs(feats.dominant_frequency_hz - 440.0) < 30.0
    assert len(feats.mfcc_mean) == 13
    assert len(feats.chroma_mean) == 12

    fp = feats.get_fingerprint_vector()
    assert len(fp) == 13 + 12 + 6
    assert abs(np.linalg.norm(fp) - 1.0) < 1e-4
