import numpy as np
import pytest

from audio_pulse.config import AudioConfig
from audio_pulse.dsp.features import AudioFeatureExtractor
from audio_pulse.recognition.classifier import ImmediateAcousticClassifier, SoundCategory
from audio_pulse.recognition.anomaly import AcousticAnomalyDetector
from audio_pulse.recognition.matcher import AudioPatternMatcher
from audio_pulse.recognition.registry import PatternRegistry
from audio_pulse.utils.synthetic_sounds import SyntheticAudioGenerator


def test_acoustic_classifier_silence():
    config = AudioConfig()
    classifier = ImmediateAcousticClassifier(config)
    extractor = AudioFeatureExtractor(config)
    silent_sig = np.zeros(2048, dtype=np.float32)
    feats = extractor.extract(silent_sig)
    res = classifier.classify(feats)
    assert res.category == SoundCategory.SILENCE


def test_acoustic_classifier_transient_clap():
    config = AudioConfig()
    classifier = ImmediateAcousticClassifier(config)
    extractor = AudioFeatureExtractor(config)
    synth = SyntheticAudioGenerator(config.sample_rate)
    clap = synth.generate_clap(0.35)
    feats = extractor.extract(clap)
    res = classifier.classify(feats)
    assert res.category == SoundCategory.TRANSIENT_IMPULSE


def test_acoustic_classifier_tonal_whistle():
    config = AudioConfig()
    classifier = ImmediateAcousticClassifier(config)
    extractor = AudioFeatureExtractor(config)
    synth = SyntheticAudioGenerator(config.sample_rate)
    whistle = synth.generate_whistle_sweep(1200.0, 1800.0, 0.8)
    feats = extractor.extract(whistle)
    res = classifier.classify(feats)
    assert res.category == SoundCategory.TONAL_HARMONIC


def test_pattern_matching_dtw_and_registry():
    config = AudioConfig()
    extractor = AudioFeatureExtractor(config)
    matcher = AudioPatternMatcher(config)
    registry = PatternRegistry(config)
    synth = SyntheticAudioGenerator(config.sample_rate)

    ref_clap = synth.generate_clap(0.35)
    registry.register_signal("Hand Clap Reference", ref_clap, "Reference clap")

    # Match identical signal -> should be high confidence match
    query_clap = synth.generate_clap(0.35)
    query_feats = extractor.extract(query_clap)
    matches = matcher.match_against_registry(query_feats, registry)

    assert len(matches) >= 1
    top_match = matches[0]
    assert top_match.pattern_name == "Hand Clap Reference"
    assert top_match.confidence > 0.70
    assert top_match.matched is True


def test_anomaly_detector():
    config = AudioConfig(anomaly_window_size=10, anomaly_sigma_threshold=2.5)
    detector = AcousticAnomalyDetector(config)
    extractor = AudioFeatureExtractor(config)

    # Establish baseline with low noise
    for _ in range(15):
        noise = np.random.randn(1024).astype(np.float32) * 0.01
        feats = extractor.extract(noise)
        detector.process_frame(feats)

    # Sudden loud burst
    burst = (np.sin(2 * np.pi * 1000.0 * np.linspace(0, 0.1, 1024)) * 0.95).astype(np.float32)
    burst_feats = extractor.extract(burst)
    report = detector.process_frame(burst_feats)

    assert report.is_anomaly is True
    assert report.anomaly_score > 50.0
