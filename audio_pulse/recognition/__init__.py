"""
Pattern recognition, acoustic event classification, and anomaly detection.
"""
from .matcher import AudioPatternMatcher, MatchResult
from .classifier import ImmediateAcousticClassifier, AcousticClassification
from .anomaly import AcousticAnomalyDetector, AnomalyReport
from .registry import PatternRegistry, SoundPattern

__all__ = [
    "AudioPatternMatcher",
    "MatchResult",
    "ImmediateAcousticClassifier",
    "AcousticClassification",
    "AcousticAnomalyDetector",
    "AnomalyReport",
    "PatternRegistry",
    "SoundPattern",
]
