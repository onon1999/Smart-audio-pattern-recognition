"""
EchoPulse - Immediate Audio Pattern Recognition & Spectral Analysis Engine
"""

__version__ = "1.0.0"
__author__ = "AI Audio Developer"

from .config import AudioConfig
from .dsp.features import AudioFeatureExtractor
from .recognition.matcher import AudioPatternMatcher
from .recognition.classifier import ImmediateAcousticClassifier
from .recognition.anomaly import AcousticAnomalyDetector
from .recognition.registry import PatternRegistry

__all__ = [
    "AudioConfig",
    "AudioFeatureExtractor",
    "AudioPatternMatcher",
    "ImmediateAcousticClassifier",
    "AcousticAnomalyDetector",
    "PatternRegistry",
]
