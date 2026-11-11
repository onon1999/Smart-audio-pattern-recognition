"""
Immediate Acoustic Event Detection & Sound Category Classifier.
"""
from dataclasses import dataclass
from enum import Enum
from ..config import AudioConfig
from ..dsp.features import AudioFeatures


class SoundCategory(str, Enum):
    SILENCE = "Silence / Quiet"
    TRANSIENT_IMPULSE = "Transient (Clap / Click / Beat)"
    TONAL_HARMONIC = "Tonal / Harmonic (Whistle / Tone / Beep)"
    SPEECH_VOICE = "Speech / Vocalization"
    BROADBAND_NOISE = "Broadband Noise (Hiss / Fan / Air)"
    COMPLEX_AUDIO = "Complex Acoustic Mixture"


@dataclass
class AcousticClassification:
    category: SoundCategory
    confidence: float
    summary: str
    primary_traits: list[str]

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "confidence": round(float(self.confidence), 4),
            "summary": self.summary,
            "primary_traits": self.primary_traits,
        }


class ImmediateAcousticClassifier:
    """Classifies audio chunks immediately based on multi-dimensional spectral and temporal features."""

    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()

    def classify(self, features: AudioFeatures) -> AcousticClassification:
        traits = []

        # 1. Check Silence
        if features.rms_db < self.config.silence_threshold_db:
            traits.append(f"Low energy ({features.rms_db:.1f} dB)")
            return AcousticClassification(
                category=SoundCategory.SILENCE,
                confidence=min(1.0, abs(features.rms_db - self.config.silence_threshold_db) / 20.0),
                summary="Acoustic background level is near silence.",
                primary_traits=traits
            )

        # 2. Check Transient (Clap / Click / Tap)
        if features.crest_factor > self.config.transient_crest_threshold and features.rms_db > -35.0:
            traits.append(f"High Crest Factor ({features.crest_factor:.2f})")
            traits.append(f"Sharp transient onset (Peak {features.peak_amplitude:.2f})")
            conf = min(1.0, 0.5 + (features.crest_factor - self.config.transient_crest_threshold) / 5.0)
            return AcousticClassification(
                category=SoundCategory.TRANSIENT_IMPULSE,
                confidence=conf,
                summary="Rapid onset and sharp peak characteristic of a clap, tap, or percussion hit.",
                primary_traits=traits
            )

        # 3. Check Tonal / Whistle / Pure Harmonic
        if features.spectral_flatness < self.config.tonal_flatness_thresh and features.dominant_frequency_hz > 80.0:
            traits.append(f"Low spectral flatness ({features.spectral_flatness:.4f})")
            traits.append(f"Dominant pitch at {features.dominant_frequency_hz:.1f} Hz")
            conf = min(1.0, 0.6 + (self.config.tonal_flatness_thresh - features.spectral_flatness) * 5.0)
            return AcousticClassification(
                category=SoundCategory.TONAL_HARMONIC,
                confidence=conf,
                summary=f"Pronounced tonal energy concentrated around {features.dominant_frequency_hz:.0f} Hz (whistle or musical tone).",
                primary_traits=traits
            )

        # 4. Check Broadband Noise
        if features.spectral_flatness > self.config.spectral_flatness_noise_thresh and features.zero_crossing_rate > 0.15:
            traits.append(f"High spectral flatness ({features.spectral_flatness:.3f})")
            traits.append(f"Broadband high zero-crossing rate ({features.zero_crossing_rate:.3f})")
            conf = min(1.0, 0.6 + (features.spectral_flatness - self.config.spectral_flatness_noise_thresh) * 2.0)
            return AcousticClassification(
                category=SoundCategory.BROADBAND_NOISE,
                confidence=conf,
                summary="Uniform energy distribution across frequencies typical of hiss, airflow, or machinery.",
                primary_traits=traits
            )

        # 5. Check Speech / Vocal Formants
        # Human speech typically has centroid in 400Hz - 2800Hz, moderate flatness, and harmonic formant structure
        if 250.0 <= features.spectral_centroid <= 3200.0 and features.spectral_flatness < 0.28:
            traits.append(f"Vocal spectral centroid ({features.spectral_centroid:.0f} Hz)")
            traits.append(f"Moderate spectral flatness ({features.spectral_flatness:.3f})")
            traits.append("Formant spectral distribution")
            return AcousticClassification(
                category=SoundCategory.SPEECH_VOICE,
                confidence=0.78,
                summary="Spectral centroid and formant distribution align with human speech or vocal sound.",
                primary_traits=traits
            )

        # 6. Complex Acoustic Activity
        traits.append(f"Centroid: {features.spectral_centroid:.0f} Hz")
        traits.append(f"Rolloff: {features.spectral_rolloff:.0f} Hz")
        traits.append(f"Energy: {features.rms_db:.1f} dB")
        return AcousticClassification(
            category=SoundCategory.COMPLEX_AUDIO,
            confidence=0.65,
            summary="Multi-component acoustic pattern with mixed harmonic and noisy content.",
            primary_traits=traits
        )

