"""
Sound pattern template registry for storing, indexing, and serializing acoustic references.
"""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np

from ..config import AudioConfig
from ..dsp.features import AudioFeatureExtractor, AudioFeatures


@dataclass
class SoundPattern:
    name: str
    description: str
    features: AudioFeatures
    fingerprint: list[float]
    sample_rate: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "sample_rate": self.sample_rate,
            "features": self.features.to_dict(),
            "fingerprint": self.fingerprint,
        }


class PatternRegistry:
    """Manages registered acoustic templates and persistence."""

    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()
        self.patterns: dict[str, SoundPattern] = {}
        self.extractor = AudioFeatureExtractor(self.config)
        self.storage_file = self.config.patterns_dir / "patterns_registry.json"
        self.load_from_disk()

    def register_signal(self, name: str, signal: np.ndarray, description: str = "") -> SoundPattern:
        """Extracts features from an audio array and registers it as a named pattern."""
        features = self.extractor.extract(signal)
        fp = features.get_fingerprint_vector().tolist()
        pattern = SoundPattern(
            name=name,
            description=description,
            features=features,
            fingerprint=fp,
            sample_rate=self.config.sample_rate
        )
        self.patterns[name] = pattern
        self.save_to_disk()
        return pattern

    def remove_pattern(self, name: str) -> bool:
        """Deletes a pattern by name."""
        if name in self.patterns:
            del self.patterns[name]
            self.save_to_disk()
            return True
        return False

    def list_patterns(self) -> list[dict]:
        """Returns summary metadata for all registered patterns."""
        return [
            {
                "name": p.name,
                "description": p.description,
                "duration_sec": p.features.duration_sec,
                "dominant_freq_hz": p.features.dominant_frequency_hz,
                "rms_db": p.features.rms_db
            }
            for p in self.patterns.values()
        ]

    def save_to_disk(self):
        """Saves registered pattern metadata to a JSON file."""
        data = {name: p.to_dict() for name, p in self.patterns.items()}
        try:
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save patterns registry: {e}")

    def load_from_disk(self):
        """Loads pattern registry from disk if it exists."""
        if not self.storage_file.exists():
            return
        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for name, item in data.items():
                feat_dict = item.get("features", {})
                mfcc_mean = np.array(feat_dict.get("mfcc_mean", np.zeros(13)), dtype=np.float32)
                chroma_mean = np.array(feat_dict.get("chroma_mean", np.zeros(12)), dtype=np.float32)
                
                # Reconstruct AudioFeatures object
                feats = AudioFeatures(
                    duration_sec=feat_dict.get("duration_sec", 0.0),
                    num_frames=feat_dict.get("num_frames", 1),
                    rms_energy=feat_dict.get("rms_energy", 0.0),
                    rms_db=feat_dict.get("rms_db", -50.0),
                    peak_amplitude=feat_dict.get("peak_amplitude", 0.0),
                    crest_factor=feat_dict.get("crest_factor", 1.0),
                    zero_crossing_rate=feat_dict.get("zero_crossing_rate", 0.0),
                    spectral_centroid=feat_dict.get("spectral_centroid_hz", 0.0),
                    spectral_spread=feat_dict.get("spectral_spread_hz", 0.0),
                    spectral_rolloff=feat_dict.get("spectral_rolloff_hz", 0.0),
                    spectral_flatness=feat_dict.get("spectral_flatness", 0.0),
                    spectral_flux=feat_dict.get("spectral_flux", 0.0),
                    dominant_frequency_hz=feat_dict.get("dominant_frequency_hz", 0.0),
                    mfcc_mean=mfcc_mean,
                    mfcc_frames=np.tile(mfcc_mean[:, np.newaxis], (1, 5)),
                    chroma_mean=chroma_mean,
                    chroma_frames=np.tile(chroma_mean[:, np.newaxis], (1, 5)),
                )
                self.patterns[name] = SoundPattern(
                    name=name,
                    description=item.get("description", ""),
                    features=feats,
                    fingerprint=item.get("fingerprint", []),
                    sample_rate=item.get("sample_rate", self.config.sample_rate)
                )
        except Exception as e:
            print(f"Warning: Could not load patterns registry from disk: {e}")

