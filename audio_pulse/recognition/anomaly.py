"""
Real-time acoustic anomaly and sudden novelty detector.
"""
from collections import deque
from dataclasses import dataclass
import numpy as np
from ..config import AudioConfig
from ..dsp.features import AudioFeatures


@dataclass
class AnomalyReport:
    is_anomaly: bool
    anomaly_score: float     # 0.0 to 100.0 score
    z_score: float
    trigger_feature: str
    message: str

    def to_dict(self) -> dict:
        return {
            "is_anomaly": self.is_anomaly,
            "anomaly_score": round(float(self.anomaly_score), 2),
            "z_score": round(float(self.z_score), 2),
            "trigger_feature": self.trigger_feature,
            "message": self.message,
        }


class AcousticAnomalyDetector:
    """Detects immediate acoustic surprises, sudden energy bursts, and spectral novelties."""

    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()
        self.window_size = self.config.anomaly_window_size
        self.energy_history = deque(maxlen=self.window_size)
        self.flux_history = deque(maxlen=self.window_size)
        self.centroid_history = deque(maxlen=self.window_size)

    def reset(self):
        """Resets running baseline history."""
        self.energy_history.clear(maxlen=self.window_size)
        self.flux_history.clear(maxlen=self.window_size)
        self.centroid_history.clear(maxlen=self.window_size)

    def process_frame(self, features: AudioFeatures) -> AnomalyReport:
        """Evaluates incoming frame against rolling baseline statistics."""
        e = features.rms_energy
        f = features.spectral_flux
        c = features.spectral_centroid

        # If not enough history yet, record and return normal
        if len(self.energy_history) < 8:
            self.energy_history.append(e)
            self.flux_history.append(f)
            self.centroid_history.append(c)
            return AnomalyReport(
                is_anomaly=False,
                anomaly_score=0.0,
                z_score=0.0,
                trigger_feature="None",
                message="Calibrating baseline background acoustic profile..."
            )

        # Baseline stats
        e_mean, e_std = float(np.mean(self.energy_history)), float(np.std(self.energy_history)) + 1e-6
        f_mean, f_std = float(np.mean(self.flux_history)), float(np.std(self.flux_history)) + 1e-6
        c_mean, c_std = float(np.mean(self.centroid_history)), float(np.std(self.centroid_history)) + 1e-6

        # Z-scores
        z_e = max(0.0, (e - e_mean) / e_std)
        z_f = max(0.0, (f - f_mean) / f_std)
        z_c = abs(c - c_mean) / c_std

        max_z = max(z_e, z_f, z_c)
        trigger = "None"
        if max_z == z_e:
            trigger = "Energy Burst"
        elif max_z == z_f:
            trigger = "Spectral Novelty / Transient"
        elif max_z == z_c:
            trigger = "Frequency Shift"

        # Update history (slow adaptation to prevent anomaly from immediately diluting baseline)
        if max_z < 3.5:
            self.energy_history.append(e)
            self.flux_history.append(f)
            self.centroid_history.append(c)

        # Compute anomaly score (0 to 100)
        # Using sigmoid scaling around sigma_threshold
        thresh = self.config.anomaly_sigma_threshold
        score = float(np.clip((max_z / thresh) * 50.0, 0.0, 100.0))
        is_anomaly = max_z >= thresh and features.rms_db > -45.0

        msg = "Acoustic characteristics within expected baseline."
        if is_anomaly:
            msg = f"Sudden acoustic anomaly detected via {trigger} (Z-Score: {max_z:.1f})!"

        return AnomalyReport(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            z_score=max_z,
            trigger_feature=trigger,
            message=msg
        )
