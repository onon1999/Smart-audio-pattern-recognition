"""
Visualizers for audio analysis: Matplotlib 4-panel spectral analysis plots and ASCII terminal HUD.
"""
import os
import tempfile
from pathlib import Path
import numpy as np

# Configure non-conflicting writable cache directory for Matplotlib on Windows
if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(tempfile.gettempdir()) / "echopulse_mpl"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import matplotlib
# Use non-interactive Agg backend for robust server/headless plotting
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..dsp.features import AudioFeatures
from ..dsp.filterbanks import compute_stft


class AudioVisualizer:
    """Generates graphical and ASCII terminal visualizations for audio patterns."""

    @staticmethod
    def create_analysis_figure(
        signal: np.ndarray,
        sr: int,
        features: AudioFeatures,
        title: str = "Audio Pattern Analysis",
        output_path: str | Path | None = None
    ) -> plt.Figure:
        """
        Creates a high-resolution 4-panel diagnostic figure:
        1. Waveform & RMS Envelope
        2. STFT Spectrogram
        3. Mel Spectrogram (40 bands)
        4. MFCC Heatmap (13 coefficients)
        """
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), constrained_layout=True)
        fig.suptitle(f"{title} ({features.duration_sec:.2f}s, SR={sr}Hz)", fontsize=14, fontweight="bold")

        # 1. Waveform
        t_sig = np.linspace(0, features.duration_sec, len(signal))
        axes[0].plot(t_sig, signal, color="#2563EB", linewidth=0.8, alpha=0.85, label="Waveform")
        axes[0].set_title(f"Time Domain Waveform (Peak: {features.peak_amplitude:.2f}, RMS: {features.rms_db:.1f} dB)", fontsize=11)
        axes[0].set_ylabel("Amplitude")
        axes[0].set_xlim(0, max(features.duration_sec, 0.01))
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # 2. STFT Spectrogram
        freqs, times, mag_spec, power_spec = compute_stft(signal, sr=sr)
        log_stft = 10.0 * np.log10(np.maximum(power_spec, 1e-9))
        im1 = axes[1].imshow(
            log_stft,
            origin="lower",
            aspect="auto",
            extent=[times[0] if len(times) else 0, times[-1] if len(times) else 1, freqs[0], freqs[-1]],
            cmap="inferno"
        )
        axes[1].set_title(f"STFT Spectrogram (Spectral Centroid: {features.spectral_centroid:.0f} Hz)", fontsize=11)
        axes[1].set_ylabel("Frequency (Hz)")
        fig.colorbar(im1, ax=axes[1], format="%+2.0f dB")

        # 3. Mel Spectrogram
        if features.mel_spectrogram.size > 0:
            im2 = axes[2].imshow(
                features.mel_spectrogram,
                origin="lower",
                aspect="auto",
                cmap="viridis"
            )
            axes[2].set_title("Log Mel-Filterbank Energies (40 Bands)", fontsize=11)
            axes[2].set_ylabel("Mel Band Index")
            fig.colorbar(im2, ax=axes[2], format="%+2.0f dB")
        else:
            axes[2].text(0.5, 0.5, "No Mel Data Available", ha="center")

        # 4. MFCC Heatmap
        if features.mfcc_frames.size > 0:
            im3 = axes[3].imshow(
                features.mfcc_frames,
                origin="lower",
                aspect="auto",
                cmap="coolwarm"
            )
            axes[3].set_title(f"Mel-Frequency Cepstral Coefficients (MFCC 1-13)", fontsize=11)
            axes[3].set_xlabel("Time Frame Index")
            axes[3].set_ylabel("MFCC Index")
            fig.colorbar(im3, ax=axes[3])
        else:
            axes[3].text(0.5, 0.5, "No MFCC Data Available", ha="center")

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=160)

        return fig

    @staticmethod
    def render_ascii_meter(val: float, min_val: float, max_val: float, width: int = 24) -> str:
        """Renders an ASCII horizontal level meter: [████████░░░░░░░░]."""
        clamped = np.clip((val - min_val) / max(1e-6, (max_val - min_val)), 0.0, 1.0)
        filled = int(round(clamped * width))
        return "█" * filled + "░" * (width - filled)

    @staticmethod
    def render_ascii_spectrum(energy_bands: list[float] | np.ndarray, num_bars: int = 16) -> str:
        """Renders vertical ASCII spectrum bars:  ▂▃▄▅▆▇█."""
        chars = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        if len(energy_bands) == 0:
            return " " * num_bars
        # Resample to num_bars
        indices = np.linspace(0, len(energy_bands) - 1, num_bars).astype(int)
        sampled = np.array(energy_bands)[indices]
        peak = np.max(sampled) if np.max(sampled) > 1e-6 else 1.0
        normalized = np.clip(sampled / peak, 0.0, 1.0)
        bar_str = "".join([chars[int(val * 8)] for val in normalized])
        return bar_str
