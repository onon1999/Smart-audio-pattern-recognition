"""
Synthetic audio pattern generator for immediate testing without external sound files or microphones.
"""
from pathlib import Path
import numpy as np
from .audio_io import save_wav


class SyntheticAudioGenerator:
    """Generates synthetic acoustic patterns (tones, claps, sweeps, chords, alarms)."""

    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate

    def generate_clap(self, duration: float = 0.35) -> np.ndarray:
        """Simulates a sharp acoustic handclap or transient click with exponential decay."""
        n_samples = int(self.sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        # Filtered white noise with rapid decay burst and secondary reflection
        noise = np.random.randn(n_samples)
        envelope = np.exp(-t * 28.0) + 0.35 * np.exp(-np.maximum(0, t - 0.02) * 32.0)
        signal = noise * envelope
        # Add high-frequency resonance
        click = np.sin(2 * np.pi * 1200.0 * t) * np.exp(-t * 80.0)
        combined = signal * 0.7 + click * 0.3
        return (combined / (np.max(np.abs(combined)) + 1e-6) * 0.9).astype(np.float32)

    def generate_whistle_sweep(self, start_freq: float = 900.0, end_freq: float = 2400.0, duration: float = 0.8) -> np.ndarray:
        """Simulates a human whistle or frequency chirp."""
        n_samples = int(self.sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        # Linear frequency modulation
        freq_t = np.linspace(start_freq, end_freq, n_samples)
        phase = 2 * np.pi * np.cumsum(freq_t) / self.sr
        # Smooth attack and release envelope
        env = np.sin(np.pi * t / duration) ** 2
        signal = np.sin(phase) * env
        # Slight breathiness / noise
        breath = np.random.randn(n_samples) * 0.02 * env
        combined = signal + breath
        return (combined / (np.max(np.abs(combined)) + 1e-6) * 0.85).astype(np.float32)

    def generate_pure_tone(self, freq: float = 440.0, duration: float = 0.8) -> np.ndarray:
        """Generates a clean sinusoidal reference tone."""
        n_samples = int(self.sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        env = np.ones(n_samples)
        # 20ms ramp-in / ramp-out to avoid click
        ramp_len = int(0.02 * self.sr)
        env[:ramp_len] = np.linspace(0, 1, ramp_len)
        env[-ramp_len:] = np.linspace(1, 0, ramp_len)
        signal = np.sin(2 * np.pi * freq * t) * env
        return signal.astype(np.float32) * 0.8

    def generate_harmonic_chord(self, freqs: tuple[float, ...] = (261.63, 329.63, 392.00), duration: float = 1.0) -> np.ndarray:
        """Generates a musical harmonic chord (e.g. C-Major triad: C4, E4, G4)."""
        n_samples = int(self.sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        env = np.sin(np.pi * t / duration) ** 1.5
        signal = np.zeros(n_samples)
        for i, f in enumerate(freqs):
            # Fundamental + overtone
            signal += (1.0 / (i + 1)) * np.sin(2 * np.pi * f * t)
            signal += (0.3 / (i + 1)) * np.sin(2 * np.pi * 2 * f * t)
        signal = signal * env
        return (signal / (np.max(np.abs(signal)) + 1e-6) * 0.85).astype(np.float32)

    def generate_alarm_pulse(self, freq: float = 1400.0, pulse_rate_hz: float = 4.0, duration: float = 1.2) -> np.ndarray:
        """Generates a pulsing high-frequency alarm signal."""
        n_samples = int(self.sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        carrier = np.sin(2 * np.pi * freq * t) + 0.4 * np.sin(2 * np.pi * (freq * 1.5) * t)
        modulator = 0.5 * (1.0 + np.sign(np.sin(2 * np.pi * pulse_rate_hz * t)))
        signal = carrier * modulator
        return (signal * 0.8).astype(np.float32)

    def generate_noise(self, noise_type: str = "pink", duration: float = 1.0) -> np.ndarray:
        """Generates white or pink acoustic noise."""
        n_samples = int(self.sr * duration)
        white = np.random.randn(n_samples)
        if noise_type == "pink":
            # 1/f noise filter approximation
            b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
            a = [1.0, -2.494956002, 2.017265875, -0.522189400]
            from scipy.signal import lfilter
            pink = lfilter(b, a, white)
            signal = pink
        else:
            signal = white
        return (signal / (np.max(np.abs(signal)) + 1e-6) * 0.5).astype(np.float32)

    def generate_speech_like(self, f0: float = 130.0, duration: float = 1.0) -> np.ndarray:
        """Simulates speech-like vowel vocalization via pulse excitation and formant resonance."""
        n_samples = int(self.sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        # Glottal pulse train
        pulse_period = int(self.sr / f0)
        excitation = np.zeros(n_samples)
        excitation[::pulse_period] = 1.0
        
        # Formants (Vowel /a/: ~800 Hz, ~1200 Hz, ~2500 Hz)
        formants = [(800.0, 80.0), (1200.0, 90.0), (2500.0, 120.0)]
        signal = np.zeros(n_samples)
        for f_c, bw in formants:
            damp = np.exp(-np.pi * bw / self.sr)
            freq_rad = 2 * np.pi * f_c / self.sr
            b = [1.0, 0.0, 0.0]
            a = [1.0, -2.0 * damp * np.cos(freq_rad), damp ** 2]
            from scipy.signal import lfilter
            signal += lfilter(b, a, excitation)
            
        env = np.sin(np.pi * t / duration) ** 2
        signal = signal * env
        return (signal / (np.max(np.abs(signal)) + 1e-6) * 0.8).astype(np.float32)

    def generate_all_samples(self, target_dir: str | Path) -> dict[str, Path]:
        """Generates all reference test WAV files and saves them to disk."""
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        samples = {
            "hand_clap": self.generate_clap(0.35),
            "whistle_chirp": self.generate_whistle_sweep(900.0, 2400.0, 0.8),
            "pure_tone_440hz": self.generate_pure_tone(440.0, 0.8),
            "c_major_chord": self.generate_harmonic_chord((261.63, 329.63, 392.00), 1.0),
            "alarm_siren": self.generate_alarm_pulse(1400.0, 4.0, 1.2),
            "ambient_noise": self.generate_noise("pink", 1.0),
            "speech_vocalization": self.generate_speech_like(130.0, 1.0),
        }
        
        saved_paths = {}
        for name, sig in samples.items():
            path = target_path / f"{name}.wav"
            save_wav(path, sig, self.sr)
            saved_paths[name] = path
            
        return saved_paths

