"""
Generates pre-packaged reference sample WAV files for benchmarking and immediate testing.
"""
import sys
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_pulse.config import AudioConfig
from audio_pulse.utils.synthetic_sounds import SyntheticAudioGenerator


def main():
    config = AudioConfig()
    synth = SyntheticAudioGenerator(sample_rate=config.sample_rate)
    samples_dir = config.samples_dir
    print(f"Generating synthetic acoustic reference samples in {samples_dir}...")
    paths = synth.generate_all_samples(samples_dir)
    for name, path in paths.items():
        print(f"  ✓ Created: {name} -> {path.name}")
    print(f"Generated {len(paths)} test audio files successfully!")


if __name__ == "__main__":
    main()
