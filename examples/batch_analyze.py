"""
Batch audio analyzer: processes a folder of WAV files, exports JSON reports and PNG diagnostic plots.
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_pulse.config import AudioConfig
from audio_pulse.dsp.features import AudioFeatureExtractor
from audio_pulse.recognition.classifier import ImmediateAcousticClassifier
from audio_pulse.recognition.matcher import AudioPatternMatcher
from audio_pulse.recognition.registry import PatternRegistry
from audio_pulse.utils.audio_io import load_wav
from audio_pulse.utils.visualizer import AudioVisualizer


def run_batch_analysis(input_dir: str | Path, output_dir: str | Path):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plots_dir = out_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    config = AudioConfig()
    extractor = AudioFeatureExtractor(config)
    classifier = ImmediateAcousticClassifier(config)
    matcher = AudioPatternMatcher(config)
    registry = PatternRegistry(config)

    wav_files = list(in_path.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files found in {in_path}")
        return

    print(f"Found {len(wav_files)} WAV files. Beginning batch analysis...")
    summary_report = []

    for idx, wav_path in enumerate(wav_files, start=1):
        try:
            signal, sr = load_wav(wav_path, target_sr=config.sample_rate)
            features = extractor.extract(signal)
            classification = classifier.classify(features)
            matches = matcher.match_against_registry(features, registry)

            # Generate diagnostic plot
            plot_file = plots_dir / f"{wav_path.stem}_diagnostics.png"
            AudioVisualizer.create_analysis_figure(
                signal,
                sr,
                features,
                title=f"Analysis — {wav_path.name}",
                output_path=plot_file
            )

            file_report = {
                "file_name": wav_path.name,
                "duration_sec": features.duration_sec,
                "rms_db": features.rms_db,
                "dominant_pitch_hz": features.dominant_frequency_hz,
                "classification": classification.to_dict(),
                "top_match": matches[0].to_dict() if matches else None,
                "plot_saved": str(plot_file.name)
            }
            summary_report.append(file_report)
            print(f"[{idx}/{len(wav_files)}] Analyzed: {wav_path.name} -> {classification.category.value}")
        except Exception as e:
            print(f"Error analyzing {wav_path.name}: {e}")

    report_json_path = out_path / "batch_analysis_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print(f"\nBatch analysis complete! Report saved to {report_json_path}")
    print(f"Diagnostic plots saved in {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description="EchoPulse Batch Audio Analyzer")
    parser.add_argument("--input", "-i", type=str, default="samples", help="Input directory containing WAV files")
    parser.add_argument("--output", "-o", type=str, default="outputs/batch_run", help="Output directory for reports & plots")
    args = parser.parse_args()
    run_batch_analysis(args.input, args.output)


if __name__ == "__main__":
    main()
