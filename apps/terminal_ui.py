"""
Interactive High-Performance Terminal HUD for Real-Time Audio Pattern Recognition.
"""
import argparse
import sys
import time
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from audio_pulse.config import AudioConfig
from audio_pulse.dsp.features import AudioFeatureExtractor
from audio_pulse.recognition.classifier import ImmediateAcousticClassifier
from audio_pulse.recognition.anomaly import AcousticAnomalyDetector
from audio_pulse.recognition.matcher import AudioPatternMatcher
from audio_pulse.recognition.registry import PatternRegistry
from audio_pulse.stream.simulated_stream import SimulatedAudioStream
from audio_pulse.stream.live_stream import LiveMicrophoneStream
from audio_pulse.utils.synthetic_sounds import SyntheticAudioGenerator
from audio_pulse.utils.visualizer import AudioVisualizer


def build_hud_panel(
    features,
    classification,
    anomaly_report,
    matches,
    mode_text: str,
    fps: float
) -> Panel:
    """Builds a Rich terminal HUD layout."""
    # Top Metrics Table
    stats_table = Table.grid(expand=True)
    stats_table.add_column(justify="left", ratio=1)
    stats_table.add_column(justify="left", ratio=1)
    stats_table.add_column(justify="left", ratio=1)

    # Meter visualizations
    vu_meter = AudioVisualizer.render_ascii_meter(features.rms_db, -60.0, 0.0, width=18)
    spectrum_bars = AudioVisualizer.render_ascii_spectrum(features.mfcc_mean, num_bars=13)

    stats_table.add_row(
        f"[bold cyan]Input Mode:[/] {mode_text}",
        f"[bold yellow]Volume VU:[/] [{vu_meter}] {features.rms_db:>5.1f} dB",
        f"[bold green]Dominant Pitch:[/] {features.dominant_frequency_hz:>6.1f} Hz"
    )
    stats_table.add_row(
        f"[bold magenta]Centroid:[/] {features.spectral_centroid:>6.1f} Hz",
        f"[bold blue]Rolloff (85%):[/] {features.spectral_rolloff:>6.1f} Hz",
        f"[bold yellow]Spectral Flatness:[/] {features.spectral_flatness:>5.3f}"
    )
    stats_table.add_row(
        f"[bold red]Crest Factor:[/] {features.crest_factor:>5.2f}",
        f"[bold green]ZCR Rate:[/] {features.zero_crossing_rate:>5.3f}",
        f"[bold white]Engine Pace:[/] {fps:>4.1f} FPS"
    )

    # Classification Badge
    cat_color = "green"
    if "Transient" in classification.category.value:
        cat_color = "bold bright_red"
    elif "Tonal" in classification.category.value:
        cat_color = "bold bright_cyan"
    elif "Speech" in classification.category.value:
        cat_color = "bold bright_yellow"
    elif "Noise" in classification.category.value:
        cat_color = "bold magenta"
    elif "Silence" in classification.category.value:
        cat_color = "dim white"

    class_text = Text()
    class_text.append(f"Detected Event: {classification.category.value}\n", style=cat_color)
    class_text.append(f"Confidence: {classification.confidence * 100.0:.1f}%\n", style="bold white")
    class_text.append(f"Summary: {classification.summary}\n", style="italic")
    if classification.primary_traits:
        class_text.append(f"Key Traits: {', '.join(classification.primary_traits)}", style="dim cyan")

    # Anomaly Alert Box
    anomaly_color = "red" if anomaly_report.is_anomaly else "dim green"
    anomaly_text = Text()
    if anomaly_report.is_anomaly:
        anomaly_text.append(f"🚨 ANOMALY ALERT! Score: {anomaly_report.anomaly_score:.1f}/100\n", style="bold blink red")
        anomaly_text.append(f"Trigger: {anomaly_report.trigger_feature} (Z={anomaly_report.z_score:.2f})\n", style="bold red")
        anomaly_text.append(anomaly_report.message, style="bright_white")
    else:
        anomaly_text.append(f"Normal Acoustic State (Score: {anomaly_report.anomaly_score:.1f}/100)\n", style="green")
        anomaly_text.append("Background baseline stable.", style="dim")

    # Pattern Match Table
    match_table = Table(title="Registered Pattern Match Candidates", expand=True, show_header=True, header_style="bold magenta")
    match_table.add_column("Pattern Name", style="bold cyan")
    match_table.add_column("Confidence", justify="right")
    match_table.add_column("Status", justify="center")
    match_table.add_column("Method", justify="center")

    for m in matches[:4]:
        status_badge = "[bold green]MATCH[/]" if m.matched else "[dim]Below Thresh[/]"
        conf_bar = f"{m.confidence * 100.0:>5.1f}%"
        match_table.add_row(m.pattern_name, conf_bar, status_badge, m.method)

    if not matches:
        match_table.add_row("No templates registered", "-", "-", "-")

    # Combine into main layout grid
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=5),
        Layout(name="middle", size=8),
        Layout(name="bottom", size=8)
    )
    layout["header"].update(Panel(stats_table, title="[bold white]Audio Stream Telemetry[/]", border_style="blue"))
    layout["middle"].split_row(
        Layout(Panel(class_text, title="[bold white]Immediate Event Classification[/]", border_style=cat_color)),
        Layout(Panel(anomaly_text, title="[bold white]Acoustic Anomaly Sentinel[/]", border_style=anomaly_color))
    )
    layout["bottom"].update(Panel(match_table, title="[bold white]Instant Pattern Recognition[/]", border_style="magenta"))

    return Panel(layout, title="[bold bright_cyan]EchoPulse AI — Immediate Audio Pattern Engine[/]", border_style="bright_cyan")


def run_terminal_hud(mode: str = "simulated", audio_file: str | None = None):
    console = Console()
    config = AudioConfig()
    extractor = AudioFeatureExtractor(config)
    classifier = ImmediateAcousticClassifier(config)
    anomaly_detector = AcousticAnomalyDetector(config)
    matcher = AudioPatternMatcher(config)
    registry = PatternRegistry(config)
    synth = SyntheticAudioGenerator(config.sample_rate)

    # Pre-register standard patterns if empty
    if len(registry.patterns) == 0:
        registry.register_signal("Hand Clap", synth.generate_clap(0.35), "Sharp transient clap")
        registry.register_signal("Whistle Sweep", synth.generate_whistle_sweep(900, 2400, 0.8), "High-pitched whistle")
        registry.register_signal("Concert A (440Hz)", synth.generate_pure_tone(440, 0.8), "Reference pure tone")
        registry.register_signal("Pulsing Siren", synth.generate_alarm_pulse(1400, 4.0, 1.0), "Alarm tone")

    console.print("[bold green]Starting EchoPulse Terminal Engine...[/]")

    if mode == "live":
        try:
            stream = LiveMicrophoneStream(config)
            stream.start()
            console.print("[bold cyan]Microphone stream active. Listening...[/]")
            mode_label = "Live Microphone"
        except Exception as e:
            console.print(f"[bold red]Could not access microphone ({e}). Falling back to Simulated Stream.[/]")
            mode = "simulated"

    if mode == "simulated":
        if audio_file:
            stream_obj = SimulatedAudioStream(wav_path=audio_file, config=config, real_time_pace=True)
            mode_label = f"Simulated: {audio_file}"
        else:
            # Generate continuous composite sequence (tone -> clap -> whistle -> noise -> alarm)
            composite = np.concatenate([
                np.zeros(int(config.sample_rate * 0.4), dtype=np.float32),
                synth.generate_clap(0.35),
                np.zeros(int(config.sample_rate * 0.4), dtype=np.float32),
                synth.generate_whistle_sweep(900, 2400, 0.8),
                np.zeros(int(config.sample_rate * 0.3), dtype=np.float32),
                synth.generate_pure_tone(440, 0.8),
                np.zeros(int(config.sample_rate * 0.3), dtype=np.float32),
                synth.generate_alarm_pulse(1400, 4.0, 1.0),
                synth.generate_noise("pink", 0.8),
            ])
            stream_obj = SimulatedAudioStream(signal=composite, config=config, real_time_pace=True)
            mode_label = "Simulated Multi-Pattern Demo"

    t_prev = time.perf_counter()
    with Live(console=console, screen=True, refresh_per_second=15) as live:
        try:
            if mode == "live":
                while True:
                    t_now = time.perf_counter()
                    dt = max(1e-4, t_now - t_prev)
                    t_prev = t_now
                    fps = 1.0 / dt

                    chunk = stream.get_recent_audio(config.chunk_duration_sec)
                    if len(chunk) < config.frame_size:
                        time.sleep(0.05)
                        continue

                    features = extractor.extract(chunk)
                    classification = classifier.classify(features)
                    anomaly = anomaly_detector.process_frame(features)
                    matches = matcher.match_against_registry(features, registry)

                    hud = build_hud_panel(features, classification, anomaly, matches, mode_label, fps)
                    live.update(hud)
                    time.sleep(0.05)
            else:
                # Loop simulation continuously
                while True:
                    for chunk in stream_obj.iter_chunks():
                        t_now = time.perf_counter()
                        dt = max(1e-4, t_now - t_prev)
                        t_prev = t_now
                        fps = 1.0 / dt

                        features = extractor.extract(chunk)
                        classification = classifier.classify(features)
                        anomaly = anomaly_detector.process_frame(features)
                        matches = matcher.match_against_registry(features, registry)

                        hud = build_hud_panel(features, classification, anomaly, matches, mode_label, fps)
                        live.update(hud)
        except KeyboardInterrupt:
            pass
        finally:
            if mode == "live" and 'stream' in locals():
                stream.stop()


def main():
    parser = argparse.ArgumentParser(description="EchoPulse AI - Immediate Audio Pattern Recognition HUD")
    parser.add_argument("--mode", choices=["live", "simulated"], default="simulated", help="Audio source mode")
    parser.add_argument("--file", type=str, default=None, help="Path to WAV file for simulated streaming")
    args = parser.parse_args()
    run_terminal_hud(mode=args.mode, audio_file=args.file)


if __name__ == "__main__":
    main()
