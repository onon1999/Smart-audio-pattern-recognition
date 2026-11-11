"""
Streamlit Web Dashboard for EchoPulse AI - Immediate Audio Pattern Recognition.
"""
import io
import sys
import time
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from audio_pulse.config import AudioConfig
from audio_pulse.dsp.features import AudioFeatureExtractor
from audio_pulse.recognition.classifier import ImmediateAcousticClassifier
from audio_pulse.recognition.anomaly import AcousticAnomalyDetector
from audio_pulse.recognition.matcher import AudioPatternMatcher
from audio_pulse.recognition.registry import PatternRegistry
from audio_pulse.utils.audio_io import load_wav, save_wav
from audio_pulse.utils.synthetic_sounds import SyntheticAudioGenerator
from audio_pulse.utils.visualizer import AudioVisualizer

# Set Page Config
st.set_page_config(
    page_title="EchoPulse AI — Audio Pattern Recognition",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "config" not in st.session_state:
    st.session_state.config = AudioConfig()
if "registry" not in st.session_state:
    st.session_state.registry = PatternRegistry(st.session_state.config)
if "synth" not in st.session_state:
    st.session_state.synth = SyntheticAudioGenerator(st.session_state.config.sample_rate)
    # Ensure standard sample library is seeded
    st.session_state.synth.generate_all_samples(st.session_state.config.samples_dir)
if "extractor" not in st.session_state:
    st.session_state.extractor = AudioFeatureExtractor(st.session_state.config)
if "classifier" not in st.session_state:
    st.session_state.classifier = ImmediateAcousticClassifier(st.session_state.config)
if "matcher" not in st.session_state:
    st.session_state.matcher = AudioPatternMatcher(st.session_state.config)
if "anomaly_detector" not in st.session_state:
    st.session_state.anomaly_detector = AcousticAnomalyDetector(st.session_state.config)

# Seed initial patterns if empty
if len(st.session_state.registry.patterns) == 0:
    st.session_state.registry.register_signal(
        "Hand Clap",
        st.session_state.synth.generate_clap(0.35),
        "Sharp impulsive percussive transient"
    )
    st.session_state.registry.register_signal(
        "Whistle Sweep",
        st.session_state.synth.generate_whistle_sweep(900, 2400, 0.8),
        "Upward frequency chirp whistle"
    )
    st.session_state.registry.register_signal(
        "Concert A4 (440Hz)",
        st.session_state.synth.generate_pure_tone(440.0, 0.8),
        "Standard musical reference pitch"
    )
    st.session_state.registry.register_signal(
        "Alarm Siren",
        st.session_state.synth.generate_alarm_pulse(1400.0, 4.0, 1.2),
        "Pulsed warning alarm"
    )

# Sidebar Controls
st.sidebar.title("🎛️ EchoPulse Controls")
st.sidebar.markdown("**Immediate Audio Pattern Recognition & Spectral Analysis Engine**")

dsp_expander = st.sidebar.expander("⚙️ DSP Engine Parameters", expanded=False)
sample_rate = dsp_expander.selectbox("Sample Rate (Hz)", [16000, 22050, 44100], index=1)
frame_size = dsp_expander.selectbox("FFT Frame Size", [512, 1024, 2048], index=1)
n_mels = dsp_expander.slider("Mel Channels", 20, 80, 40)
anomaly_sigma = dsp_expander.slider("Anomaly Sensitivity (Sigma)", 1.5, 4.5, 2.8, 0.1)

st.session_state.config.sample_rate = sample_rate
st.session_state.config.frame_size = frame_size
st.session_state.config.n_mels = n_mels
st.session_state.config.anomaly_sigma_threshold = anomaly_sigma

st.sidebar.info(
    f"📚 **Registry**: {len(st.session_state.registry.patterns)} sound templates active.\n\n"
    f"⚡ **Latency**: Sub-10ms chunk DSP evaluation."
)

# Main Navigation
tab_stream, tab_file, tab_registry, tab_lab = st.tabs([
    "⚡ Immediate Stream Analysis",
    "📊 Audio File & Spectral Inspector",
    "🧬 Sound Pattern Library",
    "🧪 Synthetic Sound Lab"
])

# -------------------------------------------------------------
# TAB 1: Immediate Stream Analysis
# -------------------------------------------------------------
with tab_stream:
    st.header("⚡ Immediate Audio Stream Recognition")
    st.caption("Inspect live or simulated incoming audio chunks in real time.")

    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        stream_source = st.selectbox(
            "Select Audio Stream Source",
            ["Simulated Multi-Pattern Stream", "Hand Clap", "Whistle Chirp", "Alarm Siren", "Concert A (440 Hz)", "Pink Noise", "Live Microphone (if supported)"]
        )
    with col_ctrl2:
        run_stream = st.button("▶️ Run Analysis Frame", type="primary", use_container_width=True)

    # Prepare signal based on selection
    if "Live Microphone" in stream_source:
        st.warning("For live microphone streaming in web browsers, use the CLI terminal HUD or click Run below for local hardware capture.")
        try:
            from audio_pulse.stream.live_stream import LiveMicrophoneStream
            live_in = LiveMicrophoneStream(st.session_state.config)
            live_in.start()
            time.sleep(0.12)
            active_chunk = live_in.get_recent_audio(0.15)
            live_in.stop()
        except Exception as e:
            st.error(f"Microphone unavailable: {e}. Switching to synthetic sample.")
            active_chunk = st.session_state.synth.generate_clap(0.35)
    elif stream_source == "Hand Clap":
        active_chunk = st.session_state.synth.generate_clap(0.35)
    elif stream_source == "Whistle Chirp":
        active_chunk = st.session_state.synth.generate_whistle_sweep(900, 2400, 0.8)
    elif stream_source == "Alarm Siren":
        active_chunk = st.session_state.synth.generate_alarm_pulse(1400, 4.0, 1.2)
    elif stream_source == "Concert A (440 Hz)":
        active_chunk = st.session_state.synth.generate_pure_tone(440, 0.8)
    elif stream_source == "Pink Noise":
        active_chunk = st.session_state.synth.generate_noise("pink", 0.8)
    else:
        # Multi-pattern composite
        active_chunk = np.concatenate([
            st.session_state.synth.generate_clap(0.35),
            st.session_state.synth.generate_whistle_sweep(900, 2000, 0.5)
        ])

    # Run immediate feature extraction
    features = st.session_state.extractor.extract(active_chunk)
    classification = st.session_state.classifier.classify(features)
    anomaly = st.session_state.anomaly_detector.process_frame(features)
    matches = st.session_state.matcher.match_against_registry(features, st.session_state.registry)

    # Top KPI Metrics Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("RMS Level", f"{features.rms_db:.1f} dB")
    m2.metric("Dominant Pitch", f"{features.dominant_frequency_hz:.1f} Hz")
    m3.metric("Spectral Centroid", f"{features.spectral_centroid:.0f} Hz")
    m4.metric("Crest Factor", f"{features.crest_factor:.2f}")
    m5.metric("Zero Crossing Rate", f"{features.zero_crossing_rate:.3f}")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🎯 Immediate Event Detection")
        cat_badge = "🟢"
        if "Transient" in classification.category.value:
            cat_badge = "⚡"
        elif "Tonal" in classification.category.value:
            cat_badge = "🎵"
        elif "Noise" in classification.category.value:
            cat_badge = "💨"
        
        st.markdown(f"### {cat_badge} `{classification.category.value}`")
        st.progress(float(classification.confidence), text=f"Confidence: {classification.confidence * 100:.1f}%")
        st.write(f"**Diagnosis:** {classification.summary}")
        if classification.primary_traits:
            st.write(f"**Identified Traits:** {', '.join(classification.primary_traits)}")

        st.divider()
        st.subheader("🚨 Acoustic Anomaly Sentinel")
        if anomaly.is_anomaly:
            st.error(f"⚠️ **ANOMALY TRIGGERED!** Score: {anomaly.anomaly_score:.1f}/100\n\n{anomaly.message}")
        else:
            st.success(f"✅ **Normal Acoustic Baseline** (Anomaly Index: {anomaly.anomaly_score:.1f}/100)\n\n{anomaly.message}")

    with col_right:
        st.subheader("🔍 Instant Pattern Recognition Matches")
        if matches:
            for m in matches[:4]:
                color = "green" if m.matched else "gray"
                status_icon = "🎯 MATCH" if m.matched else "⚪ Candidate"
                st.markdown(
                    f"<div style='padding: 10px; margin-bottom: 8px; border-radius: 8px; background-color: rgba(255,255,255,0.05); border-left: 4px solid {color};'>"
                    f"<strong>{m.pattern_name}</strong> — <code>{status_icon}</code><br/>"
                    f"Confidence: <strong>{m.confidence * 100:.1f}%</strong> | Distance: <code>{m.distance:.4f}</code> | Method: <code>{m.method}</code>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("No registered patterns to compare against. Add some in the Pattern Library tab!")

    # Waveform & Spectrogram preview of chunk
    st.subheader("📈 Signal Waveform & Instant Spectrum")
    fig, (ax_w, ax_s) = plt.subplots(1, 2, figsize=(12, 3), constrained_layout=True)
    t = np.linspace(0, features.duration_sec, len(active_chunk))
    ax_w.plot(t, active_chunk, color="#3B82F6")
    ax_w.set_title("Audio Chunk Waveform")
    ax_w.set_xlabel("Time (s)")
    ax_w.set_ylabel("Amplitude")
    ax_w.grid(True, alpha=0.3)

    if features.mel_spectrogram.size > 0:
        im = ax_s.imshow(features.mel_spectrogram, aspect="auto", origin="lower", cmap="magma")
        ax_s.set_title("Mel-Spectrogram Bands")
        ax_s.set_xlabel("Time Frames")
        ax_s.set_ylabel("Mel Band")
        fig.colorbar(im, ax=ax_s)
    st.pyplot(fig)
    plt.close(fig)


# -------------------------------------------------------------
# TAB 2: Audio File & Spectral Inspector
# -------------------------------------------------------------
with tab_file:
    st.header("📊 Deep Audio File & Spectral Inspector")
    st.caption("Upload any WAV audio file or choose from the sample gallery for complete diagnostic analysis.")

    file_col1, file_col2 = st.columns([1, 1])
    with file_col1:
        uploaded_file = st.file_uploader("Upload WAV File", type=["wav"])
    with file_col2:
        sample_choice = st.selectbox(
            "Or pick from Pre-Generated Test Library",
            ["hand_clap.wav", "whistle_chirp.wav", "pure_tone_440hz.wav", "c_major_chord.wav", "alarm_siren.wav", "ambient_noise.wav", "speech_vocalization.wav"]
        )

    inspect_sig = None
    inspect_name = ""

    if uploaded_file is not None:
        inspect_sig, _ = load_wav(io.BytesIO(uploaded_file.read()), target_sr=st.session_state.config.sample_rate)
        inspect_name = uploaded_file.name
    elif sample_choice:
        sample_path = st.session_state.config.samples_dir / sample_choice
        if sample_path.exists():
            inspect_sig, _ = load_wav(sample_path, target_sr=st.session_state.config.sample_rate)
            inspect_name = sample_choice

    if inspect_sig is not None and len(inspect_sig) > 0:
        # Audio Player
        temp_wav_buf = io.BytesIO()
        save_wav(temp_wav_buf, inspect_sig, st.session_state.config.sample_rate)
        st.audio(temp_wav_buf.getvalue(), format="audio/wav")

        full_features = st.session_state.extractor.extract(inspect_sig)
        full_classification = st.session_state.classifier.classify(full_features)
        full_matches = st.session_state.matcher.match_against_registry(full_features, st.session_state.registry)

        # Overview row
        col_c1, col_c2, col_c3 = st.columns(3)
        col_c1.info(f"**Event Classification:**\n### {full_classification.category.value}\nConfidence: {full_classification.confidence*100:.1f}%")
        col_c2.info(f"**Acoustic Profile:**\n- Duration: `{full_features.duration_sec:.2f} s`\n- RMS: `{full_features.rms_db:.1f} dB`\n- Pitch: `{full_features.dominant_frequency_hz:.1f} Hz`")
        top_match_name = full_matches[0].pattern_name if (full_matches and full_matches[0].matched) else "No exact template match"
        top_match_score = f"{full_matches[0].confidence * 100:.1f}%" if full_matches else "N/A"
        col_c3.info(f"**Top Matched Pattern:**\n### {top_match_name}\nScore: {top_match_score}")

        # 4-Panel Diagnostic Matplotlib Plot
        st.subheader("🔬 4-Panel Spectral Decomposition")
        diag_fig = AudioVisualizer.create_analysis_figure(
            inspect_sig,
            st.session_state.config.sample_rate,
            full_features,
            title=f"Spectral Diagnostics — {inspect_name}"
        )
        st.pyplot(diag_fig)
        plt.close(diag_fig)

        # Feature Data Table
        st.subheader("📋 Complete Audio Feature Descriptors")
        st.json(full_features.to_dict())


# -------------------------------------------------------------
# TAB 3: Sound Pattern Library
# -------------------------------------------------------------
with tab_registry:
    st.header("🧬 Acoustic Pattern Template Library")
    st.caption("Register reference sounds to build a customized immediate recognition model.")

    reg_col1, reg_col2 = st.columns([1, 1])

    with reg_col1:
        st.subheader("➕ Register New Acoustic Template")
        new_name = st.text_input("Pattern Name", placeholder="e.g. Door Knock, Bell Chime, Cough")
        new_desc = st.text_area("Description / Tags", placeholder="Short description of sound pattern")
        new_source = st.radio("Template Source", ["Upload Audio File", "Use Synthetic Sound Generator"], horizontal=True)

        new_signal = None
        if new_source == "Upload Audio File":
            new_file = st.file_uploader("Upload Reference WAV", type=["wav"], key="new_pat_uploader")
            if new_file:
                new_signal, _ = load_wav(io.BytesIO(new_file.read()), target_sr=st.session_state.config.sample_rate)
        else:
            synth_type = st.selectbox("Select Synthetic Sound", ["Hand Clap", "Whistle Sweep", "440Hz Pure Tone", "Harmonic Chord", "Alarm Pulse"])
            if synth_type == "Hand Clap":
                new_signal = st.session_state.synth.generate_clap(0.35)
            elif synth_type == "Whistle Sweep":
                new_signal = st.session_state.synth.generate_whistle_sweep(900, 2400, 0.8)
            elif synth_type == "440Hz Pure Tone":
                new_signal = st.session_state.synth.generate_pure_tone(440, 0.8)
            elif synth_type == "Harmonic Chord":
                new_signal = st.session_state.synth.generate_harmonic_chord()
            else:
                new_signal = st.session_state.synth.generate_alarm_pulse(1400, 4.0, 1.0)

        if st.button("Save Pattern to Registry", type="primary"):
            if new_name and new_signal is not None and len(new_signal) > 0:
                pat = st.session_state.registry.register_signal(new_name, new_signal, new_desc)
                st.success(f"Pattern '{new_name}' successfully fingerprinted and registered!")
                st.rerun()
            else:
                st.error("Please provide a pattern name and valid audio signal.")

    with reg_col2:
        st.subheader("📚 Registered Patterns")
        patterns_list = st.session_state.registry.list_patterns()
        if patterns_list:
            for p in patterns_list:
                with st.expander(f"🎵 {p['name']} ({p['duration_sec']:.2f}s)"):
                    st.write(f"**Description:** {p['description']}")
                    st.write(f"**Dominant Pitch:** `{p['dominant_freq_hz']:.1f} Hz` | **Energy:** `{p['rms_db']:.1f} dB`")
                    if st.button(f"🗑️ Delete {p['name']}", key=f"del_{p['name']}"):
                        st.session_state.registry.remove_pattern(p['name'])
                        st.success(f"Removed {p['name']}")
                        st.rerun()
        else:
            st.info("No patterns currently registered.")


# -------------------------------------------------------------
# TAB 4: Synthetic Sound Lab
# -------------------------------------------------------------
with tab_lab:
    st.header("🧪 Synthetic Acoustic Pattern Lab")
    st.caption("Generate synthetic test signals on demand to benchmark DSP and matching accuracy.")

    lab_col1, lab_col2 = st.columns([1, 1])

    with lab_col1:
        generator_choice = st.selectbox(
            "Select Generator Algorithm",
            ["Hand Clap / Transient", "Whistle Chirp (Frequency Sweep)", "Pure Sine Tone", "Harmonic Musical Chord", "Pulsed Siren Alarm", "Speech-like Vocalization", "Pink Noise"]
        )

        gen_sig = None
        if generator_choice == "Hand Clap / Transient":
            decay_dur = st.slider("Clap Duration (s)", 0.1, 1.0, 0.35, 0.05)
            gen_sig = st.session_state.synth.generate_clap(duration=decay_dur)
        elif generator_choice == "Whistle Chirp (Frequency Sweep)":
            f_start = st.slider("Start Frequency (Hz)", 200, 3000, 800, 50)
            f_end = st.slider("End Frequency (Hz)", 400, 5000, 2400, 50)
            dur_sweep = st.slider("Sweep Duration (s)", 0.2, 2.0, 0.8, 0.1)
            gen_sig = st.session_state.synth.generate_whistle_sweep(f_start, f_end, dur_sweep)
        elif generator_choice == "Pure Sine Tone":
            tone_f = st.slider("Tone Frequency (Hz)", 100, 4000, 440, 10)
            tone_dur = st.slider("Duration (s)", 0.2, 3.0, 0.8, 0.1)
            gen_sig = st.session_state.synth.generate_pure_tone(tone_f, tone_dur)
        elif generator_choice == "Harmonic Musical Chord":
            chord_dur = st.slider("Chord Duration (s)", 0.3, 3.0, 1.0, 0.1)
            gen_sig = st.session_state.synth.generate_harmonic_chord(duration=chord_dur)
        elif generator_choice == "Pulsed Siren Alarm":
            siren_f = st.slider("Carrier Frequency (Hz)", 500, 3000, 1400, 50)
            pulse_rate = st.slider("Pulse Rate (Hz)", 1.0, 10.0, 4.0, 0.5)
            gen_sig = st.session_state.synth.generate_alarm_pulse(siren_f, pulse_rate, 1.2)
        elif generator_choice == "Speech-like Vocalization":
            f0 = st.slider("Fundamental Pitch F0 (Hz)", 80, 300, 130, 5)
            gen_sig = st.session_state.synth.generate_speech_like(f0, 1.0)
        else:
            noise_dur = st.slider("Noise Duration (s)", 0.2, 3.0, 1.0, 0.1)
            gen_sig = st.session_state.synth.generate_noise("pink", noise_dur)

    with lab_col2:
        if gen_sig is not None:
            st.subheader("🔊 Audio Preview")
            buf = io.BytesIO()
            save_wav(buf, gen_sig, st.session_state.config.sample_rate)
            st.audio(buf.getvalue(), format="audio/wav")

            st.download_button(
                label="⬇️ Download WAV File",
                data=buf.getvalue(),
                file_name=f"{generator_choice.lower().replace(' ', '_')}.wav",
                mime="audio/wav"
            )

            # Quick waveform preview
            fig, ax = plt.subplots(figsize=(6, 2.5), constrained_layout=True)
            ax.plot(gen_sig[:min(len(gen_sig), 4000)], color="#10B981")
            ax.set_title("Initial Waveform Preview (First 4000 samples)")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)
            plt.close(fig)
