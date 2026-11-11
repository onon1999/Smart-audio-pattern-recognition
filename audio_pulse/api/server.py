"""
FastAPI REST and WebSocket streaming server for immediate audio pattern analysis.
"""
import io
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np

from ..config import AudioConfig
from ..dsp.features import AudioFeatureExtractor
from ..recognition.classifier import ImmediateAcousticClassifier
from ..recognition.anomaly import AcousticAnomalyDetector
from ..recognition.matcher import AudioPatternMatcher
from ..recognition.registry import PatternRegistry
from ..utils.audio_io import load_wav
from ..utils.synthetic_sounds import SyntheticAudioGenerator

app = FastAPI(
    title="EchoPulse AI - Immediate Audio Pattern Recognition API",
    description="Real-time acoustic analysis, event detection, DTW pattern matching, and anomaly detection.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared singletons
config = AudioConfig()
extractor = AudioFeatureExtractor(config)
classifier = ImmediateAcousticClassifier(config)
anomaly_detector = AcousticAnomalyDetector(config)
matcher = AudioPatternMatcher(config)
registry = PatternRegistry(config)
synth = SyntheticAudioGenerator(config.sample_rate)

# Ensure sample files exist on startup
synth.generate_all_samples(config.samples_dir)


class ChunkAnalysisRequest(BaseModel):
    samples: list[float]
    sample_rate: int = 22050


@app.get("/api")
def api_info():
    return {
        "status": "online",
        "service": "EchoPulse AI - Immediate Audio Pattern Recognition",
        "version": "1.0.0",
        "registered_patterns": len(registry.patterns),
        "endpoints": {
            "dashboard": "GET /",
            "docs": "/docs",
            "analyze_file": "POST /analyze/file",
            "analyze_chunk": "POST /analyze/chunk",
            "list_patterns": "GET /patterns",
            "register_pattern": "POST /patterns/register",
            "websocket_stream": "WS /ws/stream"
        }
    }


@app.get("/", response_class=HTMLResponse)
def dashboard_html():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EchoPulse AI — Immediate Audio Pattern Recognition</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { background-color: #0b0f19; color: #e2e8f0; font-family: ui-sans-serif, system-ui, sans-serif; }
    .glass { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }
    .glow-blue { box-shadow: 0 0 20px rgba(59, 130, 246, 0.25); }
  </style>
</head>
<body class="min-h-screen p-4 md:p-8">
  <div class="max-w-7xl mx-auto space-y-6">
    <!-- Header -->
    <header class="glass p-5 rounded-2xl flex flex-wrap items-center justify-between gap-4">
      <div class="flex items-center gap-3">
        <span class="text-3xl">🔊</span>
        <div>
          <h1 class="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">EchoPulse AI</h1>
          <p class="text-xs text-slate-400">Immediate Audio Pattern Recognition & Spectral Analysis Engine (&lt;10ms DSP)</p>
        </div>
      </div>
      <div class="flex items-center gap-3">
        <span id="conn-pill" class="px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">● Connected</span>
        <a href="/docs" target="_blank" class="px-3 py-1.5 rounded-xl bg-slate-800 text-xs text-cyan-400 hover:bg-slate-700 transition">Swagger API</a>
        <a href="https://github.com/onon1999/Smart-audio-pattern-recognition" target="_blank" class="px-3 py-1.5 rounded-xl bg-blue-600/30 text-xs text-blue-300 hover:bg-blue-600/50 transition">GitHub Repo</a>
      </div>
    </header>

    <!-- Main Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      
      <!-- Col 1: Audio Ingestion & Oscilloscope -->
      <div class="glass p-5 rounded-2xl space-y-5">
        <h2 class="text-lg font-semibold flex items-center gap-2"><span>🎙️</span> Ingestion & Oscilloscope</h2>
        
        <div class="flex gap-2">
          <button id="btn-mic" onclick="toggleMic()" class="flex-1 py-3 px-4 rounded-xl font-bold text-sm bg-blue-600 hover:bg-blue-500 text-white transition flex items-center justify-center gap-2 shadow-lg">
            <span>🎤</span> Start Live Mic
          </button>
          <button id="btn-synth-stream" onclick="toggleSynthStream()" class="flex-1 py-3 px-4 rounded-xl font-bold text-sm bg-indigo-600 hover:bg-indigo-500 text-white transition flex items-center justify-center gap-2">
            <span>⚡</span> Test Stream
          </button>
        </div>

        <div>
          <div class="flex justify-between text-xs text-slate-400 mb-1">
            <span>Waveform Oscilloscope</span>
            <span id="vol-db" class="font-mono text-cyan-400">-50.0 dB</span>
          </div>
          <canvas id="canvas-wave" width="380" height="110" class="w-full bg-slate-900/90 rounded-xl border border-slate-800"></canvas>
        </div>

        <div>
          <div class="flex justify-between text-xs text-slate-400 mb-1">
            <span>Real-Time FFT Spectrum</span>
            <span id="dom-pitch" class="font-mono text-emerald-400">0 Hz</span>
          </div>
          <canvas id="canvas-fft" width="380" height="90" class="w-full bg-slate-900/90 rounded-xl border border-slate-800"></canvas>
        </div>

        <!-- File Upload Area -->
        <div class="border-t border-slate-800 pt-4 space-y-2">
          <label class="block text-xs font-semibold text-slate-300">Analyze Audio File (WAV)</label>
          <input type="file" id="file-input" accept=".wav" onchange="uploadAudioFile()" class="block w-full text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-slate-800 file:text-cyan-400 hover:file:bg-slate-700 cursor-pointer"/>
          <p id="file-status" class="text-xs text-slate-500"></p>
        </div>
      </div>

      <!-- Col 2: Event Detection & Anomaly Sentinel -->
      <div class="glass p-5 rounded-2xl space-y-5">
        <h2 class="text-lg font-semibold flex items-center gap-2"><span>🎯</span> Immediate Event Detection</h2>
        
        <!-- Detected Event Card -->
        <div id="event-card" class="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-2">
          <div class="text-xs text-slate-400 font-medium">Identified Sound Category</div>
          <div id="cat-badge" class="text-xl font-extrabold text-cyan-400">Silence / Ambient</div>
          <div class="w-full bg-slate-800 rounded-full h-2">
            <div id="conf-bar" class="bg-cyan-500 h-2 rounded-full transition-all duration-150" style="width: 20%"></div>
          </div>
          <div class="flex justify-between text-xs text-slate-400">
            <span>Confidence: <strong id="conf-val" class="text-white">20%</strong></span>
            <span id="event-fps" class="text-slate-500 font-mono">15 FPS</span>
          </div>
        </div>

        <!-- Telemetry Metrics Grid -->
        <div class="grid grid-cols-2 gap-3">
          <div class="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80">
            <div class="text-xs text-slate-400">Spectral Centroid</div>
            <div id="stat-centroid" class="text-lg font-mono font-bold text-blue-400">0 Hz</div>
          </div>
          <div class="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80">
            <div class="text-xs text-slate-400">Spectral Rolloff</div>
            <div id="stat-rolloff" class="text-lg font-mono font-bold text-indigo-400">0 Hz</div>
          </div>
          <div class="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80">
            <div class="text-xs text-slate-400">Crest Factor</div>
            <div id="stat-crest" class="text-lg font-mono font-bold text-emerald-400">1.00</div>
          </div>
          <div class="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80">
            <div class="text-xs text-slate-400">Zero-Crossing Rate</div>
            <div id="stat-zcr" class="text-lg font-mono font-bold text-amber-400">0.00</div>
          </div>
        </div>

        <!-- Anomaly Alert Card -->
        <div id="anomaly-box" class="p-4 rounded-xl bg-emerald-950/20 border border-emerald-500/30 space-y-1">
          <div class="flex justify-between items-center">
            <span class="text-xs font-bold text-emerald-400" id="anomaly-title">✅ Acoustic Baseline Normal</span>
            <span class="text-xs font-mono text-emerald-300" id="anomaly-score">Score: 0/100</span>
          </div>
          <p class="text-xs text-slate-400" id="anomaly-desc">Environment background level is stable.</p>
        </div>
      </div>

      <!-- Col 3: Pattern Library & Quick Sound Lab -->
      <div class="glass p-5 rounded-2xl space-y-5">
        <h2 class="text-lg font-semibold flex items-center gap-2"><span>🧬</span> Pattern Match Candidates</h2>
        
        <div id="match-list" class="space-y-2">
          <div class="p-3 rounded-xl bg-slate-900/80 border border-slate-800 flex justify-between items-center">
            <div>
              <div class="font-bold text-sm text-cyan-300">Hand Clap</div>
              <div class="text-xs text-slate-500">Method: Hybrid DTW + Cosine</div>
            </div>
            <span class="px-2.5 py-1 rounded-lg text-xs font-semibold bg-slate-800 text-slate-400">Waiting</span>
          </div>
          <div class="p-3 rounded-xl bg-slate-900/80 border border-slate-800 flex justify-between items-center">
            <div>
              <div class="font-bold text-sm text-cyan-300">Whistle Sweep</div>
              <div class="text-xs text-slate-500">Method: Hybrid DTW + Cosine</div>
            </div>
            <span class="px-2.5 py-1 rounded-lg text-xs font-semibold bg-slate-800 text-slate-400">Waiting</span>
          </div>
          <div class="p-3 rounded-xl bg-slate-900/80 border border-slate-800 flex justify-between items-center">
            <div>
              <div class="font-bold text-sm text-cyan-300">Concert A (440Hz)</div>
              <div class="text-xs text-slate-500">Method: Hybrid DTW + Cosine</div>
            </div>
            <span class="px-2.5 py-1 rounded-lg text-xs font-semibold bg-slate-800 text-slate-400">Waiting</span>
          </div>
          <div class="p-3 rounded-xl bg-slate-900/80 border border-slate-800 flex justify-between items-center">
            <div>
              <div class="font-bold text-sm text-cyan-300">Alarm Siren</div>
              <div class="text-xs text-slate-500">Method: Hybrid DTW + Cosine</div>
            </div>
            <span class="px-2.5 py-1 rounded-lg text-xs font-semibold bg-slate-800 text-slate-400">Waiting</span>
          </div>
        </div>

        <div class="border-t border-slate-800 pt-4 space-y-2">
          <label class="block text-xs font-semibold text-slate-300">⚡ Test Sound Triggers (Immediate Recognition)</label>
          <div class="grid grid-cols-3 gap-2">
            <button onclick="triggerSound('clap')" class="p-2 rounded-xl bg-slate-800/90 hover:bg-slate-700 text-xs font-semibold text-cyan-300 transition border border-slate-700">👏 Clap</button>
            <button onclick="triggerSound('whistle')" class="p-2 rounded-xl bg-slate-800/90 hover:bg-slate-700 text-xs font-semibold text-indigo-300 transition border border-slate-700">😗 Whistle</button>
            <button onclick="triggerSound('tone')" class="p-2 rounded-xl bg-slate-800/90 hover:bg-slate-700 text-xs font-semibold text-emerald-300 transition border border-slate-700">🔔 440Hz</button>
            <button onclick="triggerSound('alarm')" class="p-2 rounded-xl bg-slate-800/90 hover:bg-slate-700 text-xs font-semibold text-red-300 transition border border-slate-700">🚨 Alarm</button>
            <button onclick="triggerSound('chord')" class="p-2 rounded-xl bg-slate-800/90 hover:bg-slate-700 text-xs font-semibold text-purple-300 transition border border-slate-700">🎹 Chord</button>
            <button onclick="triggerSound('noise')" class="p-2 rounded-xl bg-slate-800/90 hover:bg-slate-700 text-xs font-semibold text-amber-300 transition border border-slate-700">💨 Noise</button>
          </div>
        </div>
      </div>

    </div>
  </div>

  <script>
    let ws = null;
    let audioCtx = null;
    let micStream = null;
    let processorNode = null;
    let isStreamingMic = false;
    let synthInterval = null;

    // Connect WebSocket
    function connectWS() {
      const loc = window.location;
      const wsUrl = (loc.protocol === 'https:' ? 'wss://' : 'ws://') + loc.host + '/ws/stream';
      ws = new WebSocket(wsUrl);
      ws.onopen = () => {
        document.getElementById('conn-pill').textContent = '● Connected';
        document.getElementById('conn-pill').className = 'px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
      };
      ws.onclose = () => {
        document.getElementById('conn-pill').textContent = '○ Reconnecting...';
        document.getElementById('conn-pill').className = 'px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30';
        setTimeout(connectWS, 2000);
      };
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateDashboard(data);
      };
    }
    connectWS();

    function updateDashboard(data) {
      // Event category & confidence
      document.getElementById('cat-badge').textContent = data.category || 'Silence / Ambient';
      const conf = Math.round((data.confidence || 0) * 100);
      document.getElementById('conf-val').textContent = conf + '%';
      document.getElementById('conf-bar').style.width = conf + '%';

      // Telemetry
      document.getElementById('vol-db').textContent = (data.rms_db ? data.rms_db.toFixed(1) : '-50.0') + ' dB';
      document.getElementById('dom-pitch').textContent = Math.round(data.dominant_freq_hz || 0) + ' Hz';
      document.getElementById('stat-centroid').textContent = Math.round(data.spectral_centroid || 0) + ' Hz';

      // Anomaly Sentinel
      const anomalyBox = document.getElementById('anomaly-box');
      const anomalyTitle = document.getElementById('anomaly-title');
      const anomalyScore = document.getElementById('anomaly-score');
      const anomalyDesc = document.getElementById('anomaly-desc');

      if (data.is_anomaly) {
        anomalyBox.className = 'p-4 rounded-xl bg-red-950/40 border border-red-500/60 space-y-1 animate-pulse';
        anomalyTitle.textContent = '🚨 ACOUSTIC ANOMALY DETECTED!';
        anomalyTitle.className = 'text-xs font-bold text-red-400';
        anomalyScore.textContent = 'Score: ' + Math.round(data.anomaly_score) + '/100';
        anomalyDesc.textContent = 'Sudden acoustic deviation or energy burst detected in audio stream.';
      } else {
        anomalyBox.className = 'p-4 rounded-xl bg-emerald-950/20 border border-emerald-500/30 space-y-1';
        anomalyTitle.textContent = '✅ Acoustic Baseline Normal';
        anomalyTitle.className = 'text-xs font-bold text-emerald-400';
        anomalyScore.textContent = 'Score: ' + Math.round(data.anomaly_score || 0) + '/100';
        anomalyDesc.textContent = 'Environment background level is stable.';
      }

      // Pattern match highlight
      if (data.top_pattern) {
        const topName = data.top_pattern;
        const topConf = Math.round((data.top_pattern_confidence || 0) * 100);
        document.getElementById('match-list').innerHTML = `
          <div class="p-3 rounded-xl bg-emerald-950/30 border border-emerald-500/50 flex justify-between items-center glow-blue">
            <div>
              <div class="font-bold text-sm text-emerald-300">🎯 ${topName}</div>
              <div class="text-xs text-emerald-400">Matched with ${topConf}% confidence</div>
            </div>
            <span class="px-2.5 py-1 rounded-lg text-xs font-semibold bg-emerald-600 text-white">MATCH</span>
          </div>
          <div class="text-xs text-slate-500 text-center">Live matching updated continuously.</div>
        `;
      }
    }

    // Canvas drawing
    const canvasWave = document.getElementById('canvas-wave');
    const ctxWave = canvasWave.getContext('2d');
    const canvasFFT = document.getElementById('canvas-fft');
    const ctxFFT = canvasFFT.getContext('2d');

    function drawWaveform(samples) {
      ctxWave.clearRect(0, 0, canvasWave.width, canvasWave.height);
      ctxWave.beginPath();
      ctxWave.strokeStyle = '#38bdf8';
      ctxWave.lineWidth = 1.5;
      const step = Math.ceil(samples.length / canvasWave.width);
      const amp = canvasWave.height / 2;
      for (let i = 0; i < canvasWave.width; i++) {
        const sampleVal = samples[i * step] || 0;
        const y = amp - sampleVal * (amp * 0.9);
        if (i === 0) ctxWave.moveTo(i, y);
        else ctxWave.lineTo(i, y);
      }
      ctxWave.stroke();
    }

    function drawFFT(samples) {
      ctxFFT.clearRect(0, 0, canvasFFT.width, canvasFFT.height);
      const numBars = 32;
      const barW = (canvasFFT.width / numBars) - 2;
      const step = Math.floor(samples.length / numBars);
      for (let i = 0; i < numBars; i++) {
        let avg = 0;
        for (let j = 0; j < step; j++) avg += Math.abs(samples[i * step + j] || 0);
        avg = (avg / step) * 3.0;
        const barH = Math.min(canvasFFT.height, avg * canvasFFT.height);
        ctxFFT.fillStyle = '#818cf8';
        ctxFFT.fillRect(i * (barW + 2), canvasFFT.height - barH, barW, barH);
      }
    }

    // Live Microphone capture
    async function toggleMic() {
      const btn = document.getElementById('btn-mic');
      if (isStreamingMic) {
        if (micStream) micStream.getTracks().forEach(t => t.stop());
        if (processorNode) processorNode.disconnect();
        isStreamingMic = false;
        btn.innerHTML = '<span>🎤</span> Start Live Mic';
        btn.className = 'flex-1 py-3 px-4 rounded-xl font-bold text-sm bg-blue-600 hover:bg-blue-500 text-white transition flex items-center justify-center gap-2';
        return;
      }

      try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 22050 });
        const source = audioCtx.createMediaStreamSource(micStream);
        processorNode = audioCtx.createScriptProcessor(2048, 1, 1);

        processorNode.onaudioprocess = (e) => {
          const input = e.inputBuffer.getChannelData(0);
          drawWaveform(input);
          drawFFT(input);
          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(input.buffer);
          }
        };

        source.connect(processorNode);
        processorNode.connect(audioCtx.destination);
        isStreamingMic = true;
        btn.innerHTML = '<span>⏹️</span> Stop Mic';
        btn.className = 'flex-1 py-3 px-4 rounded-xl font-bold text-sm bg-red-600 hover:bg-red-500 text-white transition flex items-center justify-center gap-2';
      } catch (err) {
        alert('Could not access microphone: ' + err.message);
      }
    }

    // Synthetic Sound Trigger
    function triggerSound(type) {
      if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 22050 });
      const sr = 22050;
      let samples;
      if (type === 'clap') {
        const len = Math.floor(sr * 0.35);
        samples = new Float32Array(len);
        for (let i = 0; i < len; i++) {
          const t = i / sr;
          samples[i] = (Math.random() * 2 - 1) * Math.exp(-t * 28.0) * 0.85;
        }
      } else if (type === 'whistle') {
        const len = Math.floor(sr * 0.8);
        samples = new Float32Array(len);
        for (let i = 0; i < len; i++) {
          const t = i / sr;
          const f = 900 + (2400 - 900) * (t / 0.8);
          samples[i] = Math.sin(2 * Math.PI * f * t) * Math.sin(Math.PI * t / 0.8) * 0.8;
        }
      } else if (type === 'tone') {
        const len = Math.floor(sr * 0.8);
        samples = new Float32Array(len);
        for (let i = 0; i < len; i++) {
          const t = i / sr;
          samples[i] = Math.sin(2 * Math.PI * 440 * t) * 0.75;
        }
      } else if (type === 'alarm') {
        const len = Math.floor(sr * 1.0);
        samples = new Float32Array(len);
        for (let i = 0; i < len; i++) {
          const t = i / sr;
          const pulse = (Math.sin(2 * Math.PI * 4 * t) > 0) ? 1.0 : 0.0;
          samples[i] = Math.sin(2 * Math.PI * 1400 * t) * pulse * 0.8;
        }
      } else if (type === 'chord') {
        const len = Math.floor(sr * 1.0);
        samples = new Float32Array(len);
        for (let i = 0; i < len; i++) {
          const t = i / sr;
          samples[i] = (Math.sin(2 * Math.PI * 261.63 * t) + Math.sin(2 * Math.PI * 329.63 * t) + Math.sin(2 * Math.PI * 392.00 * t)) / 3.0 * 0.8;
        }
      } else {
        const len = Math.floor(sr * 0.8);
        samples = new Float32Array(len);
        for (let i = 0; i < len; i++) samples[i] = (Math.random() * 2 - 1) * 0.4;
      }

      drawWaveform(samples);
      drawFFT(samples);

      // Play through speakers
      const buf = audioCtx.createBuffer(1, samples.length, sr);
      buf.copyToChannel(samples, 0);
      const src = audioCtx.createBufferSource();
      src.buffer = buf;
      src.connect(audioCtx.destination);
      src.start();

      // Send to server in chunks
      const chunkSz = 2048;
      for (let offset = 0; offset < samples.length; offset += chunkSz) {
        const sub = samples.slice(offset, offset + chunkSz);
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(sub.buffer);
        }
      }
    }

    // Simulated stream loop
    function toggleSynthStream() {
      const btn = document.getElementById('btn-synth-stream');
      if (synthInterval) {
        clearInterval(synthInterval);
        synthInterval = null;
        btn.innerHTML = '<span>⚡</span> Test Stream';
        btn.className = 'flex-1 py-3 px-4 rounded-xl font-bold text-sm bg-indigo-600 hover:bg-indigo-500 text-white transition flex items-center justify-center gap-2';
        return;
      }

      btn.innerHTML = '<span>⏹️</span> Stop Stream';
      btn.className = 'flex-1 py-3 px-4 rounded-xl font-bold text-sm bg-red-600 hover:bg-red-500 text-white transition flex items-center justify-center gap-2';
      const seq = ['clap', 'whistle', 'tone', 'alarm', 'chord', 'noise'];
      let idx = 0;
      triggerSound(seq[idx]);
      synthInterval = setInterval(() => {
        idx = (idx + 1) % seq.length;
        triggerSound(seq[idx]);
      }, 1200);
    }

    // Upload audio file
    async function uploadAudioFile() {
      const fileInput = document.getElementById('file-input');
      const status = document.getElementById('file-status');
      if (!fileInput.files || fileInput.files.length === 0) return;

      const file = fileInput.files[0];
      status.textContent = 'Analyzing ' + file.name + '...';

      const formData = new FormData();
      formData.append('file', file);

      try {
        const res = await fetch('/analyze/file', { method: 'POST', body: formData });
        const json = await res.json();
        status.textContent = '✓ Analyzed ' + file.name + ' -> ' + json.classification.category;
        updateDashboard({
          category: json.classification.category,
          confidence: json.classification.confidence,
          rms_db: json.features.rms_db,
          dominant_freq_hz: json.features.dominant_frequency_hz,
          spectral_centroid: json.features.spectral_centroid_hz,
          is_anomaly: json.anomaly.is_anomaly,
          anomaly_score: json.anomaly.anomaly_score,
          top_pattern: json.matched_patterns.length > 0 ? json.matched_patterns[0].pattern_name : null,
          top_pattern_confidence: json.matched_patterns.length > 0 ? json.matched_patterns[0].confidence : 0,
        });
      } catch (err) {
        status.textContent = 'Error: ' + err.message;
      }
    }
  </script>
</body>
</html>"""


@app.post("/analyze/file")
async def analyze_file(file: UploadFile = File(...)):
    """Uploads an audio file and performs complete spectral pattern analysis and matching."""
    contents = await file.read()
    temp_buf = io.BytesIO(contents)
    signal, sr = load_wav(temp_buf, target_sr=config.sample_rate)

    features = extractor.extract(signal)
    classification = classifier.classify(features)
    matches = matcher.match_against_registry(features, registry)
    anomaly = anomaly_detector.process_frame(features)

    return {
        "filename": file.filename,
        "sample_rate": sr,
        "classification": classification.to_dict(),
        "features": features.to_dict(),
        "matched_patterns": [m.to_dict() for m in matches[:5]],
        "anomaly": anomaly.to_dict(),
    }


@app.post("/analyze/chunk")
def analyze_chunk(req: ChunkAnalysisRequest):
    """Processes an immediate raw audio chunk (e.g. 50ms - 100ms) with sub-10ms response time."""
    signal = np.array(req.samples, dtype=np.float32)
    features = extractor.extract(signal)
    classification = classifier.classify(features)
    anomaly = anomaly_detector.process_frame(features)
    matches = matcher.match_against_registry(features, registry)

    return {
        "classification": classification.to_dict(),
        "anomaly": anomaly.to_dict(),
        "top_match": matches[0].to_dict() if matches else None,
        "rms_db": features.rms_db,
        "dominant_freq_hz": features.dominant_frequency_hz,
    }


@app.get("/patterns")
def list_patterns():
    """Lists all registered acoustic pattern templates."""
    return {"count": len(registry.patterns), "patterns": registry.list_patterns()}


@app.post("/patterns/register")
async def register_pattern(name: str = Form(...), description: str = Form(""), file: UploadFile = File(...)):
    """Uploads a reference WAV and registers it as a named pattern template."""
    contents = await file.read()
    temp_buf = io.BytesIO(contents)
    signal, _ = load_wav(temp_buf, target_sr=config.sample_rate)
    pattern = registry.register_signal(name, signal, description=description)
    return {"status": "success", "registered": pattern.to_dict()}


@app.delete("/patterns/{name}")
def delete_pattern(name: str):
    """Removes a pattern template."""
    removed = registry.remove_pattern(name)
    return {"status": "success" if removed else "not_found", "pattern_name": name}


@app.websocket("/ws/stream")
async def websocket_audio_stream(websocket: WebSocket):
    """
    Real-time streaming WebSocket endpoint.
    Client continuously sends binary float32 PCM frames or JSON sample arrays,
    and server immediately responds with instant recognition, classification, and anomaly alerts.
    """
    await websocket.accept()
    local_anomaly = AcousticAnomalyDetector(config)
    try:
        while True:
            # Receive float bytes or text JSON
            data = await websocket.receive()
            if "bytes" in data and data["bytes"]:
                chunk = np.frombuffer(data["bytes"], dtype=np.float32)
            elif "text" in data and data["text"]:
                import json
                parsed = json.loads(data["text"])
                chunk = np.array(parsed.get("samples", []), dtype=np.float32)
            else:
                continue

            if len(chunk) == 0:
                continue

            features = extractor.extract(chunk)
            classification = classifier.classify(features)
            anomaly = local_anomaly.process_frame(features)
            matches = matcher.match_against_registry(features, registry)

            response = {
                "category": classification.category.value,
                "confidence": classification.confidence,
                "rms_db": features.rms_db,
                "dominant_freq_hz": features.dominant_frequency_hz,
                "spectral_centroid": features.spectral_centroid,
                "is_anomaly": anomaly.is_anomaly,
                "anomaly_score": anomaly.anomaly_score,
                "top_pattern": matches[0].pattern_name if (matches and matches[0].matched) else None,
                "top_pattern_confidence": matches[0].confidence if matches else 0.0,
            }
            await websocket.send_json(response)
    except WebSocketDisconnect:
        pass

