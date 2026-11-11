"""
FastAPI REST and WebSocket streaming server for immediate audio pattern analysis.
"""
import io
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
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


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "EchoPulse AI - Immediate Audio Pattern Recognition",
        "version": "1.0.0",
        "registered_patterns": len(registry.patterns),
        "endpoints": {
            "docs": "/docs",
            "analyze_file": "POST /analyze/file",
            "analyze_chunk": "POST /analyze/chunk",
            "list_patterns": "GET /patterns",
            "register_pattern": "POST /patterns/register",
            "websocket_stream": "WS /ws/stream"
        }
    }


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

