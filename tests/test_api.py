import numpy as np
from audio_pulse.api.server import root, list_patterns, analyze_chunk, ChunkAnalysisRequest


def test_api_root():
    data = root()
    assert data["status"] == "online"
    assert "service" in data
    assert "endpoints" in data


def test_api_patterns_list():
    data = list_patterns()
    assert "patterns" in data
    assert "count" in data
    assert isinstance(data["patterns"], list)


def test_api_analyze_chunk():
    samples = (np.sin(np.linspace(0, 10, 1024)) * 0.5).tolist()
    req = ChunkAnalysisRequest(samples=samples, sample_rate=22050)
    data = analyze_chunk(req)
    assert "classification" in data
    assert "anomaly" in data
    assert "rms_db" in data
    assert data["classification"]["category"] is not None
