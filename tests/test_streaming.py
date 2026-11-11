import numpy as np
import pytest

from audio_pulse.config import AudioConfig
from audio_pulse.stream.buffer import AudioRingBuffer
from audio_pulse.stream.simulated_stream import SimulatedAudioStream


def test_audio_ring_buffer_read_write():
    buf = AudioRingBuffer(capacity=100)
    data1 = np.ones(40, dtype=np.float32)
    buf.write(data1)

    read1 = buf.read_recent(40)
    assert len(read1) == 40
    assert np.all(read1 == 1.0)

    # Write overflow past capacity
    data2 = np.full(80, 2.0, dtype=np.float32)
    buf.write(data2)

    # Should retain the latest 100 samples (20 ones, then 80 twos)
    recent = buf.read_recent(100)
    assert len(recent) == 100
    assert np.all(recent[:20] == 1.0)
    assert np.all(recent[20:] == 2.0)


def test_simulated_audio_stream():
    config = AudioConfig(sample_rate=1000, chunk_duration_sec=0.1)
    sig = np.random.randn(500).astype(np.float32)
    stream = SimulatedAudioStream(signal=sig, config=config, real_time_pace=False)

    chunks = list(stream.iter_chunks())
    assert len(chunks) == 5
    for c in chunks:
        assert len(c) == 100
