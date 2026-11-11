"""
Real-time WebSocket streaming client demonstration.
Connects to ws://localhost:8000/ws/stream and streams audio chunks in real time,
printing immediate pattern recognition & anomaly alerts.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import websockets

from audio_pulse.config import AudioConfig
from audio_pulse.utils.synthetic_sounds import SyntheticAudioGenerator


async def stream_audio_client(server_uri: str = "ws://localhost:8000/ws/stream"):
    config = AudioConfig()
    synth = SyntheticAudioGenerator(config.sample_rate)

    print(f"Connecting to WebSocket stream at {server_uri}...")
    try:
        async with websockets.connect(server_uri) as ws:
            print("Connected! Generating dynamic test acoustic stream...\n")

            # Sequence of acoustic patterns to stream
            segments = [
                ("Silence / Background", np.zeros(int(config.sample_rate * 0.4), dtype=np.float32)),
                ("Transient Hand Clap", synth.generate_clap(0.35)),
                ("Silence / Background", np.zeros(int(config.sample_rate * 0.3), dtype=np.float32)),
                ("Whistle Frequency Sweep", synth.generate_whistle_sweep(900, 2400, 0.8)),
                ("Pulsing Alarm Siren", synth.generate_alarm_pulse(1400, 4.0, 1.0)),
                ("Pink Acoustic Noise", synth.generate_noise("pink", 0.6)),
            ]

            chunk_size = config.chunk_size

            for label, audio in segments:
                print(f"\n▶ Streaming Sound Segment: [{label}] ({len(audio)} samples)")
                
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i + chunk_size]
                    if len(chunk) < chunk_size:
                        padded = np.zeros(chunk_size, dtype=np.float32)
                        padded[:len(chunk)] = chunk
                        chunk = padded

                    # Send audio frame as binary float32 buffer
                    await ws.send(chunk.tobytes())

                    # Receive immediate server analysis response
                    response_text = await ws.recv()
                    data = json.loads(response_text)

                    cat = data.get("category", "Unknown")
                    conf = data.get("confidence", 0.0)
                    rms = data.get("rms_db", -99.0)
                    top_pat = data.get("top_pattern") or "None"
                    is_anomaly = data.get("is_anomaly", False)

                    anomaly_badge = "🚨 [ANOMALY]" if is_anomaly else "   [Normal]"
                    print(
                        f"  {anomaly_badge} RMS: {rms:>5.1f}dB | Event: {cat:<32} "
                        f"| Conf: {conf*100:>4.1f}% | Matched Pattern: {top_pat}"
                    )

                    await asyncio.sleep(0.08)

            print("\nStreaming finished successfully!")

    except ConnectionRefusedError:
        print("[ERROR] Could not connect to API server. Ensure the server is running:")
        print("  uvicorn audio_pulse.api.server:app --reload --port 8000")
    except Exception as e:
        print(f"[ERROR] WebSocket communication error: {e}")


def main():
    asyncio.run(stream_audio_client())


if __name__ == "__main__":
    main()
