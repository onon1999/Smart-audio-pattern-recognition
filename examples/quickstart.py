"""
EchoPulse AI — 15-Line Quickstart Example
Demonstrates immediate audio feature extraction, acoustic classification, and template pattern matching.
"""
import sys
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_pulse.config import AudioConfig
from audio_pulse.dsp.features import AudioFeatureExtractor
from audio_pulse.recognition.classifier import ImmediateAcousticClassifier
from audio_pulse.recognition.matcher import AudioPatternMatcher
from audio_pulse.recognition.registry import PatternRegistry
from audio_pulse.utils.synthetic_sounds import SyntheticAudioGenerator

# 1. Initialize engine
config = AudioConfig()
extractor = AudioFeatureExtractor(config)
classifier = ImmediateAcousticClassifier(config)
matcher = AudioPatternMatcher(config)
registry = PatternRegistry(config)
synth = SyntheticAudioGenerator(config.sample_rate)

# 2. Register reference acoustic template
clap_ref = synth.generate_clap(duration=0.35)
registry.register_signal("Hand Clap", clap_ref, "Acoustic percussive handclap")
print(f"Registered reference template: 'Hand Clap' with {len(clap_ref)} samples.")

# 3. Simulate an unknown incoming audio query
test_query = synth.generate_clap(duration=0.30)
features = extractor.extract(test_query)

# 4. Immediate Event Classification
classification = classifier.classify(features)
print(f"\n[Immediate Event Detection]")
print(f"Category:   {classification.category.value}")
print(f"Confidence: {classification.confidence * 100:.1f}%")
print(f"Summary:    {classification.summary}")

# 5. Immediate Pattern Matching
matches = matcher.match_against_registry(features, registry)
print(f"\n[Pattern Match Results]")
for m in matches:
    status = "MATCH FOUND" if m.matched else "NO MATCH"
    print(f"- {m.pattern_name}: {status} (Confidence: {m.confidence * 100:.1f}%, Distance: {m.distance:.4f})")
