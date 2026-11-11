"""
Immediate pattern matching engine using Fast Dynamic Time Warping (DTW) and Vector Cosine Similarity.
"""
from dataclasses import dataclass
import numpy as np
from ..config import AudioConfig
from ..dsp.features import AudioFeatures


@dataclass
class MatchResult:
    pattern_name: str
    confidence: float        # Normalized score between 0.0 and 1.0
    matched: bool
    distance: float
    method: str              # "dtw", "cosine", or "hybrid"
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "pattern_name": self.pattern_name,
            "confidence": round(float(self.confidence), 4),
            "matched": self.matched,
            "distance": round(float(self.distance), 4),
            "method": self.method,
            "description": self.description,
        }


class AudioPatternMatcher:
    """Matches audio features against registered acoustic templates."""

    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()

    @staticmethod
    def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Computes cosine similarity between two 1D normalized feature vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-7 or norm_b < 1e-7:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    @staticmethod
    def compute_dtw_distance(mfcc_a: np.ndarray, mfcc_b: np.ndarray) -> float:
        """
        Computes normalized Dynamic Time Warping distance between two MFCC sequence matrices
        with Cepstral Mean and Variance Normalization (CMVN).
        Input shapes: (n_mfcc, n_frames_a) and (n_mfcc, n_frames_b).
        """
        a = mfcc_a.T
        b = mfcc_b.T
        n_a, n_b = len(a), len(b)
        if n_a == 0 or n_b == 0:
            return 1.0

        # Apply CMVN along time axis
        a_norm = (a - np.mean(a, axis=0, keepdims=True)) / (np.std(a, axis=0, keepdims=True) + 1e-6)
        b_norm = (b - np.mean(b, axis=0, keepdims=True)) / (np.std(b, axis=0, keepdims=True) + 1e-6)

        # Pairwise euclidean distance matrix between frames
        diff = a_norm[:, np.newaxis, :] - b_norm[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)

        # Cost matrix initialization
        cost = np.full((n_a + 1, n_b + 1), np.inf, dtype=np.float32)
        cost[0, 0] = 0.0

        for i in range(1, n_a + 1):
            for j in range(1, n_b + 1):
                d = dist_matrix[i - 1, j - 1]
                cost[i, j] = d + min(cost[i - 1, j],
                                     cost[i, j - 1],
                                     cost[i - 1, j - 1])

        total_cost = cost[n_a, n_b]
        normalized_distance = float(total_cost / (n_a + n_b + 1e-6))
        return normalized_distance

    def match_features(self, query_features: AudioFeatures, template_features: AudioFeatures, template_name: str) -> MatchResult:
        """Compares a query audio against a single template using hybrid DTW and Cosine similarity."""
        query_fp = query_features.get_fingerprint_vector()
        template_fp = template_features.get_fingerprint_vector()
        cosine_sim = self.compute_cosine_similarity(query_fp, template_fp)

        if query_features.mfcc_frames.shape[1] >= 2 and template_features.mfcc_frames.shape[1] >= 2:
            dtw_dist = self.compute_dtw_distance(query_features.mfcc_frames, template_features.mfcc_frames)
            dtw_conf = 1.0 / (1.0 + 0.45 * dtw_dist)
            combined_conf = float(0.65 * cosine_sim + 0.35 * dtw_conf)
            is_match = (combined_conf >= 0.65) or (cosine_sim >= 0.80)

            return MatchResult(
                pattern_name=template_name,
                confidence=combined_conf,
                matched=is_match,
                distance=dtw_dist,
                method="hybrid_dtw"
            )
        else:
            is_match = cosine_sim >= self.config.cosine_similarity_threshold
            return MatchResult(
                pattern_name=template_name,
                confidence=max(0.0, cosine_sim),
                matched=is_match,
                distance=1.0 - cosine_sim,
                method="cosine"
            )

    def match_against_registry(self, query_features: AudioFeatures, registry) -> list[MatchResult]:
        """Matches query features against all patterns stored in the registry."""
        results = []
        for name, pattern in registry.patterns.items():
            res = self.match_features(query_features, pattern.features, name)
            res.description = pattern.description
            results.append(res)

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results
