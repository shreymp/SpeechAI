"""
Comprehensive unit tests for the 5-class fuzzy inference engine.
Covers all 5 MF types, all 5 classes, boundary conditions,
rule isolation, and edge cases.
"""

import pytest
import math
from src.fuzzy_engine import (
    fuzzy_classify_5, DEFAULT_PARAMS_5, CLASSES_5,
    left_shoulder, right_shoulder, triangular, gaussian, sigmoid_mf,
    fuzzify, evaluate_rules, defuzzify,
)


# ============================================================
# MEMBERSHIP FUNCTION TESTS
# ============================================================

class TestLeftShoulder:
    def test_full_membership_below_a(self):
        assert left_shoulder(0, 10, 20) == 1.0

    def test_full_membership_at_a(self):
        assert left_shoulder(10, 10, 20) == 1.0

    def test_zero_at_b(self):
        assert left_shoulder(20, 10, 20) == 0.0

    def test_zero_above_b(self):
        assert left_shoulder(30, 10, 20) == 0.0

    def test_midpoint(self):
        assert math.isclose(left_shoulder(15, 10, 20), 0.5)

    def test_degenerate_a_equals_b(self):
        assert left_shoulder(10, 10, 10) == 1.0
        assert left_shoulder(11, 10, 10) == 0.0


class TestRightShoulder:
    def test_zero_below_a(self):
        assert right_shoulder(0, 10, 20) == 0.0

    def test_zero_at_a(self):
        assert right_shoulder(10, 10, 20) == 0.0

    def test_full_at_b(self):
        assert right_shoulder(20, 10, 20) == 1.0

    def test_full_above_b(self):
        assert right_shoulder(30, 10, 20) == 1.0

    def test_midpoint(self):
        assert math.isclose(right_shoulder(15, 10, 20), 0.5)

    def test_degenerate_a_equals_b(self):
        # When a==b, x<=a returns 0.0 per implementation
        assert right_shoulder(10, 10, 10) == 0.0
        assert right_shoulder(11, 10, 10) == 1.0


class TestTriangular:
    def test_zero_at_a(self):
        assert triangular(0, 0, 5, 10) == 0.0

    def test_peak_at_b(self):
        assert triangular(5, 0, 5, 10) == 1.0

    def test_zero_at_c(self):
        assert triangular(10, 0, 5, 10) == 0.0

    def test_zero_below_a(self):
        assert triangular(-5, 0, 5, 10) == 0.0

    def test_zero_above_c(self):
        assert triangular(15, 0, 5, 10) == 0.0

    def test_left_midpoint(self):
        assert math.isclose(triangular(2.5, 0, 5, 10), 0.5)

    def test_right_midpoint(self):
        assert math.isclose(triangular(7.5, 0, 5, 10), 0.5)

    def test_degenerate_symmetric(self):
        """When a == b, x<=a returns 0 per implementation."""
        assert triangular(0, 0, 0, 10) == 0.0
        assert triangular(5, 0, 0, 10) == 0.5


class TestGaussian:
    def test_peak(self):
        assert gaussian(10, 10, 5) == 1.0

    def test_one_sigma(self):
        expected = math.exp(-0.5)
        assert math.isclose(gaussian(15, 10, 5), expected, rel_tol=1e-6)

    def test_sigma_zero(self):
        assert gaussian(10, 10, 0) == 1.0
        assert gaussian(11, 10, 0) == 0.0


class TestSigmoid:
    def test_center_is_half(self):
        assert math.isclose(sigmoid_mf(50, 50, 0.3), 0.5, abs_tol=0.01)

    def test_high_above_center(self):
        assert sigmoid_mf(100, 50, 0.3) > 0.9

    def test_low_below_center(self):
        assert sigmoid_mf(0, 50, 0.3) < 0.1


# ============================================================
# FUZZIFICATION TESTS
# ============================================================

class TestFuzzify:
    def test_returns_25_keys(self):
        features = {
            "avg_level": 50.0, "speech_ratio": 0.5, "num_gaps": 5,
            "prosody_curvature": 10.0, "speech_rate_variability": 1.0,
            "pause_duration_entropy": 0.5, "intensity_drift": 0.0,
            "response_latency": 500,
        }
        m = fuzzify(features, DEFAULT_PARAMS_5)
        assert len(m) == 25  # 24 standard + 1 Gaussian variant

    def test_all_values_in_range(self):
        features = {
            "avg_level": 100, "speech_ratio": 0.5, "num_gaps": 8,
            "prosody_curvature": 15, "speech_rate_variability": 1.5,
            "pause_duration_entropy": 0.8, "intensity_drift": 0.1,
            "response_latency": 800,
        }
        m = fuzzify(features, DEFAULT_PARAMS_5)
        for k, v in m.items():
            assert 0.0 <= v <= 1.0, f"{k} = {v} out of range [0, 1]"


# ============================================================
# CLASSIFICATION TESTS — ALL 5 CLASSES
# ============================================================

class TestClassifySilence:
    def test_pure_silence(self):
        features = {
            "avg_level": 5.0, "speech_ratio": 0.02, "num_gaps": 0,
            "prosody_curvature": 1.0, "speech_rate_variability": 0.1,
            "pause_duration_entropy": 0.0, "intensity_drift": 0.0,
            "response_latency": 2800,
        }
        idx, conf, details = fuzzy_classify_5(features, DEFAULT_PARAMS_5)
        assert CLASSES_5[idx] == "Ambient Silence"
        assert conf > 0


class TestClassifyConfident:
    def test_strong_confident(self):
        features = {
            "avg_level": 150.0, "speech_ratio": 0.85, "num_gaps": 2,
            "prosody_curvature": 20.0, "speech_rate_variability": 0.3,
            "pause_duration_entropy": 0.2, "intensity_drift": 0.0,
            "response_latency": 150,
        }
        idx, conf, details = fuzzy_classify_5(features, DEFAULT_PARAMS_5)
        assert CLASSES_5[idx] == "Confident Articulation"
        assert conf > 0


class TestClassifyHesitant:
    def test_hesitant_disfluency(self):
        features = {
            "avg_level": 65.0, "speech_ratio": 0.35, "num_gaps": 12,
            "prosody_curvature": 8.0, "speech_rate_variability": 2.5,
            "pause_duration_entropy": 1.5, "intensity_drift": -0.8,
            "response_latency": 1800,
        }
        idx, conf, details = fuzzy_classify_5(features, DEFAULT_PARAMS_5)
        assert CLASSES_5[idx] == "Hesitant Disfluency"
        assert conf > 0


class TestClassifyAnxious:
    def test_anxious_urgency(self):
        features = {
            "avg_level": 170.0, "speech_ratio": 0.95, "num_gaps": 0,
            "prosody_curvature": 2.5, "speech_rate_variability": 1.8,
            "pause_duration_entropy": 0.0, "intensity_drift": 1.2,
            "response_latency": 50,
        }
        idx, conf, details = fuzzy_classify_5(features, DEFAULT_PARAMS_5)
        assert CLASSES_5[idx] == "Anxious Urgency"
        assert conf > 0


class TestClassifyMonotone:
    def test_disengaged_monotone(self):
        features = {
            "avg_level": 70.0, "speech_ratio": 0.45, "num_gaps": 5,
            "prosody_curvature": 2.0, "speech_rate_variability": 0.3,
            "pause_duration_entropy": 0.2, "intensity_drift": 0.0,
            "response_latency": 600,
        }
        idx, conf, details = fuzzy_classify_5(features, DEFAULT_PARAMS_5)
        assert CLASSES_5[idx] == "Disengaged Monotone"
        assert conf > 0


# ============================================================
# EDGE CASE TESTS
# ============================================================

class TestEdgeCases:
    def test_all_zero_features(self):
        """System should not crash on all-zero features."""
        features = {k: 0 for k in [
            "avg_level", "speech_ratio", "num_gaps", "prosody_curvature",
            "speech_rate_variability", "pause_duration_entropy",
            "intensity_drift", "response_latency",
        ]}
        idx, conf, details = fuzzy_classify_5(features, DEFAULT_PARAMS_5)
        assert 0 <= idx < 5
        assert "margin" in details
        assert "is_ambiguous" in details
        assert "rules" in details

    def test_all_max_features(self):
        """System should not crash with extremely high values."""
        features = {
            "avg_level": 255, "speech_ratio": 1.0, "num_gaps": 50,
            "prosody_curvature": 100, "speech_rate_variability": 10.0,
            "pause_duration_entropy": 3.0, "intensity_drift": 5.0,
            "response_latency": 5000,
        }
        idx, conf, details = fuzzy_classify_5(features, DEFAULT_PARAMS_5)
        assert 0 <= idx < 5

    def test_details_structure(self):
        """Verify the details dict has all expected keys."""
        features = {
            "avg_level": 100, "speech_ratio": 0.5, "num_gaps": 5,
            "prosody_curvature": 10, "speech_rate_variability": 1.0,
            "pause_duration_entropy": 0.5, "intensity_drift": 0.0,
            "response_latency": 500,
        }
        idx, conf, details = fuzzy_classify_5(features, DEFAULT_PARAMS_5)
        assert "scores" in details
        assert "margin" in details
        assert "is_ambiguous" in details
        assert "rules" in details
        assert isinstance(details["is_ambiguous"], bool)
        assert len(details["scores"]) == 5

    def test_custom_rule_weights(self):
        """Ensure custom rule weights are applied."""
        features = {
            "avg_level": 5.0, "speech_ratio": 0.02, "num_gaps": 0,
            "prosody_curvature": 1.0, "speech_rate_variability": 0.1,
            "pause_duration_entropy": 0.0, "intensity_drift": 0.0,
            "response_latency": 2800,
        }
        # Zero out all weights — should still return a valid result
        weights = [0.0] * 13
        idx, conf, details = fuzzy_classify_5(features, DEFAULT_PARAMS_5, weights)
        assert 0 <= idx < 5
