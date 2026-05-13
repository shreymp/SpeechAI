"""
Calibrator — Noise Floor & Adaptive Threshold
===============================================
Manages per-session noise floor calibration and adaptive threshold
computation for the 5-class speech classifier.

Reference: Phase 4 of the Action Plan.
"""

import os
import json
from datetime import datetime

from src.features import (
    estimate_noise_floor,
    detect_clipping,
    THRESHOLD_OFFSET,
    DEFAULT_NOISE_FLOOR,
)


# ============================================================
# CALIBRATION DATA MANAGEMENT
# ============================================================

class NoiseCalibration:
    """
    Stores and manages per-session noise floor calibration data.
    """

    def __init__(self, data_dir=None):
        self.noise_floor = DEFAULT_NOISE_FLOOR
        self.threshold = DEFAULT_NOISE_FLOOR + THRESHOLD_OFFSET
        self.calibrated = False
        self.calibrated_at = None
        self.data_dir = data_dir

    def calibrate_from_samples(self, ambient_samples):
        """
        Calibrate noise floor from a set of ambient (silence) samples.

        Args:
            ambient_samples: List of raw 8-bit amplitude values recorded
                             during known silence/ambient conditions.

        Returns:
            (noise_floor, threshold)
        """
        if not ambient_samples:
            return self.noise_floor, self.threshold

        self.noise_floor = estimate_noise_floor(ambient_samples)
        self.threshold = self.noise_floor + THRESHOLD_OFFSET
        self.calibrated = True
        self.calibrated_at = datetime.now().isoformat()

        # Check for clipping
        if detect_clipping(ambient_samples):
            print("  ⚠️  WARNING: Signal clipping detected in ambient recording.")
            print("     The microphone may be saturated. Try moving away from loud sources.")

        return self.noise_floor, self.threshold

    def calibrate_from_recording_start(self, samples, n_samples=30):
        """
        Quick calibration using the first N samples of a recording
        (assumed to be ambient before speech onset).

        Args:
            samples: Full recording samples.
            n_samples: Number of initial samples to use for calibration.

        Returns:
            (noise_floor, threshold)
        """
        if not samples:
            return self.noise_floor, self.threshold

        ambient = samples[:min(n_samples, len(samples))]
        return self.calibrate_from_samples(ambient)

    def get_threshold(self):
        """Get the current adaptive threshold."""
        return self.threshold

    def get_noise_floor(self):
        """Get the current noise floor estimate."""
        return self.noise_floor

    def to_dict(self):
        """Serialize calibration state to dict."""
        return {
            "noise_floor": self.noise_floor,
            "threshold": self.threshold,
            "calibrated": self.calibrated,
            "calibrated_at": self.calibrated_at,
        }

    def from_dict(self, d):
        """Load calibration state from dict."""
        self.noise_floor = d.get("noise_floor", DEFAULT_NOISE_FLOOR)
        self.threshold = d.get("threshold", DEFAULT_NOISE_FLOOR + THRESHOLD_OFFSET)
        self.calibrated = d.get("calibrated", False)
        self.calibrated_at = d.get("calibrated_at", None)

    def save(self, filepath=None):
        """Save calibration data to JSON."""
        if filepath is None and self.data_dir:
            filepath = os.path.join(self.data_dir, "noise_calibration.json")
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)

    def load(self, filepath=None):
        """Load calibration data from JSON."""
        if filepath is None and self.data_dir:
            filepath = os.path.join(self.data_dir, "noise_calibration.json")
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    d = json.load(f)
                self.from_dict(d)
                return True
            except Exception:
                return False
        return False

    def online_adapt(self, new_noise_estimate, alpha=0.2):
        """
        Phase D online adaptation: update noise floor using exponential
        moving average to compensate for environmental drift over time.

        Args:
            new_noise_estimate: Fresh noise floor estimate from latest recording.
            alpha: Learning rate (0–1). Higher = faster adaptation. Default 0.2.

        Returns:
            Updated (noise_floor, threshold).
        """
        if not self.calibrated:
            # First calibration — use the new estimate directly
            self.noise_floor = new_noise_estimate
        else:
            # EMA: new_floor = alpha * new_estimate + (1-alpha) * old_floor
            self.noise_floor = alpha * new_noise_estimate + (1 - alpha) * self.noise_floor

        self.threshold = self.noise_floor + THRESHOLD_OFFSET
        self.calibrated = True
        self.calibrated_at = datetime.now().isoformat()
        return self.noise_floor, self.threshold


# ============================================================
# SESSION LOGGER
# ============================================================

class SessionLogger:
    """
    Logs classification results over time for longitudinal tracking.
    Saves to data/session_log.csv.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.log_path = os.path.join(data_dir, "session_log.csv")
        self._ensure_header()

    def _ensure_header(self):
        """Create the CSV file with header if it doesn't exist."""
        if not os.path.exists(self.log_path):
            os.makedirs(self.data_dir, exist_ok=True)
            with open(self.log_path, 'w') as f:
                f.write("timestamp,class_name,confidence,margin,is_ambiguous,"
                        "avg_level,speech_ratio,num_gaps,prosody_curvature,"
                        "speech_rate_variability,pause_duration_entropy,"
                        "intensity_drift,response_latency\n")

    def log_classification(self, class_name, confidence, margin, is_ambiguous, features):
        """
        Append a classification result to the session log.

        Args:
            class_name: String name of the classified category.
            confidence: Integer confidence percentage.
            margin: Float margin between top-2 scores.
            is_ambiguous: Boolean ambiguity flag.
            features: Dict of 8 feature values.
        """
        timestamp = datetime.now().isoformat()
        row = (
            f"{timestamp},{class_name},{confidence},{margin:.4f},{is_ambiguous},"
            f"{features.get('avg_level', 0):.1f},"
            f"{features.get('speech_ratio', 0):.3f},"
            f"{features.get('num_gaps', 0)},"
            f"{features.get('prosody_curvature', 0):.3f},"
            f"{features.get('speech_rate_variability', 0):.3f},"
            f"{features.get('pause_duration_entropy', 0):.3f},"
            f"{features.get('intensity_drift', 0):.4f},"
            f"{features.get('response_latency', 0):.0f}\n"
        )

        with open(self.log_path, 'a') as f:
            f.write(row)
