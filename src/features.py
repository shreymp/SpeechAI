"""
Feature Extraction Pipeline — 8-Variable System
=================================================
Extracts 8 acoustic features from raw 8-bit sound-level samples (0–255)
captured at ~50 Hz over a 3-second window (~150 samples).

Features:
  1. avg_level            — Mean amplitude
  2. speech_ratio         — Soft VAD temporal density
  3. num_gaps             — Filtered gap count (≥100ms)
  4. prosody_curvature    — Mean absolute 2nd derivative of envelope
  5. speech_rate_variability — Std dev of per-window syllable rate
  6. pause_duration_entropy  — Shannon entropy of gap duration distribution
  7. intensity_drift      — Linear regression slope of amplitude
  8. response_latency     — Time to first above-threshold sample (ms)

Reference: Phase 2 & Phase 4 of the Action Plan.
"""

import math


# ============================================================
# DEFAULT CONSTANTS
# ============================================================

DEFAULT_NOISE_FLOOR = 25       # Default noise floor if not calibrated
MIN_GAP_SAMPLES = 5            # 100ms at 50Hz — minimum gap duration filter
WINDOW_SAMPLES = 25            # 500ms at 50Hz — sub-window for SRV
SAMPLE_INTERVAL_MS = 20        # 20ms per sample at 50Hz
SOFT_VAD_STEEPNESS = 0.3       # Sigmoid steepness for soft VAD
THRESHOLD_OFFSET = 15          # Added to noise floor for adaptive threshold

# Entropy bin edges in samples: <8, 8–15, 15–40, >40
# Note: first edge must be > MIN_GAP_SAMPLES (5) to avoid empty first bin
ENTROPY_BINS = [8, 15, 40]


# ============================================================
# PREPROCESSING
# ============================================================

def smooth_samples(samples):
    """Apply 3-point moving average to suppress single-sample noise spikes."""
    n = len(samples)
    if n <= 2:
        return list(samples)

    smoothed = [samples[0]]
    for i in range(1, n - 1):
        smoothed.append((samples[i - 1] + samples[i] + samples[i + 1]) / 3.0)
    smoothed.append(samples[-1])
    return smoothed


def estimate_noise_floor(samples):
    """
    Estimate noise floor from the first quarter of the recording.
    Uses the 20th percentile of the first 25% of samples.
    """
    n = len(samples)
    if n == 0:
        return DEFAULT_NOISE_FLOOR

    quarter = max(1, n // 4)
    sorted_quarter = sorted(samples[:quarter])
    idx = max(0, len(sorted_quarter) // 5)  # 20th percentile
    return sorted_quarter[idx]


def detect_clipping(samples, threshold_pct=0.20):
    """
    Detect if the signal is saturated (clipped).
    Returns True if >threshold_pct of samples equal 255.
    """
    if not samples:
        return False
    count_255 = sum(1 for s in samples if s >= 255)
    return (count_255 / len(samples)) > threshold_pct


# ============================================================
# FEATURE EXTRACTION — INDIVIDUAL FEATURES
# ============================================================

def compute_avg_level(smoothed):
    """Feature 1: Arithmetic mean of all smoothed amplitude samples."""
    n = len(smoothed)
    if n == 0:
        return 0.0
    return sum(smoothed) / n


def compute_speech_ratio_soft(smoothed, threshold):
    """
    Feature 2: Speech ratio using soft VAD (sigmoid membership).
    Replaces the hard SPEECH_THRESHOLD with a smooth sigmoid transition.
    """
    n = len(smoothed)
    if n == 0:
        return 0.0

    k = SOFT_VAD_STEEPNESS
    total = 0.0
    for s in smoothed:
        # Sigmoid: 1 / (1 + exp(-k * (s - threshold)))
        exponent = -k * (s - threshold)
        # Clamp to avoid overflow
        if exponent > 500:
            total += 0.0
        elif exponent < -500:
            total += 1.0
        else:
            total += 1.0 / (1.0 + math.exp(exponent))
    return total / n


def compute_gaps(smoothed, threshold):
    """
    Feature 3: Count speech-to-silence transitions with minimum gap duration filter.
    Also returns the list of gap durations for pause_duration_entropy computation.
    """
    n = len(smoothed)
    if n == 0:
        return 0, []

    is_speech = [s >= threshold for s in smoothed]
    gaps = []
    gap_start = None

    for i, speaking in enumerate(is_speech):
        if not speaking:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                gap_duration = i - gap_start
                if gap_duration >= MIN_GAP_SAMPLES:
                    gaps.append(gap_duration)
                gap_start = None

    # Handle trailing gap
    if gap_start is not None:
        gap_duration = n - gap_start
        if gap_duration >= MIN_GAP_SAMPLES:
            gaps.append(gap_duration)

    return len(gaps), gaps


def compute_prosody_curvature(smoothed):
    """
    Feature 4: Mean absolute second derivative of the amplitude envelope.
    Captures how 'dynamically shaped' the speech is.
    """
    n = len(smoothed)
    if n < 3:
        return 0.0

    total = 0.0
    count = 0
    for i in range(1, n - 1):
        dd = smoothed[i + 1] - 2.0 * smoothed[i] + smoothed[i - 1]
        total += abs(dd)
        count += 1

    return total / count if count > 0 else 0.0


def compute_speech_rate_variability(smoothed, threshold):
    """
    Feature 5: Standard deviation of per-sub-window rising-edge crossing counts.
    Approximates syllable rate variability across 500ms sub-windows.
    """
    n = len(smoothed)
    if n < WINDOW_SAMPLES:
        return 0.0

    is_speech = [s >= threshold for s in smoothed]
    crossing_counts = []

    for w in range(0, n - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
        crossings = 0
        for i in range(w + 1, w + WINDOW_SAMPLES):
            if is_speech[i] and not is_speech[i - 1]:
                crossings += 1
        crossing_counts.append(crossings)

    if len(crossing_counts) <= 1:
        return 0.0

    cc_mean = sum(crossing_counts) / len(crossing_counts)
    variance = sum((c - cc_mean) ** 2 for c in crossing_counts) / len(crossing_counts)
    return math.sqrt(variance)


def compute_pause_duration_entropy(gaps):
    """
    Feature 6: Shannon entropy of the distribution of gap durations.
    Bins: [<5 samples, 5–15, 15–40, >40] (approximately <100ms, 100–300ms, 300–800ms, >800ms).
    """
    if len(gaps) < 2:
        return 0.0

    bins = [0, 0, 0, 0]
    for g in gaps:
        if g < ENTROPY_BINS[0]:
            bins[0] += 1
        elif g < ENTROPY_BINS[1]:
            bins[1] += 1
        elif g < ENTROPY_BINS[2]:
            bins[2] += 1
        else:
            bins[3] += 1

    total_gaps = sum(bins)
    if total_gaps == 0:
        return 0.0

    entropy = 0.0
    for b in bins:
        if b > 0:
            p = b / total_gaps
            entropy -= p * math.log2(p)

    return entropy


def compute_intensity_drift(smoothed):
    """
    Feature 7: Linear regression slope of amplitude over time.
    Positive drift = getting louder; negative drift = trailing off.
    Uses the normal equation: b = (n*Σ(i*x) - Σi*Σx) / (n*Σi² - (Σi)²)
    """
    n = len(smoothed)
    if n < 2:
        return 0.0

    sum_i = n * (n - 1) / 2.0
    sum_i2 = n * (n - 1) * (2 * n - 1) / 6.0
    sum_ix = sum(i * smoothed[i] for i in range(n))
    sum_x = sum(smoothed)

    denom = n * sum_i2 - sum_i * sum_i
    if denom == 0:
        return 0.0

    return (n * sum_ix - sum_i * sum_x) / denom


def compute_response_latency(smoothed, threshold):
    """
    Feature 8: Time delay (in ms) from start of recording to first above-threshold sample.
    If no speech detected, returns the full window duration.
    """
    n = len(smoothed)
    onset_idx = n  # default: no speech detected

    for i in range(n):
        if smoothed[i] >= threshold:
            onset_idx = i
            break

    return onset_idx * SAMPLE_INTERVAL_MS


# ============================================================
# COMPLETE FEATURE EXTRACTION
# ============================================================

DEFAULT_FEATURES = {
    "avg_level": 0.0,
    "speech_ratio": 0.0,
    "num_gaps": 0,
    "prosody_curvature": 0.0,
    "speech_rate_variability": 0.0,
    "pause_duration_entropy": 0.0,
    "intensity_drift": 0.0,
    "response_latency": 3000.0,
}

FEATURE_NAMES_8 = list(DEFAULT_FEATURES.keys())


def extract_features_v2(samples, noise_floor=None):
    """
    Extract all 8 features from raw sound-level samples.

    Args:
        samples: List of 8-bit integers (0–255) from micro:bit sound_level().
        noise_floor: Pre-calibrated noise floor. If None, estimated from data.

    Returns:
        Dict with all 8 feature values.
    """
    n = len(samples)
    if n == 0:
        return DEFAULT_FEATURES.copy()

    # ── Preprocessing ──
    smoothed = smooth_samples(samples)

    # Adaptive noise floor
    if noise_floor is None:
        noise_floor = estimate_noise_floor(smoothed)
    threshold = noise_floor + THRESHOLD_OFFSET

    # ── Feature 1: avg_level ──
    avg_level = compute_avg_level(smoothed)

    # ── Feature 2: speech_ratio (soft VAD) ──
    speech_ratio = compute_speech_ratio_soft(smoothed, threshold)

    # ── Feature 3: num_gaps (with duration filter) ──
    num_gaps, gap_durations = compute_gaps(smoothed, threshold)

    # ── Feature 4: prosody_curvature ──
    prosody_curvature = compute_prosody_curvature(smoothed)

    # ── Feature 5: speech_rate_variability ──
    srv = compute_speech_rate_variability(smoothed, threshold)

    # ── Feature 6: pause_duration_entropy ──
    pde = compute_pause_duration_entropy(gap_durations)

    # ── Feature 7: intensity_drift ──
    intensity_drift = compute_intensity_drift(smoothed)

    # ── Feature 8: response_latency ──
    response_latency = compute_response_latency(smoothed, threshold)

    return {
        "avg_level": avg_level,
        "speech_ratio": speech_ratio,
        "num_gaps": num_gaps,
        "prosody_curvature": prosody_curvature,
        "speech_rate_variability": srv,
        "pause_duration_entropy": pde,
        "intensity_drift": intensity_drift,
        "response_latency": response_latency,
    }



