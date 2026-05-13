# Membership Functions and Parameter Tuning

## Overview
The Fuzzy Logic engine in SpeechAI maps real-valued acoustic features into linguistic terms (e.g., LOW, MED, HIGH) using mathematical curves called Membership Functions (MFs).

## Membership Function Types
The `fuzzy_engine.py` implements the following MFs:
- `left_shoulder(x, a, b)`: Full membership (1.0) below `a`, linearly decreasing to 0.0 at `b`.
- `right_shoulder(x, a, b)`: Zero membership below `a`, linearly increasing to 1.0 at `b`.
- `triangular(x, a, b, c)`: Zero at `a` and `c`, peaking at `b`.
- `gaussian(x, mu, sigma)`: A bell curve centered at `mu`.
- `sigmoid(x, center, steepness)`: An S-shaped curve used primarily for soft Voice Activity Detection (VAD).

## Feature to Linguistic Mapping
The 8 extracted features are mapped to 24 distinct membership values.

| Feature | Linguistic Terms |
|---------|------------------|
| `avg_level` | `lvl_lo`, `lvl_md`, `lvl_hi` |
| `speech_ratio` | `rat_lo`, `rat_md`, `rat_hi` |
| `num_gaps` | `gap_fw`, `gap_sm`, `gap_mn` |
| `prosody_curvature` | `curv_lo`, `curv_md`, `curv_hi` |
| `speech_rate_variability`| `srv_lo`, `srv_md`, `srv_hi` |
| `pause_duration_entropy` | `pde_lo`, `pde_md`, `pde_hi` |
| `intensity_drift` | `drift_neg`, `drift_zero`, `drift_pos` |
| `response_latency` | `latency_lo`, `latency_md`, `latency_hi` |

## Training and Auto-Calibration
The `trainer.py` calculates the optimal parameters for these MFs based on sample data. It uses the statistics (min, mean, max) for each target class to set the `a`, `b`, and `c` anchor points.

For each feature, the system assigns one of the 5 classes to represent the LOW state, one for MED, and one for HIGH, effectively deriving real-world anchors from training samples. For example, `avg_level` assumes "Ambient Silence" is LOW, "Confident Articulation" is HIGH, and "Disengaged Monotone" is MED.
