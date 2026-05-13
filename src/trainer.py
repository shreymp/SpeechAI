"""
Fuzzy Logic — Training & Validation Pipeline (5-Class)
=======================================================
Trains and validates triangular membership function parameters
for the 5-class speech classifier.

Can be called from START_HERE.py via run_training() or standalone via CLI.
"""

import json
import os
import time
import math
import random
import argparse
from datetime import datetime

from src.fuzzy_engine import (
    CLASSES_5, DEFAULT_PARAMS_5, fuzzy_classify_5,
)
from src.visualizer import display_training_summary
from src.serial_utils import get_microbit_port

# Optional: serial for micro:bit data push
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# ============================================================
# CONSTANTS
# ============================================================

BAUD_RATE = 115200
TRAIN_RATIO = 0.7
N_CLASSES = 5

# 8 features used in the 5-class system
FEATURE_NAMES_8 = [
    "avg_level", "speech_ratio", "num_gaps",
    "prosody_curvature", "speech_rate_variability",
    "pause_duration_entropy", "intensity_drift", "response_latency",
]

# Mapping: which classes map to LOW / MED / HIGH for each variable.
# For each variable we pick the class whose typical values are lowest,
# middle, and highest to anchor the three MF terms.
#
# Mapping rationale (from Action Plan Phase 3 acoustic indicators):
#   avg_level:  Silence(0) is lowest, Confident(1) is highest, Monotone(4) is mid
#   speech_ratio: Silence(0) is lowest, Confident(1) is highest, Hesitant(2) is mid
#   num_gaps: Silence(0) is lowest (0-1), Hesitant(2) is highest (8-20), Monotone(4) is mid
#   prosody_curvature: Silence(0) is lowest, Confident(1) is highest, Monotone(4) is mid
#   SRV: Silence(0) is lowest, Hesitant(2) is highest, Anxious(3) is mid
#   PDE: Silence(0) is lowest, Hesitant(2) is highest, Monotone(4) is mid
#   intensity_drift: Hesitant(2) is lowest (NEG/trailing off), Anxious(3) is highest (POS), Monotone(4) is mid (ZERO)
#   response_latency: Confident(1) is lowest (fast), Silence(0) is highest (no speech), Hesitant(2) is mid
FEATURE_CONFIG_5 = [
    # (feat_name, [class_for_LOW, class_for_HIGH, class_for_MED], floor, ceil, is_ratio)
    ("avg_level",               [0, 1, 4], 0,    255,  False),
    ("speech_ratio",            [0, 1, 2], 0.0,  1.0,  True),
    ("num_gaps",                [0, 2, 4], 0,    25,   False),
    ("prosody_curvature",       [0, 1, 4], 0,    50,   False),
    ("speech_rate_variability", [0, 2, 3], 0,    5.0,  True),
    ("pause_duration_entropy",  [0, 2, 4], 0,    2.0,  True),
    ("intensity_drift",         [2, 3, 4], -2.0, 2.0,  True),
    ("response_latency",        [1, 0, 2], 0,    3000, False),
]

PARAM_NAMES_5 = [
    ["LEVEL_LOW",   "LEVEL_MED",   "LEVEL_HIGH"],
    ["RATIO_LOW",   "RATIO_MED",   "RATIO_HIGH"],
    ["GAPS_FEW",    "GAPS_SOME",   "GAPS_MANY"],
    ["CURV_LOW",    "CURV_MED",    "CURV_HIGH"],
    ["SRV_LOW",     "SRV_MED",     "SRV_HIGH"],
    ["PDE_LOW",     "PDE_MED",     "PDE_HIGH"],
    ["DRIFT_NEG",   "DRIFT_ZERO",  "DRIFT_POS"],
    ["LATENCY_LOW", "LATENCY_MED", "LATENCY_HIGH"],
]

# Predefined 5-class synthetic data for testing
PREDEFINED_DATA_5 = [
    # Class 0: Ambient Silence
    {"class": 0, "avg_level": 5.0,  "speech_ratio": 0.02, "num_gaps": 0, "prosody_curvature": 1.0, "speech_rate_variability": 0.1, "pause_duration_entropy": 0.0, "intensity_drift": 0.01, "response_latency": 2800},
    {"class": 0, "avg_level": 12.0, "speech_ratio": 0.05, "num_gaps": 1, "prosody_curvature": 1.5, "speech_rate_variability": 0.2, "pause_duration_entropy": 0.0, "intensity_drift": -0.02, "response_latency": 2500},
    {"class": 0, "avg_level": 8.0,  "speech_ratio": 0.01, "num_gaps": 0, "prosody_curvature": 0.8, "speech_rate_variability": 0.0, "pause_duration_entropy": 0.0, "intensity_drift": 0.0, "response_latency": 3000},
    # Class 1: Confident Articulation
    {"class": 1, "avg_level": 140.0, "speech_ratio": 0.85, "num_gaps": 2, "prosody_curvature": 18.0, "speech_rate_variability": 0.4, "pause_duration_entropy": 0.2, "intensity_drift": 0.1, "response_latency": 150},
    {"class": 1, "avg_level": 160.0, "speech_ratio": 0.90, "num_gaps": 1, "prosody_curvature": 22.0, "speech_rate_variability": 0.3, "pause_duration_entropy": 0.1, "intensity_drift": 0.15, "response_latency": 100},
    {"class": 1, "avg_level": 120.0, "speech_ratio": 0.78, "num_gaps": 3, "prosody_curvature": 15.0, "speech_rate_variability": 0.5, "pause_duration_entropy": 0.3, "intensity_drift": 0.05, "response_latency": 200},
    # Class 2: Hesitant Disfluency
    {"class": 2, "avg_level": 65.0, "speech_ratio": 0.35, "num_gaps": 12, "prosody_curvature": 8.0, "speech_rate_variability": 2.5, "pause_duration_entropy": 1.5, "intensity_drift": -0.8, "response_latency": 1800},
    {"class": 2, "avg_level": 55.0, "speech_ratio": 0.28, "num_gaps": 15, "prosody_curvature": 6.0, "speech_rate_variability": 3.0, "pause_duration_entropy": 1.8, "intensity_drift": -1.0, "response_latency": 2000},
    {"class": 2, "avg_level": 80.0, "speech_ratio": 0.42, "num_gaps": 10, "prosody_curvature": 10.0, "speech_rate_variability": 2.0, "pause_duration_entropy": 1.2, "intensity_drift": -0.5, "response_latency": 1500},
    # Class 3: Anxious Urgency
    {"class": 3, "avg_level": 150.0, "speech_ratio": 0.92, "num_gaps": 1, "prosody_curvature": 3.0, "speech_rate_variability": 1.5, "pause_duration_entropy": 0.1, "intensity_drift": 0.8, "response_latency": 80},
    {"class": 3, "avg_level": 170.0, "speech_ratio": 0.95, "num_gaps": 0, "prosody_curvature": 2.5, "speech_rate_variability": 1.8, "pause_duration_entropy": 0.0, "intensity_drift": 1.2, "response_latency": 50},
    {"class": 3, "avg_level": 130.0, "speech_ratio": 0.88, "num_gaps": 2, "prosody_curvature": 4.0, "speech_rate_variability": 1.2, "pause_duration_entropy": 0.2, "intensity_drift": 0.5, "response_latency": 120},
    # Class 4: Disengaged Monotone
    {"class": 4, "avg_level": 70.0,  "speech_ratio": 0.45, "num_gaps": 5, "prosody_curvature": 2.0, "speech_rate_variability": 0.3, "pause_duration_entropy": 0.2, "intensity_drift": 0.0, "response_latency": 600},
    {"class": 4, "avg_level": 60.0,  "speech_ratio": 0.40, "num_gaps": 6, "prosody_curvature": 1.5, "speech_rate_variability": 0.2, "pause_duration_entropy": 0.1, "intensity_drift": -0.05, "response_latency": 800},
    {"class": 4, "avg_level": 80.0,  "speech_ratio": 0.50, "num_gaps": 4, "prosody_curvature": 2.5, "speech_rate_variability": 0.4, "pause_duration_entropy": 0.3, "intensity_drift": 0.02, "response_latency": 500},
]


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_parameters_5(train_data):
    """Compute triangular MF parameters from 5-class training data."""
    class_data = {i: [] for i in range(N_CLASSES)}
    for sample in train_data:
        class_data[sample["class"]].append(sample)

    for ci in range(N_CLASSES):
        if len(class_data[ci]) == 0:
            print(f"  WARNING: No data for '{CLASSES_5[ci]}'. Using defaults.")
            return DEFAULT_PARAMS_5.copy()

    params = {}
    for feat_idx, (feat_name, class_map, floor_val, ceil_val, is_ratio) in enumerate(FEATURE_CONFIG_5):
        term_stats = []
        for term_idx in range(3):
            ci = class_map[term_idx]
            values = [s[feat_name] for s in class_data[ci] if feat_name in s]
            if not values:
                return DEFAULT_PARAMS_5.copy()
            term_stats.append((min(values), sum(values) / len(values), max(values)))

        lo_min, lo_mean, lo_max = term_stats[0]
        md_min, md_mean, md_max = term_stats[1]
        hi_min, hi_mean, hi_max = term_stats[2]

        min_margin = 0.05 if is_ratio else 5.0
        margin_1 = max(abs(md_mean - lo_mean) * 0.5, min_margin)
        margin_2 = max(abs(hi_mean - md_mean) * 0.5, min_margin)

        if not is_ratio:
            ceil_val = max(ceil_val, hi_max * 1.5)

        low_a, low_b, low_c = floor_val, lo_mean, lo_max + margin_1
        med_a, med_b, med_c = md_min - margin_1, md_mean, md_max + margin_2
        high_a, high_b, high_c = hi_min - margin_2, hi_mean, ceil_val

        # Clamp
        low_a = max(floor_val, low_a); low_c = min(ceil_val, low_c)
        med_a = max(floor_val, med_a); med_c = min(ceil_val, med_c)
        high_a = max(floor_val, high_a); high_c = min(ceil_val, high_c)

        # Ensure overlap between adjacent MFs to eliminate dead zones.
        # LOW.c must overlap with MED.a, MED.c must overlap with HIGH.a.
        overlap_margin = 0.02 if is_ratio else 2.0
        low_c = max(low_c, med_a - overlap_margin)
        med_a = min(med_a, low_c + overlap_margin)
        med_c = max(med_c, high_a - overlap_margin)
        high_a = min(high_a, med_c + overlap_margin)

        low_b = max(low_a, min(low_b, low_c))
        med_b = max(med_a, min(med_b, med_c))
        high_b = max(high_a, min(high_b, high_c))

        def rnd(v):
            return round(v, 2) if is_ratio else round(v)

        names = PARAM_NAMES_5[feat_idx]
        params[names[0]] = (rnd(low_a), rnd(low_b), rnd(low_c))
        params[names[1]] = (rnd(med_a), rnd(med_b), rnd(med_c))
        params[names[2]] = (rnd(high_a), rnd(high_b), rnd(high_c))

    return params


def optimize_rule_weights(train_data, params, n_iterations=1500):
    """Optimize per-rule weights via random search (1500 iterations for 13-dim space)."""
    n_rules = 13
    best_weights = [1.0] * n_rules
    best_acc = _evaluate_accuracy(train_data, params, best_weights)

    for _ in range(n_iterations):
        candidate = best_weights.copy()
        idx = random.randint(0, n_rules - 1)
        candidate[idx] = round(random.uniform(0.2, 1.8), 2)
        acc = _evaluate_accuracy(train_data, params, candidate)
        if acc > best_acc:
            best_acc = acc
            best_weights = candidate

    return best_weights, best_acc


def _evaluate_accuracy(data, params, rule_weights):
    """Compute accuracy of 5-class classifier on data."""
    correct = 0
    for s in data:
        feats = {k: s[k] for k in FEATURE_NAMES_8 if k in s}
        idx, _, _ = fuzzy_classify_5(feats, params, rule_weights)
        if idx == s["class"]:
            correct += 1
    return correct / len(data) if data else 0


def validate_5(test_data, params, rule_weights=None):
    """Validate 5-class classifier. Returns (accuracy, confusion, results)."""
    confusion = [[0] * N_CLASSES for _ in range(N_CLASSES)]
    results = []
    correct = 0
    for sample in test_data:
        feats = {k: sample[k] for k in FEATURE_NAMES_8 if k in sample}
        pred_idx, confidence, details = fuzzy_classify_5(feats, params, rule_weights)
        actual_idx = sample["class"]
        confusion[actual_idx][pred_idx] += 1
        is_correct = (pred_idx == actual_idx)
        if is_correct:
            correct += 1
        results.append({
            "actual": CLASSES_5[actual_idx], "predicted": CLASSES_5[pred_idx],
            "correct": is_correct, "confidence": confidence,
            "features": feats, "details": details,
        })
    accuracy = correct / len(test_data) if test_data else 0
    return accuracy, confusion, results


def train_test_split_5(data, train_ratio=TRAIN_RATIO):
    """Stratified, randomized train/test split for 5 classes."""
    by_class = {i: [] for i in range(N_CLASSES)}
    for sample in data:
        by_class[sample["class"]].append(sample)
    train_data, test_data = [], []
    for ci in range(N_CLASSES):
        samples = by_class[ci][:]
        random.shuffle(samples)
        n_train = max(1, int(len(samples) * train_ratio))
        train_data.extend(samples[:n_train])
        test_data.extend(samples[n_train:])
    return train_data, test_data


def loocv_accuracy(data, params_fn, n_classes=N_CLASSES):
    """
    Leave-One-Out Cross-Validation for small datasets (<20 samples).
    Trains on all-but-one, tests on the held-out sample.
    Returns overall accuracy.
    """
    correct = 0
    for i in range(len(data)):
        train = data[:i] + data[i+1:]
        test_sample = data[i]
        p = params_fn(train)
        feats = {k: test_sample[k] for k in FEATURE_NAMES_8 if k in test_sample}
        idx, _, _ = fuzzy_classify_5(feats, p)
        if idx == test_sample["class"]:
            correct += 1
    return correct / len(data) if data else 0


def compute_cohens_kappa(confusion, n_classes=N_CLASSES):
    """Compute Cohen's Kappa from a confusion matrix."""
    n = sum(sum(row) for row in confusion)
    if n == 0:
        return 0.0
    p_o = sum(confusion[i][i] for i in range(n_classes)) / n
    p_e = sum(
        (sum(confusion[i][j] for j in range(n_classes)) *
         sum(confusion[j][i] for j in range(n_classes)))
        for i in range(n_classes)
    ) / (n * n)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def compute_ambiguity_rate(results):
    """Compute the percentage of classifications flagged as ambiguous."""
    if not results:
        return 0.0
    ambiguous = sum(1 for r in results if r["details"].get("is_ambiguous", False))
    return ambiguous / len(results)


# ============================================================
# SAVE / REPORT
# ============================================================

def save_params_5(params, rule_weights, accuracy, filepath):
    """Save trained 5-class parameters to JSON."""
    with open(filepath, 'w') as f:
        json.dump({
            "trained_at": datetime.now().isoformat(),
            "accuracy": accuracy,
            "n_classes": N_CLASSES,
            "rule_weights": rule_weights,
            "parameters": {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in params.items()},
        }, f, indent=2)
    print(f"  ✅ AI parameters saved.")


def save_report_5(accuracy, confusion, results, params, rule_weights, filepath):
    """Save 5-class validation report to text file."""
    with open(filepath, 'w') as f:
        f.write("FUZZY LOGIC VOICE CLASSIFIER — 5-CLASS VALIDATION REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"{'=' * 60}\n\n")
        n_test = len(results)
        n_correct = sum(1 for r in results if r["correct"])
        f.write(f"Test samples: {n_test}\n")
        f.write(f"Overall Accuracy: {accuracy * 100:.1f}% ({n_correct}/{n_test})\n\n")
        f.write("Confusion Matrix:\n")
        header = " " * 22
        for cls in CLASSES_5:
            header += f"{cls[:8]:>10}"
        f.write(header + "\n")
        for i, cls in enumerate(CLASSES_5):
            row = f"Actual {cls[:16]:<16}"
            for j in range(N_CLASSES):
                row += f"{confusion[i][j]:>10}"
            f.write(row + "\n")
        f.write(f"\n{'=' * 60}\nTrained Parameters:\n\n")
        for name in sorted(params.keys()):
            v = params[name]
            f.write(f"  {name:<15} = {v}\n")
        f.write(f"\n  Rule Weights = {rule_weights}\n")
    print(f"  📝 Detailed report saved.")


def print_report_5(accuracy, confusion, results, params, rule_weights):
    """Print 5-class validation report to terminal."""
    display_training_summary(accuracy, confusion, N_CLASSES)

    misses = [r for r in results if not r["correct"]]
    if misses:
        print(f"\n  The AI got {len(misses)} wrong:")
        for r in misses[:5]:
            print(f"    Was '{r['actual']}', but AI guessed '{r['predicted']}'")
    else:
        print("\n  🎯 Perfect! No mistakes at all!")
    print()


# ============================================================
# MICRO:BIT PUSH
# ============================================================

def push_params_to_microbit(params, rule_weights):
    """Send trained parameters to micro:bit if connected."""
    if not HAS_SERIAL:
        print("  [i] pyserial not installed. Cannot push to micro:bit.")
        return

    port = get_microbit_port()
    if not port:
        print("  [i] micro:bit not found. Cannot push parameters.")
        return

    try:
        print("  Sending trained parameters to micro:bit...")
        ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        # Build param string — keep backward compat format for micro:bit.
        # The micro:bit runs a 3-class standalone classifier, so we send
        # the subset of params it understands. We use a fixed R6_WEIGHT
        # of 0.6 (the 3-class default) since the 5-class rule weights
        # have different semantic meaning and should not be transferred.
        R6_WEIGHT_3CLASS = 0.6
        parts = [str(R6_WEIGHT_3CLASS)]
        keys_3class = [
            "LEVEL_LOW", "LEVEL_MED", "LEVEL_HIGH",
            "RATIO_LOW", "RATIO_MED", "RATIO_HIGH",
            "GAPS_FEW", "GAPS_SOME", "GAPS_MANY",
        ]
        # Use 5-class params where available, else skip
        for k in keys_3class:
            if k in params:
                v = params[k]
                a, b, c = (v[0], v[1], v[2]) if isinstance(v, (list, tuple)) else (0, 0, 0)
                parts.extend([str(a), str(b), str(c)])
        # Add VAR params from defaults for micro:bit backward compat
        for k in ["VAR_LOW", "VAR_MED", "VAR_HIGH"]:
            parts.extend(["0", "0", "0"])

        param_str = ",".join(parts)
        msg = f"CMD_SAVE_PARAMS:{param_str}\n"
        for i in range(0, len(msg), 32):
            ser.write(msg[i:i+32].encode('utf-8'))
            ser.flush()
            time.sleep(0.05)
        ser.close()
        print("  ✅ micro:bit updated!")
    except Exception as e:
        print(f"  ⚠️ Could not update micro:bit. It may be disconnected.")


# ============================================================
# CALLABLE API FOR START_HERE.py
# ============================================================

def run_training(data_path, data_dir):
    """
    Run the full 5-class training pipeline on calibration data.

    Args:
        data_path: Path to calibration_data.json
        data_dir: Directory to save outputs

    Returns:
        Path to trained_params.json on success, None on failure.
    """
    os.makedirs(data_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  TRAINING 5-CLASS FUZZY LOGIC CLASSIFIER")
    print("=" * 60)

    # Load data
    try:
        with open(data_path, 'r') as f:
            obj = json.load(f)
        data = obj["samples"]
        print(f"\n  Loaded {len(data)} voice samples.")
    except Exception as e:
        print(f"  ❌ Could not load voice data. Try recording again.")
        return None

    if not data:
        print("  ❌ No samples found in calibration data.")
        return None

    # Upgrade data to 8 features if legacy variance is present
    for sample in data:
        if "speech_rate_variability" not in sample and "variance" in sample:
            sample["speech_rate_variability"] = sample["variance"] / 500.0
        if "pause_duration_entropy" not in sample:
            sample["pause_duration_entropy"] = sample.get("num_gaps", 0) * 0.1

    # Check if data has 5-class features or legacy 4-feature format
    has_8_features = all(
        feat in data[0] for feat in FEATURE_NAMES_8
    ) if data else False

    if not has_8_features:
        print("  ⚠️  Data uses legacy 4-feature format.")
        print("     Training with default 5-class parameters.")
        trained_params = DEFAULT_PARAMS_5.copy()
        rule_weights = [1.0] * 13
        accuracy = 0.0
        confusion = None
        results = []
    else:
        try:
            class_counts = [0] * N_CLASSES
            for sample in data:
                ci = sample.get("class", 0)
                if ci < N_CLASSES:
                    class_counts[ci] += 1
            print(f"\n  Data summary:")
            for i, cls in enumerate(CLASSES_5):
                print(f"    {cls}: {class_counts[i]} samples")

            train_data, test_data = train_test_split_5(data)
            print(f"\n  Training set: {len(train_data)} samples")
            print(f"  Test set:     {len(test_data)} samples")

            print("\n  Training membership functions...")
            trained_params = train_parameters_5(train_data)

            print("  Optimizing rule weights...")
            rule_weights, train_acc = optimize_rule_weights(train_data, trained_params)
            print(f"    Learning accuracy so far: {train_acc * 100:.1f}%")

            accuracy, confusion, results = validate_5(test_data, trained_params, rule_weights)
            print_report_5(accuracy, confusion, results, trained_params, rule_weights)
        except Exception as e:
            print(f"\n  ❌ Training pipeline error: {e}")
            print("     Falling back to default parameters.")
            trained_params = DEFAULT_PARAMS_5.copy()
            rule_weights = [1.0] * 13
            accuracy = 0.0
            confusion = None
            results = []

    # Save outputs
    params_path = os.path.join(data_dir, "trained_params.json")
    report_path = os.path.join(data_dir, "validation_report.txt")
    save_params_5(trained_params, rule_weights, accuracy, params_path)

    if has_8_features and confusion is not None:
        save_report_5(accuracy, confusion, results, trained_params, rule_weights, report_path)

    # Push to micro:bit
    push_params_to_microbit(trained_params, rule_weights)

    if accuracy >= 0.80:
        print(f"\n  🌟 Great result! AI accuracy: {accuracy * 100:.1f}%")
    elif has_8_features:
        print(f"\n  ⚠️ AI accuracy is {accuracy * 100:.1f}% — could be better!")
        print("    Try recording more voice samples to help the AI learn.")

    return params_path


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train & Validate 5-Class Fuzzy Voice Classifier")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predefined", action="store_true", help="Use synthetic 5-class test data")
    group.add_argument("--load", type=str, metavar="FILE", help="Load data from JSON file")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    if args.predefined:
        data_path = os.path.join(data_dir, "calibration_data.json")
        os.makedirs(data_dir, exist_ok=True)
        with open(data_path, 'w') as f:
            json.dump({"classes": CLASSES_5, "samples": PREDEFINED_DATA_5}, f, indent=2)
        run_training(data_path, data_dir)
    else:
        run_training(args.load, data_dir)
