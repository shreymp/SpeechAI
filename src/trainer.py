"""
Fuzzy Logic — Training & Validation Pipeline
=============================================
Trains and validates triangular membership function parameters.

Can be called from run.py via run_training() or standalone via CLI.
"""

import json
import os
import sys
import time
import math
import argparse
from datetime import datetime

# Optional: serial for micro:bit data collection
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

CLASSES = ["Background Noise", "Confident Speaking", "Hesitant Speaking"]
CLASS_INDICES = {name: i for i, name in enumerate(CLASSES)}
SPEECH_THRESHOLD = 40
BAUD_RATE = 115200
RECORD_DURATION_S = 3
SAMPLE_INTERVAL_S = 0.020
SAMPLES_PER_CLASS = 10
TRAIN_RATIO = 0.7

DEFAULT_PARAMS = {
    "LEVEL_LOW": (0, 0, 50), "LEVEL_MED": (20, 70, 130), "LEVEL_HIGH": (80, 160, 255),
    "RATIO_LOW": (0.0, 0.0, 0.25), "RATIO_MED": (0.10, 0.40, 0.70), "RATIO_HIGH": (0.50, 0.80, 1.0),
    "GAPS_FEW": (0, 0, 4), "GAPS_SOME": (2, 6, 12), "GAPS_MANY": (8, 15, 25),
    "VAR_LOW": (0, 0, 500), "VAR_MED": (200, 1000, 2500), "VAR_HIGH": (1500, 3500, 6000),
}

R6_WEIGHT = 0.6

PREDEFINED_DATA = [
    {"class": 0, "avg_level": 5.0, "speech_ratio": 0.02, "num_gaps": 0, "variance": 30.0},
    {"class": 0, "avg_level": 12.0, "speech_ratio": 0.05, "num_gaps": 1, "variance": 80.0},
    {"class": 0, "avg_level": 8.0, "speech_ratio": 0.01, "num_gaps": 0, "variance": 15.0},
    {"class": 0, "avg_level": 20.0, "speech_ratio": 0.08, "num_gaps": 0, "variance": 120.0},
    {"class": 0, "avg_level": 3.0, "speech_ratio": 0.00, "num_gaps": 0, "variance": 10.0},
    {"class": 0, "avg_level": 15.0, "speech_ratio": 0.04, "num_gaps": 1, "variance": 60.0},
    {"class": 0, "avg_level": 10.0, "speech_ratio": 0.03, "num_gaps": 0, "variance": 45.0},
    {"class": 0, "avg_level": 7.0, "speech_ratio": 0.01, "num_gaps": 0, "variance": 25.0},
    {"class": 0, "avg_level": 18.0, "speech_ratio": 0.06, "num_gaps": 1, "variance": 95.0},
    {"class": 0, "avg_level": 25.0, "speech_ratio": 0.10, "num_gaps": 0, "variance": 150.0},
    {"class": 1, "avg_level": 140.0, "speech_ratio": 0.85, "num_gaps": 2, "variance": 800.0},
    {"class": 1, "avg_level": 160.0, "speech_ratio": 0.90, "num_gaps": 1, "variance": 600.0},
    {"class": 1, "avg_level": 120.0, "speech_ratio": 0.78, "num_gaps": 3, "variance": 1000.0},
    {"class": 1, "avg_level": 180.0, "speech_ratio": 0.92, "num_gaps": 1, "variance": 500.0},
    {"class": 1, "avg_level": 130.0, "speech_ratio": 0.82, "num_gaps": 2, "variance": 750.0},
    {"class": 1, "avg_level": 150.0, "speech_ratio": 0.88, "num_gaps": 1, "variance": 650.0},
    {"class": 1, "avg_level": 110.0, "speech_ratio": 0.75, "num_gaps": 3, "variance": 900.0},
    {"class": 1, "avg_level": 170.0, "speech_ratio": 0.91, "num_gaps": 2, "variance": 550.0},
    {"class": 1, "avg_level": 145.0, "speech_ratio": 0.86, "num_gaps": 1, "variance": 700.0},
    {"class": 1, "avg_level": 155.0, "speech_ratio": 0.89, "num_gaps": 2, "variance": 620.0},
    {"class": 2, "avg_level": 65.0, "speech_ratio": 0.35, "num_gaps": 12, "variance": 3200.0},
    {"class": 2, "avg_level": 55.0, "speech_ratio": 0.28, "num_gaps": 15, "variance": 4000.0},
    {"class": 2, "avg_level": 80.0, "speech_ratio": 0.42, "num_gaps": 10, "variance": 2800.0},
    {"class": 2, "avg_level": 45.0, "speech_ratio": 0.22, "num_gaps": 18, "variance": 3800.0},
    {"class": 2, "avg_level": 70.0, "speech_ratio": 0.38, "num_gaps": 11, "variance": 3500.0},
    {"class": 2, "avg_level": 60.0, "speech_ratio": 0.30, "num_gaps": 14, "variance": 4200.0},
    {"class": 2, "avg_level": 75.0, "speech_ratio": 0.40, "num_gaps": 9, "variance": 2500.0},
    {"class": 2, "avg_level": 50.0, "speech_ratio": 0.25, "num_gaps": 16, "variance": 3900.0},
    {"class": 2, "avg_level": 85.0, "speech_ratio": 0.45, "num_gaps": 8, "variance": 2600.0},
    {"class": 2, "avg_level": 58.0, "speech_ratio": 0.32, "num_gaps": 13, "variance": 3600.0},
]


def tri_mf(x, a, b, c):
    """Triangular membership function → [0.0, 1.0]"""
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    else:
        return (c - x) / (c - b) if c != b else 1.0


def left_mf(x, a, b, c):
    """Left shoulder membership function → [0.0, 1.0]"""
    if x <= b:
        return 1.0
    elif x >= c:
        return 0.0
    else:
        return (c - x) / (c - b) if c != b else 1.0


def right_mf(x, a, b, c):
    """Right shoulder membership function → [0.0, 1.0]"""
    if x <= a:
        return 0.0
    elif x >= b:
        return 1.0
    else:
        return (x - a) / (b - a) if b != a else 1.0


def extract_features(samples):
    """Compute features from raw sound level values (0-255)."""
    n = len(samples)
    if n == 0:
        return {"avg_level": 0, "speech_ratio": 0.0, "num_gaps": 0, "variance": 0}
    avg = sum(samples) / n
    var_sum = sum((s - avg) ** 2 for s in samples)
    variance = var_sum / n
    above_count = 0
    num_gaps = 0
    in_gap = True
    for s in samples:
        if s >= SPEECH_THRESHOLD:
            above_count += 1
            if in_gap:
                in_gap = False
        else:
            if not in_gap:
                num_gaps += 1
                in_gap = True
    speech_ratio = above_count / n
    return {"avg_level": avg, "speech_ratio": speech_ratio, "num_gaps": num_gaps, "variance": variance}


def fuzzy_classify(features, params, r6_weight=R6_WEIGHT):
    """Run Mamdani-style fuzzy inference. Returns (class_idx, confidence%, details)."""
    avg = features["avg_level"]
    ratio = features["speech_ratio"]
    gaps = features["num_gaps"]
    var = features["variance"]

    lvl_lo = left_mf(avg, *params["LEVEL_LOW"])
    lvl_md = tri_mf(avg, *params["LEVEL_MED"])
    lvl_hi = right_mf(avg, *params["LEVEL_HIGH"])
    rat_lo = left_mf(ratio, *params["RATIO_LOW"])
    rat_md = tri_mf(ratio, *params["RATIO_MED"])
    rat_hi = right_mf(ratio, *params["RATIO_HIGH"])
    gap_fw = left_mf(gaps, *params["GAPS_FEW"])
    gap_sm = tri_mf(gaps, *params["GAPS_SOME"])
    gap_mn = right_mf(gaps, *params["GAPS_MANY"])
    var_lo = left_mf(var, *params["VAR_LOW"])
    var_md = tri_mf(var, *params["VAR_MED"])
    var_hi = right_mf(var, *params["VAR_HIGH"])

    r1 = min(lvl_lo, rat_lo)
    r6 = min(lvl_lo, rat_md) * r6_weight
    r2 = min(lvl_hi, rat_hi, gap_fw)
    r5 = min(lvl_md, rat_md, var_lo)
    r3 = min(rat_md, gap_mn)
    r4 = min(var_hi, gap_sm)

    scores = [max(r1, r6), max(r2, r5), max(r3, r4)]
    total = sum(scores)
    best_idx = max(range(3), key=lambda i: scores[i])
    confidence = int((scores[best_idx] / total) * 100) if total > 0 else 0

    details = {
        "scores": {CLASSES[i]: round(scores[i], 4) for i in range(3)},
        "rules": {"R1": round(r1, 4), "R2": round(r2, 4), "R3": round(r3, 4),
                  "R4": round(r4, 4), "R5": round(r5, 4), "R6": round(r6, 4)},
        "memberships": {
            "level": {"low": round(lvl_lo, 4), "med": round(lvl_md, 4), "high": round(lvl_hi, 4)},
            "ratio": {"low": round(rat_lo, 4), "med": round(rat_md, 4), "high": round(rat_hi, 4)},
            "gaps": {"few": round(gap_fw, 4), "some": round(gap_sm, 4), "many": round(gap_mn, 4)},
            "var": {"low": round(var_lo, 4), "med": round(var_md, 4), "high": round(var_hi, 4)},
        },
        "confidence": confidence,
    }
    return best_idx, confidence, details


def train_parameters(train_data):
    """Compute triangular MF parameters from training data."""
    class_data = {0: [], 1: [], 2: []}
    for sample in train_data:
        class_data[sample["class"]].append(sample)

    for ci in range(3):
        if len(class_data[ci]) == 0:
            print(f"  WARNING: No data for '{CLASSES[ci]}'. Using defaults.")
            return DEFAULT_PARAMS.copy()

    feature_config = [
        ("avg_level", [0, 2, 1], 0, 255, False),
        ("speech_ratio", [0, 2, 1], 0.0, 1.0, True),
        ("num_gaps", [1, 0, 2], 0, 25, False),
        ("variance", [0, 1, 2], 0, 6000, False),
    ]
    param_names = [
        ["LEVEL_LOW", "LEVEL_MED", "LEVEL_HIGH"],
        ["RATIO_LOW", "RATIO_MED", "RATIO_HIGH"],
        ["GAPS_FEW", "GAPS_SOME", "GAPS_MANY"],
        ["VAR_LOW", "VAR_MED", "VAR_HIGH"],
    ]

    params = {}
    for feat_idx, (feat_name, class_map, floor_val, ceil_val, is_ratio) in enumerate(feature_config):
        term_stats = []
        for term_idx in range(3):
            ci = class_map[term_idx]
            values = [s[feat_name] for s in class_data[ci]]
            term_stats.append((min(values), sum(values) / len(values), max(values)))

        lo_min, lo_mean, lo_max = term_stats[0]
        md_min, md_mean, md_max = term_stats[1]
        hi_min, hi_mean, hi_max = term_stats[2]

        min_margin = 0.05 if is_ratio else 5.0
        margin_1 = max(abs(md_mean - lo_mean) * 0.3, min_margin)
        margin_2 = max(abs(hi_mean - md_mean) * 0.3, min_margin)

        if not is_ratio:
            ceil_val = max(ceil_val, hi_max * 1.5)

        low_a, low_b, low_c = floor_val, lo_mean, lo_max + margin_1
        med_a, med_b, med_c = md_min - margin_1, md_mean, md_max + margin_2
        high_a, high_b, high_c = hi_min - margin_2, hi_mean, ceil_val

        # Clamp and enforce a <= b <= c
        for vals in [(low_a, low_b, low_c), (med_a, med_b, med_c), (high_a, high_b, high_c)]:
            pass
        low_a = max(floor_val, low_a); low_c = min(ceil_val, low_c)
        med_a = max(floor_val, med_a); med_c = min(ceil_val, med_c)
        high_a = max(floor_val, high_a); high_c = min(ceil_val, high_c)
        low_b = max(low_a, min(low_b, low_c))
        med_b = max(med_a, min(med_b, med_c))
        high_b = max(high_a, min(high_b, high_c))

        def rnd(v):
            return round(v, 2) if is_ratio else round(v)

        names = param_names[feat_idx]
        params[names[0]] = (rnd(low_a), rnd(low_b), rnd(low_c))
        params[names[1]] = (rnd(med_a), rnd(med_b), rnd(med_c))
        params[names[2]] = (rnd(high_a), rnd(high_b), rnd(high_c))

    return params


def optimize_r6_weight(train_data, params):
    """Grid search for best R6 weight."""
    best_weight, best_acc = R6_WEIGHT, 0.0
    for weight in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        correct = sum(1 for s in train_data
                      if fuzzy_classify({k: s[k] for k in ["avg_level", "speech_ratio", "num_gaps", "variance"]},
                                        params, r6_weight=weight)[0] == s["class"])
        acc = correct / len(train_data) if train_data else 0
        if acc > best_acc:
            best_acc, best_weight = acc, weight
    return best_weight, best_acc


def validate(test_data, params, r6_weight=R6_WEIGHT):
    """Validate classifier on test data. Returns (accuracy, confusion_matrix, results)."""
    confusion = [[0] * 3 for _ in range(3)]
    results = []
    correct = 0
    for sample in test_data:
        features = {k: sample[k] for k in ["avg_level", "speech_ratio", "num_gaps", "variance"]}
        pred_idx, confidence, details = fuzzy_classify(features, params, r6_weight)
        actual_idx = sample["class"]
        confusion[actual_idx][pred_idx] += 1
        is_correct = (pred_idx == actual_idx)
        if is_correct:
            correct += 1
        results.append({
            "actual": CLASSES[actual_idx], "predicted": CLASSES[pred_idx],
            "correct": is_correct, "confidence": confidence,
            "features": features, "details": details,
        })
    accuracy = correct / len(test_data) if test_data else 0
    return accuracy, confusion, results


def train_test_split(data, train_ratio=TRAIN_RATIO):
    """Stratified train/test split."""
    by_class = {0: [], 1: [], 2: []}
    for sample in data:
        by_class[sample["class"]].append(sample)
    train_data, test_data = [], []
    for ci in range(3):
        samples = by_class[ci]
        n_train = max(1, int(len(samples) * train_ratio))
        train_data.extend(samples[:n_train])
        test_data.extend(samples[n_train:])
    return train_data, test_data


def save_params(params, r6_weight, accuracy, filepath):
    """Save trained parameters to JSON."""
    with open(filepath, 'w') as f:
        json.dump({
            "trained_at": datetime.now().isoformat(),
            "accuracy": accuracy,
            "r6_weight": r6_weight,
            "parameters": {k: list(v) for k, v in params.items()},
        }, f, indent=2)
    print(f"  Parameters saved to {filepath}")


def save_report(accuracy, confusion, results, params, r6_weight, filepath):
    """Save validation report to text file."""
    with open(filepath, 'w') as f:
        f.write("FUZZY LOGIC VOICE CLASSIFIER — VALIDATION REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"{'=' * 60}\n\n")
        n_test = len(results)
        n_correct = sum(1 for r in results if r["correct"])
        f.write(f"Test samples: {n_test}\n")
        f.write(f"Overall Accuracy: {accuracy * 100:.1f}% ({n_correct}/{n_test})\n")
        f.write(f"R6 Weight: {r6_weight}\n\n")
        f.write("Confusion Matrix:\n")
        header = " " * 18
        for cls in CLASSES:
            header += f"{cls[:6]:>8}"
        f.write(header + "\n")
        for i, cls in enumerate(CLASSES):
            row = f"Actual {cls[:14]:<14}"
            for j in range(3):
                row += f"{confusion[i][j]:>8}"
            f.write(row + "\n")
        f.write("\nPer-class Metrics:\n")
        for i, cls in enumerate(CLASSES):
            tp = confusion[i][i]
            fp = sum(confusion[j][i] for j in range(3)) - tp
            fn = sum(confusion[i][j] for j in range(3)) - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            f.write(f"  {cls:<22} P={p:.2f}  R={r:.2f}  F1={f1:.2f}\n")
        f.write(f"\n{'=' * 60}\nTrained Parameters:\n\n")
        for name, (a, b, c) in sorted(params.items()):
            f.write(f"  {name:<13} = ({a}, {b}, {c})\n")
        f.write(f"\n  R6_WEIGHT = {r6_weight}\n")
    print(f"  Report saved to {filepath}")


def print_report(accuracy, confusion, results, params, r6_weight):
    """Print validation report to terminal."""
    print("\n" + "=" * 60)
    print("  VALIDATION REPORT")
    print("=" * 60)
    n_test = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    print(f"\n  Test samples: {n_test}")
    print(f"  Overall Accuracy: {accuracy * 100:.1f}% ({n_correct}/{n_test})")
    print(f"  R6 Weight: {r6_weight}")
    print("\n  Confusion Matrix:")
    print("  " + " " * 22 + "Predicted")
    header = "  " + " " * 18
    for cls in CLASSES:
        header += f"{cls[:6]:>8}"
    print(header)
    for i, cls in enumerate(CLASSES):
        row = f"  Actual {cls[:14]:<14}"
        for j in range(3):
            row += f"{confusion[i][j]:>8}"
        print(row)
    misses = [r for r in results if not r["correct"]]
    if misses:
        print(f"\n  Misclassified ({len(misses)}):")
        for r in misses:
            f = r["features"]
            print(f"    actual={r['actual']}, predicted={r['predicted']}")
            print(f"      avg={f['avg_level']:.1f}, ratio={f['speech_ratio']:.2f}, "
                  f"gaps={f['num_gaps']}, var={f['variance']:.0f}")
    else:
        print("\n  No misclassifications!")
    print("\n" + "=" * 60)


# ============================================================
# CALLABLE API FOR run.py
# ============================================================

def run_training(data_path, data_dir):
    """
    Run the full training pipeline on calibration data.

    Args:
        data_path: Path to calibration_data.json
        data_dir: Directory to save outputs (trained_params.json, etc.)

    Returns:
        Path to trained_params.json on success, None on failure.
    """
    os.makedirs(data_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  TRAINING FUZZY LOGIC CLASSIFIER")
    print("=" * 60)

    # Load data
    try:
        with open(data_path, 'r') as f:
            obj = json.load(f)
        data = obj["samples"]
        print(f"\n  Loaded {len(data)} samples from {data_path}")
    except Exception as e:
        print(f"  ERROR loading data: {e}")
        return None

    # Show data summary
    class_counts = [0, 0, 0]
    for sample in data:
        class_counts[sample["class"]] += 1
    print(f"\n  Data summary:")
    for i, cls in enumerate(CLASSES):
        print(f"    {cls}: {class_counts[i]} samples")

    # Train/test split
    train_data, test_data = train_test_split(data)
    print(f"\n  Training set: {len(train_data)} samples")
    print(f"  Test set:     {len(test_data)} samples")

    # Train
    print("\n  Training membership functions...")
    trained_params = train_parameters(train_data)
    print("  Trained parameters:")
    for name, (a, b, c) in sorted(trained_params.items()):
        print(f"    {name:<13} = ({a}, {b}, {c})")

    # Optimize R6
    print("\n  Optimizing R6 weight...")
    best_r6, train_acc = optimize_r6_weight(train_data, trained_params)
    print(f"    Best R6 weight: {best_r6}")
    print(f"    Training accuracy: {train_acc * 100:.1f}%")

    # Validate
    accuracy, confusion, results = validate(test_data, trained_params, best_r6)
    print_report(accuracy, confusion, results, trained_params, best_r6)

    # Save outputs
    params_path = os.path.join(data_dir, "trained_params.json")
    report_path = os.path.join(data_dir, "validation_report.txt")
    save_params(trained_params, best_r6, accuracy, params_path)
    save_report(accuracy, confusion, results, trained_params, best_r6, report_path)
    
    if HAS_SERIAL:
        def get_microbit_port():
            ports = list(serial.tools.list_ports.comports())
            for p in ports:
                # Check VID/PID (0x0D28:0x0204 is the standard micro:bit identifier)
                if p.vid == 0x0D28 and p.pid == 0x0204:
                    return p.device
                # Check description for common micro:bit/mbed strings
                desc = p.description.lower()
                if "micro:bit" in desc or "mbed" in desc:
                    return p.device
                # Check device name (Mac /dev/cu.usbmodemXXXX)
                if "usbmodem" in p.device.lower():
                    return p.device
            return None
            
        port = get_microbit_port()
        if port:
            try:
                print("  Sending trained parameters to micro:bit...")
                ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
                time.sleep(2)
                # Format: R6, LL1,LL2,LL3, LM1...
                parts = [str(best_r6)]
                keys = [
                    "LEVEL_LOW", "LEVEL_MED", "LEVEL_HIGH",
                    "RATIO_LOW", "RATIO_MED", "RATIO_HIGH",
                    "GAPS_FEW", "GAPS_SOME", "GAPS_MANY",
                    "VAR_LOW", "VAR_MED", "VAR_HIGH"
                ]
                for k in keys:
                    a, b, c = trained_params[k]
                    parts.extend([str(a), str(b), str(c)])
                param_str = ",".join(parts)
                msg = f"CMD_SAVE_PARAMS:{param_str}\n"
                for i in range(0, len(msg), 32):
                    ser.write(msg[i:i+32].encode('utf-8'))
                    ser.flush()
                    time.sleep(0.05)
                ser.close()
                print("  ✅ micro:bit updated! Standalone mode ready.")
            except Exception as e:
                print(f"  ⚠️ Could not send params to micro:bit: {e}")

    if accuracy >= 0.80:
        print(f"\n  Target accuracy (>=80%) MET: {accuracy * 100:.1f}%")
    else:
        print(f"\n  Target accuracy (>=80%) NOT MET: {accuracy * 100:.1f}%")
        print("    Consider recalibrating with clearer audio samples.")

    return params_path


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train & Validate Fuzzy Voice Mood Classifier")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predefined", action="store_true", help="Use synthetic test data")
    group.add_argument("--load", type=str, metavar="FILE", help="Load data from JSON file")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    if args.predefined:
        # Save predefined data then train on it
        data_path = os.path.join(data_dir, "calibration_data.json")
        os.makedirs(data_dir, exist_ok=True)
        with open(data_path, 'w') as f:
            json.dump({"classes": CLASSES, "samples": PREDEFINED_DATA}, f, indent=2)
        run_training(data_path, data_dir)
    else:
        run_training(args.load, data_dir)
