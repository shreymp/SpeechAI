#!/usr/bin/env python3
"""
Fuzzy Logic Voice Mood Classifier — Single Entry Point
=======================================================
Run this ONE script to do everything:
  1. Calibrate (if needed)
  2. Train (if needed)
  3. Launch the classification server

Usage:
  python START_HERE.py
"""

import os
import sys
import json
import logging

from src.fuzzy_engine import CLASSES_5

# ============================================================
# CONFIGURATION
# ============================================================

# Configure logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

CALIBRATION_DATA = os.path.join(DATA_DIR, "calibration_data.json")
TRAINED_PARAMS = os.path.join(DATA_DIR, "trained_params.json")

# Ensure the base directory is in the path for module imports
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ============================================================
# STATE MANAGEMENT
# ============================================================

class ProjectState:
    """Manages the state of calibration and training data."""

    @staticmethod
    def _load_json(filepath):
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"  [!] Warning: Data file corrupted. Please clear data and restart.")
            return None
        except Exception as e:
            logger.error(f"  [✗] Error: Failed to read saved data. Please check file permissions or clear data.")
            return None

    @staticmethod
    def _save_json(filepath, data):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def get_calibration_data(cls):
        return cls._load_json(CALIBRATION_DATA)

    @classmethod
    def get_trained_params(cls):
        return cls._load_json(TRAINED_PARAMS)

    @classmethod
    def has_sufficient_calibration(cls):
        data = cls.get_calibration_data()
        # 5 classes * 3 samples = 15 minimum samples
        return bool(data and len(data.get("samples", [])) >= 15)

    @classmethod
    def has_trained_model(cls):
        data = cls.get_trained_params()
        # Ensure 'parameters' exists AND is not an empty dictionary
        return bool(data and data.get("parameters"))

    @classmethod
    def initialize_empty_files(cls):
        """Create empty but valid JSON files in the data directory."""
        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(CALIBRATION_DATA):
            cls._save_json(CALIBRATION_DATA, {
                "classes": list(CLASSES_5),
                "samples": []
            })
        if not os.path.exists(TRAINED_PARAMS):
            cls._save_json(TRAINED_PARAMS, {
                "trained_at": None,
                "accuracy": 0.0,
                "n_classes": 5,
                "rule_weights": [1.0] * 13,
                "parameters": {}
            })


# ============================================================
# UI HELPERS
# ============================================================

def prompt_user(msg, default="n"):
    """Safely prompt the user, handling Ctrl+C/Ctrl+D gracefully."""
    try:
        response = input(f"{msg} (y/n) [{default}]: ").strip().lower()
        if not response:
            return default == "y"
        return response == 'y'
    except (EOFError, KeyboardInterrupt):
        print("\n\n  Exiting SpeechAI. Goodbye!")
        sys.exit(0)


def print_banner():
    """Print a standardized welcome banner."""
    box_width = 64
    indent = "  "
    banner = f"""
{indent}╔{'═' * box_width}╗
{indent}║{' ' * 64}║
{indent}║{'🎤  AI Voice Mood Detector — Student Edition 🎤':^64}║
{indent}║{' ' * 64}║
{indent}║    Classes: Silence, Confident, Hesitant,{' ' * 23}║
{indent}║             Anxious, Monotone{' ' * 42}║
{indent}║{' ' * 64}║
{indent}╚{'═' * box_width}╝
  IMPORTANT: Flash 'FLASH_TO_MICROBIT.py' to your device
  ONCE before starting. (Reflashing will reset AI brain data!)
"""
    print(banner)


def print_menu():
    """Print the main menu with dynamic status."""
    has_cal = ProjectState.has_sufficient_calibration()
    has_brain = ProjectState.has_trained_model()

    print("  What would you like to do?")
    print("\n    [1] Run Full Pipeline (Capture Data → Train AI → Start Detector)")
    print("        → Recommended for initial setup.")

    print("\n    [2] Add More Voice Samples")
    print("        → Collect additional data to improve AI accuracy.")

    if has_cal:
        print("\n    [3] Retrain AI Brain")
        print("        → Refresh the AI model using existing voice data.")
    else:
        print("\n    [3] [LOCKED] Retrain AI Brain")
        print("        → (Requires Step 1: Voice Data)")

    if has_brain:
        print("\n    [4] Launch Live Detector")
        print("        → Run real-time inference on your voice.")
    else:
        print("\n    [4] [LOCKED] Launch Live Detector")
        print("        → (Requires Step 2: Trained AI Brain)")

    print("\n    [5] Check System Status")
    print("\n    [6] Clear All Data")
    print("\n    [7] Mass Data Collection (Custom Duration)")
    print("\n    [8] Prepare Micro:bit for Standalone Mode")
    print("\n    [q] Quit\n")


def show_status():
    """Print the current status of calibration and training."""
    box_width = 64
    indent = "  "
    
    print(f"\n{indent}System Status:")
    print(f"{indent}{'─' * box_width}")

    # Calibration data
    cal_data = ProjectState.get_calibration_data()
    samples = cal_data.get("samples", []) if cal_data else []
    n_samples = len(samples)

    if n_samples >= 15:
        print(f"{indent}[✓] Voice Data: Found {n_samples} samples.")
    elif n_samples > 0:
        print(f"{indent}[!] Voice Data: Insufficient data ({n_samples} samples).")
    else:
        print(f"{indent}[✗] Voice Data: No data found. Run Option [1] or [2].")

    # Trained params
    train_data = ProjectState.get_trained_params()
    params = train_data.get("parameters", {}) if train_data else {}

    if params:
        acc = train_data.get("accuracy", "?")
        if isinstance(acc, (float, int)):
            acc_val = acc * 100
            acc_str = f"{acc_val:.1f}%"
            if acc_val >= 93.0:
                print(f"{indent}[🌟] AI Brain: Trained (Accuracy: {acc_str})")
            else:
                print(f"{indent}[!] AI Brain: Trained (Accuracy: {acc_str}). Recommendation: Collect samples.")
        else:
            print(f"{indent}[✓] AI Brain: Trained.")
    else:
        print(f"{indent}[✗] AI Brain: Not ready. Run Option [1] or [3] to train.")

    print(f"{indent}{'─' * box_width}\n")


def clear_all_data():
    """Delete all files in the data directory and re-initialize empty data files."""
    box_width = 64
    indent = "  "
    print(f"\n{indent}╔{'═' * box_width}╗")
    print(f"{indent}║{'CLEAR ALL DATA 🧹':^64}║")
    print(f"{indent}╚{'═' * box_width}╝\n")
    print(f"{indent}[!] WARNING: This will permanently delete all voice samples")
    print(f"{indent}    and the trained AI brain. This action cannot be undone.")

    if not prompt_user("  Are you sure you want to delete all data?", default="n"):
        print("  Operation cancelled. Data is safe.")
        return

    deleted_anything = False
    data_files = [
        "calibration_data.json", "calibration_log.txt", "noise_calibration.json",
        "trained_params.json", "validation_report.txt", "session_log.csv",
        "scientific_poster_data.txt"
    ]
    for filename in data_files:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"  [✓] Deleted: {filename}")
                deleted_anything = True
            except Exception as e:
                print(f"  [✗] Error deleting {filename}: {e}")

    if deleted_anything:
        print("\n  [✓] All data cleared successfully.")
    else:
        print("\n  [i] No data files were found to delete.")

    # Re-initialize empty files so the project never operates without valid state
    ProjectState.initialize_empty_files()
    print("  [✓] Empty data files initialized. Ready to record new samples.")


# ============================================================
# PIPELINE STEPS
# ============================================================

def step_mass_record_data():
    """Run the calibration capture process with a custom duration."""
    box_width = 64
    indent = "  "
    print(f"\n{indent}╔{'═' * box_width}╗")
    print(f"{indent}║{'MASS DATA RECORDING 🎙️':^64}║")
    print(f"{indent}╚{'═' * box_width}╝\n")
    print("  This feature allows for high-volume voice data collection.")
    print("  Specify the duration (seconds) per category to be recorded.")
    print("  Example: 60 seconds ≈ 20 samples per category.")
    print()
    print("  Requirements:")
    print("    1. Micro:bit must be connected via USB.\n")

    try:
        time_str = input("  How many seconds per category do you want to record? [default: 30]: ").strip()
        time_sec = int(time_str) if time_str else 30
    except ValueError:
        print("  [!] Invalid input. Defaulting to 30 seconds.")
        time_sec = 30

    if time_sec < 3:
        time_sec = 3

    if not prompt_user(f"  Ready to start recording {time_sec} seconds per category?", default="y"):
        print("  Okay, skipping voice collection for now.")
        return False

    from src.capture import run_capture
    result = run_capture(DATA_DIR, record_time=time_sec)

    if result:
        print(f"\n  [✓] Data collection complete. File saved: {result}")
        return True

    print("\n  [✗] Error: Data collection failed. Please check your connection.")
    return False


def step_calibrate():
    """Run the calibration capture process."""
    box_width = 64
    indent = "  "
    print(f"\n{indent}╔{'═' * box_width}╗")
    print(f"{indent}║{'STEP 1: COLLECTING VOICE DATA 🎙️':^64}║")
    print(f"{indent}╚{'═' * box_width}╝\n")
    print("  Before we start listening, make sure:")
    print("    1. Your micro:bit is plugged into this computer via USB.")
    print("    2. Close any other code windows or Serial tools.")
    print("    3. You will need to press Button A on the micro:bit to start recording each mood.\n")

    if not prompt_user("  Ready to start recording? (Press Button A on the device when prompted)", default="y"):
        print("  Okay, skipping voice collection for now.")
        return False

    # Lazy import to avoid loading heavy dependencies immediately
    from src.capture import run_capture

    result = run_capture(DATA_DIR, record_time=15) # 15s for 5 classes
    if result:
        print(f"\n  [✓] Data collection complete. File saved: {result}")
        return True

    print("\n  [✗] Error: Failed to extract voice data. Please try recording again.")
    return False


def step_train(_retry_depth=0):
    """Run the training pipeline."""
    box_width = 64
    indent = "  "
    print(f"\n{indent}╔{'═' * box_width}╗")
    print(f"{indent}║{'STEP 2: TEACHING THE AI 🧠':^64}║")
    print(f"{indent}╚{'═' * box_width}╝\n")

    # Verify calibration data exists and has sufficient samples
    cal_data = ProjectState.get_calibration_data()
    if not cal_data:
        print(f"\n  [✗] Error: No voice data found. Please complete Step 1 first.")
        return False

    samples = cal_data.get("samples", [])
    if len(samples) < 15:
        print(f"\n  [✗] Error: Insufficient voice data ({len(samples)} samples, need ≥15).")
        return False

    print(f"  Found {len(samples)} voice samples across {len(CLASSES_5)} classes.")

    from src.trainer import run_training

    print("\n  Training AI Brain in progress... Please wait.")
    result = run_training(CALIBRATION_DATA, DATA_DIR)
    if result:
        print(f"\n  [✓] AI training complete.")

        # Check accuracy to guide students
        train_data = ProjectState.get_trained_params()
        if train_data and "accuracy" in train_data:
            acc = train_data["accuracy"]
            if isinstance(acc, (int, float)):
                acc_val = acc * 100
                if acc_val < 93.0:
                    print(f"\n  [!] AI accuracy is {acc_val:.1f}% (target: ≥93.0%).")
                    print("      More voice samples usually improve accuracy.")

                    if _retry_depth >= 2:
                        print("      Maximum retries reached. You can add more samples")
                        print("      later using Option [2] or Option [7].")
                    else:
                        print()
                        if prompt_user("  Record additional samples now to improve accuracy?", default="y"):
                            print("\n  Initiating additional collection...")
                            if step_calibrate():
                                return step_train(_retry_depth=_retry_depth + 1)
                            print("  Additional collection did not complete. Proceeding with current model.\n")
                else:
                    print(f"\n  [🌟] High Accuracy achieved: {acc_val:.1f}%. Excellent!")

        return True

    print("\n  [✗] Error: AI training failed.")
    print("      Check that calibration_data.json is valid and has complete feature data.")
    return False


def step_serve():
    """Start the classification server."""
    box_width = 64
    indent = "  "
    print(f"\n{indent}╔{'═' * box_width}╗")
    print(f"{indent}║{'STEP 3: LIVE MOOD DETECTION ✨':^64}║")
    print(f"{indent}╚{'═' * box_width}╝\n")
    print("  Before we start the magic, make sure:")
    print("    1. Your micro:bit is plugged into this computer via USB.\n")

    if not ProjectState.has_trained_model():
        print("  [!] Warning: AI Brain not trained. System will use generic settings.")
        print("      Accuracy may be significantly reduced.\n")

    if not prompt_user("  Launch the Live Detector display?", default="y"):
        print("  Action deferred.")
        return

    from src.server import run_server
    print("\n  [🚀] Launching Live Detector... Speak into the Micro:bit to observe results.")
    run_server(TRAINED_PARAMS)


# ============================================================
# FULL PIPELINE
# ============================================================

def run_full_pipeline():
    """Run the complete pipeline: calibrate → train → serve."""
    needs_calibration = not ProjectState.has_sufficient_calibration()
    needs_training = not ProjectState.has_trained_model()

    if needs_calibration:
        print("\n  [i] Insufficient voice data detected. Starting collection...")
        if not step_calibrate():
            return
        needs_training = True

    elif needs_training:
        print("\n  [i] Voice data found. AI model requires training.")
    else:
        print("\n  [✓] System Ready: AI model is already trained.")
        show_status()
        if not prompt_user("  Proceed directly to Live Detector?", default="y"):
            if prompt_user("  Would you like to record new voice samples instead?", default="n"):
                if step_calibrate():
                    needs_training = True
                else:
                    return

    if needs_training:
        if not step_train():
            print("\n  [✗] Pipeline halted: Training failed. Exiting.")
            return

    print("\n  [✓] Pipeline complete. System is ready.\n")

    if prompt_user("  Launch Live Detector now?", default="y"):
        step_serve()




def step_prepare_standalone():
    """Prepare the micro:bit for standalone mode by pushing weights."""
    from src.trainer import push_params_to_microbit
    box_width = 64
    indent = "  "
    print(f"\n{indent}╔{'═' * box_width}╗")
    print(f"{indent}║{'PREPARE MICRO:BIT FOR STANDALONE MODE 🚀':^64}║")
    print(f"{indent}╚{'═' * box_width}╝\n")

    if not ProjectState.has_sufficient_calibration():
        print(f"{indent}[✗] Error: Insufficient voice data. Please complete Step 1.")
        return False

    train_data = ProjectState.get_trained_params()
    if not train_data or "parameters" not in train_data:
        print(f"{indent}[✗] Error: AI brain not found. Please complete Step 2.")
        return False

    print("  Make sure:")
    print("    1. Your micro:bit is plugged into this computer via USB.\n")

    if not prompt_user("  Ready to push the most recent weights to the micro:bit?", default="y"):
        print("  Okay, cancelled.")
        return False

    trained_params = train_data["parameters"]
    rule_weights = train_data.get("rule_weights", [1.0]*13)

    push_params_to_microbit(trained_params, rule_weights)


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    ProjectState.initialize_empty_files()
    print_banner()
    show_status()
    print_menu()

    try:
        while True:
            choice = input("  Enter your choice [1-8, or q to quit]: ").strip().lower()

            if choice == '1':
                run_full_pipeline()
                print("\n  Pipeline action completed. Returning to menu.\n")
                show_status()
                print_menu()

            elif choice == '2':
                if step_calibrate():
                    if not step_train():
                        print("  [i] Training was not completed. Check diagnostics above.\n")
                else:
                    print("  [i] Data collection did not complete.\n")
                show_status()
                print_menu()

            elif choice == '3':
                if ProjectState.has_sufficient_calibration():
                    step_train()
                else:
                    print("  [!] Warning: Insufficient voice data. Run Option [1] or [2] first.\n")
                show_status()
                print_menu()

            elif choice == '4':
                if ProjectState.has_trained_model():
                    step_serve()
                else:
                    print("  [!] Warning: AI brain not trained. Run Option [1] or [3] first.\n")
                show_status()
                print_menu()

            elif choice == '5':
                show_status()
                print_menu()

            elif choice == '6':
                clear_all_data()
                show_status()
                print_menu()

            elif choice == '7':
                if step_mass_record_data():
                    if not step_train():
                        print("  [i] Training was not completed. Check diagnostics above.\n")
                else:
                    print("  [i] Mass data collection did not complete.\n")
                show_status()
                print_menu()

            elif choice == '8':
                step_prepare_standalone()
                show_status()
                print_menu()

            elif choice == 'q':
                print("  Exiting SpeechAI system. Goodbye!")
                break

            else:
                print("  [!] Invalid choice. Please enter 1-8 or q.\n")

    except (EOFError, KeyboardInterrupt):
        print("\n\n  Exiting SpeechAI system. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
