#!/usr/bin/env python3
"""
Fuzzy Logic Voice Mood Classifier — Single Entry Point
=======================================================
Run this ONE script to do everything:
  1. Calibrate (if needed)
  2. Train (if needed)
  3. Launch the classification server

Usage:
  python run.py
"""

import os
import sys
import json
import logging
import re

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
            logger.warning(f"  ⚠️ Warning: Corrupted JSON file at {filepath}")
            return None
        except Exception as e:
            logger.error(f"  ❌ Error reading {filepath}: {e}")
            return None

    @classmethod
    def get_calibration_data(cls):
        return cls._load_json(CALIBRATION_DATA)

    @classmethod
    def get_trained_params(cls):
        return cls._load_json(TRAINED_PARAMS)

    @classmethod
    def has_sufficient_calibration(cls):
        data = cls.get_calibration_data()
        return bool(data and len(data.get("samples", [])) >= 9)

    @classmethod
    def has_trained_model(cls):
        data = cls.get_trained_params()
        return bool(data and "parameters" in data)


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
        print("\n\n  See you later! 👋")
        sys.exit(0)


def print_banner():
    """Print a nice welcome banner."""
    banner = """
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║  🎤  AI Voice Mood Detector - Student Edition 🎤      ║
║                                                       ║
║   Detects: Background Noise / Confident / Hesitant    ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝
  IMPORTANT: Flash 'FLASH_TO_MICROBIT.py' to your device 
  ONCE before starting. (Reflashing wipes AI weights!)
"""
    print(banner)


def print_menu():
    """Print the main menu with dynamic status."""
    has_cal = ProjectState.has_sufficient_calibration()
    has_brain = ProjectState.has_trained_model()
    
    print("  What would you like to do?")
    print("\n    [1] Do Everything! (Collect Voice → Teach AI → Start Live Detector)")
    print("        → Recommended for first-time use.")
    
    print("\n    [2] Add More Voice Samples")
    print("        → Add more data to make the AI smarter.")
    
    if has_cal:
        print("\n    [3] Re-teach the AI")
        print("        → Run the learning math again using your existing data.")
    else:
        print("\n    [3] [Locked] Re-teach the AI")
        print("        → (Requires Step 1: Voice Data)")
        
    if has_brain:
        print("\n    [4] Start the Live Detector!")
        print("        → See the AI guess your mood in real-time.")
    else:
        print("\n    [4] [Locked] Start the Live Detector!")
        print("        → (Requires Step 2: Trained AI Brain)")
    
    print("\n    [5] Check AI Status")
    print("    [6] Clear All Data")
    print("    [7] Mass Data Recording (Custom Time)")
    print("    [8] Generate Scientific Poster Data")
    print("    [9] Prepare Micro:bit for Standalone Mode")
    print("    [q] Quit\n")


def show_status():
    """Print the current status of calibration and training."""
    print("\n  Current Project Status:")
    print("  " + "-" * 50)

    # Calibration data
    cal_data = ProjectState.get_calibration_data()
    if cal_data:
        n_samples = len(cal_data.get("samples", []))
        print(f"  ✅ Voice Data: We have {n_samples} samples saved.")
    else:
        print("  ❌ Voice Data: Not found. We need to record your voice!")

    # Trained params
    train_data = ProjectState.get_trained_params()
    if train_data and "parameters" in train_data:
        acc = train_data.get("accuracy", "?")
        trained_at = train_data.get("trained_at", "?")
        if isinstance(acc, (float, int)):
            acc_val = acc * 100
            acc_str = f"{acc_val:.1f}%"
            if acc_val >= 93.0:
                print(f"  🌟 AI Brain: Trained! It's super smart (Accuracy: {acc_str})")
            else:
                print(f"  ⚠️ AI Brain: Trained, but it could be smarter (Accuracy: {acc_str}) - Try adding more samples!")
        else:
            print(f"  ✅ AI Brain: Trained!")
    else:
        print("  ❌ AI Brain: The AI hasn't learned from your data yet.")

    print("  " + "-" * 50 + "\n")


def clear_all_data():
    """Delete all files in the data directory."""
    print("\n" + "=" * 60)
    print("  CLEAR ALL DATA 🧹")
    print("=" * 60 + "\n")
    print("  ⚠️ WARNING: This will delete ALL your recorded voice samples")
    print("     and your AI's trained brain. You will have to start completely")
    print("     from scratch.")
    
    if not prompt_user("  Are you ABSOLUTELY sure you want to delete everything?", default="n"):
        print("  Phew! Your data is safe. Cancelled.")
        return
        
    deleted_anything = False
    for filename in ["calibration_data.json", "calibration_log.txt", "trained_params.json", "validation_report.txt"]:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"  🗑️ Deleted {filename}")
                deleted_anything = True
            except Exception as e:
                print(f"  ❌ Error deleting {filename}: {e}")
                
    if deleted_anything:
        print("\n  ✅ All data cleared! You are starting fresh. Rerun START_HERE.py!")
    else:
        print("\n  🤷 No data found to delete! You were already starting fresh.")


# ============================================================
# PIPELINE STEPS
# ============================================================

def step_mass_record_data():
    """Run the calibration capture process with a custom duration."""
    print("\n" + "=" * 60)
    print("  MASS DATA RECORDING 🎙️")
    print("=" * 60 + "\n")
    print("  This allows you to record a large volume of real voice data at once.")
    print("  You will input the number of seconds you want to record per category.")
    print("  (e.g., 60 seconds = 20 samples per category)")
    print()
    print("  Before we start listening, make sure:")
    print("    1. Your micro:bit is plugged into this computer via USB.\n")

    try:
        time_str = input("  How many seconds per category do you want to record? [default: 30]: ").strip()
        time_sec = int(time_str) if time_str else 30
    except ValueError:
        print("  Invalid number. Defaulting to 30 seconds.")
        time_sec = 30
        
    if time_sec < 3:
        time_sec = 3

    if not prompt_user(f"  Ready to start recording {time_sec} seconds per category?", default="y"):
        print("  Okay, skipping voice collection for now.")
        return False

    from src.capture import run_capture
    result = run_capture(DATA_DIR, record_time=time_sec)
    
    if result:
        print(f"\n  ✅ Awesome! Your mass data was saved to: {result}")
        return True
    
    print("\n  ❌ Oops, we couldn't collect your voice. Let's try again.")
    return False


def step_calibrate():
    """Run the calibration capture process."""
    print("\n" + "=" * 60)
    print("  STEP 1: COLLECTING VOICE DATA 🎙️")
    print("=" * 60 + "\n")
    print("  Before we start listening, make sure:")
    print("    1. Your micro:bit is plugged into this computer via USB.")
    print("    2. Close any other code windows or Serial tools.\n")

    if not prompt_user("  Ready to start recording?", default="y"):
        print("  Okay, skipping voice collection for now.")
        return False

    # Lazy import to avoid loading heavy dependencies immediately
    from src.capture import run_capture

    result = run_capture(DATA_DIR)
    if result:
        print(f"\n  ✅ Awesome! Your voice data was saved to: {result}")
        return True
    
    print("\n  ❌ Oops, we couldn't collect your voice. Let's try again.")
    return False


def step_train():
    """Run the training pipeline."""
    print("\n" + "=" * 60)
    print("  STEP 2: TEACHING THE AI 🧠")
    print("=" * 60)

    if not ProjectState.has_sufficient_calibration():
        print("\n  ❌ We don't have enough voice samples yet. Let's do Step 1 first!")
        return False

    from src.trainer import run_training

    print("\n  The AI is crunching the numbers... Please wait a moment.")
    result = run_training(CALIBRATION_DATA, DATA_DIR)
    if result:
        print(f"\n  ✅ Success! The AI has finished learning from your voice.")
        
        # Check accuracy to guide students
        train_data = ProjectState.get_trained_params()
        if train_data and "accuracy" in train_data:
            acc = train_data["accuracy"]
            if isinstance(acc, (int, float)):
                acc_val = acc * 100
                if acc_val < 93.0:
                    print(f"\n  ⚠️  Hmm, the AI's accuracy is only {acc_val:.1f}%.")
                    print("      That's a bit low! (We want at least 93.0% for the best results)")
                    print("      The AI might get confused between your moods.")
                    print("\n  💡  PRO TIP: You can make it much smarter by giving it more voice samples!")
                    
                    if prompt_user("  Would you like to record more samples right now to fix this?", default="y"):
                        print("\n  Great idea! Let's get some more samples...")
                        if step_calibrate():
                            # Re-run training with the new data
                            return step_train()
                        else:
                            return False
                else:
                    print(f"\n  🌟  Wow! The AI's accuracy is {acc_val:.1f}%. That is excellent!")
                    
        return True
    
    print("\n  ❌ Oh no, the AI failed to learn. Let's try again.")
    return False





def step_serve():
    """Start the classification server."""
    print("\n" + "=" * 60)
    print("  STEP 3: LIVE MOOD DETECTION ✨")
    print("=" * 60 + "\n")
    print("  Before we start the magic, make sure:")
    print("    1. Your micro:bit is plugged into this computer via USB.\n")

    if not ProjectState.has_trained_model():
        print("  ⚠️  The AI hasn't been taught your voice yet — it's going to guess using generic settings.")
        print("      (It might not be very accurate!)\n")

    if not prompt_user("  Ready to launch the Live Display?", default="y"):
        print("  Okay, we'll wait.")
        return

    from src.server import run_server
    print("\n  🚀 Starting the Live Display! Talk into the micro:bit to see it work...")
    run_server(TRAINED_PARAMS)


# ============================================================
# FULL PIPELINE
# ============================================================

def run_full_pipeline():
    """Run the complete pipeline: calibrate → train → serve."""
    needs_calibration = not ProjectState.has_sufficient_calibration()
    needs_training = not ProjectState.has_trained_model()

    if needs_calibration:
        print("\n  📋 You don't have enough voice data yet. Let's collect some!")
        if not step_calibrate():
            print("\n  We can't move forward without voice data. See you next time!")
            return
        needs_training = True

    elif needs_training:
        print("\n  📋 Great! You have voice data, but the AI hasn't learned it yet.")
    else:
        print("\n  ✅ Good news: Your AI is already trained and ready to go!")
        show_status()
        if not prompt_user("  Would you like to skip straight to the Live Display?", default="y"):
            if prompt_user("  Would you like to record new voice samples instead?", default="n"):
                if step_calibrate():
                    needs_training = True
                else:
                    return
            else:
                needs_training = True  # We will just re-train

    if needs_training:
        if not step_train():
            print("\n  We can't move forward without a smart AI. See you next time!")
            return

    print("\n  ✅ Everything is perfectly set up!\n")
    
    if prompt_user("  Start the Live Mood Detection?", default="y"):
        step_serve()


def step_generate_poster_data():
    """Compiles all important information for a scientific poster."""
    print("\n" + "=" * 60)
    print("  SCIENTIFIC POSTER DATA COMPILATION 📊")
    print("=" * 60 + "\n")
    
    cal_data = ProjectState.get_calibration_data()
    train_data = ProjectState.get_trained_params()
    
    if not cal_data and not train_data:
        print("  ❌ No data available to compile. Please collect data and train the AI first.")
        return
        
    poster_lines = []
    poster_lines.append("=" * 50)
    poster_lines.append("    AI VOICE MOOD DETECTOR - SCIENTIFIC DATA")
    poster_lines.append("=" * 50 + "\n")
    
    if cal_data:
        classes = cal_data.get("classes", [])
        samples = cal_data.get("samples", [])
        poster_lines.append("1. DATASET OVERVIEW")
        poster_lines.append("-" * 20)
        poster_lines.append(f"Total Samples Collected: {len(samples)}")
        poster_lines.append(f"Classes: {', '.join(classes)}\n")
        
        # Calculate samples per class
        class_counts = {c: 0 for c in range(len(classes))}
        
        # Calculate average features per class
        features = ["avg_level", "speech_ratio", "num_gaps", "variance"]
        class_features = {c: {f: [] for f in features} for c in range(len(classes))}
        
        for s in samples:
            c = s.get("class", 0)
            class_counts[c] = class_counts.get(c, 0) + 1
            for f in features:
                if f in s:
                    class_features[c][f].append(s[f])
                    
        for i, c_name in enumerate(classes):
            poster_lines.append(f"Class: {c_name} ({class_counts.get(i, 0)} samples)")
            if class_counts.get(i, 0) > 0:
                for f in features:
                    vals = class_features[i][f]
                    avg = sum(vals) / len(vals) if vals else 0
                    poster_lines.append(f"  - Avg {f}: {avg:.2f}")
            poster_lines.append("")
            
    if train_data:
        poster_lines.append("2. AI MODEL PERFORMANCE")
        poster_lines.append("-" * 20)
        trained_at = train_data.get("trained_at", "Unknown")
        acc = train_data.get("accuracy", 0)
        
        poster_lines.append(f"Last Trained: {trained_at}")
        if isinstance(acc, (int, float)):
            poster_lines.append(f"Model Accuracy: {acc * 100:.2f}%")
        else:
            poster_lines.append(f"Model Accuracy: {acc}")
        poster_lines.append("")
        
        poster_lines.append("3. FUZZY LOGIC PARAMETERS (MEMBERSHIP FUNCTIONS)")
        poster_lines.append("-" * 20)
        params = train_data.get("parameters", {})
        for param_name, values in params.items():
            if isinstance(values, list) and len(values) == 3:
                poster_lines.append(f"{param_name}: Min={values[0]:.2f}, Peak={values[1]:.2f}, Max={values[2]:.2f}")
            else:
                poster_lines.append(f"{param_name}: {values}")
        poster_lines.append("")
        
    poster_text = "\n".join(poster_lines)
    print(poster_text)
    
    # Save to file
    out_file = os.path.join(DATA_DIR, "scientific_poster_data.txt")
    try:
        with open(out_file, "w") as f:
            f.write(poster_text)
        print(f"\n  ✅ Successfully saved this data to: {out_file}")
    except Exception as e:
        print(f"\n  ❌ Failed to save to file: {e}")


def step_prepare_standalone():
    """Prepare the micro:bit for standalone mode by pushing weights."""
    print("\n" + "=" * 60)
    print("  PREPARE MICRO:BIT FOR STANDALONE MODE 🚀")
    print("=" * 60 + "\n")
    
    if not ProjectState.has_sufficient_calibration():
        print("  ❌ Adequate samples are not present. Please record voice data first (Step 1).")
        return False
        
    train_data = ProjectState.get_trained_params()
    if not train_data or "parameters" not in train_data:
        print("  ❌ AI Brain not found. Please train the AI first (Step 2).")
        return False
        
    print("  Make sure:")
    print("    1. Your micro:bit is plugged into this computer via USB.\n")
    
    if not prompt_user("  Ready to push the most recent weights to the micro:bit?", default="y"):
        print("  Okay, cancelled.")
        return False

    import time
    try:
        import serial
        import serial.tools.list_ports
    except ImportError:
        print("  ❌ 'pyserial' is not installed. Run 'pip install pyserial'.")
        return False

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
    if not port:
        print("  ❌ micro:bit not found. Make sure it is plugged in.")
        return False

    trained_params = train_data["parameters"]
    best_r6 = train_data.get("r6_weight", 0.6)

    try:
        print("  Sending trained parameters to micro:bit...")
        ser = serial.Serial(port, 115200, timeout=0.1)
        time.sleep(2)
        parts = [str(best_r6)]
        keys = [
            "LEVEL_LOW", "LEVEL_MED", "LEVEL_HIGH",
            "RATIO_LOW", "RATIO_MED", "RATIO_HIGH",
            "GAPS_FEW", "GAPS_SOME", "GAPS_MANY",
            "VAR_LOW", "VAR_MED", "VAR_HIGH"
        ]
        for k in keys:
            if k in trained_params:
                a, b, c = trained_params[k]
                parts.extend([str(a), str(b), str(c)])
            else:
                print(f"  ❌ Missing parameter '{k}' in weights.")
                ser.close()
                return False

        param_str = ",".join(parts)
        msg = f"CMD_SAVE_PARAMS:{param_str}\n"
        for i in range(0, len(msg), 32):
            ser.write(msg[i:i+32].encode('utf-8'))
            ser.flush()
            time.sleep(0.05)
        ser.close()
        print("  ✅ micro:bit updated! Standalone mode ready.")
        return True
    except Exception as e:
        print(f"  ⚠️ Could not send params to micro:bit: {e}")
        return False


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print_banner()
    show_status()
    print_menu()

    try:
        while True:
            choice = input("  Enter your choice [1-9, or q to quit]: ").strip().lower()

            if choice == '1':
                run_full_pipeline()
                break
            elif choice == '2':
                if step_calibrate():
                    step_train()
                break
            elif choice == '3':
                if ProjectState.has_sufficient_calibration():
                    step_train()
                    break
                else:
                    print("  ⚠️  You need to record voice data (Step 1) before you can teach the AI!\n")
            elif choice == '4':
                if ProjectState.has_trained_model():
                    step_serve()
                    break
                else:
                    print("  ⚠️  The AI hasn't learned your voice yet (Step 2). Run Option 1 or 3 first!\n")
            elif choice == '5':
                show_status()
                print_menu()
            elif choice == '6':
                clear_all_data()
                break
            elif choice == '7':
                if step_mass_record_data():
                    step_train()
                break
            elif choice == '8':
                step_generate_poster_data()
                break
            elif choice == '9':
                step_prepare_standalone()
                break
            elif choice == 'q':
                print("  See you later! 👋")
                break
            else:
                print("  Hmm, that wasn't a valid choice. Try typing 1, 2, 3, 4, 5, 6, 7, 8, 9, or q.\n")
                
    except (EOFError, KeyboardInterrupt):
        print("\n\n  See you later! 👋")
        sys.exit(0)


if __name__ == "__main__":
    main()
