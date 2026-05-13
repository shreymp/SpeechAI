"""
Calibration Capture — PC Companion Script (5-Class)
=====================================================
Connects to the micro:bit over USB serial while calibration runs,
captures all serial output, and saves:
  - calibration_data.json   (the JSON training data)
  - calibration_log.txt     (the full serial output log)

Can be run standalone or called from START_HERE.py via run_capture().

Usage (standalone):
  python src/capture.py

Requirements:
  pip install -r requirements.txt
"""

import serial
import serial.tools.list_ports
import time
import os
import json
import re
from datetime import datetime

from src.fuzzy_engine import CLASSES_5
from src.serial_utils import get_microbit_port


# ============================================================
# CONFIGURATION
# ============================================================

BAUD_RATE = 115200
N_CLASSES = 5
MAX_CAPTURE_RETRIES = 1

# Class short letters for the micro:bit display
CLASS_LETTERS = ["S", "C", "H", "A", "M"]  # Silence, Confident, Hesitant, Anxious, Monotone


# ============================================================
# ROBUST MARKER DETECTION (handles micro:bit serial corruption)
# ============================================================

def _is_json_data_marker(line):
    """Detect JSON DATA marker despite serial corruption at 115200 baud.

    The micro:bit sends '--- JSON DATA ---' but UART corruption
    commonly drops characters (e.g. '--- JSN DATA ---').
    We check for the structural pattern '--- ... DATA' instead.
    """
    stripped = line.strip()
    return stripped.startswith("---") and "DATA" in stripped


def _is_calibration_complete(line):
    """Detect CALIBRATION COMPLETE marker despite serial corruption.

    The micro:bit sends 'CALIBRATION COMPLETE' but UART corruption
    commonly drops characters (e.g. 'CALIBRATION COMLETE').
    Uses prefix detection to distinguish from 'Starting Calibration...'.
    """
    l = line.upper().strip()
    if "CALIBRATION" not in l and "ALIBRATION" not in l:
        return False
    # Find position of the calibration word
    cal_pos = l.find("CALIBRATION")
    if cal_pos == -1:
        cal_pos = l.find("ALIBRATION")
    # "Starting Calibration..." has text before "CALIBRATION"
    if cal_pos > 0 and l[:cal_pos].strip():
        return False
    return True


class SerialCaptureStats:
    """Tracks serial health during a capture session for diagnostics."""

    def __init__(self):
        self.total_lines = 0
        self.decode_errors = 0
        self.empty_lines = 0
        self.json_marker_detected = False
        self.calibration_complete = False
        self.idle_timeout_triggered = False
        self.disconnect_triggered = False

    def report(self):
        if self.total_lines == 0:
            return
        corruption_rate = self.decode_errors / max(self.total_lines, 1)
        diagnostics = []
        if self.decode_errors:
            diagnostics.append(f"{self.decode_errors} decode errors ({corruption_rate:.1%})")
        if self.empty_lines:
            diagnostics.append(f"{self.empty_lines} empty lines")
        if not self.json_marker_detected:
            diagnostics.append("JSON marker not detected")
        if self.idle_timeout_triggered:
            diagnostics.append("idle timeout")
        if self.disconnect_triggered:
            diagnostics.append("disconnect")
        if diagnostics:
            print(f"  [i] Serial diagnostics: {', '.join(diagnostics)}")
        self._clear()

    def _clear(self):
        self.total_lines = 0
        self.decode_errors = 0
        self.empty_lines = 0
        self.json_marker_detected = False
        self.calibration_complete = False
        self.idle_timeout_triggered = False
        self.disconnect_triggered = False


# ============================================================
# JSON EXTRACTION
# ============================================================

def _try_parse_json(json_text):
    """Try to parse JSON text, with trailing-comma recovery."""
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        fixed = re.sub(r',\s*([}\]])', r'\1', json_text)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None


def _extract_json_standard(lines):
    """Extract JSON using the --- JSON DATA --- marker (robust to serial corruption)."""
    json_lines = []
    capturing = False
    for line in lines:
        if _is_json_data_marker(line):
            capturing = True
            continue
        if capturing:
            if line.startswith("=" * 10) or "--- Parameters" in line or _is_calibration_complete(line):
                break
            stripped = line.strip()
            if stripped:
                json_lines.append(stripped)
    if not json_lines:
        return None
    return _try_parse_json("\n".join(json_lines))


def _extract_json_by_structure(lines):
    """
    Fallback: find JSON by looking for lines that start with '{'.
    Also uses early-parse detection to stop as soon as valid JSON
    is assembled, making it robust against corrupted CALIBRATION COMPLETE
    markers (a known issue with micro:bit V2 UART at 115200 baud).
    """
    json_lines = []
    capturing = False
    for line in lines:
        stripped = line.strip()
        if not capturing and stripped.startswith("{"):
            capturing = True
            json_lines.append(stripped)
            continue
        if capturing:
            if _is_calibration_complete(line):
                break
            if stripped:
                json_lines.append(stripped)
                # Early-parse: once the JSON object is complete (has ]}),
                # try to parse immediately. This handles cases where the
                # CALIBRATION COMPLETE marker is corrupted or missing.
                result = _try_parse_json("\n".join(json_lines))
                if result is not None:
                    return result
    if not json_lines:
        return None
    return _try_parse_json("\n".join(json_lines))


def extract_json_from_log(lines):
    """
    Find the JSON block in the captured serial output.
    The micro:bit prints it between the JSON DATA markers.

    Tries multiple strategies:
    1. Standard marker-based extraction (--- JSON DATA ---)
    2. Structure-based extraction (find JSON by leading '{')
       — handles corrupted/noisy marker lines
    """
    result = _extract_json_standard(lines)
    if result is not None:
        return result
    result = _extract_json_by_structure(lines)
    if result is not None:
        return result
    return None


def extract_features_from_log(lines):
    """
    Fallback: extract features directly from the serial log,
    even if the JSON block is corrupted.
    Returns a list of sample dicts if successful, None otherwise.
    """
    samples = []
    current_phase = -1

    for line in lines:
        # Detect phase from micro:bit output
        if "Phase 1" in line or "Silence" in line or "BG Noise" in line:
            current_phase = 0
        elif "Phase 2" in line or "Confident" in line:
            current_phase = 1
        elif "Phase 3" in line or "Hesitant" in line:
            current_phase = 2
        elif "Phase 4" in line or "Anxious" in line:
            current_phase = 3
        elif "Phase 5" in line or "Monotone" in line:
            current_phase = 4

        # Extract numbers positionally to avoid failing on text corruption (e.g. 'volume0.1' or 'enegy')
        # Use both "Sample" and "volume" to handle serial corruption that drops
        # characters from the "Sample" prefix (e.g. "Sampl" or "Smple") or the
        # "=" sign in volume= (e.g. "volume3.8")
        if ("Sample" in line or "volume" in line) and ":" in line:
            try:
                parts = line.split(":", 1)[1]
                nums = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', parts)
                if len(nums) >= 7 and current_phase >= 0:
                    avg = float(nums[0])
                    ratio = float(nums[1])
                    gaps = int(float(nums[2]))
                    var = float(nums[3])
                    curv = float(nums[4])
                    drift = float(nums[5])
                    lat = int(float(nums[6]))

                    samples.append({
                        "class": current_phase,
                        "avg_level": avg,
                        "speech_ratio": ratio,
                        "num_gaps": gaps,
                        "variance": var,
                        "prosody_curvature": curv,
                        "intensity_drift": drift,
                        "response_latency": lat,
                        "speech_rate_variability": var / 500.0,
                        "pause_duration_entropy": gaps * 0.1,
                    })
            except Exception:
                continue

    # Need at least some samples per class
    min_samples = N_CLASSES * 3  # 3 per class minimum
    return samples if len(samples) >= min_samples else None


# ============================================================
# MAIN CAPTURE FUNCTION
# ============================================================

def run_capture(data_dir, record_time=15, _retries=MAX_CAPTURE_RETRIES):
    """
    Connect to micro:bit, capture calibration serial output, and save
    the JSON data to data_dir/calibration_data.json.

    Args:
        data_dir: Directory path to save output files.
        record_time: Total recording time in seconds (default 15 for 5 classes).
        _retries: Internal — number of automatic retries remaining.

    Returns:
        Path to the saved calibration_data.json, or None on failure.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Retry loop replaces fragile recursion pattern
    attempt = 0
    while attempt <= _retries:
        if attempt > 0:
            print(f"\n  [!] Extraction failed. Retry {attempt}/{_retries}...")
            print("  The micro:bit will restart. Press Button A when prompted.")
            print()
            time.sleep(3)
        result = _run_capture_once(data_dir, record_time)
        if result is not None:
            return result
        attempt += 1

    print("\n  [✗] Data extraction failed after all retries.")
    print("  The full serial log has been saved for debugging.")
    return None


def _run_capture_once(data_dir, record_time):
    json_output = os.path.join(data_dir, "calibration_data.json")
    log_output = os.path.join(data_dir, "calibration_log.txt")

    print()
    print("=" * 60)
    print("  Calibration Capture — 5-Class Serial Logger")
    print("=" * 60)
    print()
    print("  This captures serial output from the micro:bit and")
    print("  saves calibration data automatically.")
    print()
    print(f"  Classes: {', '.join(CLASSES_5)}")
    print()
    print("  Make sure FLASH_TO_MICROBIT.py is FLASHED on the micro:bit!")
    print()

    # Find micro:bit
    port = get_microbit_port()
    attempt = 0
    while not port:
        attempt += 1
        if attempt == 1:
            print("  Waiting for micro:bit... (connect via USB)")
        if attempt > 30:
            print("  ERROR: micro:bit not found after 60 seconds.")
            print("  Make sure it's plugged in and FLASH_TO_MICROBIT.py is flashed.")
            return None
        time.sleep(2)
        port = get_microbit_port()

    print(f"  Found micro:bit on {port}")
    print()

    try:
        try:
            ser = serial.Serial(port, BAUD_RATE, timeout=0.5, rtscts=True)
        except (ValueError, serial.SerialException, OSError):
            ser = serial.Serial(port, BAUD_RATE, timeout=0.5)
        time.sleep(2)

        # Command micro:bit to enter Calibration mode
        cmd = f"CMD_MODE_CALIBRATE:{record_time}\n"
        ser.write(cmd.encode('utf-8'))
        ser.flush()
    except Exception as e:
        print(f"  ❌ Connection error. Make sure the micro:bit is plugged in.")
        print("  Make sure no other program is using the serial port.")
        return None

    print("  Listening for calibration output...")
    print("  (Follow the prompts on the micro:bit: press A to record)")
    print()
    print("-" * 60)

    stats = SerialCaptureStats()
    all_lines = []
    in_json_block = False
    json_block_start = 0.0
    last_activity = time.time()
    IDLE_TIMEOUT = 300
    MAX_JSON_TIMEOUT = 30

    try:
        while True:
            try:
                if ser.in_waiting > 0:
                    raw = ser.readline()
                    try:
                        line = raw.decode('utf-8', errors='replace').rstrip('\r\n')
                    except Exception:
                        line = str(raw)

                    # Send ACK to micro:bit — confirms line receipt so it can
                    # send the next line. This provides deterministic back-pressure
                    # that prevents UART buffer overflow and character loss.
                    if line.strip():
                        try:
                            ser.write(b'\x06')
                            ser.flush()
                        except Exception:
                            pass

                    stats.total_lines += 1
                    if not line:
                        stats.empty_lines += 1

                    # Count decode errors: the 'replace' marker \ufffd indicates
                    # bytes that could not be decoded as UTF-8
                    if "\ufffd" in line:
                        stats.decode_errors += 1

                    # Always capture the line for JSON extraction
                    all_lines.append(line)
                    last_activity = time.time()

                    # --- UI filtering: hide raw JSON from the student ---
                    if _is_json_data_marker(line):
                        stats.json_marker_detected = True
                        in_json_block = True
                        json_block_start = time.time()
                        print("  📦 Transmitting voice data to PC...", end="", flush=True)
                    elif _is_calibration_complete(line):
                        stats.calibration_complete = True
                        in_json_block = False
                        print("\n")
                        break
                    elif not in_json_block and line.strip().startswith("{"):
                        # JSON DATA marker was corrupted, detect by content
                        in_json_block = True
                        json_block_start = time.time()
                        print("  📦 Transmitting voice data to PC...", end="", flush=True)
                    elif in_json_block:
                        if "Traceback" in line or "MemoryError" in line or "Exception" in line:
                            print(f"\n  [!] Micro:bit crashed during transmission: {line}")
                            break
                        # Safety timeout: if all markers are corrupted,
                        # don't wait longer than 30 seconds for JSON to finish
                        if time.time() - json_block_start > MAX_JSON_TIMEOUT:
                            print("\n  [i] JSON transmission complete (timeout).")
                            break
                        # Silently capture JSON lines — don't show to student
                        print(".", end="", flush=True)
                    else:
                        # Print all non-JSON output (phase headers, sample
                        # features, prompts, countdowns, etc.)
                        print(line)

                else:
                    if time.time() - last_activity > IDLE_TIMEOUT:
                        stats.idle_timeout_triggered = True
                        print("\n  No activity for 5 minutes. Stopping.")
                        break
                    time.sleep(0.05)

            except serial.SerialException:
                stats.disconnect_triggered = True
                print("\n  micro:bit disconnected.")
                break

    except KeyboardInterrupt:
        print("\n\n  Stopped by user (Ctrl+C).")

    if ser.is_open:
        ser.close()

    print()
    print("-" * 60)
    print()

    if not all_lines:
        print("  No data captured.")
        return None

    # --- Save full log ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_content = f"Calibration Log — {timestamp}\n{'=' * 60}\n\n"
    log_content += "\n".join(all_lines)

    with open(log_output, 'w') as f:
        f.write(log_content)
    print(f"  Full log saved.")

    # --- Extract and save JSON ---
    json_data = extract_json_from_log(all_lines)

    if json_data is None:
        print("  Data format issue detected. Using backup extraction...")
        # Count how many sample-like lines exist (for diagnostics)
        sample_line_count = sum(1 for l in all_lines if "volume=" in l or "Sample" in l)
        samples = extract_features_from_log(all_lines)
        if samples:
            print(f"  Backup extraction recovered {len(samples)} samples from {sample_line_count} data lines.")
            json_data = {
                "classes": list(CLASSES_5),
                "samples": samples,
            }
        else:
            print(f"  Backup extraction failed: found {sample_line_count} data lines but could not extract valid samples.")

    if json_data:
        # If calibration data already exists, append the new samples
        if os.path.exists(json_output):
            try:
                with open(json_output, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if "samples" in existing_data and isinstance(existing_data["samples"], list):
                    existing_data["samples"].extend(json_data.get("samples", []))
                    json_data = existing_data
            except Exception as e:
                print(f"  Note: Previous data could not be merged. Saving new data only.")

        # Ensure classes list reflects 5-class system
        json_data["classes"] = list(CLASSES_5)

        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
            f.flush()
        sample_count = len(json_data.get("samples", []))
        print(f"  ✅ Voice data saved! ({sample_count} total samples)")
        stats.report()
        return json_output

    stats.report()
    print("\n  [✗] Data extraction failed. The full serial log has been saved for debugging.")
    return None


# ============================================================
# STANDALONE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    result = run_capture(data_dir)
    if result:
        print(f"\n  Success! Data saved to {result}")
    else:
        print("\n  Capture failed.")
