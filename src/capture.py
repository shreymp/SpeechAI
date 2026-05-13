"""
Calibration Capture — PC Companion Script
==========================================
Connects to the micro:bit over USB serial while calibrate.py runs,
captures all serial output, and saves:
  - calibration_data.json   (the JSON training data)
  - calibration_log.txt     (the full serial output log)

Can be run standalone or called from run.py via run_capture().

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


# ============================================================
# CONFIGURATION
# ============================================================

BAUD_RATE = 115200


# ============================================================
# SERIAL HELPERS
# ============================================================

def get_microbit_port():
    """Auto-detect micro:bit serial port (Windows, Mac, Linux)."""
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


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json_from_log(lines):
    """
    Find the JSON block in the captured serial output.
    The micro:bit prints it between the JSON DATA markers.
    """
    json_lines = []
    capturing = False

    for line in lines:
        if "JSON DATA" in line:
            capturing = True
            continue
        if capturing:
            # Stop at the next separator or empty marker
            if line.startswith("=" * 10) or line.startswith("--- Parameters") or "CALIBRATION COMPLETE" in line:
                break
            stripped = line.strip()
            if stripped:
                json_lines.append(stripped)

    if not json_lines:
        return None

    json_text = "\n".join(json_lines)
    try:
        data = json.loads(json_text)
        return data
    except json.JSONDecodeError:
        # Try to fix common issues from serial corruption
        fixed = re.sub(r',\s*([}\]])', r'\1', json_text)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None


def extract_features_from_log(lines):
    """
    Fallback: extract features directly from the 'Done! Features:' blocks
    in the serial log, even if the JSON block is corrupted.
    Returns a list of sample dicts if successful, None otherwise.
    """
    samples = []
    current_phase = -1  # 0=BG, 1=Confident, 2=Hesitant

    # Track which phase we're in
    for line in lines:
        if "Phase 1:" in line or "BG Noise" in line:
            current_phase = 0
        elif "Phase 2:" in line or "Confident" in line:
            current_phase = 1
        elif "Phase 3:" in line or "Hesitant" in line:
            current_phase = 2

        # Look for feature lines in the raw data section
        if "avg=" in line and "ratio=" in line and "gaps=" in line and "var=" in line:
            try:
                # Parse: "  Sample N: avg=X  ratio=X  gaps=X  var=X"
                parts = line.strip()
                avg = float(re.search(r'avg[_=]*(\d+\.?\d*)', parts).group(1))
                ratio = float(re.search(r'ratio[_=]*(\d+\.?\d*)', parts).group(1))
                gaps = int(re.search(r'gaps[_=]*(\d+)', parts).group(1))
                var = float(re.search(r'var[_=]*(\d+\.?\d*)', parts).group(1))

                samples.append({
                    "class": current_phase,
                    "avg_level": avg,
                    "speech_ratio": ratio,
                    "num_gaps": gaps,
                    "variance": var,
                })
            except (AttributeError, ValueError):
                continue

    # Also try parsing from the "Done! Features:" blocks
    if len(samples) < 9:
        samples = []
        current_phase = -1
        pending_features = {}

        for line in lines:
            stripped = line.strip()

            if "Phase 1:" in line:
                current_phase = 0
            elif "Phase 2:" in line:
                current_phase = 1
            elif "Phase 3:" in line:
                current_phase = 2

            # Parse individual feature lines
            if "avg_level" in stripped and "=" in stripped:
                try:
                    val = float(re.search(r'=\s*(\d+\.?\d*)', stripped).group(1))
                    pending_features["avg_level"] = val
                except (AttributeError, ValueError):
                    pass
            elif "speech_ratio" in stripped and "=" in stripped:
                try:
                    val = float(re.search(r'=\s*(\d+\.?\d*)', stripped).group(1))
                    pending_features["speech_ratio"] = val
                except (AttributeError, ValueError):
                    pass
            elif "num_gaps" in stripped and "=" in stripped:
                try:
                    val = int(re.search(r'=\s*(\d+)', stripped).group(1))
                    pending_features["num_gaps"] = val
                except (AttributeError, ValueError):
                    pass
            elif "variance" in stripped and "=" in stripped:
                try:
                    val = float(re.search(r'=\s*(\d+\.?\d*)', stripped).group(1))
                    pending_features["variance"] = val
                except (AttributeError, ValueError):
                    pass

            # When we have all 4 features, save the sample
            if len(pending_features) == 4 and current_phase >= 0:
                pending_features["class"] = current_phase
                samples.append(pending_features.copy())
                pending_features = {}

    return samples if len(samples) >= 9 else None


# ============================================================
# MAIN CAPTURE FUNCTION
# ============================================================

def run_capture(data_dir, record_time=9):
    """
    Connect to micro:bit, capture calibration serial output, and save
    the JSON data to data_dir/calibration_data.json.

    Args:
        data_dir: Directory path to save output files.

    Returns:
        Path to the saved calibration_data.json, or None on failure.
    """
    json_output = os.path.join(data_dir, "calibration_data.json")
    log_output = os.path.join(data_dir, "calibration_log.txt")

    os.makedirs(data_dir, exist_ok=True)

    print()
    print("=" * 60)
    print("  Calibration Capture — Serial Logger")
    print("=" * 60)
    print()
    print("  This captures serial output from the micro:bit and")
    print("  saves calibration data automatically.")
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
        if attempt > 30:  # ~60 seconds
            print("  ERROR: micro:bit not found after 60 seconds.")
            print("  Make sure it's plugged in and FLASH_TO_MICROBIT.py is flashed.")
            return None
        time.sleep(2)
        port = get_microbit_port()

    print(f"  Found micro:bit on {port}")
    print()

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.5)
        time.sleep(2)  # Allow micro:bit to reset after connection
        
        # Command micro:bit to enter Calibration mode
        cmd = f"CMD_MODE_CALIBRATE:{record_time}\n"
        ser.write(cmd.encode('utf-8'))
        ser.flush()
    except Exception as e:
        print(f"  Serial error: {e}")
        print("  Make sure no other program is using the serial port.")
        return None

    print("  Listening for calibration output...")
    print("  (Follow the prompts on the micro:bit: press A to record)")
    print()
    print("-" * 60)

    all_lines = []
    calibration_complete = False
    json_section_done = False
    last_activity = time.time()
    IDLE_TIMEOUT = 300  # 5 minutes

    try:
        while True:
            try:
                if ser.in_waiting > 0:
                    raw = ser.readline()
                    try:
                        line = raw.decode('utf-8', errors='replace').rstrip('\r\n')
                    except Exception:
                        line = str(raw)

                    # Print to terminal so user can see everything
                    print(line)
                    all_lines.append(line)
                    last_activity = time.time()

                    if "CALIBRATION COMPLETE" in line:
                        calibration_complete = True

                    # Detect end of JSON block (last line is " ]}")
                    if calibration_complete and ("]}" in line):
                        json_section_done = True

                    if json_section_done and (time.time() - last_activity > 2):
                        break

                else:
                    if json_section_done and (time.time() - last_activity > 2):
                        break
                    if calibration_complete and (time.time() - last_activity > 5):
                        break
                    if time.time() - last_activity > IDLE_TIMEOUT:
                        print("\n  No activity for 5 minutes. Stopping.")
                        break
                    time.sleep(0.05)

            except serial.SerialException:
                print("\n  micro:bit disconnected.")
                break

    except KeyboardInterrupt:
        print("\n\n  Stopped by user (Ctrl+C).")

    # Close serial
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
    print(f"  Log saved to: {log_output}")

    # --- Extract and save JSON ---
    # Try the JSON block first
    json_data = extract_json_from_log(all_lines)

    # If JSON parsing failed (serial corruption), try feature extraction fallback
    if json_data is None:
        print("  JSON block was corrupted. Using fallback feature extraction...")
        samples = extract_features_from_log(all_lines)
        if samples:
            json_data = {
                "classes": ["Background Noise", "Confident Speaking", "Hesitant Speaking"],
                "samples": samples,
            }

    if json_data:
        # If calibration data already exists, append the new samples to it!
        if os.path.exists(json_output):
            try:
                with open(json_output, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if "samples" in existing_data and isinstance(existing_data["samples"], list):
                    existing_data["samples"].extend(json_data.get("samples", []))
                    json_data = existing_data
            except Exception as e:
                print(f"  Warning: Could not read existing data to append. Overwriting instead. ({e})")

        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        sample_count = len(json_data.get("samples", []))
        print(f"  Calibration data saved: {json_output} ({sample_count} total samples)")
        return json_output
    else:
        print("  ERROR: Could not extract calibration data from serial output.")
        print("  Check the log file for details:", log_output)
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
