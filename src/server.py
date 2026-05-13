"""
Fuzzy Logic Voice Classifier — PC Server (5-Class)
====================================================
Receives audio from micro:bit via USB serial, runs 5-class fuzzy
classification, and sends results back.

Can be called from START_HERE.py via run_server() or standalone.
"""

import serial
import serial.tools.list_ports
import time
import os
import json

from src.features import extract_features_v2
from src.fuzzy_engine import (
    CLASSES_5, RESULT_TAGS_5, DEFAULT_PARAMS_5,
    fuzzy_classify_5,
)
from src.calibrator import NoiseCalibration, SessionLogger
from src.visualizer import display_classification_result
from src.serial_utils import get_microbit_port

BAUD_RATE = 115200
RECORDING_TIMEOUT = 10  # seconds — abort if no END_STREAM received
MAX_PORT_RETRIES = 30  # max iterations waiting for micro:bit


def load_params(trained_params_path):
    """Load trained 5-class parameters if available, else use defaults."""
    params = DEFAULT_PARAMS_5.copy()
    rule_weights = None

    if os.path.exists(trained_params_path):
        try:
            with open(trained_params_path, 'r') as f:
                data = json.load(f)
            trained = data.get("parameters", {})
            for key in params:
                if key in trained:
                    params[key] = tuple(trained[key])
            rule_weights = data.get("rule_weights", None)
            acc = data.get("accuracy", None)
            if isinstance(acc, (int, float)):
                print(f"  Loaded trained AI (accuracy: {acc * 100:.1f}%)")
            else:
                print(f"  Loaded trained AI parameters.")
            return params, rule_weights
        except Exception as e:
            print(f"  Warning: Could not load saved AI data. Using defaults.")

    print("  Using default 5-class parameters.")
    return params, rule_weights


def run_server(trained_params_path):
    """
    Start the 5-class classification server. Connects to micro:bit
    and listens for audio streams.

    Args:
        trained_params_path: Path to trained_params.json
    """
    print()
    print("=" * 60)
    print("  Fuzzy Logic Voice Classifier — 5-Class PC Server")
    print("=" * 60)

    # Load parameters
    params, rule_weights = load_params(trained_params_path)

    # Set up data directory for session logging
    data_dir = os.path.dirname(trained_params_path)
    calibrator = NoiseCalibration(data_dir)
    logger = SessionLogger(data_dir)

    # Connect to micro:bit
    port = get_microbit_port()
    port_attempts = 0
    while not port:
        port_attempts += 1
        if port_attempts > MAX_PORT_RETRIES:
            print("  ERROR: micro:bit not found after 60 seconds.")
            print("  Make sure it's plugged in and FLASH_TO_MICROBIT.py is flashed.")
            return
        print("  Waiting for micro:bit... (connect via USB)")
        time.sleep(2)
        port = get_microbit_port()

    print(f"  Connecting to {port}...")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        ser.write(b"CMD_MODE_UI\n")
        ser.flush()
    except Exception as e:
        print(f"  ❌ Connection error. Make sure the micro:bit is plugged in and no other program is using it.")
        return

    print("  Server ready! Listening for commands...")
    print()
    print("  Instructions:")
    print("    1. Press A on micro:bit to record audio")
    print("    2. Press B on micro:bit to classify")
    print("    3. Press Ctrl+C to stop the server")
    print()

    # State
    recording_data = []
    is_recording = False
    start_rec_time = 0
    buffer_tail = b""
    last_samples = None
    last_features = None

    while True:
        try:
            if is_recording:
                if ser.in_waiting > 0:
                    chunk = ser.read(ser.in_waiting)
                    recording_data.append(chunk)
                    buffer_tail += chunk
                    if len(buffer_tail) > 30:
                        buffer_tail = buffer_tail[-30:]
                    if b"END_STREAM" in buffer_tail:
                        combined = b''.join(recording_data)
                        if b"END_STREAM" in combined:
                            is_recording = False
                            duration = time.time() - start_rec_time
                            raw_bytes = combined.split(b"END_STREAM")[0]
                            # Strip leading START_STREAM\n marker sent before recording
                            if raw_bytes.startswith(b"START_STREAM\n"):
                                raw_bytes = raw_bytes[len(b"START_STREAM\n"):]
                            # Strip trailing padding (\x00 bytes + newlines)
                            raw_bytes = raw_bytes.rstrip(b'\x00\n')
                            last_samples = list(raw_bytes)
                            print(f"  ✅ Audio received ({duration:.1f}s). Analyzing...")

                            # Extract features with current noise floor (needed to
                            # determine response latency for ambient calibration)
                            last_features = extract_features_v2(
                                last_samples,
                                noise_floor=calibrator.get_noise_floor()
                            )

                            # Calibrate noise floor using samples before speech onset.
                            # Uses response_latency to determine how many initial
                            # samples are truly ambient, avoiding the feedback loop
                            # where speech contaminates the noise floor estimate.
                            latency_ms = last_features.get('response_latency', 600)
                            n_ambient = max(10, int(latency_ms / 20))
                            calibrator.calibrate_from_recording_start(last_samples, n_samples=n_ambient)

                            # Re-extract features with updated noise floor for accuracy
                            last_features = extract_features_v2(
                                last_samples,
                                noise_floor=calibrator.get_noise_floor()
                            )

                            print(f"  📊 Audio analyzed. Press B on micro:bit to classify.")

                            ser.write(b"STATUS_REC_DONE\n")
                            recording_data = []
                            buffer_tail = b""
                elif time.time() - start_rec_time > RECORDING_TIMEOUT:
                    # Timeout — no END_STREAM received (micro:bit may have disconnected)
                    is_recording = False
                    print(f"  ⚠️  Recording timeout ({RECORDING_TIMEOUT}s) — no END_STREAM received.")
                    recording_data = []
                    buffer_tail = b""
            else:
                if ser.in_waiting > 0:
                    try:
                        line = ser.readline()
                        try:
                            line_str = line.decode('utf-8').strip()
                        except UnicodeDecodeError:
                            line_str = ""
                        if line_str:
                            if line_str == "START_STREAM":
                                print("\n  🎤 Recording... Speak into the micro:bit now!")
                                is_recording = True
                                start_rec_time = time.time()
                                recording_data = []
                                buffer_tail = b""
                            elif line_str == "CMD_CLASSIFY":
                                if last_features is None:
                                    print("  No audio data. Record first.")
                                    ser.write(b"STATUS_ERROR\n")
                                else:
                                    print("\n  🧠 Classifying your voice...")
                                    class_idx, confidence, details = fuzzy_classify_5(
                                        last_features, params, rule_weights
                                    )
                                    class_name = CLASSES_5[class_idx]
                                    result_tag = RESULT_TAGS_5[class_idx]

                                    display_classification_result(
                                        class_idx, confidence, details, last_features
                                    )

                                    margin = details.get("margin", 0)
                                    is_ambiguous = details.get("is_ambiguous", False)
                                    logger.log_classification(
                                        class_name, confidence, margin,
                                        is_ambiguous, last_features
                                    )

                                    msg = f"RESULT_{result_tag}:{confidence}\n"
                                    ser.write(msg.encode('utf-8'))
                    except serial.SerialException:
                        print(f"  ⚠️ Serial connection lost during command processing.")
                        raise
                    except (KeyError, TypeError, ValueError) as e:
                        print(f"  ⚠️ Classification error: {e}")

            time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n  Stopping server...")
            break
        except Exception as e:
            print(f"  ⚠️ Connection interrupted. Reconnecting...")
            try:
                ser.close()
                time.sleep(1)
                ser.open()
            except Exception:
                break

    if ser.is_open:
        ser.close()
    print("  Server stopped.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    params_path = os.path.join(base_dir, "data", "trained_params.json")
    run_server(params_path)
