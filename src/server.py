"""
Fuzzy Logic Voice Classifier — PC Server
=========================================
Receives audio from micro:bit via USB serial, runs fuzzy classification,
and sends results back. Can be called from run.py via run_server().
"""

import serial
import serial.tools.list_ports
import time
import os
import json

BAUD_RATE = 115200
SPEECH_THRESHOLD = 40
CLASSES = ["Background Noise", "Confident Speaking", "Hesitant Speaking"]
RESULT_TAGS = ["BACKGROUND", "CONFIDENT", "HESITANT"]
R6_WEIGHT = 0.6

# Default membership function parameters
DEFAULT_PARAMS = {
    "LEVEL_LOW": (0, 0, 50), "LEVEL_MED": (20, 70, 130), "LEVEL_HIGH": (80, 160, 255),
    "RATIO_LOW": (0.0, 0.0, 0.25), "RATIO_MED": (0.10, 0.40, 0.70), "RATIO_HIGH": (0.50, 0.80, 1.0),
    "GAPS_FEW": (0, 0, 4), "GAPS_SOME": (2, 6, 12), "GAPS_MANY": (8, 15, 25),
    "VAR_LOW": (0, 0, 500), "VAR_MED": (200, 1000, 2500), "VAR_HIGH": (1500, 3500, 6000),
}


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
    """Extract features from raw sound level values (0-255)."""
    n = len(samples)
    if n == 0:
        return {"avg_level": 0, "speech_ratio": 0.0, "num_gaps": 0, "variance": 0}
    avg = sum(samples) / n
    variance = sum((s - avg) ** 2 for s in samples) / n
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


def fuzzy_classify(features, params):
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

    r6_weight = params.get("_r6_weight", R6_WEIGHT)
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
        "memberships": {
            "level": (round(lvl_lo, 3), round(lvl_md, 3), round(lvl_hi, 3)),
            "ratio": (round(rat_lo, 3), round(rat_md, 3), round(rat_hi, 3)),
            "gaps": (round(gap_fw, 3), round(gap_sm, 3), round(gap_mn, 3)),
            "var": (round(var_lo, 3), round(var_md, 3), round(var_hi, 3)),
        },
        "rules": {"R1": round(r1, 4), "R2": round(r2, 4), "R3": round(r3, 4),
                  "R4": round(r4, 4), "R5": round(r5, 4), "R6": round(r6, 4)},
    }
    return best_idx, confidence, details


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


def load_params(trained_params_path):
    """Load trained parameters if available, else use defaults."""
    params = DEFAULT_PARAMS.copy()

    if os.path.exists(trained_params_path):
        try:
            with open(trained_params_path, 'r') as f:
                data = json.load(f)
            trained = data.get("parameters", {})
            for key in params:
                if key in trained:
                    params[key] = tuple(trained[key])
            r6 = data.get("r6_weight", R6_WEIGHT)
            params["_r6_weight"] = r6
            acc = data.get("accuracy", "?")
            print(f"  Loaded trained parameters (accuracy: {acc})")
            return params
        except Exception as e:
            print(f"  Warning: Could not load trained params: {e}")

    print("  Using default parameters.")
    return params


def run_server(trained_params_path):
    """
    Start the classification server. Connects to micro:bit and listens
    for audio streams.

    Args:
        trained_params_path: Path to trained_params.json
    """
    print()
    print("=" * 60)
    print("  Fuzzy Logic Voice Classifier — PC Server")
    print("=" * 60)

    # Load parameters
    params = load_params(trained_params_path)

    # Connect to micro:bit
    port = get_microbit_port()
    while not port:
        print("  Waiting for micro:bit... (connect via USB)")
        print("  Make sure FLASH_TO_MICROBIT.py is flashed on the micro:bit!")
        time.sleep(2)
        port = get_microbit_port()

    print(f"  Connecting to {port}...")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
        time.sleep(2)  # Allow micro:bit to reset
        
        # Command micro:bit to enter UI mode
        ser.write(b"CMD_MODE_UI\n")
        ser.flush()
    except Exception as e:
        print(f"  Serial error: {e}")
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
                            if raw_bytes.startswith(b'\n'):
                                raw_bytes = raw_bytes[1:]
                            samples = list(raw_bytes)
                            print(f"  Recording received: {len(samples)} samples in {duration:.1f}s")
                            last_features = extract_features(samples)
                            print(f"  Features:")
                            print(f"    avg_level:    {last_features['avg_level']:.1f}")
                            print(f"    speech_ratio: {last_features['speech_ratio']:.2f}")
                            print(f"    num_gaps:     {last_features['num_gaps']}")
                            print(f"    variance:     {last_features['variance']:.0f}")
                            ser.write(b"STATUS_REC_DONE\n")
                            print(f"  -> Sent: STATUS_REC_DONE")
                            recording_data = []
                            buffer_tail = b""
            else:
                if ser.in_waiting > 0:
                    try:
                        line = ser.readline()
                        try:
                            line_str = line.decode('utf-8').strip()
                        except Exception:
                            line_str = ""
                        if line_str:
                            print(f"  Received: {line_str}")
                            if line_str == "START_STREAM":
                                print("\n  -- Starting audio stream reception --")
                                is_recording = True
                                start_rec_time = time.time()
                                recording_data = []
                                buffer_tail = b""
                            elif line_str == "CMD_CLASSIFY":
                                if last_features is None:
                                    print("  No audio data. Record first.")
                                    ser.write(b"STATUS_ERROR\n")
                                else:
                                    print("\n  -- Running fuzzy classification --")
                                    class_idx, confidence, details = fuzzy_classify(last_features, params)
                                    class_name = CLASSES[class_idx]
                                    result_tag = RESULT_TAGS[class_idx]
                                    print(f"\n  ╔══════════════════════════════════════╗")
                                    print(f"  ║  RESULT: {class_name:<28} ║")
                                    print(f"  ║  Confidence: {confidence}%{' ' * (24 - len(str(confidence)))}║")
                                    print(f"  ╚══════════════════════════════════════╝")
                                    print(f"\n  Scores:")
                                    for cls, score in details["scores"].items():
                                        bar = "█" * int(score * 30)
                                        print(f"    {cls:<22} {score:.4f}  {bar}")
                                    print()
                                    msg = f"RESULT_{result_tag}:{confidence}\n"
                                    ser.write(msg.encode('utf-8'))
                                    print(f"  -> Sent: {msg.strip()}")
                    except Exception as e:
                        print(f"  Serial read error: {e}")

            time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n  Stopping server...")
            break
        except Exception as e:
            print(f"  Loop error: {e}")
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
