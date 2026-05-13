"""
Universal Micro:bit App — Student Edition
=========================================
Flash this ONCE. It handles everything:
1. Calibration (Collecting Voice)
2. Live PC Detection (Streaming — Full 5-Class)
3. Standalone Mode (Unplugged — Simplified 3-Class)

It listens for commands from the PC over USB.
"""

from microbit import *
import math
import gc

try:
    uart.init(baudrate=115200, flow=UARTFlow.RTS_CTS)
except:
    uart.init(baudrate=115200)

RECORD_DURATION_MS = 3000
SAMPLE_INTERVAL_MS = 20
SPEECH_THRESHOLD = 40

MODE_STANDALONE = 0
MODE_PC_STREAM = 1

current_mode = MODE_STANDALONE
saved_params = None

# ============================================================
# PARAMETER STORAGE
# ============================================================

def save_params(data_str):
    try:
        with open('params.txt', 'w') as f:
            f.write(data_str)
        return True
    except:
        return False

def load_params():
    try:
        with open('params.txt', 'r') as f:
            data = f.read().strip()
            if not data: return None

            # Parse simple comma-separated list of values
            parts = data.split(',')
            if len(parts) < 37: return None

            p = {"R6_WEIGHT": float(parts[0])}
            idx = 1
            keys = [
                "LEVEL_LOW", "LEVEL_MED", "LEVEL_HIGH",
                "RATIO_LOW", "RATIO_MED", "RATIO_HIGH",
                "GAPS_FEW", "GAPS_SOME", "GAPS_MANY",
                "VAR_LOW", "VAR_MED", "VAR_HIGH"
            ]
            for k in keys:
                p[k] = (float(parts[idx]), float(parts[idx+1]), float(parts[idx+2]))
                idx += 3
            return p
    except:
        return None

# ============================================================
# CORE AUDIO & FEATURES
# ============================================================

def record_samples(anim_mode=0):
    samples = []
    start = running_time()
    tick = 0
    while running_time() - start < RECORD_DURATION_MS:
        val = microphone.sound_level()
        samples.append(val)

        # Animations
        if anim_mode == 0:  # Pulsing dot
            elapsed = running_time() - start
            b = min(9, max(0, int(4.5 + 4.5 * math.sin(elapsed / 200))))
            display.set_pixel(2, 2, b)
        elif anim_mode == 1:  # Progress bar
            pct = (running_time() - start) / RECORD_DURATION_MS
            pixels_on = int(pct * 25)
            img = Image("00000:00000:00000:00000:00000")
            for i in range(pixels_on):
                img.set_pixel(i % 5, 4 - (i // 5), 7)
            display.show(img)

        if current_mode == MODE_PC_STREAM:
            uart.write(bytes([val]))

        sleep(SAMPLE_INTERVAL_MS)

    display.clear()
    gc.collect()
    return samples

def extract_features(samples):
    n = len(samples)
    if n == 0: return (0, 0.0, 0, 0, 0.0, 0.0, 0)

    total = sum(samples)
    avg = total / n

    var_sum = 0
    for s in samples:
        d = s - avg
        var_sum += d * d
    variance = var_sum / n

    above_count = 0
    num_gaps = 0
    in_gap = True
    first_speech_idx = -1

    for i, s in enumerate(samples):
        if s >= SPEECH_THRESHOLD:
            above_count += 1
            if in_gap:
                in_gap = False
                if first_speech_idx < 0:
                    first_speech_idx = i
        else:
            if not in_gap:
                num_gaps += 1
                in_gap = True

    speech_ratio = above_count / n

    # Prosody curvature (mean absolute 2nd derivative)
    curv = 0.0
    if n > 2:
        curv_sum = 0.0
        for i in range(1, n - 1):
            curv_sum += abs(samples[i+1] - 2*samples[i] + samples[i-1])
        curv = curv_sum / (n - 2)

    # Intensity drift (linear slope via simplified normal equation)
    drift = 0.0
    if n > 1:
        x_mean = (n - 1) / 2.0
        num_d = 0.0
        den_d = 0.0
        for i in range(n):
            dx = i - x_mean
            num_d += dx * (samples[i] - avg)
            den_d += dx * dx
        if den_d > 0:
            drift = num_d / den_d

    # Response latency (ms until first speech sample)
    latency = n * SAMPLE_INTERVAL_MS  # default: full window = no speech
    if first_speech_idx >= 0:
        latency = first_speech_idx * SAMPLE_INTERVAL_MS

    return (avg, speech_ratio, num_gaps, variance, curv, drift, latency)

# ============================================================
# CALIBRATION FLOW
# ============================================================

def safe_print(msg):
    """Print a line over UART and wait for ACK (0x06) from PC before returning.

    The PC sends 0x06 to confirm receipt, providing back-pressure that
    prevents UART buffer overflow and character loss at 115200 baud.
    Times out after 2 seconds to avoid hanging if PC disconnects.
    """
    uart.write((str(msg) + '\n').encode('utf-8'))
    deadline = running_time() + 2000
    while running_time() < deadline:
        if uart.any():
            b = uart.read(1)
            if b and b[0] == 0x06:
                return True
    return False


def do_calibration(total_time_sec=15):
    # Calculate number of 3-second samples to match the requested duration
    num_samples = max(1, total_time_sec // (RECORD_DURATION_MS // 1000))
    display.scroll("CAL", delay=110)
    classes = ["Silence", "Confident", "Hesitant", "Anxious", "Monotone"]
    letters = ["S", "C", "H", "A", "M"]

    class_data = [[], [], [], [], []]

    safe_print("\nStarting Calibration...")
    for ci in range(5):
        safe_print("\n=== Phase {}: {} ===".format(ci + 1, classes[ci]))
        display.show(letters[ci])
        sleep(1000)

        safe_print("  --> Press Button A to start continuous {}s recording...".format(num_samples * 3))
        display.scroll("A", delay=110, wait=False)
        while not button_a.was_pressed():
            sleep(50)

        for i in range(3, 0, -1):
            safe_print("    {}...".format(i))
            display.show(str(i))
            sleep(800)

        safe_print("    [Recording continuously for {} seconds]".format(num_samples * 3))
        display.clear()

        for si in range(num_samples):
            samples = record_samples(anim_mode=0)
            feats = extract_features(samples)
            class_data[ci].append(feats)

            safe_print("    Sample {}/{}: volume={:.1f}  speech={:.2f}  pauses={}  energy={:.0f}  melody={:.1f}  trend={:.3f}  reaction={}ms".format(si + 1, num_samples, *feats))

        display.show(Image.YES)
        sleep(1000)

    display.scroll("DONE", delay=110)

    # Send data to PC — 7-feature JSON for 5-class compatibility
    safe_print("\n--- JSON DATA ---")
    safe_print('{"classes": ["Ambient Silence", "Confident Articulation", "Hesitant Disfluency", "Anxious Urgency", "Disengaged Monotone"],')
    safe_print(' "samples": [')
    for ci in range(5):
        for si, feats in enumerate(class_data[ci]):
            a, r, g, v, curv, drift, lat = feats
            trail = "," if (ci < 4 or si < len(class_data[ci]) - 1) else ""
            safe_print('  {"class": %d, "avg_level": %.1f, "speech_ratio": %.2f, "num_gaps": %d, "variance": %.0f, "prosody_curvature": %.1f, "intensity_drift": %.3f, "response_latency": %d, "speech_rate_variability": %.2f, "pause_duration_entropy": %.1f}%s' % (ci, a, r, g, v, curv, drift, lat, v / 500.0, g * 0.1, trail))
        gc.collect()
    safe_print(' ]}')
    safe_print("CALIBRATION COMPLETE")
    display.show(Image.HAPPY)

# ============================================================
# FUZZY STANDALONE
# ============================================================

def tri_mf(x, a, b, c):
    if x <= a or x >= c: return 0.0
    elif a < x <= b: return (x - a) / (b - a) if b != a else 1.0
    else: return (c - x) / (c - b) if c != b else 1.0

def left_mf(x, a, b, c):
    if x <= b: return 1.0
    elif x >= c: return 0.0
    else: return (c - x) / (c - b) if c != b else 1.0

def right_mf(x, a, b, c):
    if x <= a: return 0.0
    elif x >= b: return 1.0
    else: return (x - a) / (b - a) if b != a else 1.0

def classify_standalone(avg_level, speech_ratio, num_gaps, variance, curv, drift, latency):
    global saved_params
    if not saved_params: return ("Unknown", 0)

    p = saved_params

    lvl_lo = left_mf(avg_level, *p["LEVEL_LOW"])
    lvl_md = tri_mf(avg_level, *p["LEVEL_MED"])
    lvl_hi = right_mf(avg_level, *p["LEVEL_HIGH"])
    rat_lo = left_mf(speech_ratio, *p["RATIO_LOW"])
    rat_md = tri_mf(speech_ratio, *p["RATIO_MED"])
    rat_hi = right_mf(speech_ratio, *p["RATIO_HIGH"])
    gap_fw = left_mf(num_gaps, *p["GAPS_FEW"])
    gap_sm = tri_mf(num_gaps, *p["GAPS_SOME"])
    gap_mn = right_mf(num_gaps, *p["GAPS_MANY"])

    # Compute variance MFs using VAR params
    var_lo = left_mf(variance, *p.get("VAR_LOW", (0, 0, 200)))
    var_hi = right_mf(variance, *p.get("VAR_HIGH", (500, 800, 1200)))

    # Rules per Action Plan Phase 1:
    # R1: lvl_lo ∧ rat_lo → Silence
    r1 = min(lvl_lo, rat_lo)
    # R6: lvl_lo ∧ rat_md → Silence (dampened)
    r6 = min(lvl_lo, rat_md) * p["R6_WEIGHT"]
    # R2: lvl_hi ∧ rat_hi ∧ gap_fw → Confident
    r2 = min(lvl_hi, rat_hi, gap_fw)
    # R5: lvl_md ∧ rat_md ∧ var_lo → Confident
    r5 = min(lvl_md, rat_md, var_lo)
    # R3: rat_md ∧ gap_mn → Hesitant
    r3 = min(rat_md, gap_mn)
    # R4: var_hi ∧ gap_sm → Hesitant
    r4 = min(var_hi, gap_sm)

    score_bg = max(r1, r6)
    score_conf = max(r2, r5)  # R5 → Confident (fixed)
    score_hes = max(r3, r4)   # R4 → Hesitant (added)

    scores = [
        ("Silence", score_bg),
        ("Confident", score_conf),
        ("Hesitant", score_hes),
    ]

    best_name = "Unknown"
    best_score = 0.0
    total = score_bg + score_conf + score_hes

    for n, s in scores:
        if s > best_score:
            best_score = s
            best_name = n

    conf = int((best_score / total) * 100) if total > 0 else 0
    return (best_name, conf)

def show_result(name, conf):
    if name == "Silence":
        display.show(Image("00000:00000:00900:00000:00000"))
        sleep(1500)
    elif name == "Confident":
        display.show(Image.YES)
        sleep(500)
        display.show(Image.HAPPY)
        sleep(1000)
    elif name == "Hesitant":
        display.show(Image.CONFUSED)
        sleep(1500)
    elif name == "Anxious":
        display.show(Image.SURPRISED)
        sleep(1500)
    elif name == "Monotone":
        display.show(Image.SAD)
        sleep(1500)

    display.scroll(name, delay=110)
    display.scroll(str(conf) + "%", delay=110)

# ============================================================
# MAIN LOOP
# ============================================================

def main():
    global current_mode, saved_params
    saved_params = load_params()
    display.scroll("RDY", delay=110)

    uart_buffer = ""

    while True:
        # Show mode indicator
        if current_mode == MODE_PC_STREAM:
            display.show(Image.MUSIC_QUAVERS)
        else:
            if saved_params:
                display.show(Image.TARGET)
            else:
                display.show(Image.SQUARE_SMALL)

        # Check UART Commands
        if uart.any():
            # Heartbeat indicator
            curr_brightness = display.get_pixel(4, 0)
            display.set_pixel(4, 0, 9 if curr_brightness == 0 else 0)

            try:
                chunk = uart.read()
                if chunk:
                    uart_buffer += str(chunk, 'utf-8').replace('\x06', '')
            except:
                pass

            if '\n' in uart_buffer:
                lines = uart_buffer.split('\n')
                uart_buffer = lines.pop()
                for msg in lines:
                    msg = msg.strip()
                    if not msg:
                        continue

                    if msg.startswith("CMD_MODE_CALIBRATE"):
                        parts = msg.split(":")
                        time_sec = 15
                        if len(parts) == 2:
                            try:
                                time_sec = int(parts[1])
                            except:
                                pass
                        do_calibration(time_sec)
                    elif msg == "CMD_MODE_UI":
                        current_mode = MODE_PC_STREAM
                        display.scroll("PC", delay=110, wait=False)
                    elif msg.startswith("CMD_SAVE_PARAMS:"):
                        data = msg.replace("CMD_SAVE_PARAMS:", "").strip()
                        if save_params(data):
                            saved_params = load_params()
                            display.show(Image.YES)
                            sleep(1000)
                        else:
                            display.show(Image.NO)
                            sleep(1000)
                    elif msg.startswith("RESULT_"):
                        # PC sent result
                        parts = msg.replace("RESULT_", "").split(":")
                        res = parts[0]
                        conf = int(parts[1]) if len(parts) > 1 else 100
                        res = res.replace("BACKGROUND", "Silence").replace("SILENCE", "Silence")
                        res = res.replace("CONFIDENT", "Confident")
                        res = res.replace("HESITANT", "Hesitant")
                        res = res.replace("ANXIOUS", "Anxious")
                        res = res.replace("MONOTONE", "Monotone")
                        show_result(res, conf)

        # Button A Action
        if button_a.was_pressed():
            if current_mode == MODE_PC_STREAM:
                # Stream Mode
                for i in range(3, 0, -1):
                    display.show(str(i))
                    sleep(800)
                uart.write(b"START_STREAM\n")
                record_samples(anim_mode=1)
                uart.write(bytes([0]) * 5)
                uart.write(b"\nEND_STREAM\n")
                display.show(Image.CLOCK12)
            else:
                # Standalone Mode
                if not saved_params:
                    display.scroll("Teach Me First!", delay=110)
                else:
                    for i in range(3, 0, -1):
                        display.show(str(i))
                        sleep(800)
                    samples = record_samples(anim_mode=0)
                    if len(samples) > 0:
                        display.show(Image.ALL_CLOCKS[0])
                        feats = extract_features(samples)
                        name, conf = classify_standalone(*feats)
                        show_result(name, conf)

        # Button B Action
        if button_b.was_pressed():
            if current_mode == MODE_PC_STREAM:
                uart.write(b"CMD_CLASSIFY\n")
                display.show(Image.ALL_CLOCKS[6])
            else:
                display.scroll("A=Listen", delay=110)

        sleep(50)

main()
