# Fuzzy Logic Voice Mood Classifier — Detailed Guide (Student Edition)

> **Project goal:** Classify audio from a BBC micro:bit V2's built-in microphone into five categories:
> - **Ambient Silence** — ambient / no speech
> - **Confident Articulation** — clear, steady voice, few pauses
> - **Hesitant Disfluency** — broken speech, gaps, variable loudness
> - **Anxious Urgency** — rapid, loud, pressured speech
> - **Disengaged Monotone** — flat, low-energy, monotonous delivery

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [How It Works (The Science)](#2-how-it-works)
3. [Step-by-Step: Collecting Voice Data (Calibration)](#3-step-by-step-collecting-voice-data)
4. [Step-by-Step: Teaching the AI (Training)](#4-step-by-step-teaching-the-ai)
5. [Mode A: PC-Connected Classification](#5-mode-a-pc-connected-classification)
6. [Mode B: Standalone micro:bit Classification](#6-mode-b-standalone-microbit-classification)
7. [Using START_HERE.py (Recommended)](#7-using-start_herepy-recommended)
8. [Understanding the Output](#8-understanding-the-output)
9. [Tips for Best Results](#9-tips-for-best-results)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

### Hardware
- **BBC micro:bit V2** (must be V2 — it has the built-in microphone)
- **USB cable** (micro-USB to connect the micro:bit to your computer)

### Software (on your PC/Mac)
- **Python 3.7+** installed
- **micro:bit Python Editor (V3)** — [python.microbit.org/v/3](https://python.microbit.org/v/3)
  - This is the recommended way to flash code onto the micro:bit directly from your browser.
  - No installation required! Just click "Connect" and "Flash".

### Python Packages
Open Terminal and run:
```bash
pip install -r requirements.txt
```
This installs `pyserial` (for USB communication).

---

## 2. How It Works

This project uses **fuzzy logic** — a type of AI that works with degrees of truth instead of just true/false. Here's the pipeline:

### What the micro:bit does
1. Records **3 seconds** of audio from its built-in microphone
2. Measures the **sound level** (~50 times per second)

### What the PC classifier does (5-class system)
From those sound level readings, it extracts **8 features**:

| Feature | What it measures | Example values |
|---------|-----------------|----------------|
| **avg_level** | How loud is it overall? | Silence: 2–10, Speech: 60–170 |
| **speech_ratio** | What % of time has sound above threshold? | Silence: 0.00, Speech: 0.30–0.95 |
| **num_gaps** | How many pauses are there? | Confident: 1–3, Hesitant: 8–15+ |
| **prosody_curvature** | How dynamic is the voice? (2nd derivative) | Flat: 1–3, Dynamic: 15–25+ |
| **speech_rate_variability** | How consistent is the speaking pace? | Steady: 0.1–0.5, Variable: 2.0–3.0 |
| **pause_duration_entropy** | How irregular are the pauses? | Regular: 0–0.3, Irregular: 1.2–1.8 |
| **intensity_drift** | Is volume trending up or down? | Stable: ~0, Trailing off: -0.5 to -1.0 |
| **response_latency** | How long before speech starts? | Quick: 50–200ms, Slow: 1500–3000ms |

Then it uses **triangular membership functions** and **13 fuzzy rules** to classify the audio into one of **5 mood categories**:

```
IF level is LOW  AND ratio is LOW  AND latency is HIGH  → Ambient Silence
IF level is HIGH AND ratio is HIGH AND gaps are FEW     → Confident Articulation
IF ratio is MED  AND gaps are MANY AND entropy is HIGH  → Hesitant Disfluency
IF level is HIGH AND ratio is HIGH AND curvature is LOW → Anxious Urgency
IF level is MED  AND curvature is LOW AND SRV is LOW   → Disengaged Monotone
```

### What the standalone micro:bit does (3-class simplified)
When unplugged from the PC, the micro:bit runs a **simplified 3-class** classifier using 4 features (avg_level, speech_ratio, num_gaps, variance) and 6 rules to detect: Silence, Confident, or Hesitant.

### Why collecting your voice matters
Every room and every voice is different. Collecting voice data records **your** voice in **your** environment so the AI knows what each mood sounds like for you.

---

## 3. Step-by-Step: Collecting Voice Data

This step collects audio samples so the system can learn your voice.

### What you need
- micro:bit V2 plugged into your computer via USB
- A relatively quiet room

### Step 1: Flash the Universal App (One Time Only!)
1. Open the [micro:bit Python Editor](https://python.microbit.org/v/3) in your browser.
2. Click **Open** and select `FLASH_TO_MICROBIT.py` from this folder.
3. Click **Connect** and then **Flash** — wait for the yellow light to stop blinking.
4. **Important:** Close the web editor's Serial panel if it's open. Only one program can use the serial port at a time.

### Step 2: Start the capture script
Open **Terminal** and run:
```bash
cd "path/to/project_folder"
python START_HERE.py
```
Choose option **[1]** or **[2]** to start collecting voice data. The program will automatically tell the micro:bit to enter voice collection mode.

### Step 3: Record your samples
The micro:bit will guide you through **5 phases**:

#### Phase 1 — Ambient Silence (letter "S" on display)
> Record what the room sounds like when nobody is talking.
- **Press Button A** to start recording
- **Stay quiet** for 3 seconds — just let the room noise be recorded

#### Phase 2 — Confident Articulation (letter "C" on display)
> Record clear, steady speech — like you're presenting to a class.
- **Press Button A** to start recording
- **Speak clearly and continuously** for 3 seconds

#### Phase 3 — Hesitant Disfluency (letter "H" on display)
> Record broken, uncertain speech — like you're unsure of what to say.
- **Press Button A** to start recording
- **Speak with pauses and hesitation**: "Um... I think... maybe... the answer is..."

#### Phase 4 — Anxious Urgency (letter "A" on display)
> Record fast, pressured speech — like you're rushing or stressed.
- **Press Button A** to start recording
- **Speak quickly and intensely** for 3 seconds

#### Phase 5 — Disengaged Monotone (letter "M" on display)
> Record flat, low-energy speech — like you're bored or disinterested.
- **Press Button A** to start recording
- **Speak in a flat, monotone voice** for 3 seconds

### Step 4: Automatic data capture
The capture script automatically:
- Displays all serial output in your terminal
- Saves `data/calibration_data.json` with all samples (7 features per sample)
- Saves `data/calibration_log.txt` with the full log

**You don't need to copy-paste anything!** The data is saved automatically.

---

## 4. Step-by-Step: Teaching the AI

Teaching computes the optimal classifier parameters from your voice data.

### If using START_HERE.py (recommended)
Teaching happens **automatically** after you collect your voice — you don't need to do anything extra. The program detects that you have data and trains the AI automatically.

**The 93% Rule:** The script will automatically check how smart the AI is after teaching it. If its accuracy is below **93.0%**, it will warn you that the AI might get confused and will ask you if you want to record more voice samples to make it smarter!

### What happens during teaching
1. **Data split**: Your samples are randomly shuffled and split into training (70%) and testing (30%)
2. **Parameter computation**: The system calculates the best triangular membership function shapes to separate your five classes using all 8 features
3. **Rule weight optimization**: 13 fuzzy rules are individually weighted for best accuracy
4. **Validation**: It tests accuracy on the held-out test samples
5. **Wireless Update**: The PC automatically beams a simplified version of the AI brain to the micro:bit's internal memory!

If accuracy is below 93%, consider recalibrating with clearer audio or recording more samples!

---

## 5. Mode A: PC-Connected Classification (Live Detector)

> In this mode, the micro:bit records audio and **streams it to your PC** over USB. The PC runs the full 5-class classification with all 8 features and sends the result back.

### Step 1: Make sure the micro:bit is plugged in
You do not need to flash any new files! The Universal App handles this automatically.

### Step 2: Start the server
```bash
python START_HERE.py
```
Choose option **[4] Start the Live Detector!** or **[1] Do Everything!** which will start it automatically after teaching the AI.

### Step 3: Use the micro:bit
The micro:bit shows a music note icon when ready.

| Action | What happens |
|--------|-------------|
| **Press A** | 3-2-1 countdown → records 3 seconds → streams to PC |
| **Press B** | Sends "classify" command → PC runs fuzzy logic → result appears on LED |

### Step 4: Read the results

**On the micro:bit LED:**
- **Ambient Silence** → dim center pixel
- **Confident Articulation** → checkmark → happy face
- **Hesitant Disfluency** → confused face
- **Anxious Urgency** → surprised face
- **Disengaged Monotone** → sad face

**On your PC terminal:**
```
╔══════════════════════════════════════════════════════╗
║  🎤 Speech Classification Result                     ║
╠══════════════════════════════════════════════════════╣
║  Class: Confident Articulation     Confidence: 84%   ║
╠══════════════════════════════════════════════════════╣
║  Silence    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.02   ║
║  Confident  ████████████████████████░░░░░░░  0.72   ║
║  Hesitant   ███░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.10   ║
║  Anxious    █████░░░░░░░░░░░░░░░░░░░░░░░░░  0.14   ║
║  Monotone   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.02   ║
╚══════════════════════════════════════════════════════╝
```

### Step 5: Stop the server
Press **Ctrl+C** in the terminal.

---

## 6. Mode B: Standalone micro:bit Classification

> In this mode, **a simplified 3-class classifier runs on the micro:bit** — no PC connection needed after flashing. Great for demos!

### Important Note
The standalone micro:bit runs a **simplified 3-class version** (Silence, Confident, Hesitant) using 4 features and 6 rules. For the full 5-class experience with all 8 features and 13 rules, use **Mode A (PC-Connected)**.

### Step 1: Make sure you've trained
Run `python START_HERE.py` and complete voice collection + teaching first. During the teaching phase, the PC automatically beams the trained parameters to the micro:bit!

### Step 2: Unplug the micro:bit
Simply **disconnect the USB cable** — the micro:bit now works on its own!

### Step 3: Use it
Power the micro:bit with a battery pack or USB power source.

| Action | What happens |
|--------|-------------|
| **Press A** | 3-2-1 countdown → records 3 seconds → classifies → shows result |
| **Press B** | Shows instructions ("A=Listen") |

### What the LED shows after classification
- **Silence** → dim center pixel → scrolls "Silence XX%"
- **Confident** → checkmark + smiley → scrolls "Confident XX%"
- **Hesitant** → confused face → scrolls "Hesitant XX%"

---

## 7. Using START_HERE.py (Recommended)

`START_HERE.py` is the single entry point that handles everything. Just run:

```bash
python START_HERE.py
```

### Menu Options

| Option | Description |
|--------|-------------|
| **[1] Do Everything!** | Collect Voice → Teach AI → Start Live Detector (recommended for first-time use) |
| **[2] Add More Voice Samples** | Add more data to make the AI smarter |
| **[3] Re-teach the AI** | Run the learning math again using your existing data |
| **[4] Start the Live Detector!** | See the AI guess your mood in real-time (5-class, PC) |
| **[5] Check AI Status** | View current calibration and training state |
| **[6] Clear All Data** | Delete all recorded data and start fresh |
| **[7] Mass Data Recording** | Record a large volume of voice data at once |
| **[8] Prepare Micro:bit** | Push trained weights for standalone mode (3-class) |

### Typical workflow

**First time:**
1. Flash `FLASH_TO_MICROBIT.py` to micro:bit (you only ever do this once)
2. `python START_HERE.py` → choose **[1]** → follow prompts → auto-trains → start live detector
3. Press A on micro:bit to stream audio, press B to classify

**After first time:**
1. `python START_HERE.py` → choose **[1]** or **[4]** → live detector starts immediately (already trained)

**Need to re-teach the AI (moved rooms, different person):**
1. `python START_HERE.py` → choose **[2]** → redo voice collection → auto-retrains

---

## 8. Understanding the Output

### Features explained (8-feature system)

When you see features printed during classification:

```
avg_level:    85.3     ← Average loudness (0-255 scale)
speech_ratio: 0.45     ← 45% of the time had sound above threshold
num_gaps:     7        ← 7 pauses detected during recording
prosody_curv: 12.50    ← How dynamic the voice pitch/volume changes are
SRV:          1.80     ← How variable the speaking pace is
PDE:          1.20     ← How irregular the pause durations are
drift:        -0.500   ← Volume is trending downward
latency:      350ms    ← 350ms before speech started
```

### What good voice data looks like

| Class | avg_level | speech_ratio | num_gaps | prosody_curvature | SRV | PDE |
|-------|-----------|-------------|----------|-------------------|-----|-----|
| **Silence** | 2–10 | 0.00–0.05 | 0 | 0–2 | 0 | 0 |
| **Confident** | 120–170 | 0.75–0.95 | 1–3 | 15–25 | 0.2–0.5 | 0.1–0.3 |
| **Hesitant** | 55–80 | 0.25–0.45 | 8–15 | 5–10 | 2.0–3.0 | 1.2–1.8 |
| **Anxious** | 130–170 | 0.85–0.95 | 0–2 | 2–4 | 1.2–1.8 | 0.0–0.2 |
| **Monotone** | 60–80 | 0.40–0.55 | 4–6 | 1.5–3 | 0.2–0.4 | 0.1–0.3 |

Key differences:
- **Silence** has very low level and almost zero speech ratio
- **Confident** has high level with high curvature (dynamic voice)
- **Hesitant** has many gaps and high pause entropy
- **Anxious** has high level but LOW curvature (flat, rushed)
- **Monotone** has medium level with very low curvature and variability

---

## 9. Tips for Best Results

### During voice collection
- 🎤 **Speak close to the micro:bit** — about 6 inches (15 cm) from the microphone hole on the front
- 🤫 **Be actually quiet** during silence samples — don't whisper
- 🗣️ **Speak continuously** during confident samples — read a sentence from a book out loud
- 🤔 **Actually hesitate** during hesitant samples — say "um," pause, start and stop
- 😰 **Talk fast** during anxious samples — rush through words like you're stressed
- 😐 **Be monotone** during disengaged samples — speak flatly like you're bored

### If accuracy is low (Below 93%)
- **Add more voice samples** (Option [2] in `START_HERE.py`)
- **Record in a quieter room**
- **Speak louder** and closer to the microphone
- **Exaggerate the difference** between the 5 moods
- Make sure silence samples are truly silent (no typing, no fan noise near mic)

### For demos
- Use **PC Mode (Mode A)** for the full 5-class experience
- Use **standalone mode** (Mode B) for portability — note it's 3-class only
- Power with a battery pack for portability
- The micro:bit works best in the same environment where it was calibrated

---

## 10. Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'serial'` | Run `pip install pyserial` |
| "Waiting for micro:bit..." won't stop | Make sure the micro:bit is plugged in via USB. Close the micro:bit web editor's Serial panel. |
| Voice data looks wrong (all zeros) | The micro:bit might not be V2. V1 has no microphone. |
| Classifier always says "Silence" | Recalibrate — you may not have been loud enough. Speak 6 inches from the mic. |
| Classifier always says "Hesitant" | Recalibrate — try speaking more continuously during Confident samples. |
| Training accuracy below 93% | Recalibrate with clearer, more distinct samples for each class, or add more data! |
| micro:bit shows `NO` after recording | No audio data was captured. Ensure it's a V2 board. |
| Serial port not detected on Mac | Check System Preferences → Security & Privacy. Port is usually `/dev/cu.usbmodemXXXX`. |
| Can't flash — Editor says "no micro:bit" | Unplug and replug the USB cable. Make sure you are using a Chrome, Edge, or Opera browser. |

### Quick Reference: What to Flash

| What you're doing | Flash this file |
|-------------------|----------------|
| **Everything!** (Collecting Voice, PC Detector, Standalone) | `FLASH_TO_MICROBIT.py` |

> 🌟 **The Magic of the Universal App:** You only need to flash this file ONCE at the very beginning of the project. The PC will automatically send hidden signals to the micro:bit to change its modes!

> ⚠️ **Never flash files from the `src/` folder.** Those run on your computer, not the micro:bit.

> 📝 **PC vs. Standalone:** The PC provides full 5-class detection with 8 features and 13 rules. The standalone micro:bit provides a simplified 3-class version (Silence/Confident/Hesitant) with 4 features.
