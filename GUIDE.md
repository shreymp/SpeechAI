# Fuzzy Logic Voice Mood Classifier — Detailed Guide (Student Edition)

> **Project goal:** Classify audio from a BBC micro:bit V2's built-in microphone into three categories:
> - **Background Noise** — ambient / no speech
> - **Confident Speaking** — clear, steady voice, few pauses
> - **Hesitant Speaking** — broken speech, gaps, variable loudness

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [How It Works (The Science)](#2-how-it-works)
3. [Step-by-Step: Collecting Voice Data (Calibration)](#3-step-by-step-collecting-voice-data)
4. [Step-by-Step: Teaching the AI (Training)](#4-step-by-step-teaching-the-ai)
5. [Mode A: PC-Connected Classification](#5-mode-a-pc-connected-classification)
6. [Mode B: Standalone micro:bit Classification](#6-mode-b-standalone-microbit-classification)
7. [Using run.py (Recommended)](#7-using-runpy-recommended)
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

### What the classifier does
From those sound level readings, it extracts **4 features**:

| Feature | What it measures | Example values |
|---------|-----------------|----------------|
| **avg_level** | How loud is it overall? | Silence: 2–5, Speech: 20–40 |
| **speech_ratio** | What % of time has sound above threshold? | Silence: 0.00, Speech: 0.25–0.45 |
| **num_gaps** | How many pauses are there? | Confident: 13–17, Hesitant: 15–23 |
| **variance** | How much does loudness jump around? | Steady: 30, Variable: 800–1000 |

Then it uses **triangular membership functions** and **fuzzy rules** to classify the audio:

```
IF level is LOW  AND speech_ratio is LOW  → Background Noise
IF level is HIGH AND speech_ratio is HIGH → Confident Speaking
IF speech_ratio is MED AND gaps are MANY  → Hesitant Speaking
```

### Why collecting your voice matters
Every room and every voice is different. Collecting voice data records **your** voice in **your** environment so the AI knows what "quiet," "confident," and "hesitant" sound like for you.

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
The micro:bit will guide you through **3 phases**, 3 samples each (9 recordings total):

#### Phase 1 — Background Noise (letter "B" on display)
> Record what the room sounds like when nobody is talking.
- The display shows `B`, then `1`
- **Press Button A** to start recording
- **Stay quiet** for 3 seconds — just let the room noise be recorded
- A ✓ appears when done
- Repeat for samples 2 and 3

#### Phase 2 — Confident Speaking (letter "C" on display)
> Record clear, steady speech — like you're presenting to a class.
- The display shows `C`, then `1`
- **Press Button A** to start recording
- **Speak clearly and continuously** for 3 seconds
  - Example: Read a sentence from a book out loud
  - Speak at a normal volume, close to the micro:bit (~6 inches)
- Repeat for samples 2 and 3

#### Phase 3 — Hesitant Speaking (letter "H" on display)
> Record broken, uncertain speech — like you're unsure of what to say.
- The display shows `H`, then `1`
- **Press Button A** to start recording
- **Speak with pauses and hesitation** for 3 seconds
  - Example: "Um... I think... maybe... the answer is..."
  - Say a few words, pause, say more, pause
- Repeat for samples 2 and 3

### Step 4: Automatic data capture
The capture script automatically:
- Displays all serial output in your terminal
- Saves `data/calibration_data.json` with all 9 samples
- Saves `data/calibration_log.txt` with the full log

**You don't need to copy-paste anything!** The data is saved automatically.

---

## 4. Step-by-Step: Teaching the AI

Teaching computes the optimal classifier parameters from your voice data.

### If using START_HERE.py (recommended)
Teaching happens **automatically** after you collect your voice — you don't need to do anything extra. The program detects that you have data and trains the AI automatically. 

**The 93% Rule:** The script will automatically check how smart the AI is after teaching it. If its accuracy is below **93.0%**, it will warn you that the AI might get confused and will ask you if you want to record more voice samples to make it smarter!

### What happens during teaching
1. **Data split**: Your 9 samples are split into 6 for training and 3 for testing
2. **Parameter computation**: The system calculates the best triangular membership function shapes to separate your three classes
3. **Validation**: It tests accuracy on the held-out test samples
4. **Wireless Update**: The PC automatically beams the new AI brain over the USB cable directly to the micro:bit's internal memory!

If accuracy is below 93%, consider recalibrating with clearer audio or recording more samples!

---

## 5. Mode A: PC-Connected Classification (Live Detector)

> In this mode, the micro:bit records audio and **streams it to your PC** over USB. The PC runs the classification and sends the result back to the micro:bit.

### Step 1: Make sure the micro:bit is plugged in
You do not need to flash any new files! The Universal App handles this automatically.

### Step 2: Start the server
```bash
python START_HERE.py
```
Choose option **[4] Start the Live Detector!** or **[1] Do Everything!** which will start it automatically after teaching the AI.

You should see:
```
  🚀 Starting the Live Display! Talk into the micro:bit to see it work...
```

### Step 3: Use the micro:bit
The micro:bit shows a music note icon when ready.

| Action | What happens |
|--------|-------------|
| **Press A** | 3-2-1 countdown → records 3 seconds → streams to PC |
| **Press B** | Sends "classify" command → PC runs fuzzy logic → result appears on LED |

### Step 4: Read the results

**On the micro:bit LED:**
- **Background Noise** → dim center pixel, gentle breathing animation
- **Confident Speaking** → happy face, bouncing animation
- **Hesitant Speaking** → confused face, pulsing animation

**On your PC terminal:**
```
╔══════════════════════════════════════╗
║  RESULT: Confident Speaking          ║
║  Confidence: 87%                     ║
╚══════════════════════════════════════╝
```

### Step 5: Stop the server
Press **Ctrl+C** in the terminal.

---

## 6. Mode B: Standalone micro:bit Classification

> In this mode, **everything runs on the micro:bit** — no PC connection needed after flashing. Great for demos!

### Step 1: Make sure you've trained
Run `python START_HERE.py` and complete voice collection + teaching first. During the teaching phase, the PC automatically beams the new parameters to the micro:bit!

### Step 2: Unplug the micro:bit
Because the PC automatically beamed the trained parameters to the micro:bit, it is already programmed! 
Simply **disconnect the USB cable** — the micro:bit now works on its own!

### Step 3: Use it
Power the micro:bit with a battery pack or USB power source.

| Action | What happens |
|--------|-------------|
| **Press A** | 3-2-1 countdown → records 3 seconds → classifies → shows result |
| **Press B** | Shows instructions ("A=Listen") |

### What the LED shows after classification
- **Background Noise** → dim center pixel → scrolls "Background Noise XX%"
- **Confident Speaking** → checkmark + smiley animation → scrolls "Confident Speaking XX%"
- **Hesitant Speaking** → confused face with pulsing → scrolls "Hesitant Speaking XX%"

---

## 7. Using START_HERE.py (Recommended)

`START_HERE.py` is the single entry point that handles everything. Just run:

```bash
python START_HERE.py
```

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

### Features explained

When you see features printed during voice collection or classification:

```
avg_level    = 24.4     ← Average loudness (0-255 scale)
speech_ratio = 0.28     ← 28% of the time had sound above threshold
num_gaps     = 13       ← 13 pauses detected during recording
variance     = 891      ← How much the volume jumped around
```

### What good voice data looks like

| Class | avg_level | speech_ratio | num_gaps | variance |
|-------|-----------|-------------|----------|----------|
| **Background Noise** | 1–10 | 0.00–0.05 | 0–1 | 10–120 |
| **Confident Speaking** | 20–40+ | 0.25–0.50 | 10–20 | 600–1100 |
| **Hesitant Speaking** | 15–35 | 0.20–0.40 | 15–25+ | 700–1000 |

Key differences:
- **BG Noise** has very low level and almost zero speech ratio
- **Confident** has higher level and speech ratio than hesitant
- **Hesitant** has more gaps than confident

---

## 9. Tips for Best Results

### During voice collection
- 🎤 **Speak close to the micro:bit** — about 6 inches (15 cm) from the microphone hole on the front
- 🤫 **Be actually quiet** during background noise samples — don't whisper
- 🗣️ **Speak continuously** during confident samples — read a sentence without pausing
- 🤔 **Actually hesitate** during hesitant samples — say "um," pause, start and stop

### If accuracy is low (Below 93%)
- **Add more voice samples** (Option [2] in `START_HERE.py`)
- **Record in a quieter room**
- **Speak louder** and closer to the microphone
- **Exaggerate the difference** between confident and hesitant
- Make sure background noise samples are truly silent (no typing, no fan noise near mic)

### For demos
- Use **standalone mode** (Mode B) — no PC or cables needed
- Power with a battery pack for portability
- The micro:bit works best in the same environment where it was calibrated

---

## 10. Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'serial'` | Run `pip install pyserial` |
| "Waiting for micro:bit..." won't stop | Make sure the micro:bit is plugged in via USB. Close the micro:bit web editor's Serial panel. |
| Voice data looks wrong (all zeros) | The micro:bit might not be V2. V1 has no microphone. |
| Classifier always says "Background Noise" | Recalibrate — you may not have been loud enough. Speak 6 inches from the mic. |
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
