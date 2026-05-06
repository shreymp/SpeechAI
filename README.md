# 🎤 Fuzzy Logic Voice Mood Classifier

> A hands-on edge-computing project that classifies vocal moods in real-time using a BBC micro:bit V2 and fuzzy logic.

## Description

This project demonstrates the principles of edge AI and machine learning through an interactive, student-friendly system. By utilizing the onboard microphone of a BBC micro:bit V2, the system captures audio data, extracts acoustic features, and applies a custom-trained fuzzy logic classifier to determine the speaker's mood. Designed for educational environments, it offers an accessible introduction to artificial intelligence, digital signal processing, and hardware integration.

The system classifies voice audio into three distinct moods:
- **Background Noise** — silence / ambient
- **Confident Speaking** — clear, steady voice
- **Hesitant Speaking** — broken speech with pauses

## Quick Start (3 Steps)

### Step 1: Install Python packages
```bash
pip install -r requirements.txt
```

### Step 2: Flash the universal script
Open [python.microbit.org/v/3](https://python.microbit.org/v/3) in your browser, drag **`FLASH_TO_MICROBIT.py`** into the editor, and click **Flash**.

### Step 3: Run the program
```bash
python START_HERE.py
```

That's it! The program will guide you through everything:
1. **Calibrate** — Record 9 audio samples (3 per mood)
2. **Train** — Automatically compute the classifier parameters
3. **Classify** — Start the live classification server

📖 **Need more detail?** See [GUIDE.md](GUIDE.md) for full step-by-step instructions, how the science works, tips for best results, and troubleshooting.

---

## Project Structure

```
├── START_HERE.py           ★ Run this! Single entry point
├── FLASH_TO_MICROBIT.py    ★ Flash this to your micro:bit
├── GUIDE.md                Detailed student guide
├── requirements.txt        Python dependencies
├── README.md               This file
│
├── src/                    Backend Python scripts
│   ├── capture.py          Captures calibration serial data
│   ├── trainer.py          Trains the fuzzy logic model
│   └── server.py           Classification server
│
└── data/                   Auto-generated data files
    ├── calibration_data.json  Calibration samples
    ├── trained_params.json    Trained model parameters
    └── validation_report.txt  Accuracy report
```

## Requirements

- **Hardware:** BBC micro:bit V2 + USB cable
- **Software:** Python 3.7+, [micro:bit Python Editor V3](https://python.microbit.org/v/3)
- **Packages:** `pip install pyserial`

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: serial` | Run `pip install pyserial` |
| micro:bit not detected | Plug in USB, close any browser Serial panels |
| Low accuracy | Recalibrate: speak louder, closer to mic |
| Always says "Background Noise" | Recalibrate with option [2] in START_HERE.py |
