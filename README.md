# 🎤 Fuzzy Logic Voice Mood Classifier (5-Class)

> A hands-on edge-computing project that classifies vocal moods in real-time using a BBC micro:bit V2 and fuzzy logic.

## Description

This project demonstrates the principles of edge AI and machine learning through an interactive, student-friendly system. By utilizing the onboard microphone of a BBC micro:bit V2, the system captures audio data, extracts 8 distinct acoustic features, and applies a custom-trained 5-class Mamdani fuzzy logic classifier to determine the speaker's mood. Designed for educational environments, it offers an accessible introduction to artificial intelligence, digital signal processing, explainable AI, and hardware integration.

The system evaluates 13 fuzzy rules to classify voice audio into five distinct states:
- **Ambient Silence** — background noise, no speech
- **Confident Articulation** — clear, steady, paced voice
- **Hesitant Disfluency** — broken speech with irregular pauses
- **Anxious Urgency** — fast, highly variable speech
- **Disengaged Monotone** — flat, low-energy speech

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
1. **Calibrate** — Record 15 audio samples (3 per class)
2. **Train** — Automatically compute the classifier parameters and optimize rule weights
3. **Classify** — Start the live classification server with rule-trace explainability

📖 **Need more detail?** See [GUIDE.md](GUIDE.md) for full step-by-step instructions, how the science works, tips for best results, and troubleshooting. Technical documentation on the architecture and fuzzy engine can be found in the `docs/` folder.

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
│   ├── capture.py          Captures 5-class calibration serial data
│   ├── features.py         8-variable feature extraction
│   ├── fuzzy_engine.py     Membership functions and rule evaluation
│   ├── trainer.py          Trains the fuzzy logic model
│   ├── server.py           Classification server
│   ├── calibrator.py       Noise floor & adaptive thresholding
│   └── visualizer.py       Terminal UI and explainability
│
├── docs/                   Technical documentation
│   ├── architecture.md
│   └── membership_functions.md
│
├── tests/                  Unit testing suite
│   ├── test_features.py
│   └── test_fuzzy_engine.py
│
└── data/                   Auto-generated data files
    ├── calibration_data.json  Calibration samples
    ├── trained_params.json    Trained model parameters
    ├── session_log.csv        Longitudinal session history
    └── validation_report.txt  Accuracy report
```

## Requirements

- **Hardware:** BBC micro:bit V2 + USB cable
- **Software:** Python 3.9+, [micro:bit Python Editor V3](https://python.microbit.org/v/3)
- **Packages:** `pip install -r requirements.txt`

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: serial` | Run `pip install pyserial` |
| micro:bit not detected | Plug in USB, close any browser Serial panels |
| Low accuracy | Recalibrate: speak louder, closer to mic, exaggerate moods |
| Always says "Silence" | Ensure micro:bit is close enough to pick up your voice during calibration |
