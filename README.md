# Mood_detect

A real-time facial emotion recognition system built with Python and OpenCV, using DeepFace for emotion classification.

---

## Overview

Mood_detect captures live webcam footage, detects faces, and classifies facial expressions into one of seven emotion categories. Results are displayed as an on-screen overlay and logged to a CSV file for further analysis.

**Detected emotions:** Angry · Disgust · Fear · Happy · Sad · Surprise · Neutral

---

## Features

| Feature | Description |
|---|---|
| Emotion Detection | 7-class classification via DeepFace |
| Confidence Bars | Per-emotion confidence scores displayed live |
| History Graph | 30-second rolling emotion timeline |
| Multi-face Support | Each detected face tracked and labeled independently |
| Mood Logging | Emotions saved to `mood_log.csv` with timestamps |
| Sound Feedback | Audio beep on emotion change |
| Alert System | Desktop notification after 3s of sustained anger or fear |

---

## Requirements

- Python 3.9
- Anaconda (recommended)

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/RashiTinkX/Mood_detect.git
cd Mood_detect
```

**2. Create and activate a conda environment**
```bash
conda create -n mooddetect python=3.9 -y
conda activate mooddetect
```

**3. Install dependencies**
```bash
pip install tensorflow==2.10.0
pip install "numpy<2"
pip install opencv-python deepface plyer
```

---

## Usage

```bash
conda activate mooddetect
python mood_swing.py
```

> On first run, DeepFace will automatically download its emotion model (~100MB). This only happens once.

Press **Q** to exit.

---

## Output

A `mood_log.csv` file is generated automatically in the project directory:

```
timestamp,           face_id, emotion, confidence
2026-02-22 12:55:00, 0,       happy,   0.923
```

---

## Project Structure

```
Mood_detect/
├── mood_swing.py                        # Main application
├── haarcascade_frontalface_default.xml  # Face detection model (OpenCV/Intel)
├── requirements.txt                     # Python dependencies
└── README.md
```

---

## Credits

| Component | Source |
|---|---|
| Face detection | [OpenCV Haar Cascade](https://github.com/opencv/opencv) — Intel Corporation |
| Emotion model | [DeepFace](https://github.com/serengil/deepface) — Sefik Ilkin Serengil |
