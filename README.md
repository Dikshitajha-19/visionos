# VisionOS — Hand Detection & Gesture Control System

Real-time hand detection and gesture recognition system built with Python, MediaPipe, OpenCV, and Flask.

---

## Features

| Mode | Description |
|------|-------------|
| 1 · AirDraw | Draw neon lines in the air with your index finger. Palm clears, fist pauses. Change colour with the button. |
| 2 · Gestures | Recognises 14 gestures: Hi, Bye, Namaste, Heart, Peace, Thumbs Up/Down, OK, Fist, Rock On, Call Me, Pinky Promise, Pointing, Flying Kiss |
| 3 · Finger Counter | Counts total extended fingers across up to 2 hands (max 10) |
| 4 · Rock Paper Scissors | Play vs AI — 3-second countdown, AI picks randomly, winner displayed |
| 5 · Face Puzzle | Frame any area with two hands, hold still 2 s to auto-capture, then drag tiles with a pinch gesture to solve |
| 6 · Shape Guesser | Draw any shape, AI auto-identifies it: Circle, Triangle, Square, Rectangle, Pentagon, Hexagon, Star, Ellipse, Octagon & more |

**Bonus features implemented:**
- Skeleton view, bounding box view, or both (toggle in UI)
- Multi-hand support (up to 2 hands simultaneously)
- Hand tracking across frames via MediaPipe

---

## Setup

### Requirements
- Python 3.11 (mediapipe does not support Python 3.12/3.13)
- Webcam

### Install

```bash
# 1. Clone / download this project
cd visionos

# 2. Create virtual environment with Python 3.11
py -3.11 -m venv venv

# 3. Activate
# Windows:
venv\Scripts\activate
# Mac / Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run
python app.py
```

### Open in browser
```
http://127.0.0.1:5000
```

---

## Project Structure

```
visionos/
├── app.py              # Backend — Flask + MediaPipe + OpenCV
├── requirements.txt    # Python dependencies
├── README.md
└── templates/
    └── index.html      # Frontend — vanilla JS + CSS
```

---

## AI Tools Used

| Tool | How it was used |
|------|----------------|
| Claude (Anthropic) | Used for guidance on implementation approaches, also helped in code writing, debugging suggestions, and resolving MediaPipe-related issues |

AI assistance was used selectively for problem-solving and learning.  
The system design, integration, testing, and final implementation decisions were independently handled and validated through hands-on development.
---

## Challenges

- **mediapipe version API break** — `mp.solutions` was removed in 0.10.30+. Solved by pinning to `0.10.9` which retains the stable solutions API.
- **Python 3.13 incompatibility** — mediapipe wheels only exist for ≤ 3.12. Solved by installing Python 3.11 side-by-side and creating a dedicated venv.
- **Camera startup delay** — Fixed by using `cv2.CAP_DSHOW` backend on Windows (DirectShow) which bypasses slow device enumeration.
- **Puzzle auto-capture** — Replaced manual SPACE-bar trigger with a stability detector: tracks bounding-box delta across frames, captures after 2 s of stillness with a visual countdown ring.

---

## Tech Stack

- **Computer Vision:** MediaPipe 0.10.9 (21-landmark hand model), OpenCV 4.10
- **Backend:** Python 3.11, Flask 3.0, Flask-CORS
- **Frontend:** Vanilla HTML/CSS/JavaScript (no framework needed)
