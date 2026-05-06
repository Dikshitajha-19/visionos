# VisionOS — Hand Detection & Gesture Control System

Real-time hand detection and gesture recognition system built with Python, MediaPipe, OpenCV, and Flask.

---
## screenshot
<img width="1781" height="1232" alt="Screenshot 2026-05-06 205229" src="https://github.com/user-attachments/assets/3a5e9e51-0780-43c0-ada6-44419c458cab" />
<img width="1112" height="1317" alt="Screenshot (122)" src="https://github.com/user-attachments/assets/4d6eb9f3-613d-4a5f-a199-4db6486ac425" />
<img width="1566" height="1253" alt="Screenshot (120)" src="https://github.com/user-attachments/assets/72754780-9de8-454a-aedf-a2b68114d946" />
<img width="927" height="1302" alt="Screenshot (124)" src="https://github.com/user-attachments/assets/189ae46b-5a44-4ed7-a4bd-1b9b3ab268a0" />
<img width="1646" height="1239" alt="Screenshot (125)" src="https://github.com/user-attachments/assets/94d5871f-eca4-4a0d-9885-1daf43c89fae" />
<img width="1582" height="1341" alt="Screenshot (126)" src="https://github.com/user-attachments/assets/dc553fee-ce6c-420a-ab27-eae7ac0efd28" />
<img width="1936" height="1317" alt="Screenshot 2026-05-06 211621" src="https://github.com/user-attachments/assets/f7354afb-590b-4447-9437-33f0b9bef97f" />



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

## ⚠️ Camera Initialization Note

When you start the application, the camera and hand detection model may take **3–4 minutes** to fully initialize, especially on the first run.

Please be patient and wait until the video feed appears before interacting with the system.

**Do not refresh or restart the app during this time.**

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
