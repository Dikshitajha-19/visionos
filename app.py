"""
VisionOS v2 - Complete Hand Detection & Gesture System
Fixes: Fast startup, RPS vs AI, Auto-capture puzzle, Drag tiles,
       Rich gestures, Stable AirDraw, AI Shape Guesser
"""

import cv2
import numpy as np
import random
import time
import math
import threading
from collections import deque
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import mediapipe as mp

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# MediaPipe — lazy init so camera starts instantly
# ─────────────────────────────────────────────
_hands_detector = None
_mp_drawing = None
_mp_hands = None

def get_detector():
    global _hands_detector, _mp_drawing, _mp_hands
    if _hands_detector is None:
        _mp_hands = mp.solutions.hands
        _mp_drawing = mp.solutions.drawing_utils
        _hands_detector = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
            model_complexity=0       # fastest model
        )
    return _hands_detector, _mp_drawing, _mp_hands

# Pre-warm detector in background thread so it's ready by the time user clicks start
def _prewarm():
    try:
        det, _, mp_h = get_detector()
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
        det.process(rgb)
    except Exception:
        pass

threading.Thread(target=_prewarm, daemon=True).start()

# ─────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────
state = {
    "mode": 1, "running": False,

    # AirDraw
    "draw_canvas": None,
    "draw_color_idx": 0,
    "prev_point": None,
    "draw_points": deque(maxlen=2000),   # stable smoothed trail

    # RPS
    "rps_score": {"player": 0, "ai": 0, "ties": 0},
    "rps_state": "waiting",   # waiting|countdown|freeze|result
    "rps_countdown_start": 0,
    "rps_frozen_gesture": "",
    "rps_ai_gesture": "",
    "rps_result": "",
    "rps_result_time": 0,

    # Puzzle
    "puzzle_tiles": None,
    "puzzle_positions": None,
    "puzzle_original_positions": None,
    "puzzle_size": None,
    "puzzle_bbox": None,
    "puzzle_captured": False,
    "puzzle_hold_start": None,   # for auto-capture
    "puzzle_hold_duration": 1.5, # seconds of stable hands to auto-capture
    "dragging_tile_idx": None,
    "drag_offset": (0, 0),
    "pinch_prev": None,

    # Shape guesser
    "shape_canvas": None,
    "shape_draw_points": [],
    "shape_result": "",
    "shape_result_time": 0,
    "shape_drawing": False,
    "shape_prev": None,
    "shape_last_point_time": 0,
    "shape_analyzed": False,
}

cap = None

DRAW_COLORS = [
    (0, 255, 200),   # neon teal
    (255, 80, 80),   # red
    (80, 150, 255),  # blue
    (255, 220, 0),   # yellow
    (200, 80, 255),  # purple
    (80, 255, 80),   # green
]

# ─────────────────────────────────────────────
# Gesture Helpers
# ─────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

def draw_skeleton_manual(img, lm_list, h, w, color=(100,100,255)):
    pts = [(int(lm.x*w), int(lm.y*h)) for lm in lm_list]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 2)
    for pt in pts:
        cv2.circle(img, pt, 4, (0,255,200), -1)

def lm_to_list(hand_landmarks):
    return hand_landmarks.landmark

def get_finger_states(lm, handedness="Right"):
    fingers = []
    if handedness == "Right":
        fingers.append(lm[4].x < lm[3].x)
    else:
        fingers.append(lm[4].x > lm[3].x)
    for tip, pip in [(8,6),(12,10),(16,14),(20,18)]:
        fingers.append(lm[tip].y < lm[pip].y)
    return fingers

def count_fingers(lm, handedness="Right"):
    return sum(get_finger_states(lm, handedness))

def is_open_palm(lm, handedness="Right"):
    return sum(get_finger_states(lm, handedness)) >= 4

def is_fist(lm, handedness="Right"):
    return sum(get_finger_states(lm, handedness)[1:]) == 0

def tip_pos(lm, idx, w, h):
    return (int(lm[idx].x * w), int(lm[idx].y * h))

def dist2d(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def pinch_distance(lm, w, h):
    t = tip_pos(lm, 4, w, h)
    i = tip_pos(lm, 8, w, h)
    return dist2d(t, i), ((t[0]+i[0])//2, (t[1]+i[1])//2)

def detect_rps(lm, handedness="Right"):
    fs = get_finger_states(lm, handedness)
    ext = sum(fs)
    if ext <= 1:      return "Rock"
    if fs[1] and fs[2] and not fs[3] and not fs[4]: return "Scissors"
    if ext >= 4:      return "Paper"
    return "?"

def rps_winner(p, a):
    if p == a: return "TIE"
    wins = {("Rock","Scissors"),("Paper","Rock"),("Scissors","Paper")}
    return "WIN" if (p, a) in wins else "LOSE"

# ─────────────────────────────────────────────
# Mode 1: Neon AirDraw (Smooth & Stable)
# ─────────────────────────────────────────────
def smooth_point(new_pt, prev_pt, alpha=0.5):
    """Exponential moving average for smooth drawing."""
    if prev_pt is None:
        return new_pt
    return (int(alpha * new_pt[0] + (1-alpha) * prev_pt[0]),
            int(alpha * new_pt[1] + (1-alpha) * prev_pt[1]))

def mode_airdraw(img, multi_hl, multi_hd, h, w):
    if state["draw_canvas"] is None or state["draw_canvas"].shape[:2] != (h, w):
        state["draw_canvas"] = np.zeros((h, w, 3), dtype=np.uint8)

    canvas = state["draw_canvas"]
    color = DRAW_COLORS[state["draw_color_idx"]]

    if multi_hl:
        lm = lm_to_list(multi_hl[0])
        handed = multi_hd[0].classification[0].label if multi_hd else "Right"
        draw_skeleton_manual(img, lm, h, w)

        tip = tip_pos(lm, 8, w, h)

        if is_open_palm(lm, handed):
            state["draw_canvas"] = np.zeros((h, w, 3), dtype=np.uint8)
            state["prev_point"] = None
            canvas = state["draw_canvas"]
            cv2.putText(img, "CLEARED", (w//2-60, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,100,100), 3)
        elif is_fist(lm, handed):
            state["prev_point"] = None
            cv2.putText(img, "PAUSED", (w//2-55, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200,200,100), 2)
        else:
            # Smooth the tip position
            smooth = smooth_point(tip, state["prev_point"], alpha=0.6)
            if state["prev_point"] is not None:
                cv2.line(canvas, state["prev_point"], smooth, color, 6)
                # Soft glow
                glow = np.zeros_like(canvas)
                cv2.line(glow, state["prev_point"], smooth, color, 18)
                glow = cv2.GaussianBlur(glow, (11, 11), 0)
                canvas = cv2.addWeighted(canvas, 1.0, glow, 0.5, 0)
                state["draw_canvas"] = canvas
            state["prev_point"] = smooth
            cv2.circle(img, smooth, 7, color, -1)
    else:
        state["prev_point"] = None

    img = cv2.addWeighted(img, 0.75, canvas, 1.0, 0)

    # HUD
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Index=Draw  Fist=Pause  Palm=Clear", (10, 30), font, 0.55, (150,150,150), 1)
    c_name = ["TEAL","RED","BLUE","YELLOW","PURPLE","GREEN"][state["draw_color_idx"]]
    cv2.putText(img, f"Color: {c_name}", (10, h-50), font, 0.6, color, 2)
    return img

# ─────────────────────────────────────────────
# Mode 2: Rock Paper Scissors vs AI
# ─────────────────────────────────────────────
RPS_CHOICES = ["Rock", "Paper", "Scissors"]
RPS_ASCII = {
    "Rock":     ["  ___  ", " /   \\ ", "|     |", "|     |", " \\___/ "],
    "Paper":    [" _____ ", "/     \\", "|     |", "|     |", "\\_____|"],
    "Scissors": ["  _  _ ", " | || |", " | || |", "  \\_/\\_/", "  scissors"],
}

def draw_rps_hand(img, choice, x, y, color):
    icons = {"Rock":"✊","Paper":"✋","Scissors":"✌"}
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(img, (x, y), (x+160, y+100), (30,30,30), -1)
    cv2.rectangle(img, (x, y), (x+160, y+100), color, 2)
    cv2.putText(img, icons.get(choice, "?"), (x+10, y+60), font, 2.0, color, 3)
    cv2.putText(img, choice, (x+10, y+90), font, 0.6, color, 2)

def mode_rps(img, multi_hl, multi_hd, h, w):
    font = cv2.FONT_HERSHEY_SIMPLEX
    now = time.time()

    # Detect current gesture (always show live)
    live_gesture = "?"
    if multi_hl:
        lm = lm_to_list(multi_hl[0])
        handed = multi_hd[0].classification[0].label if multi_hd else "Right"
        draw_skeleton_manual(img, lm, h, w)
        live_gesture = detect_rps(lm, handed)

    # === WAITING ===
    if state["rps_state"] == "waiting":
        cv2.putText(img, "Show your gesture!", (w//2-130, 50), font, 1.0, (200,200,200), 2)
        cv2.putText(img, "Press SPACE or button to play", (w//2-180, 80), font, 0.65, (150,150,150), 1)
        if live_gesture != "?":
            cv2.putText(img, f"Ready: {live_gesture}", (w//2-80, h-60), font, 1.2, (0,255,180), 3)

    # === COUNTDOWN ===
    elif state["rps_state"] == "countdown":
        elapsed = now - state["rps_countdown_start"]
        remaining = 3 - int(elapsed)
        if remaining <= 0:
            # Freeze player gesture and pick AI
            frozen = live_gesture if live_gesture != "?" else "Rock"
            ai = random.choice(RPS_CHOICES)
            result = rps_winner(frozen, ai)
            state.update({
                "rps_frozen_gesture": frozen,
                "rps_ai_gesture": ai,
                "rps_result": result,
                "rps_state": "result",
                "rps_result_time": now,
            })
            if result == "WIN":   state["rps_score"]["player"] += 1
            elif result == "LOSE": state["rps_score"]["ai"] += 1
            else:                  state["rps_score"]["ties"] += 1
        else:
            # Big countdown number
            text = str(remaining + 1)
            ts = cv2.getTextSize(text, font, 6, 15)[0]
            cv2.putText(img, text, ((w-ts[0])//2, (h+ts[1])//2), font, 6, (0,220,255), 15)
            cv2.putText(img, "GET READY!", (w//2-90, 55), font, 1.0, (255,200,0), 2)
            if live_gesture != "?":
                cv2.putText(img, f"Your gesture: {live_gesture}", (20, h-20), font, 0.8, (0,255,180), 2)

    # === RESULT ===
    elif state["rps_state"] == "result":
        pg  = state["rps_frozen_gesture"]
        ag  = state["rps_ai_gesture"]
        res = state["rps_result"]
        color = (0,220,80) if res=="WIN" else (60,60,255) if res=="LOSE" else (200,180,0)

        # VS panel in center
        panel_x, panel_y = w//2-220, h//2-120
        cv2.rectangle(img, (panel_x, panel_y), (panel_x+440, panel_y+230), (15,15,15), -1)
        cv2.rectangle(img, (panel_x, panel_y), (panel_x+440, panel_y+230), color, 3)

        # Player side
        cv2.putText(img, "YOU", (panel_x+20, panel_y+30), font, 0.8, (200,200,200), 2)
        draw_rps_hand(img, pg, panel_x+10, panel_y+40, (0,255,180))

        # AI side
        cv2.putText(img, "AI", (panel_x+300, panel_y+30), font, 0.8, (200,200,200), 2)
        draw_rps_hand(img, ag, panel_x+270, panel_y+40, (255,80,80))

        # VS text
        cv2.putText(img, "VS", (panel_x+200, panel_y+100), font, 1.2, (200,200,200), 3)

        # Result
        cv2.putText(img, res, (panel_x+155, panel_y+200), font, 2.0, color, 5)

        if now - state["rps_result_time"] > 3.5:
            state["rps_state"] = "waiting"

    # Score bar top
    sc = state["rps_score"]
    cv2.rectangle(img, (0,0), (w,40), (15,15,15), -1)
    cv2.putText(img, f"YOU: {sc['player']}   TIE: {sc['ties']}   AI: {sc['ai']}", (w//2-160, 28), font, 0.85, (200,200,200), 2)

    return img

# ─────────────────────────────────────────────
# Mode 3: Finger Counter
# ─────────────────────────────────────────────
def mode_finger_counter(img, multi_hl, multi_hd, h, w):
    font = cv2.FONT_HERSHEY_SIMPLEX
    total = 0
    if multi_hl:
        for i, hl in enumerate(multi_hl):
            lm = lm_to_list(hl)
            handed = multi_hd[i].classification[0].label if multi_hd else "Right"
            draw_skeleton_manual(img, lm, h, w)
            total += count_fingers(lm, handed)
    num = str(total)
    s, t = 7, 18
    ts = cv2.getTextSize(num, font, s, t)[0]
    cv2.putText(img, num, ((w-ts[0])//2+4, (h+ts[1])//2+4), font, s, (0,0,0), t+4)
    cv2.putText(img, num, ((w-ts[0])//2, (h+ts[1])//2), font, s, (0,255,180), t)
    n = len(multi_hl) if multi_hl else 0
    cv2.putText(img, f"{n} hand(s) detected", (20, h-20), font, 0.8, (180,180,180), 2)
    return img

# ─────────────────────────────────────────────
# Mode 4: Cat / Sign Language Gestures
# ─────────────────────────────────────────────
GESTURE_LIBRARY = {
    # name: (display_text, emoji, description)
    "namaste":      ("NAMASTE",         "🙏", "Both palms together"),
    "thumbs_up":    ("THUMBS UP",       "👍", "Only thumb extended"),
    "thumbs_down":  ("THUMBS DOWN",     "👎", "Thumb down"),
    "peace":        ("PEACE",           "✌️", "Index + Middle up"),
    "peace_both":   ("DOUBLE PEACE",    "✌️✌️", "Both hands peace"),
    "fist":         ("POWER FIST",      "✊", "All fingers closed"),
    "open_palm":    ("HELLO / HI",      "👋", "Open palm wave"),
    "wave_bye":     ("BYE BYE",         "👋", "Big open palm"),
    "heart_finger": ("FINGER HEART",    "🤍", "Thumb+Index cross"),
    "ok":           ("OK",              "👌", "Thumb+Index circle"),
    "call_me":      ("CALL ME",         "🤙", "Pinky+Thumb out"),
    "middle":       ("MIDDLE FINGER",   "🖕", "Only middle up"),
    "point_up":     ("POINTING UP",     "☝️", "Only index up"),
    "rock_on":      ("ROCK ON",         "🤘", "Index+Pinky out"),
    "flying_kiss":  ("FLYING KISS",     "😘", "Fingers to lips"),
    "i_love_you":   ("I LOVE YOU",      "🤟", "Thumb+Index+Pinky"),
    "stop":         ("STOP",            "🛑", "All 5 fingers flat"),
}

def detect_cat_gesture(lm, handedness="Right"):
    fs = get_finger_states(lm, handedness)
    thumb, idx, mid, ring, pinky = fs
    ext = sum(fs)

    if not thumb and not idx and not mid and not ring and not pinky:
        return "fist"
    if ext >= 5:
        return "stop"
    if ext == 4 and not thumb:
        return "wave_bye"
    if ext >= 4 and thumb:
        return "open_palm"
    if thumb and not idx and not mid and not ring and not pinky:
        return "thumbs_up"
    # thumbs down: thumb extended but pointing down (tip y > wrist y)
    if thumb and not idx and not mid and not ring and not pinky:
        if lm[4].y > lm[0].y:
            return "thumbs_down"
    if idx and mid and not ring and not pinky and not thumb:
        return "peace"
    if idx and mid and ring and not pinky and not thumb:
        return "middle"  # approximate
    if not idx and mid and not ring and not pinky and not thumb:
        return "middle"
    if idx and not mid and not ring and not pinky and not thumb:
        return "point_up"
    if idx and pinky and not mid and not ring and not thumb:
        return "rock_on"
    if thumb and idx and pinky and not mid and not ring:
        return "i_love_you"
    if thumb and pinky and not idx and not mid and not ring:
        return "call_me"
    # OK: thumb+index tips close
    d = dist2d((lm[4].x, lm[4].y), (lm[8].x, lm[8].y))
    if d < 0.07 and mid and ring and pinky:
        return "ok"
    # Finger heart: thumb+index crossed (x distance very small)
    dx = abs(lm[4].x - lm[8].x)
    dy = abs(lm[4].y - lm[8].y)
    if dx < 0.04 and dy < 0.06 and not mid and not ring and not pinky:
        return "heart_finger"
    # Flying kiss: fingers bunched near mouth area (y > 0.5)
    if lm[8].y > 0.55 and ext <= 2:
        return "flying_kiss"
    return None

def mode_cat_gestures(img, multi_hl, multi_hd, h, w):
    font = cv2.FONT_HERSHEY_SIMPLEX
    gesture_key = None
    is_both = False

    if multi_hl and len(multi_hl) >= 2:
        lm0 = lm_to_list(multi_hl[0])
        lm1 = lm_to_list(multi_hl[1])
        d = dist2d((lm0[0].x, lm0[0].y), (lm1[0].x, lm1[0].y))
        if d < 0.25:
            gesture_key = "namaste"
        else:
            # Both peace?
            h0 = multi_hd[0].classification[0].label if multi_hd else "Right"
            h1 = multi_hd[1].classification[0].label if multi_hd else "Left"
            g0 = detect_cat_gesture(lm0, h0)
            g1 = detect_cat_gesture(lm1, h1)
            draw_skeleton_manual(img, lm0, h, w)
            draw_skeleton_manual(img, lm1, h, w)
            if g0 == "peace" and g1 == "peace":
                gesture_key = "peace_both"
            elif g0:
                gesture_key = g0
            elif g1:
                gesture_key = g1
    elif multi_hl:
        lm = lm_to_list(multi_hl[0])
        handed = multi_hd[0].classification[0].label if multi_hd else "Right"
        draw_skeleton_manual(img, lm, h, w)
        gesture_key = detect_cat_gesture(lm, handed)

    if gesture_key and gesture_key in GESTURE_LIBRARY:
        txt, emoji, desc = GESTURE_LIBRARY[gesture_key]
        # Big center panel
        panel_w = 480
        px = (w - panel_w) // 2
        py = h - 130
        cv2.rectangle(img, (px, py), (px+panel_w, py+120), (0,0,0), -1)
        cv2.rectangle(img, (px, py), (px+panel_w, py+120), (0,200,150), 2)
        # Emoji-like big text
        cv2.putText(img, txt,  (px+20, py+55), font, 1.6, (0,255,180), 3)
        cv2.putText(img, desc, (px+20, py+95), font, 0.75, (150,200,180), 2)
        cv2.putText(img, emoji if emoji.isascii() else "", (px+panel_w-80, py+70), font, 1.5, (255,220,80), 3)

    # Guide in corner
    cv2.putText(img, "Show gestures to screen!", (10, 30), font, 0.6, (100,100,100), 1)
    return img

# ─────────────────────────────────────────────
# Mode 5: Face Puzzle (Auto-capture + Drag tiles)
# ─────────────────────────────────────────────
def mode_face_puzzle(img, multi_hl, multi_hd, h, w):
    font = cv2.FONT_HERSHEY_SIMPLEX
    now = time.time()

    if not state["puzzle_captured"]:
        # Show bbox when 2 hands present
        if multi_hl and len(multi_hl) >= 2:
            pts = []
            for hl in multi_hl:
                lm = lm_to_list(hl)
                draw_skeleton_manual(img, lm, h, w)
                for p in lm:
                    pts.append((int(p.x*w), int(p.y*h)))
            if pts:
                x1 = max(0, min(p[0] for p in pts)-30)
                y1 = max(0, min(p[1] for p in pts)-30)
                x2 = min(w, max(p[0] for p in pts)+30)
                y2 = min(h, max(p[1] for p in pts)+30)
                state["puzzle_bbox"] = (x1,y1,x2,y2)

                # Auto-capture: start hold timer
                if state["puzzle_hold_start"] is None:
                    state["puzzle_hold_start"] = now
                held = now - state["puzzle_hold_start"]
                progress = min(held / state["puzzle_hold_duration"], 1.0)

                # Draw box
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,180), 2)

                # Progress arc around box
                bar_w = int((x2-x1) * progress)
                cv2.rectangle(img, (x1, y2+4), (x1+bar_w, y2+12), (0,255,180), -1)
                cv2.rectangle(img, (x1, y2+4), (x2, y2+12), (40,40,40), 1)

                secs_left = max(0, state["puzzle_hold_duration"] - held)
                cv2.putText(img, f"Hold still... {secs_left:.1f}s", (x1, y1-10), font, 0.75, (0,255,180), 2)

                if progress >= 1.0:
                    _do_capture_puzzle(img, h, w)
                    state["puzzle_hold_start"] = None
        else:
            state["puzzle_hold_start"] = None
            cv2.putText(img, "Show BOTH hands to frame puzzle area", (w//2-230, h//2), font, 0.9, (150,150,150), 2)
            cv2.putText(img, "Hold still 1.5s to auto-capture", (w//2-195, h//2+35), font, 0.75, (100,100,100), 1)
            if multi_hl:
                lm = lm_to_list(multi_hl[0])
                draw_skeleton_manual(img, lm, h, w)

    else:
        # Draw shuffled tiles
        tiles = state["puzzle_tiles"]
        positions = state["puzzle_positions"]
        orig = state["puzzle_original_positions"]
        size = state["puzzle_size"]
        if tiles and positions and size:
            tw, th = size

            # Check pinch for dragging
            drag_pt = None
            if multi_hl:
                lm = lm_to_list(multi_hl[0])
                draw_skeleton_manual(img, lm, h, w)
                pd, pmid = pinch_distance(lm, w, h)
                if pd < 40:   # pinching
                    drag_pt = pmid
                    cv2.circle(img, pmid, 15, (255,220,0), 3)

            # Handle drag
            if drag_pt:
                if state["dragging_tile_idx"] is None:
                    # Pick up tile
                    for i, (px, py) in enumerate(positions):
                        if px <= drag_pt[0] <= px+tw and py <= drag_pt[1] <= py+th:
                            state["dragging_tile_idx"] = i
                            state["drag_offset"] = (drag_pt[0]-px, drag_pt[1]-py)
                            break
                else:
                    # Move tile
                    di = state["dragging_tile_idx"]
                    ox, oy = state["drag_offset"]
                    positions[di] = (drag_pt[0]-ox, drag_pt[1]-oy)
            else:
                # Drop tile — snap to nearest original slot
                if state["dragging_tile_idx"] is not None:
                    di = state["dragging_tile_idx"]
                    px, py = positions[di]
                    best_slot = min(range(len(orig)), key=lambda s: dist2d((px,py), orig[s]))
                    # Swap
                    for j, pos in enumerate(positions):
                        if j != di and abs(pos[0]-orig[best_slot][0]) < 10 and abs(pos[1]-orig[best_slot][1]) < 10:
                            positions[j] = positions[di]
                            break
                    positions[di] = orig[best_slot]
                    state["dragging_tile_idx"] = None

            # Draw tiles
            for i, (tile, (px, py)) in enumerate(zip(tiles, positions)):
                try:
                    if i == state["dragging_tile_idx"]:
                        # Draw dragged tile with highlight
                        overlay_tile(img, tile, px, py, tw, th, highlight=True)
                    else:
                        overlay_tile(img, tile, px, py, tw, th)
                except Exception:
                    pass

        cv2.putText(img, "Pinch fingers together + drag to move tiles", (10, h-15), font, 0.6, (150,150,150), 1)
        cv2.putText(img, "SPACE = Reset puzzle", (10, h-35), font, 0.55, (100,100,100), 1)

    return img

def overlay_tile(img, tile, px, py, tw, th, highlight=False):
    h, w = img.shape[:2]
    # Clamp
    x1 = max(0, px); y1 = max(0, py)
    x2 = min(w, px+tw); y2 = min(h, py+th)
    if x2 <= x1 or y2 <= y1:
        return
    tile_crop = tile[y1-py:y2-py, x1-px:x2-px]
    img[y1:y2, x1:x2] = tile_crop
    color = (0,255,220) if highlight else (80,80,80)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2 if highlight else 1)

def _do_capture_puzzle(img, h, w):
    bbox = state.get("puzzle_bbox") or (w//4, h//4, 3*w//4, 3*h//4)
    x1,y1,x2,y2 = bbox
    region = img[y1:y2, x1:x2].copy()
    bh, bw = region.shape[:2]
    bw = (bw//3)*3; bh = (bh//3)*3
    if bw == 0 or bh == 0: return
    region = cv2.resize(region, (bw, bh))
    tw, th = bw//3, bh//3
    tiles = [region[r*th:(r+1)*th, c*tw:(c+1)*tw].copy() for r in range(3) for c in range(3)]
    cx = w//2 - bw//2; cy = h//2 - bh//2
    orig_positions = [(cx+c*tw, cy+r*th) for r in range(3) for c in range(3)]
    shuffled = tiles[:]
    random.shuffle(shuffled)
    state.update({
        "puzzle_tiles": shuffled,
        "puzzle_positions": list(orig_positions),
        "puzzle_original_positions": list(orig_positions),
        "puzzle_size": (tw, th),
        "puzzle_captured": True,
        "dragging_tile_idx": None,
    })

# ─────────────────────────────────────────────
# Mode 6: AI Shape Guesser
# ─────────────────────────────────────────────
def detect_shape(canvas, h, w):
    """Detect shape drawn on canvas using contour analysis."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    # Dilate to connect strokes
    kernel = np.ones((9,9), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=3)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return "DRAW SOMETHING FIRST"

    # Get largest contour
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 500:
        return "TOO SMALL - DRAW BIGGER"

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.03 * peri, True)
    vertices = len(approx)

    # Circularity
    circularity = 4 * math.pi * area / (peri * peri) if peri > 0 else 0
    # Aspect ratio
    x, y, bw, bh = cv2.boundingRect(c)
    aspect = bw / bh if bh > 0 else 1

    # Convex hull for convexity
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area > 0 else 1

    if circularity > 0.80:
        if 0.85 < aspect < 1.15:
            return "CIRCLE  ⬤"
        else:
            return "ELLIPSE / OVAL  ⬭"

    if vertices == 3:
        return "TRIANGLE  △"

    if vertices == 4:
        if 0.85 < aspect < 1.15:
            return "SQUARE  ■"
        else:
            return "RECTANGLE  ▬"

    if vertices == 5:
        return "PENTAGON  ⬠"

    if vertices == 6:
        return "HEXAGON  ⬡"

    if vertices == 7 or vertices == 8:
        return "HEPTAGON / OCTAGON"

    if vertices > 8 and circularity > 0.6:
        return "CIRCLE  ⬤"

    if convexity < 0.7 and vertices > 5:
        return "STAR  ★"

    if vertices == 2 or (peri > 0 and area/peri < 5):
        return "LINE  ╱"

    if vertices > 10:
        return "ZIGZAG / WAVE  ≋"

    return "UNDEFINED SHAPE (?)"

def mode_shape_guesser(img, multi_hl, multi_hd, h, w):
    font = cv2.FONT_HERSHEY_SIMPLEX
    now = time.time()

    if state["shape_canvas"] is None or state["shape_canvas"].shape[:2] != (h, w):
        state["shape_canvas"] = np.zeros((h, w, 3), dtype=np.uint8)

    canvas = state["shape_canvas"]
    color = (255, 255, 255)

    if multi_hl:
        lm = lm_to_list(multi_hl[0])
        handed = multi_hd[0].classification[0].label if multi_hd else "Right"
        draw_skeleton_manual(img, lm, h, w)
        tip = tip_pos(lm, 8, w, h)

        if is_fist(lm, handed):
            # Fist = trigger analysis
            if not state["shape_analyzed"] and cv2.countNonZero(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)) > 100:
                state["shape_result"] = detect_shape(canvas, h, w)
                state["shape_result_time"] = now
                state["shape_analyzed"] = True
            state["shape_prev"] = None
            cv2.putText(img, "ANALYZING...", (w//2-100, h//2), font, 1.2, (255,220,0), 3)

        elif is_open_palm(lm, handed):
            # Palm = clear
            state["shape_canvas"] = np.zeros((h, w, 3), dtype=np.uint8)
            state["shape_result"] = ""
            state["shape_analyzed"] = False
            state["shape_prev"] = None
            canvas = state["shape_canvas"]
            cv2.putText(img, "CLEARED!", (w//2-70, h//2), font, 1.5, (255,100,100), 3)

        else:
            # Index finger draws — smooth
            smooth = smooth_point(tip, state["shape_prev"], alpha=0.5)
            if state["shape_prev"]:
                cv2.line(canvas, state["shape_prev"], smooth, color, 5)
            state["shape_prev"] = smooth
            state["shape_analyzed"] = False
            state["shape_last_point_time"] = now
            cv2.circle(img, smooth, 6, (255,255,255), -1)
    else:
        state["shape_prev"] = None
        # Auto-analyze after 1.5s of no drawing
        if (not state["shape_analyzed"] and
            state["shape_last_point_time"] > 0 and
            now - state["shape_last_point_time"] > 1.5 and
            cv2.countNonZero(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)) > 100):
            state["shape_result"] = detect_shape(canvas, h, w)
            state["shape_result_time"] = now
            state["shape_analyzed"] = True

    # Blend canvas
    draw_overlay = np.zeros_like(img)
    draw_overlay[canvas > 0] = (200, 220, 255)
    img = cv2.addWeighted(img, 1.0, draw_overlay, 0.9, 0)

    # Show result
    if state["shape_result"]:
        res = state["shape_result"]
        ts = cv2.getTextSize(res, font, 1.1, 3)[0]
        rx = (w - ts[0]) // 2
        cv2.rectangle(img, (rx-15, h-80), (rx+ts[0]+15, h-20), (0,0,0), -1)
        cv2.rectangle(img, (rx-15, h-80), (rx+ts[0]+15, h-20), (0,200,150), 2)
        cv2.putText(img, res, (rx, h-30), font, 1.1, (0,255,180), 3)

    # HUD
    cv2.putText(img, "Index=Draw  Fist=Guess  Palm=Clear  (also auto-guesses)", (8, 28), font, 0.5, (120,120,120), 1)
    return img

# ─────────────────────────────────────────────
# Overlay UI
# ─────────────────────────────────────────────
MODE_NAMES = {
    1:"AIRDRAW", 2:"ROCK PAPER SCISSORS", 3:"FINGER COUNTER",
    4:"SIGN LANGUAGE", 5:"FACE PUZZLE", 6:"SHAPE GUESSER"
}
MODE_COLORS = {
    1:(0,255,200), 2:(80,80,255), 3:(200,200,80),
    4:(255,160,0), 5:(180,80,255), 6:(80,200,255)
}

def draw_overlay_ui(img, mode, h, w):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = MODE_COLORS.get(mode, (255,255,255))
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h-34), (w, h), (8,8,8), -1)
    img = cv2.addWeighted(img, 0.35, overlay, 0.65, 0)
    cv2.putText(img, f"[{mode}] {MODE_NAMES.get(mode,'')}", (10, h-10), font, 0.7, color, 2)
    cv2.putText(img, "Keys 1-6: switch mode", (w-210, h-10), font, 0.5, (80,80,80), 1)
    return img

# ─────────────────────────────────────────────
# Video Generator
# ─────────────────────────────────────────────
def generate_frames():
    global cap
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduce latency

    det, mp_draw, mp_h = get_detector()

    while state["running"]:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = det.process(rgb)
        multi_hl = results.multi_hand_landmarks
        multi_hd = results.multi_handedness
        mode = state["mode"]

        if   mode == 1: img = mode_airdraw(img, multi_hl, multi_hd, h, w)
        elif mode == 2: img = mode_rps(img, multi_hl, multi_hd, h, w)
        elif mode == 3: img = mode_finger_counter(img, multi_hl, multi_hd, h, w)
        elif mode == 4: img = mode_cat_gestures(img, multi_hl, multi_hd, h, w)
        elif mode == 5: img = mode_face_puzzle(img, multi_hl, multi_hd, h, w)
        elif mode == 6: img = mode_shape_guesser(img, multi_hl, multi_hd, h, w)

        img = draw_overlay_ui(img, mode, h, w)
        ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    if cap: cap.release()

# ─────────────────────────────────────────────
# Flask Routes
# ─────────────────────────────────────────────
@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start():
    state["running"] = True
    return jsonify({"status": "started"})

@app.route('/stop', methods=['POST'])
def stop():
    state["running"] = False
    return jsonify({"status": "stopped"})

@app.route('/set_mode', methods=['POST'])
def set_mode():
    mode = int(request.json.get("mode", 1))
    state["mode"] = mode
    if mode == 1: state["draw_canvas"] = None; state["prev_point"] = None
    if mode == 2: state["rps_state"] = "waiting"
    if mode == 5: state["puzzle_captured"] = False; state["puzzle_tiles"] = None; state["puzzle_hold_start"] = None
    if mode == 6: state["shape_canvas"] = None; state["shape_result"] = ""; state["shape_analyzed"] = False; state["shape_prev"] = None; state["shape_last_point_time"] = 0
    return jsonify({"status": "ok", "mode": mode})

@app.route('/action', methods=['POST'])
def action():
    act = request.json.get("action", "")
    if act == "space":
        m = state["mode"]
        if m == 1:
            state["draw_canvas"] = None; state["prev_point"] = None
        elif m == 2 and state["rps_state"] == "waiting":
            state["rps_state"] = "countdown"
            state["rps_countdown_start"] = time.time()
        elif m == 5:
            state["puzzle_captured"] = False; state["puzzle_tiles"] = None; state["puzzle_hold_start"] = None
        elif m == 6:
            state["shape_canvas"] = None; state["shape_result"] = ""; state["shape_analyzed"] = False; state["shape_prev"] = None; state["shape_last_point_time"] = 0
    elif act == "color":
        state["draw_color_idx"] = (state["draw_color_idx"] + 1) % len(DRAW_COLORS)
    elif act == "reset_rps":
        state["rps_score"] = {"player":0,"ai":0,"ties":0}; state["rps_state"] = "waiting"
    return jsonify({"status": "ok"})

@app.route('/status')
def status():
    return jsonify({
        "running": state["running"], "mode": state["mode"],
        "rps_score": state["rps_score"], "rps_state": state["rps_state"],
        "puzzle_captured": state["puzzle_captured"],
    })

if __name__ == '__main__':
    print("🚀 VisionOS starting — pre-warming AI model...")
    app.run(debug=False, host='127.0.0.1', port=5000, threaded=True)