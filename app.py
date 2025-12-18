from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import cv2
import mediapipe as mp
import numpy as np
import io
import math
from collections import deque

app = FastAPI(title="Live Hand Gesture Detection - Advanced")

# Serve static files (index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8
)


# -------------------------------
# 🎯 Enhanced Gesture Detection
# -------------------------------


# history deque to stabilize prediction
gesture_history = deque(maxlen=5)


def finger_angle(a, b, c):
    """Return angle ABC (in degrees) from 3 landmark points."""
    ab = np.array([a.x - b.x, a.y - b.y])
    cb = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle


def detect_gesture(landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # --- Detect which fingers are open ---
    # Thumb
    if landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for i in range(1, 5):
        fingers.append(
            1 if landmarks[tip_ids[i]].y < landmarks[tip_ids[i] - 2].y else 0
        )

    total_fingers = fingers.count(1)

    # --- Pre-calc helper distances ---
    thumb_tip, index_tip = landmarks[4], landmarks[8]
    dist_thumb_index = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)

    # --- Gesture Detection Rules ---
    gesture = "🤔 Gesture Not Defined"

    if total_fingers == 0:
        gesture = "✊ Fist"

    elif total_fingers == 5:
        gesture = "🖐 Open Hand (5)"

    elif total_fingers == 1:
        if fingers == [1, 0, 0, 0, 0]:
            gesture = "👍 Thumbs Up"
        elif fingers == [0, 1, 0, 0, 0]:
            gesture = "☝️ One"
        else:
            gesture = "☝️ One (1)"

    elif total_fingers == 2:
        if fingers[1:3] == [1, 1]:
            if landmarks[8].y < landmarks[12].y:
                gesture = "✌️ Two (2)"
            else:
                gesture = "✌️ Peace Down"
        elif fingers == [1, 0, 0, 0, 1]:
            gesture = "🤙 Call Me"
        else:
            gesture = "✌️ Two (2)"

    elif total_fingers == 3:
        if fingers[0:3] == [1, 1, 1]:
            gesture = "🤟 Love (3)"
        elif fingers[1:4] == [1, 1, 1]:
            gesture = "3️⃣ Three (3)"
        else:
            gesture = "✋ Three (3)"

    elif total_fingers == 4:
        if fingers[0] == 0 and fingers[1:] == [1, 1, 1, 1]:
            gesture = "🖖 Four (4)"
        else:
            gesture = "✋ Four (4)"

    elif total_fingers == 5:
        gesture = "🖐 Open Hand (5)"

    # OK Sign
    if dist_thumb_index < 0.05 and fingers[2:] == [1, 1, 1]:
        gesture = "👌 OK Sign"

    # Middle finger only
    if fingers == [0, 0, 1, 0, 0]:
        gesture = "🖕 Middle Finger"

    # Rock sign
    if fingers == [0, 1, 0, 0, 1]:
        gesture = "🤘 Rock"

    # Punch (fist but wrist extended)
    wrist = landmarks[0]
    mid_finger_base = landmarks[9]
    if total_fingers == 0 and abs(wrist.y - mid_finger_base.y) < 0.1:
        gesture = "👊 Punch"

    # Stabilize with short-term history
    gesture_history.append(gesture)
    stable_gesture = max(set(gesture_history), key=gesture_history.count)

    return stable_gesture


# -------------------------------
# 🧠 API Routes
# -------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve webcam UI"""
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """Process a webcam frame and return annotated response"""
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    gesture = "None"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand_landmarks.landmark)

    cv2.putText(frame, gesture, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    _, img_encoded = cv2.imencode(".jpg", frame)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")
