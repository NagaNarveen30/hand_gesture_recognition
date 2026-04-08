import os
import cv2
import mediapipe as mp
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "gesture_model", "weights", "best.pt")

# ── Load model ─────────────────────────────────────────────────────────────
model = YOLO(MODEL_PATH)

# ── MediaPipe ──────────────────────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils

# ── Gesture → robot command mapping ───────────────────────────────────────
COMMANDS = {
    "fist":        "STOP",
    "index_left":  "TURN LEFT",
    "index_right": "TURN RIGHT",
    "open_palm":   "FORWARD",
    "point":       "REVERSE",
    "thumbs_up":   "SPEED UP",
    "thumbs_down": "SLOW DOWN",
}

# ── Colours (BGR) ──────────────────────────────────────────────────────────
BOX_COLOR  = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)
CMD_COLOR  = (0, 200, 255)

def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

def draw_prediction(frame, label, conf, cmd, x1, y1, x2, y2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
    cv2.putText(frame, cmd, (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, CMD_COLOR, 2)

def main():
    cap = cv2.VideoCapture(0)
    prev_tick = cv2.getTickCount()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ── FPS ────────────────────────────────────────────────────────
            curr_tick = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (curr_tick - prev_tick)
            prev_tick = curr_tick
            draw_fps(frame, fps)

            # ── MediaPipe landmarks ────────────────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            # ── YOLO detection + classification ───────────────────────────
            detections = model(frame, verbose=False)[0]
            for box in detections.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf  = float(box.conf[0])
                label = model.names[int(box.cls[0])]
                cmd   = COMMANDS.get(label, "UNKNOWN")
                draw_prediction(frame, label, conf, cmd, x1, y1, x2, y2)

            cv2.imshow("Hand Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()