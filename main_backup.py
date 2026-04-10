import os
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

# ── Paths & Setup ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model/keras_Model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "model/labels.txt")

# Load Teachable Machine Model
np.set_printoptions(suppress=True)
model = load_model(MODEL_PATH, compile=False)
class_names = [line.strip() for line in open(LABELS_PATH, "r").readlines()]

# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Visual Settings
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)

def main():
    cap = cv2.VideoCapture(0)
    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1, # Set to 1 for better stability with TM
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        ) as hands:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                h, w, c = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # 1. Get Bounding Box from landmarks
                        x_max = 0
                        y_max = 0
                        x_min = w
                        y_min = h
                        for lm in hand_landmarks.landmark:
                            x, y = int(lm.x * w), int(lm.y * h)
                            if x > x_max: x_max = x
                            if x < x_min: x_min = x
                            if y > y_max: y_max = y
                            if y < y_min: y_min = y

                        # Add padding to the crop
                        offset = 40
                        x_min, y_min = max(0, x_min - offset), max(0, y_min - offset)
                        x_max, y_max = min(w, x_max + offset), min(h, y_max + offset)

                        # 2. Crop and Prepare for Teachable Machine
                        try:
                            hand_img = frame[y_min:y_max, x_min:x_max]
                            img_input = cv2.resize(hand_img, (224, 224), interpolation=cv2.INTER_AREA)
                            img_input = np.asarray(img_input, dtype=np.float32).reshape(1, 224, 224, 3)
                            img_input = (img_input / 127.5) - 1

                            # 3. Predict
                            prediction = model.predict(img_input, verbose=0)
                            index = np.argmax(prediction)
                            label = class_names[index][2:] # Remove the "0 " prefix
                            conf = prediction[0][index]

                            # 4. Draw resultsds
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), BOX_COLOR, 2)
                            cv2.putText(frame, f"{label} ({int(conf*100)}%)", (x_min, y_min - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, BOX_COLOR, 2)
                            
                            # Optional: Draw landmarks
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        except Exception as e:
                            pass

                cv2.imshow("Hand Gesture Classifier", frame)
                if cv2.waitKey(1) & 0xFF == 27: break
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # This part runs even if the code crashes!
        print("Releasing Camera...")    
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()