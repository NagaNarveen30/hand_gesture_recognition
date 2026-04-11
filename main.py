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

def main():
    cap = cv2.VideoCapture(0) # Use 0 for default webcam, change to 1 for external camera (webcam)
    
    # Attempt to set HD resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        ) as hands:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # OPTIONAL: Flip if your model was trained with 'Mirror' on
                # frame = cv2.flip(frame, 1) 

                h, w, c = frame.shape
                # 1. Convert to RGB for MediaPipe and Model
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                # Default 'AI View' if no hand is detected
                ai_view = cv2.resize(frame, (224, 224)) 
                label = "No Hand Detected"
                conf_text = ""

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Calculate Bounding Box
                        x_coords = [lm.x for lm in hand_landmarks.landmark]
                        y_coords = [lm.y for lm in hand_landmarks.landmark]
                        
                        x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                        y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

                        # Padding (Crucial for TM models)
                        offset = 60 
                        x_min, y_min = max(0, x_min - offset), max(0, y_min - offset)
                        x_max, y_max = min(w, x_max + offset), min(h, y_max + offset)
                        
                        # 2. Extract the AI Feed (Crop from the RGB frame!)
                        hand_crop_rgb = rgb_frame[y_min:y_max, x_min:x_max]
                        
                        if hand_crop_rgb.size != 0:
                            # Prepare for TM (224x224, Normalized)
                            ai_view = cv2.resize(hand_crop_rgb, (224, 224), interpolation=cv2.INTER_AREA)
                            img_input = np.asarray(ai_view, dtype=np.float32).reshape(1, 224, 224, 3)
                            img_input = (img_input / 127.5) - 1 

                            # 3. Predict
                            prediction = model.predict(img_input, verbose=0)
                            index = np.argmax(prediction)
                            label = class_names[index][2:]
                            conf = prediction[0][index]
                            conf_text = f"{int(conf*100)}%"

                            # Draw on Main Window
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            cv2.putText(frame, f"{label} {conf_text}", (x_min, y_min - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # ── Display Dual Windows ─────────────────────────────────────
                # Convert ai_view back to BGR for display only
                ai_view_display = cv2.cvtColor(ai_view, cv2.COLOR_RGB2BGR)
                
                # Show Main Feed
                cv2.imshow("Main View (MediaPipe)", frame)
                
                # Show AI Feed (What the model is actually judging)
                cv2.imshow("AI Feed (224x224)", ai_view_display)

                if cv2.waitKey(1) & 0xFF == 27: break # ESC to quit

    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Closing application...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()