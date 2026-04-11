# Hand Gesture Recognition with Teachable Machine

A real-time hand gesture recognition system using **Google's Teachable Machine** for model training and **MediaPipe** for hand detection.

## Project Overview

This project recognizes hand gestures in real-time using a webcam. The system detects hand landmarks using MediaPipe and classifies gestures (Fist, Like, OK, One, Open Palm) using a TensorFlow/Keras model trained with Google's Teachable Machine.

### Features
- Real-time webcam input processing
- Multi-hand detection (up to 2 hands)
- High-confidence gesture classification
- Dual display: Main view with MediaPipe landmarks and AI feed (224x224)
- Easy-to-use inference pipeline

---

## Prerequisites

Before you begin, ensure you have the following installed on your system:
- **Python 3.8 or higher**
- **Git** (for cloning the repository)
- **Webcam** (for real-time inference)
- **pip** (Python package manager)

---

## Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/NagaNarveen30/hand_gesture_recognition.git
cd <folder name>
```

### Step 2: Create a Virtual Environment

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies included:**
- `opencv-python` - Image processing and webcam capture
- `mediapipe` - Hand landmark detection
- `numpy` - Numerical computations
- `tensorflow` / `keras` - Model inference
- `absl-py`, `astunparse`, `attrs`, `certifi` - Supporting libraries

---

## Project Structure

```
priya_dob/
├── main.py                 # Main inference script
├── main_backup.py          # Backup version
├── requirements.txt        # Python dependencies
├── project.tm              # Teachable Machine project file
├── README.md              # This file
│
├── model/
│   ├── keras_model.h5     # Trained Teachable Machine model
│   └── labels.txt         # Class labels (gesture names)
│
└── dataset/               # Original training data
    ├── Fist/              # Gesture samples
    ├── Like/
    ├── OK/
    ├── One/
    └── Open palm/
```

---

## Model Information

### Model Source: Google Teachable Machine

The model used in this project is trained using **[Google's Teachable Machine](https://teachablemachine.withgoogle.com/)**.

**Model Details:**
- **Model File:** `model/keras_model.h5` (Keras/TensorFlow format)
- **Labels File:** `model/labels.txt`
- **Training Method:** Transfer learning using MobileNet backbone
- **Input Size:** 224x224 pixels
- **Classes:** 5 hand gestures
  1. Fist
  2. Like
  3. OK
  4. One
  5. Open palm

### How the Model Was Trained

1. Collected hand gesture images for each class using Teachable Machine's built-in webcam capture
2. Used Teachable Machine's web interface to train the model
3. Exported the model in Keras (.h5) format
4. Integrated the model into this recognition pipeline

**To retrain or modify the model:**
1. Upload `project.tm` to [Teachable Machine](https://teachablemachine.withgoogle.com/)
2. Collect additional samples if needed
3. Train the model
4. Export as Keras format
5. Replace the files in the `model/` directory

---

## Running the Application

### Basic Usage

Once the environment is set up and dependencies are installed, run the inference:

```bash
python main.py
```

### What Happens

1. **Webcam Initialization:** Opens your default webcam (index 0)
2. **Real-time Detection:** 
   - MediaPipe detects hand landmarks in the video stream
   - Hand bounding box is extracted with 60-pixel padding
   - Input is resized to 224x224 pixels (required by the model)
3. **Classification:** 
   - Model predicts the gesture class
   - Confidence percentage is displayed
4. **Dual Display:**
   - **Main View:** Shows video feed with hand bounding box and gesture label
   - **AI Feed:** Shows the 224x224 input the model actually processes

### Controls

| Key | Action |
|-----|--------|
| **ESC** | Close the application and stop inference |

---

## Resources

- **Teachable Machine:** https://teachablemachine.withgoogle.com/
- **MediaPipe Documentation:** https://mediapipe.dev/
- **Keras Model Guide:** https://keras.io/
- **OpenCV Documentation:** https://docs.opencv.org/

---

## License

This project uses models trained with Google's Teachable Machine and open-source libraries (MediaPipe, TensorFlow, OpenCV).

---

**Happy gesture recognition! 🖐️**
