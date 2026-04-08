import os
import yaml
from ultralytics import YOLO

# Dynamically resolve dataset path
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "dataset")
DATA_YAML = os.path.join(DATA_DIR, "data.yaml")

# Patch the 'path' field in data.yaml to the absolute dataset directory
with open(DATA_YAML, "r") as f:
    data_cfg = yaml.safe_load(f)

data_cfg["path"] = DATA_DIR

with open(DATA_YAML, "w") as f:
    yaml.dump(data_cfg, f)

# Load pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train
results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=16,
    name="gesture_model",
    patience=10,
)

# Evaluate
metrics = model.val()
print(f"mAP50:     {metrics.box.map50:.4f}")
print(f"mAP50-95:  {metrics.box.map:.4f}")
print(f"\nBest model saved to: runs/detect/gesture_model/weights/best.pt")