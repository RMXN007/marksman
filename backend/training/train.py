from ultralytics import YOLO
from pathlib import Path
import os

# Resolve paths relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_YAML_PATH = BASE_DIR / "backend" / "dataset" / "data.yaml"
PROJECT_DIR = BASE_DIR / "backend" / "runs"

def train_model():
    # Load pretrained nano model for transfer learning
    model = YOLO("yolov8n.pt")
    
    print("Starting quick CPU training for demo purposes...")
    
    # Train the model with smaller image size and fewer epochs for fast CPU execution
    results = model.train(
        data=str(DATA_YAML_PATH),
        epochs=10,        # Reduced for quick training demo
        patience=5,       # Early stopping
        batch=16,
        imgsz=320,        # Significantly reduced from 640 to speed up CPU training
        project=str(PROJECT_DIR),
        name="bullet_hole_detection",
        device="cpu",     # Enforce CPU since GPU is seemingly unavailable (0G memory)
        pretrained=True,
        # Sensible augmentations
        degrees=5.0,
        translate=0.1,
        scale=0.2,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )
    
    # Evaluate
    metrics = model.val()
    
    print("\n--- Training Complete ---")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"Precision: {metrics.box.mp}")
    print(f"Recall: {metrics.box.mr}")

if __name__ == "__main__":
    train_model()
