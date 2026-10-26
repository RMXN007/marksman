from ultralytics import YOLO
import torch
from pathlib import Path
import os
import shutil

# Resolve base paths dynamically
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_YAML = BASE_DIR / "backend" / "dataset" / "data.yaml"
RUNS_DIR = BASE_DIR / "backend" / "runs"
BEST_MODEL_DEST = BASE_DIR / "backend" / "models" / "best.pt"

def main():
    print("========================================")
    print("Airgun Target Analysis - Serious GPU Training")
    print("========================================")
    
    # Check GPU Availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU Detected: {gpu_name} ({vram:.2f} GB VRAM)")
        device = "0"
    else:
        print("❌ WARNING: CUDA is NOT available. PyTorch is using CPU.")
        print("Training will be extremely slow. Please fix PyTorch CUDA installation before proceeding.")
        return # Abort if no GPU
        
    print("\nInitializing YOLOv8 Small model...")
    # Using 'small' variant as it perfectly balances speed/accuracy for 6GB VRAM
    model = YOLO("yolov8s.pt")
    
    print("\nStarting Training Pipeline...")
    results = model.train(
        data=str(DATA_YAML),
        project=str(RUNS_DIR),
        name="experiment_01_yolov8s_640",
        
        # Core Architecture Configuration
        device=device,
        epochs=100,         # Sufficient ceiling
        patience=20,        # Early stopping if no improvement
        batch=8,            # Conservative batch size for 6GB VRAM at 640px
        imgsz=640,          # 640px captures small 4.5mm objects far better than 320px
        
        # Optimization
        amp=True,           # Automatic Mixed Precision for memory efficiency
        optimizer="auto",
        
        # Augmentation specifically tuned for Target Images
        # We limit extreme geometric distortions that could warp the hole shape
        degrees=10.0,       # Max rotation +/- 10 deg
        translate=0.1,      # Max translation 10%
        scale=0.2,          # Max scale +/- 20%
        shear=0.0,          # Disable shear (keeps holes circular)
        perspective=0.0,
        flipud=0.0,         # Targets usually aren't upside down
        fliplr=0.5,         # Horizontal flip is fine
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        
        # Save behavior
        save=True,
        exist_ok=False      # Prevents accidental overwriting of past experiments
    )
    
    print("\n========================================")
    print("Training Completed.")
    print(f"Experiment saved to: {RUNS_DIR / 'experiment_01_yolov8s_640'}")
    
    # We do NOT automatically overwrite backend/models/best.pt 
    # as per rule: "Do not overwrite the current model blindly before training results have been inspected."
    print("\n[!] The newly trained 'best.pt' is located inside the experiment folder.")
    print(f"Evaluate the metrics. If satisfied, manually copy it to:")
    print(f"{BEST_MODEL_DEST}")

if __name__ == '__main__':
    main()
