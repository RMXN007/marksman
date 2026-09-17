import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
import datetime

# Resolve paths relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "backend" / "models" / "best.pt"

def run_inference(image_path, model_path):
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Running inference on {image_path}...")
    results = model(image_path)
    
    # Process results
    for result in results:
        boxes = result.boxes
        print(f"\nDetected {len(boxes)} bullet holes.")
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            print(f" - Bounding Box: ({x1}, {y1}) to ({x2}, {y2}) | Confidence: {conf:.2f}")
            
        res_img = result.plot()
        
        out_dir = BASE_DIR / "backend" / "inference" / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        img_name = Path(image_path).stem
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_filename = f"{img_name}_inference_{timestamp}.jpg"
        out_path = out_dir / out_filename
        
        cv2.imwrite(str(out_path), res_img)
        print(f"\nInference completed.\nDetected holes: {len(boxes)}")
        try:
            print(f"Output saved to:\n{out_path.relative_to(BASE_DIR)}")
        except ValueError:
            print(f"Output saved to:\n{out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to best.pt model")
    args = parser.parse_args()
    
    run_inference(args.image, args.model)
