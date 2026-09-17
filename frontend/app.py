import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

st.set_page_config(page_title="Bullet Hole Detection Demo", layout="wide")
st.title("🎯 Bullet Hole Detection Demo")
st.write("Upload an image of a target to detect bullet holes using the trained YOLO model.")

# Resolve the model path relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "backend" / "models" / "best.pt"

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

model_path = st.text_input("Path to best.pt", str(DEFAULT_MODEL_PATH))

if os.path.exists(model_path):
    model = load_model(model_path)
    st.success("Model loaded successfully!")
else:
    st.warning("Model not found at the specified path. Please enter the correct path.")
    st.stop()

uploaded_file = st.file_uploader("Choose a target image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tfile:
        tfile.write(uploaded_file.read())
        temp_path = tfile.name

    st.write("Running inference...")
    
    # Run YOLO model
    results = model(temp_path)
    result = results[0]
    
    # Extract info
    num_holes = len(result.boxes)
    boxes = result.boxes
    
    res_img = result.plot()
    
    # Convert BGR to RGB for Streamlit
    res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
    
    st.image(res_img_rgb, caption=f"Detected {num_holes} Bullet Holes", use_container_width=True)
    
    st.markdown(f"### Total Detected Bullet Holes: **{num_holes}**")
    
    if num_holes > 0:
        st.write("### Detection Details")
        det_data = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            det_data.append({"X1": x1, "Y1": y1, "X2": x2, "Y2": y2, "Confidence": f"{conf:.2f}"})
        
        df = pd.DataFrame(det_data)
        st.table(df)
        
    os.remove(temp_path)
