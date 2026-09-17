# Airgun Target Analysis

A computer vision project for detecting extremely small 4.5mm airgun bullet impacts on paper targets using YOLOv8, complete with a local Python inference backend and an interactive Streamlit frontend.

## Project Structure

This project has been cleanly separated into frontend and backend logic, free of absolute path dependencies:

```text
anu project/
├── backend/
│   ├── dataset/             # The 80/20 train/val image split & YAML config
│   ├── inference/           # Python scripts for CLI inference testing
│   ├── models/              # The trained weights (best.pt)
│   ├── runs/                # Complete Ultralytics output (metrics, graphs, etc)
│   ├── training/            # Python scripts to replicate the YOLO CPU training
│   ├── requirements.txt     # Python dependencies for the project
│   └── *.md                 # Complete, detailed documentation of the architecture
│
├── frontend/
│   └── app.py               # The interactive Streamlit user interface
│
├── run_frontend.bat         # 1-click launcher for Windows
└── README.md                # This file
```

## Running the Application

### 1-Click Launch (Windows)
Double-click `run_frontend.bat`. 
If it's your first time running the project, this script will automatically create a Python virtual environment (`.venv`), install all requirements from `backend/requirements.txt`, and launch the web app.

### Manual Launch (Command Line)
If you prefer to run it manually from the root `anu project` directory:
```bash
# 1. Create a virtual environment
py -m venv .venv

# 2. Activate the environment
.venv\Scripts\activate

# 3. Install dependencies
py -m pip install -r backend\requirements.txt

# 4. Launch the application!
py -m streamlit run frontend\app.py
```

## Running Inference via CLI
If you want to test the model in your terminal rather than the web app:
```bash
# Make sure your virtual environment is active, then run:
py backend\inference\inference.py --image "backend\dataset\images\val\IMG-20260315-WA0001.jpg"
```
*(This automatically defaults to using `backend/models/best.pt`)*

## Current Model Metrics (YOLOv8 Nano)
- **Total images:** 518
- **Total bounding boxes:** 4,042
- **Train/validation split:** 414/104 images
- **Training device:** CPU (10 epochs at 320x320 resolution)
- **Training time:** ~14 minutes
- **Precision:** 0.407
- **Recall:** 0.362
- **mAP50:** 0.271
- **mAP50-95:** 0.112

*Note: The current labels were generated via preliminary heuristic scripts. The model quality will be improved in future iterations through better manual annotation in Label Studio, higher resolution training, and GPU tuning.*
