# Marksman — AI-Powered Airgun Target Analysis

> **Release Version**: `v1.0 — College Demonstration Release`  
> **Status**: Stable & Verified  
> **Author**: Marksman Team  

Marksman is a computer vision and ballistics analysis platform designed for detecting 4.5mm airgun pellet impacts on paper targets using YOLOv8, calculating geometric grouping and dispersion metrics, and presenting comprehensive visual diagnostics and downloadable reports through an interactive Streamlit application.

Developed as a demonstrable college engineering project combining computer vision, object detection, and spatial ballistics analysis.

---

## 🎯 Problem Statement & Motivation

In recreational and competitive airgun shooting, shooters fire multiple shots into standardized paper targets to evaluate shot groupings and adjust sights (windage and elevation). Traditionally, analyzing target groups involves manual measurement using calipers or scoring gauges, which is tedious, prone to parallax errors, and lacks automated digital record-keeping.

Marksman solves this by providing:
1. **Automated pellet hole detection** using custom-trained deep learning object detection (YOLOv8s).
2. **Deterministic mathematical ballistics analytics** (center-of-mass, radial RMS, group dispersion, and maximum spread).
3. **Target center calibration** to decouple optical photo centers from the physical target bullseye.
4. **Automated export deliverables** including raw impact data (CSV), summary ballistics (CSV), and an executive multi-page audit report (PDF).

---

## 🌟 Key Capabilities & Features

1. **Dual Operational Modes**:
   - **Synthetic Demo Gallery**: Evaluates known synthetic target scenarios with mathematically exact ground-truth coordinates. Provides baseline demonstrations of grouping calculations independent of detector error.
   - **Live AI Analysis**: Runs deep learning inference using a trained YOLOv8s model on uploaded paper target photos to locate pellet holes.

2. **Modular Target Analysis Engine (`backend/analysis/target_engine.py`)**:
   - Reusable, framework-agnostic geometric calculations.
   - Individual impact offsets ($\Delta X$, $\Delta Y$) and radial distances relative to a calibrated target center.
   - **Mean Impact Point (MIP)**: Center-of-mass coordinate for assessing windage/elevation sight zeroing.
   - **Radial RMS ($D_{\text{RMS}}$)**: Root-mean-square distance from the target center ($D_{\text{RMS}} = \sqrt{\frac{1}{N}\sum r_i^2}$, measures zeroing/sighting accuracy).
   - **Group RMS Spread ($S_{\text{RMS}}$)**: Root-mean-square dispersion relative to the Mean Impact Point ($S_{\text{RMS}} = \sqrt{\frac{1}{N}\sum \|p_i - p_{\text{MIP}}\|^2}$, measures grouping tightness).
   - **Maximum Spread (Extreme Spread)**: Farthest pairwise distance between any two shots in the group.
   - Optional illustrative ring scoring for synthetic benchmark comparison (clearly segregated from official ISSF scoring).

3. **Visual Diagnostics Dashboard**:
   - Target diagram with numbered impacts, calibrated bullseye center crosshair, MIP diamond marker, and dispersion circle.
   - Polar coordinate chart displaying angular distribution and radial distance from $(0,0)$.
   - Cartesian offset chart illustrating shot placement relative to $(0,0)$.
   - Per-impact coordinate tables and one-click export buttons for per-impact CSV, summary metrics CSV, and multi-page PDF reports.

4. **Target Center Calibration**:
   - Avoids assuming that optical image centers align with paper target bullseyes.
   - Provides preset geometric defaults and manual pixel coordinate inputs $(X, Y)$ bounded by image dimensions.

5. **Multi-Page PDF Target Analysis Report (`backend/analysis/report_generator.py`)**:
   - Clean, professional downloadable PDF report generated using matplotlib's native PDF backend (zero extra dependencies).
   - Page 1 includes side-by-side original and annotated target imagery, summary metrics table, and technical integrity notices.
   - Page 2 includes per-impact coordinate audit logs, model specifications, detection confidence scores, and coordinate metadata.

---

## 📐 Mathematical Formulas

All measurements are computed in image pixel space ($\text{px}$) relative to calibrated target center $(C_x, C_y)$:

1. **Offsets**:
   $$\Delta X_i = X_i - C_x, \quad \Delta Y_i = Y_i - C_y$$
2. **Radial Distance**:
   $$r_i = \sqrt{\Delta X_i^2 + \Delta Y_i^2}$$
3. **Mean Impact Point (MIP)**:
   $$\bar{X} = \frac{1}{N} \sum_{i=1}^N X_i, \quad \bar{Y} = \frac{1}{N} \sum_{i=1}^N Y_i$$
4. **Radial RMS ($D_{\text{RMS}}$)**:
   $$D_{\text{RMS}} = \sqrt{\frac{1}{N} \sum_{i=1}^N r_i^2}$$
5. **Group RMS Spread ($S_{\text{RMS}}$)**:
   $$S_{\text{RMS}} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left[(X_i - \bar{X})^2 + (Y_i - \bar{Y})^2\right]}$$
6. **Maximum Pairwise Spread (Extreme Spread)**:
   $$E_{\text{spread}} = \max_{i < j} \sqrt{(X_i - X_j)^2 + (Y_i - Y_j)^2}$$

---

## 🏗️ Project Architecture

```text
marksman/
├── backend/
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── target_engine.py     # Core mathematical and grouping analysis engine
│   │   ├── visualizer.py        # Target diagrams, polar charts, and Cartesian plots
│   │   └── report_generator.py  # Professional multi-page PDF report generation
│   ├── dataset/                 # 80/20 train/val image split & YAML config
│   ├── inference/               # CLI inference utility scripts
│   ├── models/                  # Baseline models (best.pt)
│   ├── runs/                    # Model training experiments & validation metrics
│   │   └── experiment_01_yolov8s_640/
│   │       ├── weights/
│   │       │   └── best.pt      # Primary trained YOLOv8s model weights (21.5 MB)
│   │       └── results.csv      # Epoch-by-epoch loss & validation metrics
│   ├── tests/                   # Automated unit & integration test suite
│   │   ├── __init__.py
│   │   ├── test_target_engine.py
│   │   ├── test_integration.py
│   │   ├── test_verification_suite.py
│   │   └── test_exports.py
│   ├── training/                # GPU training scripts & smoke test suite
│   └── requirements.txt         # Project dependencies
│
├── frontend/
│   └── app.py                   # Demonstrable Streamlit UI & dashboard
│
├── run_frontend.bat             # 1-click launcher for Windows
└── README.md                    # Project documentation
```

---

## 🚀 Installation & Running

### Prerequisites
- Python 3.10, 3.11, or 3.12 (Tested on Python 3.11 / 3.12)
- Windows 10/11 (or Linux / macOS)
- Git

### Windows 1-Click Launch
Double-click `run_frontend.bat` from the project root. If running for the first time, this script configures the virtual environment, installs dependencies from `backend/requirements.txt`, and launches the application.

### Manual Command-Line Launch
```bash
# 1. Clone repository (or open project root directory)
cd "anu project"

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
# source .venv/bin/activate

# 4. Install dependencies
pip install -r backend\requirements.txt

# 5. Launch the Streamlit dashboard
streamlit run frontend\app.py
```

### Running Automated Test Suite
```bash
python -m unittest discover -s backend/tests
```
*(All 19 unit, integration, export, and verification tests execute in ~7-10 seconds)*

---

## 🤖 Model Specifications & Weights Setup

The primary model used in the application is **YOLOv8 Small (`YOLOv8s`)** trained at 640x640 resolution.

- **Primary Weights Path**: `backend/runs/experiment_01_yolov8s_640/weights/best.pt`
- **File Size**: `21.47 MB` (Well within GitHub's 100 MB file limit; suitable for direct Git hosting or GitHub Releases)
- **Architecture**: Ultralytics YOLOv8s (11.2M parameters)
- **Training Device**: NVIDIA GeForce RTX 3050 Laptop GPU (AMP enabled)
- **Input Resolution**: 640 x 640 px
- **Validation Dataset**: 104 independent target images
- **Best Validation Epoch**: Epoch 48 of 68 (Early stopping triggered at patience 20)

### Verified Validation Metrics:
- **Precision ($P$)**: `0.625`
- **Recall ($R$)**: `0.707`
- **mAP@50**: `0.720`
- **mAP@50-95**: `0.497`

> [!NOTE]
> **Validation Metrics Limitation**: Validation performance metrics reflect aggregate detector accuracy on the 104-image test split under controlled conditions. They do not guarantee 100% detection accuracy on arbitrary real-world target photos with varying lighting, torn paper edges, non-perpendicular camera angles, or overlapping holes.

---

## 📖 Example Workflow

1. **Launch App**: Open `http://localhost:8501` in your browser.
2. **Select Mode**:
   - Choose **Synthetic Demo Gallery** for instant verification of exact math across 3 predefined grouping scenarios.
   - Choose **Live AI Analysis** for real photographic target evaluation.
3. **Calibrate Bullseye Center**:
   - Check the target center coordinates. If the camera shot is offset from the center of the paper target card, adjust the Target Center X/Y coordinates to place $(0,0)$ on the bullseye.
4. **Inspect Diagnostics**:
   - Review the **YOLO Bounding Box** detection vs. **Spatial Target Overlay**.
   - Check **Radial RMS** (sighting zero accuracy) and **Group RMS Spread** (group compactness).
   - Inspect the **Polar Distribution Chart** for directional fliers and **Cartesian Offset Chart** for sight elevation/windage adjustment.
5. **Download Analysis Deliverables**:
   - Click `Download Per-Impact CSV` for tabular hole coordinate data.
   - Click `Download Summary CSV` for group statistics.
   - Click `Download Analysis PDF Report` for a printable, high-resolution multi-page audit document.

---

## 🖼️ Application Screenshots & Visuals

| Synthetic Demo Gallery | Live AI Detection & Overlay |
| :---: | :---: |
| *(Annotated target with MIP diamond, group RMS circle, and extreme spread line)* | *(Dual view: YOLOv8s bounding boxes side-by-side with spatial ballistics overlay)* |

| Polar Impact Distribution | Cartesian Offset Chart |
| :---: | :---: |
| *(Polar angular distribution and Radial RMS circle)* | *(Offsets relative to calibrated Center (0,0) with MIP indicator)* |

---

## ⚠️ Known Limitations & Technical Integrity

1. **Approximate Centers**: Impact centers derived from YOLO bounding-box midpoints $\left(\frac{x_1+x_2}{2}, \frac{y_1+y_2}{2}\right)$ are approximations. Sub-millimeter ring gauge accuracy requires homographic perspective unwarping and sub-pixel edge circle fitting.
2. **Confidence vs. Shooting Skill**: Detection confidence ($0.0 - 1.0$) measures the model's certainty that an object is a pellet hole; it has no correlation with the shooter's marksmanship accuracy.
3. **Target Calibration**: Photos taken with perspective distortion or off-center angles require geometric calibration. The system allows manual coordinate calibration, but physical metric conversion ($\text{mm} / \text{px}$) requires knowing target ring scale.
4. **Scoring Systems**: Ring scores in synthetic mode are illustrative heuristics for comparing scenarios. They are not official ISSF electronic scoring target (EST) certifications.

---

## 🔧 Troubleshooting

- **Error: "Model file not found"**: Ensure the trained weights exist at `backend/runs/experiment_01_yolov8s_640/weights/best.pt`. You can also enter a custom path in the "Detector Settings & Model Configuration" expander.
- **Detector finds 0 impacts**: Adjust the confidence slider down to `0.15`–`0.20` in the Detector Settings expander. Ensure the target photo is clear and well-lit.
- **Streamlit Port In Use**: Run on a different port: `streamlit run frontend/app.py --server.port 8502`.

---

## 🔮 Future Improvements

- Automatic perspective unwarping via target corner / border quad detection.
- Sub-pixel circle fitting for overlapping pellet holes (doubles).
- Scale calibration using known target dimensions (e.g. 17x17 cm target card standard) to convert pixel dispersion directly to millimeters and Minute of Angle (MOA).
