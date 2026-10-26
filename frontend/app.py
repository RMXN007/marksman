import math
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Add repository root to python sys.path to enable backend module imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.analysis.target_engine import analyze_target_impacts, GroupMetrics
from backend.analysis.visualizer import (
    render_annotated_target,
    plot_radial_polar_chart,
    plot_offset_scatter_chart,
)
from backend.analysis.report_generator import generate_marksman_pdf_report

st.set_page_config(
    page_title="Marksman | AI-Powered Airgun Target Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished, clean, demonstrable presentation
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .legend-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        align-items: center;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 8px 14px;
        margin-bottom: 12px;
        font-size: 0.85rem;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .badge-synthetic {
        display: inline-block;
        background-color: #fef3c7;
        color: #92400e;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #fde68a;
    }
    .badge-live {
        display: inline-block;
        background-color: #dbeafe;
        color: #1e40af;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #bfdbfe;
    }
    /* Theme-aware Sidebar Title: Deep Navy in Light Theme (#172554), Near-White in Dark Theme (#F8FAFC) */
    .sidebar-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: var(--text-color, #172554);
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
        line-height: 1.25;
        letter-spacing: -0.01em;
    }
    @media (prefers-color-scheme: dark) {
        .sidebar-title {
            color: #F8FAFC !important;
        }
    }
    @media (prefers-color-scheme: light) {
        .sidebar-title {
            color: #172554 !important;
        }
    }
    [data-theme="dark"] .sidebar-title,
    [data-testid="stAppViewContainer"][data-theme="dark"] .sidebar-title,
    .stApp[data-theme="dark"] .sidebar-title,
    section[data-testid="stSidebar"][data-theme="dark"] .sidebar-title {
        color: #F8FAFC !important;
    }
    [data-theme="light"] .sidebar-title,
    [data-testid="stAppViewContainer"][data-theme="light"] .sidebar-title,
    .stApp[data-theme="light"] .sidebar-title,
    section[data-testid="stSidebar"][data-theme="light"] .sidebar-title {
        color: #172554 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

EXPERIMENT_MODEL = ROOT_DIR / "backend" / "runs" / "experiment_01_yolov8s_640" / "weights" / "best.pt"
LEGACY_MODEL = ROOT_DIR / "backend" / "models" / "best.pt"

# Known Synthetic Demonstration Scenarios
DEMO_SCENARIOS = {
    "Demo 1 — Tight grouping": {
        "description": "Compact pellet cluster near the target center (minimal dispersion).",
        "impacts": [(401, 391), (418, 404), (409, 416), (390, 408), (414, 385), (397, 399)],
        "center": (400.0, 400.0),
    },
    "Demo 2 — Off-centre grouping": {
        "description": "Tight group shifted rightward/downward from center (sight zeroing calibration needed).",
        "impacts": [(465, 382), (479, 394), (461, 405), (472, 413), (486, 402), (468, 391)],
        "center": (400.0, 400.0),
    },
    "Demo 3 — Wider spread": {
        "description": "Broader shot dispersion highlighting extreme spread and group RMS metrics.",
        "impacts": [(365, 372), (437, 378), (390, 425), (425, 431), (405, 390), (448, 410)],
        "center": (400.0, 400.0),
    },
}


@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path: str):
    return YOLO(model_path)


# Sidebar controls & system overview
with st.sidebar:
    st.markdown('<div class="sidebar-title">🎯 Marksman</div>', unsafe_allow_html=True)
    st.caption("AI-Powered Airgun Target Analysis · College Prototype")
    st.markdown("---")

    mode = st.radio(
        "Application Mode",
        ["Synthetic Demo Gallery", "Live AI Analysis"],
        help="Switch between synthetic ground-truth targets and real YOLOv8 model inference.",
    )

    st.markdown("---")
    st.markdown("### ℹ️ System Specifications")
    st.markdown(
        """
        - **Primary Model**: YOLOv8s (640px)
        - **Validation mAP@50**: 0.720
        - **Validation Precision**: 0.625
        - **Validation Recall**: 0.707
        - **Target Type**: 17x17cm Airgun Paper Target
        """
    )
    st.markdown("---")
    st.caption("Developed for computer vision & ballistics target analysis demonstrations.")


# Main App Title
st.markdown('<div class="main-header">🎯 Marksman — Target Analysis Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Computer vision pellet-hole detection, geometric dispersion analysis, and grouping metrics.</div>',
    unsafe_allow_html=True,
)


def render_external_legend():
    """Render a clean horizontal legend outside diagrams so images remain unobstructed."""
    st.markdown(
        """
        <div class="legend-bar">
            <span class="legend-item"><span style="color:#0284c7; font-weight:bold; font-size:1.1rem;">⌖</span> <b>Target Center (0,0)</b></span>
            <span class="legend-item"><span style="color:#d97706; font-size:1.1rem;">◆</span> <b>Mean Impact Point (MIP)</b></span>
            <span class="legend-item"><span style="color:#d97706; font-weight:bold;">◌</span> <b>Group RMS Circle</b></span>
            <span class="legend-item"><span style="color:#dc2626; font-size:1.1rem;">●</span> <b>Impact Positions (#)</b></span>
            <span class="legend-item"><span style="color:#64748b; font-weight:bold;">―</span> <b>Extreme Spread Span</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# MODE 1: SYNTHETIC DEMO GALLERY
# ==============================================================================
if mode == "Synthetic Demo Gallery":
    st.markdown(
        """
        <span class="badge-synthetic">Mode: Synthetic Demo</span>
        &nbsp; **Ground-truth evaluation:** Impact coordinates in this mode are geometrically exact by construction.
        They demonstrate analytical calculations (MIP, dispersion RMS, extreme spread) independently of AI detection errors.
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    col_ctrl, col_desc = st.columns([1.5, 2])
    with col_ctrl:
        scenario_name = st.selectbox("Select Target Scenario", list(DEMO_SCENARIOS.keys()))
    scenario = DEMO_SCENARIOS[scenario_name]
    with col_desc:
        st.info(f"**Scenario Profile:** {scenario['description']}")

    # Target center configuration
    with st.expander("⚙️ Calibrate Target Center Coordinates", expanded=False):
        c_col1, c_col2 = st.columns(2)
        cx = c_col1.number_input("Target Center X (px)", value=float(scenario["center"][0]), step=5.0)
        cy = c_col2.number_input("Target Center Y (px)", value=float(scenario["center"][1]), step=5.0)
        calibrated_center = (cx, cy)

    # Perform Analysis Engine calculations
    raw_points = scenario["impacts"]
    metrics: GroupMetrics = analyze_target_impacts(raw_points, target_center=calibrated_center)

    render_external_legend()

    # Visual layout: Target diagram left, metrics right
    col_vis, col_stats = st.columns([1.2, 1.0])

    with col_vis:
        st.markdown("#### Annotated Target Diagram")
        target_img = render_annotated_target(metrics, is_synthetic=True)
        st.image(target_img, use_container_width=True, caption=f"{scenario_name} · Exact Ground-Truth Diagram")

    with col_stats:
        st.markdown("#### Summary Grouping Metrics")
        m1, m2 = st.columns(2)
        m1.metric("Impact Count", metrics.count)
        m2.metric("Mean Distance to Center", f"{metrics.mean_radial_distance:.1f} px", help="Average radial distance from target center.")

        m3, m4 = st.columns(2)
        m3.metric("Radial RMS (D_RMS)", f"{metrics.radial_rms:.1f} px", help="Root-mean-square distance from the target center (sighting accuracy).")
        m4.metric("Group RMS Spread", f"{metrics.group_rms:.1f} px", help="Root-mean-square dispersion around the Mean Impact Point (group tightness).")

        m5, m6 = st.columns(2)
        m5.metric("Max Spread (Extreme)", f"{metrics.max_spread:.1f} px", help="Maximum pairwise distance between any two shots.")
        if metrics.total_illustrative_score is not None:
            m6.metric("Illustrative Score", f"{metrics.total_illustrative_score} / {metrics.max_possible_score}")

        st.markdown("---")
        st.markdown("#### Sight Zeroing Offsets")
        st.write(f"**Horizontal (ΔX):** `{metrics.mean_offset_x:+.1f} px` | **Vertical (ΔY):** `{metrics.mean_offset_y:+.1f} px`")
        if abs(metrics.mean_offset_x) > 15 or abs(metrics.mean_offset_y) > 15:
            st.warning("⚠️ Group center of mass is shifted from target center. Sights may require windage/elevation adjustment.")
        else:
            st.success("✅ Group center of mass is well-centered on the target bullseye.")

    # Diagnostic charts
    st.markdown("---")
    st.markdown("### 📊 Distribution & Dispersion Diagnostics")
    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        fig_polar = plot_radial_polar_chart(metrics)
        st.pyplot(fig_polar, use_container_width=True)

    with c_chart2:
        fig_offset = plot_offset_scatter_chart(metrics)
        st.pyplot(fig_offset, use_container_width=True)

    # Per-Impact Table and Export
    st.markdown("---")
    st.markdown("### 📋 Impact Coordinates & Radial Data")

    table_data = []
    for rec in metrics.impacts:
        table_data.append({
            "Impact #": rec.index,
            "Center X (px)": rec.x,
            "Center Y (px)": rec.y,
            "Offset ΔX (px)": rec.offset_x,
            "Offset ΔY (px)": rec.offset_y,
            "Distance to Center (px)": rec.radial_distance,
            "Illustrative Score": rec.illustrative_score,
        })
    df_impacts = pd.DataFrame(table_data)
    st.dataframe(df_impacts, use_container_width=True, hide_index=True)

    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        st.download_button(
            "📥 Download Per-Impact CSV",
            data=df_impacts.to_csv(index=False).encode("utf-8"),
            file_name=f"marksman_synthetic_{scenario_name.split()[0].lower()}_impacts.csv",
            mime="text/csv",
        )
    with exp_col2:
        summary_df = pd.DataFrame([metrics.to_summary_dict()])
        st.download_button(
            "📥 Download Summary CSV",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name=f"marksman_synthetic_{scenario_name.split()[0].lower()}_summary.csv",
            mime="text/csv",
        )
    with exp_col3:
        pdf_bytes = generate_marksman_pdf_report(
            metrics=metrics,
            original_image=None,
            annotated_image=target_img,
            is_synthetic=True,
            scenario_or_filename=scenario_name,
            model_name="N/A (Synthetic Ground Truth)",
        )
        st.download_button(
            "📄 Download Analysis PDF Report",
            data=pdf_bytes,
            file_name=f"marksman_synthetic_{scenario_name.split()[0].lower()}_report.pdf",
            mime="application/pdf",
        )

# ==============================================================================
# MODE 2: LIVE AI ANALYSIS
# ==============================================================================
else:
    st.markdown(
        """
        <span class="badge-live">Mode: Live AI Analysis</span>
        &nbsp; **Real-world Computer Vision:** Upload an airgun paper target image to detect pellet holes using the trained YOLOv8s detector.
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Notice: Coordinates are estimated from bounding-box midpoints. "
        "Confidence reflects detector certainty, not marksman shooting accuracy. Target center calibration is required for valid radial metrics."
    )
    st.write("")

    # Model Configuration & Inspection
    default_model_str = str(EXPERIMENT_MODEL)
    if not EXPERIMENT_MODEL.exists():
        st.error(
            f"❌ Primary model weights not found at `{EXPERIMENT_MODEL}`. "
            "Never silently falling back to older models. Please specify the path to Experiment 01 `best.pt` below."
        )

    with st.expander("🔧 Detector Settings & Model Configuration", expanded=False):
        col_m1, col_m2 = st.columns([3, 1])
        model_path_input = col_m1.text_input("Path to Model Weights (best.pt)", value=default_model_str)
        conf_thresh = col_m2.slider("Confidence Threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05)

    if not os.path.isfile(model_path_input):
        st.warning(f"Model file not found at `{model_path_input}`. Please ensure Experiment 01 weights are present.")
        st.stop()

    # Image upload
    uploaded_file = st.file_uploader(
        "Upload a target image (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload an image of an airgun paper target."
    )

    if uploaded_file is None:
        st.info("👆 Please upload a target image to begin AI detection and ballistics analysis.")
    else:
        try:
            # Decode image
            image_bytes = uploaded_file.getvalue()
            file_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image_bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

            if image_bgr is None:
                st.error("Failed to decode image. Please ensure the file is a valid image (JPG, PNG).")
                st.stop()

            img_h, img_w = image_bgr.shape[:2]

            # Interactive Target Center Calibration Workflow
            st.markdown("---")
            st.markdown("### 🎯 Target Center Calibration")
            st.write(
                f"Image Dimensions: **{img_w} x {img_h} px**. "
                "In real photographs, the bullseye rarely aligns with the image center. Use the calibration controls to position the target center (0,0)."
            )

            calib_mode = st.radio(
                "Calibration Method",
                ["Geometric Presets", "Manual Pixel Coordinates (X, Y)"],
                horizontal=True,
            )

            if calib_mode == "Geometric Presets":
                preset = st.selectbox(
                    "Select Target Center Location Preset",
                    [
                        "Image Geometric Center (Default)",
                        "Custom Target Bullseye (Adjust Below)",
                    ],
                )
                if preset == "Image Geometric Center (Default)":
                    calib_cx, calib_cy = float(img_w / 2.0), float(img_h / 2.0)
                else:
                    cp1, cp2 = st.columns(2)
                    calib_cx = cp1.number_input("Target Center X (px)", min_value=0.0, max_value=float(img_w), value=float(img_w / 2.0), step=1.0)
                    calib_cy = cp2.number_input("Target Center Y (px)", min_value=0.0, max_value=float(img_h), value=float(img_h / 2.0), step=1.0)
            else:
                cp1, cp2 = st.columns(2)
                calib_cx = cp1.number_input("Target Center X (px)", min_value=0.0, max_value=float(img_w), value=float(img_w / 2.0), step=1.0)
                calib_cy = cp2.number_input("Target Center Y (px)", min_value=0.0, max_value=float(img_h), value=float(img_h / 2.0), step=1.0)

            calibrated_target_center = (calib_cx, calib_cy)
            st.caption(f"Active Calibrated Target Center: **X = {calib_cx:.1f} px, Y = {calib_cy:.1f} px**")

            # Load YOLO and predict
            with st.spinner("Executing YOLOv8s detection on target image..."):
                model = load_yolo_model(model_path_input)
                detection_result = model.predict(
                    source=image_bgr,
                    conf=conf_thresh,
                    imgsz=640,
                    verbose=False,
                )[0]

            # Extract detections
            detected_points: List[Tuple[float, float]] = []
            detection_confidences: List[float] = []
            detection_boxes: List[Tuple[float, float, float, float]] = []

            if detection_result.boxes is not None:
                for box in detection_result.boxes:
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    conf = float(box.conf[0])
                    xc, yc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    detected_points.append((xc, yc))
                    detection_confidences.append(conf)
                    detection_boxes.append((x1, y1, x2, y2))

            # Run Analysis Engine on detections
            ai_metrics: GroupMetrics = analyze_target_impacts(
                detected_points,
                target_center=calibrated_target_center,
                confidences=detection_confidences,
                ring_radii=None,  # No real scoring without physical scale calibration
                ring_values=None,
            )

            # Visual presentation
            st.markdown("---")
            st.markdown("### 🎯 Detection & Spatial Analysis")
            render_external_legend()

            v_col1, v_col2 = st.columns(2)
            with v_col1:
                yolo_plot_rgb = cv2.cvtColor(detection_result.plot(), cv2.COLOR_BGR2RGB)
                st.image(yolo_plot_rgb, caption=f"YOLOv8s Detections ({len(detected_points)} holes)", use_container_width=True)

            with v_col2:
                base_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                annotated_overlay = render_annotated_target(
                    ai_metrics,
                    is_synthetic=False,
                    base_image=base_pil,
                )
                st.image(annotated_overlay, caption="Spatial Target Overlay (Center, MIP & Dispersion)", use_container_width=True)

            # Metrics
            st.markdown("---")
            st.markdown("#### AI Grouping & Dispersion Statistics")

            if ai_metrics.count == 0:
                st.warning("⚠️ The detector found no bullet holes at the current confidence threshold. Try lowering the threshold.")
            else:
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Detected Impacts", ai_metrics.count)
                s2.metric("Mean Radial Dist", f"{ai_metrics.mean_radial_distance:.1f} px", help="Average radial distance from calibrated target center.")
                s3.metric("Group RMS Spread", f"{ai_metrics.group_rms:.1f} px", help="Root-mean-square dispersion around the Mean Impact Point (tightness).")
                s4.metric("Max Spread", f"{ai_metrics.max_spread:.1f} px", help="Maximum pairwise distance between any two detected impacts.")

                # Diagnostic charts for live target
                st.markdown("---")
                st.markdown("### 📊 Live Impact Distribution Charts")
                ch1, ch2 = st.columns(2)
                with ch1:
                    fig_polar_ai = plot_radial_polar_chart(ai_metrics)
                    st.pyplot(fig_polar_ai, use_container_width=True)
                with ch2:
                    fig_offset_ai = plot_offset_scatter_chart(ai_metrics)
                    st.pyplot(fig_offset_ai, use_container_width=True)

                # Detections table
                st.markdown("---")
                st.markdown("### 📋 Detected Impact Details")
                rows = []
                for idx, (rec, box) in enumerate(zip(ai_metrics.impacts, detection_boxes), start=1):
                    rows.append({
                        "Impact #": rec.index,
                        "Confidence": rec.confidence,
                        "Approx Center X (px)": rec.x,
                        "Approx Center Y (px)": rec.y,
                        "Offset ΔX (px)": rec.offset_x,
                        "Offset ΔY (px)": rec.offset_y,
                        "Distance to Center (px)": rec.radial_distance,
                        "Bounding Box [X1, Y1, X2, Y2] (px)": f"[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]",
                    })
                df_live = pd.DataFrame(rows)
                st.dataframe(df_live, use_container_width=True, hide_index=True)

                d_col1, d_col2, d_col3 = st.columns(3)
                with d_col1:
                    st.download_button(
                        "📥 Download Detections CSV",
                        data=df_live.to_csv(index=False).encode("utf-8"),
                        file_name=f"marksman_detections_{Path(uploaded_file.name).stem}.csv",
                        mime="text/csv",
                    )
                with d_col2:
                    st.download_button(
                        "📥 Download Summary CSV",
                        data=pd.DataFrame([ai_metrics.to_summary_dict()]).to_csv(index=False).encode("utf-8"),
                        file_name=f"marksman_summary_{Path(uploaded_file.name).stem}.csv",
                        mime="text/csv",
                    )
                with d_col3:
                    live_pdf_bytes = generate_marksman_pdf_report(
                        metrics=ai_metrics,
                        original_image=base_pil,
                        annotated_image=annotated_overlay,
                        is_synthetic=False,
                        scenario_or_filename=uploaded_file.name,
                        model_name="YOLOv8s (640px) Detector",
                        conf_threshold=conf_thresh,
                        bounding_boxes=detection_boxes,
                    )
                    st.download_button(
                        "📄 Download Analysis PDF Report",
                        data=live_pdf_bytes,
                        file_name=f"marksman_analysis_{Path(uploaded_file.name).stem}_report.pdf",
                        mime="application/pdf",
                    )

        except Exception as exc:
            st.error(f"Inference error: {exc}")
