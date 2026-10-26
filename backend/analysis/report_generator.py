"""PDF Analysis Report Generator for Marksman.

Generates a multi-section PDF target ballistics analysis report using
matplotlib's native PdfPages backend (zero additional dependencies required).
Includes original and annotated target imagery, summary metrics, calibration center,
model parameters, per-impact details, and clear technical notices.
"""

from __future__ import annotations

import datetime
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image

from backend.analysis.target_engine import GroupMetrics, ImpactRecord


def generate_marksman_pdf_report(
    metrics: GroupMetrics,
    original_image: Optional[Image.Image],
    annotated_image: Image.Image,
    is_synthetic: bool = False,
    scenario_or_filename: str = "Target Analysis",
    model_name: str = "YOLOv8s (640px)",
    conf_threshold: Optional[float] = None,
    bounding_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
) -> bytes:
    """Generate a clean, high-resolution multi-page PDF target analysis report.

    Args:
        metrics: GroupMetrics dataclass from target analysis engine.
        original_image: PIL Image of the original upload (or None for synthetic).
        annotated_image: PIL Image of the annotated target with center and MIP.
        is_synthetic: True if report is for synthetic ground truth.
        scenario_or_filename: Name of scenario or uploaded file.
        model_name: Description of detection model.
        conf_threshold: Confidence threshold used during inference.
        bounding_boxes: Optional list of (x1, y1, x2, y2) detection boxes.

    Returns:
        Bytes of the generated PDF file.
    """
    buf = io.BytesIO()
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with PdfPages(buf) as pdf:
        # ==========================================
        # PAGE 1: Executive Overview, Images & Summary Metrics
        # ==========================================
        fig1 = plt.figure(figsize=(8.5, 11), dpi=150)
        fig1.patch.set_facecolor("#ffffff")

        # Header Title
        fig1.text(0.08, 0.94, "MARKSMAN — TARGET ANALYSIS REPORT", fontsize=17, fontweight="bold", color="#0f172a")
        report_sub = "SYNTHETIC GROUND-TRUTH DEMONSTRATION" if is_synthetic else "COMPUTER VISION TARGET EVALUATION (YOLOv8s)"
        fig1.text(0.08, 0.915, report_sub, fontsize=10, fontweight="bold", color="#0284c7" if not is_synthetic else "#d97706")
        fig1.text(0.08, 0.895, f"Target: {scenario_or_filename}  |  Generated: {timestamp_str}", fontsize=9, color="#64748b")

        # Top Horizontal Rule
        line_ax = fig1.add_axes([0.08, 0.88, 0.84, 0.002])
        line_ax.patch.set_facecolor("#cbd5e1")
        line_ax.axis("off")

        # Display Images (Original & Annotated side-by-side)
        if original_image is not None and not is_synthetic:
            # Side by side: Original on left, Annotated on right
            ax_img1 = fig1.add_axes([0.08, 0.54, 0.40, 0.32])
            ax_img1.imshow(original_image)
            ax_img1.set_title("Original Target Image", fontsize=10, fontweight="bold", color="#1e293b", pad=6)
            ax_img1.axis("off")

            ax_img2 = fig1.add_axes([0.52, 0.54, 0.40, 0.32])
            ax_img2.imshow(annotated_image)
            ax_img2.set_title("Spatial Analysis Overlay", fontsize=10, fontweight="bold", color="#1e293b", pad=6)
            ax_img2.axis("off")
        else:
            # Centered Annotated Target for synthetic demo
            ax_img = fig1.add_axes([0.25, 0.54, 0.50, 0.32])
            ax_img.imshow(annotated_image)
            ax_img.set_title("Annotated Target Diagram", fontsize=10, fontweight="bold", color="#1e293b", pad=6)
            ax_img.axis("off")

        # Section 2: Summary Metrics Cards
        fig1.text(0.08, 0.49, "SUMMARY BALLISTICS & DISPERSION METRICS", fontsize=11, fontweight="bold", color="#0f172a")

        metric_box_data = [
            ("Impact Count", f"{metrics.count}"),
            ("Calibrated Center (0,0)", f"({metrics.target_center[0]:.1f}, {metrics.target_center[1]:.1f}) px"),
            ("Mean Impact Point (MIP)", f"({metrics.mean_impact_point[0]:.1f}, {metrics.mean_impact_point[1]:.1f}) px"),
            ("Mean Center Offset", f"ΔX: {metrics.mean_offset_x:+.1f} px, ΔY: {metrics.mean_offset_y:+.1f} px"),
            ("Mean Radial Distance", f"{metrics.mean_radial_distance:.1f} px"),
            ("Radial RMS (D_RMS)", f"{metrics.radial_rms:.1f} px"),
            ("Group RMS Spread (S_RMS)", f"{metrics.group_rms:.1f} px"),
            ("Max Spread (Extreme Spread)", f"{metrics.max_spread:.1f} px"),
        ]

        if is_synthetic and metrics.total_illustrative_score is not None:
            metric_box_data.append(("Illustrative Ring Score", f"{metrics.total_illustrative_score} / {metrics.max_possible_score}"))

        # Table of Summary Metrics
        summary_rows = [[lbl, val] for lbl, val in metric_box_data]
        ax_summary = fig1.add_axes([0.08, 0.22, 0.84, 0.25])
        ax_summary.axis("off")

        table_summary = ax_summary.table(
            cellText=summary_rows,
            colLabels=["Metric Description", "Computed Value (pixel-based)"],
            loc="center",
            cellLoc="left",
            colWidths=[0.55, 0.45],
        )
        table_summary.auto_set_font_size(False)
        table_summary.set_fontsize(8.5)
        table_summary.scale(1.0, 1.35)

        for (row_idx, col_idx), cell in table_summary.get_celld().items():
            cell.set_edgecolor("#e2e8f0")
            if row_idx == 0:
                cell.set_facecolor("#f1f5f9")
                cell.set_text_props(weight="bold", color="#1e293b")
            else:
                cell.set_facecolor("#ffffff" if row_idx % 2 == 0 else "#f8fafc")

        # Bottom Notice Box
        notice_ax = fig1.add_axes([0.08, 0.05, 0.84, 0.13])
        notice_ax.patch.set_facecolor("#fefce8")
        notice_ax.patch.set_edgecolor("#fde047")
        notice_ax.axis("off")

        notice_text = (
            "IMPORTANT TECHNICAL NOTICES & METHODOLOGY:\n"
            "1. Measurements are in image pixel space (px) and represent relative geometric distances.\n"
            "2. AI coordinates are estimated from bounding-box midpoints and are subject to optical/perspective distortion.\n"
            "3. Detection confidence represents neural network classification certainty, not shooter marksmanship skill.\n"
            "4. Ring scores (if shown) are illustrative demonstration heuristics and NOT official ISSF scoring certifications."
        )
        fig1.text(0.10, 0.07, notice_text, fontsize=7.5, color="#713f12", linespacing=1.4)

        pdf.savefig(fig1)
        plt.close(fig1)

        # ==========================================
        # PAGE 2: Per-Impact Coordinate Table & Model Specifications
        # ==========================================
        fig2 = plt.figure(figsize=(8.5, 11), dpi=150)
        fig2.patch.set_facecolor("#ffffff")

        # Header Title
        fig2.text(0.08, 0.94, "MARKSMAN — IMPACT LEVEL DATA & AUDIT LOG", fontsize=15, fontweight="bold", color="#0f172a")
        fig2.text(0.08, 0.915, f"Detailed Coordinates for {scenario_or_filename}  |  Total Impacts: {metrics.count}", fontsize=9.5, color="#64748b")

        line_ax2 = fig2.add_axes([0.08, 0.90, 0.84, 0.002])
        line_ax2.patch.set_facecolor("#cbd5e1")
        line_ax2.axis("off")

        # Model and Configuration Metadata Block
        meta_ax = fig2.add_axes([0.08, 0.80, 0.84, 0.085])
        meta_ax.patch.set_facecolor("#f8fafc")
        meta_ax.patch.set_edgecolor("#e2e8f0")
        meta_ax.axis("off")

        conf_str = f"{conf_threshold:.2f}" if conf_threshold is not None else "N/A (Synthetic)"
        meta_text = (
            f"Model Architecture: {model_name}\n"
            f"Confidence Threshold: {conf_str}  |  Input Image Size: 640x640 px\n"
            f"Validation Performance: mAP@50 = 0.720, mAP@50-95 = 0.497, Precision = 0.625, Recall = 0.707\n"
            f"Coordinate System: Origin (0,0) at top-left. Offsets: ΔX = X - X_center, ΔY = Y - Y_center."
        )
        fig2.text(0.10, 0.81, meta_text, fontsize=8, color="#334155", linespacing=1.35)

        # Per-Impact Table
        fig2.text(0.08, 0.77, "PER-IMPACT COORDINATES & RADIAL DISTANCES", fontsize=11, fontweight="bold", color="#0f172a")

        table_headers = [
            "Impact #",
            "Center X (px)",
            "Center Y (px)",
            "Offset ΔX (px)",
            "Offset ΔY (px)",
            "Radial Dist (px)",
            "Confidence",
        ]

        impact_rows = []
        for rec in metrics.impacts[:35]:  # Fits cleanly on single page
            conf_disp = f"{rec.confidence:.3f}" if rec.confidence is not None else ("Exact" if is_synthetic else "N/A")
            impact_rows.append([
                f"#{rec.index}",
                f"{rec.x:.1f}",
                f"{rec.y:.1f}",
                f"{rec.offset_x:+.1f}",
                f"{rec.offset_y:+.1f}",
                f"{rec.radial_distance:.1f}",
                conf_disp,
            ])

        if len(impact_rows) == 0:
            impact_rows = [["None", "-", "-", "-", "-", "-", "-"]]

        ax_table = fig2.add_axes([0.08, 0.12, 0.84, 0.62])
        ax_table.axis("off")

        table = ax_table.table(
            cellText=impact_rows,
            colLabels=table_headers,
            loc="top",
            cellLoc="center",
            colWidths=[0.12, 0.14, 0.14, 0.14, 0.14, 0.16, 0.16],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.0)
        table.scale(1.0, 1.3)

        for (row_idx, col_idx), cell in table.get_celld().items():
            cell.set_edgecolor("#e2e8f0")
            if row_idx == 0:
                cell.set_facecolor("#0284c7")
                cell.set_text_props(weight="bold", color="#ffffff")
            else:
                cell.set_facecolor("#ffffff" if row_idx % 2 == 0 else "#f8fafc")

        # Page 2 Footer
        fig2.text(0.08, 0.05, f"Marksman Target Ballistics · Generated {timestamp_str} · Page 2 of 2", fontsize=7.5, color="#94a3b8")

        pdf.savefig(fig2)
        plt.close(fig2)

    buf.seek(0)
    return buf.getvalue()
