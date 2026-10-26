"""Tests for CSV and PDF export consistency and data alignment."""

import csv
import io
import unittest
import numpy as np
import pandas as pd
from PIL import Image

from backend.analysis.target_engine import analyze_target_impacts
from backend.analysis.visualizer import render_annotated_target
from backend.analysis.report_generator import generate_marksman_pdf_report


class TestExportsAndConsistency(unittest.TestCase):
    def setUp(self):
        self.center = (320.0, 320.0)
        self.points = [(300.0, 310.0), (340.0, 330.0), (320.0, 300.0)]
        self.confs = [0.92, 0.85, 0.78]
        self.metrics = analyze_target_impacts(self.points, target_center=self.center, confidences=self.confs)

    def test_summary_csv_matches_metrics(self):
        summary_dict = self.metrics.to_summary_dict()
        df = pd.DataFrame([summary_dict])
        csv_str = df.to_csv(index=False)
        reader = list(csv.DictReader(io.StringIO(csv_str)))
        self.assertEqual(len(reader), 1)
        row = reader[0]

        self.assertEqual(int(row["Total Impacts"]), 3)
        self.assertAlmostEqual(float(row["Target Center X (px)"]), 320.0)
        self.assertAlmostEqual(float(row["Target Center Y (px)"]), 320.0)
        self.assertAlmostEqual(float(row["Mean Radial Distance (px)"]), self.metrics.mean_radial_distance, places=1)
        self.assertAlmostEqual(float(row["Radial RMS (D_RMS) (px)"]), self.metrics.radial_rms, places=1)
        self.assertAlmostEqual(float(row["Group RMS Spread (S_RMS) (px)"]), self.metrics.group_rms, places=1)
        self.assertAlmostEqual(float(row["Max Spread (Extreme Spread) (px)"]), self.metrics.max_spread, places=1)

    def test_impact_csv_matches_records(self):
        rows = []
        for rec in self.metrics.impacts:
            rows.append({
                "Impact #": rec.index,
                "Confidence": rec.confidence,
                "Center X (px)": rec.x,
                "Center Y (px)": rec.y,
                "Offset ΔX (px)": rec.offset_x,
                "Offset ΔY (px)": rec.offset_y,
                "Distance to Center (px)": rec.radial_distance,
            })
        df = pd.DataFrame(rows)
        csv_str = df.to_csv(index=False)
        reader = list(csv.DictReader(io.StringIO(csv_str)))
        self.assertEqual(len(reader), 3)

        for i, row in enumerate(reader):
            rec = self.metrics.impacts[i]
            self.assertEqual(int(row["Impact #"]), rec.index)
            self.assertAlmostEqual(float(row["Center X (px)"]), rec.x)
            self.assertAlmostEqual(float(row["Center Y (px)"]), rec.y)
            self.assertAlmostEqual(float(row["Offset ΔX (px)"]), rec.offset_x)
            self.assertAlmostEqual(float(row["Offset ΔY (px)"]), rec.offset_y)
            self.assertAlmostEqual(float(row["Distance to Center (px)"]), rec.radial_distance)
            self.assertAlmostEqual(float(row["Confidence"]), rec.confidence)

    def test_pdf_report_generation_live(self):
        orig = Image.new("RGB", (640, 640), "white")
        annotated = render_annotated_target(self.metrics, is_synthetic=False, base_image=orig)

        pdf_bytes = generate_marksman_pdf_report(
            metrics=self.metrics,
            original_image=orig,
            annotated_image=annotated,
            is_synthetic=False,
            scenario_or_filename="test_target.jpg",
            model_name="YOLOv8s (640px) Detector",
            conf_threshold=0.25,
        )

        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        # PDF magic number header: %PDF-
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))

    def test_pdf_report_generation_synthetic(self):
        annotated = render_annotated_target(self.metrics, is_synthetic=True)
        pdf_bytes = generate_marksman_pdf_report(
            metrics=self.metrics,
            original_image=None,
            annotated_image=annotated,
            is_synthetic=True,
            scenario_or_filename="Demo 1",
            model_name="N/A (Synthetic Ground Truth)",
        )
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))
        self.assertGreater(len(pdf_bytes), 1000)


if __name__ == "__main__":
    unittest.main()
