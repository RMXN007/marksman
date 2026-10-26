"""End-to-end integration tests for Marksman target pipeline and edge cases."""

import os
import unittest
from pathlib import Path
import cv2
import numpy as np

from backend.analysis.target_engine import analyze_target_impacts
from backend.analysis.visualizer import render_annotated_target, plot_radial_polar_chart, plot_offset_scatter_chart
from ultralytics import YOLO


class TestMarksmanIntegration(unittest.TestCase):
    """Test full pipeline, edge cases, model inference, and export summaries."""

    @classmethod
    def setUpClass(cls):
        cls.root_dir = Path(__file__).resolve().parent.parent.parent
        cls.model_path = cls.root_dir / "backend" / "runs" / "experiment_01_yolov8s_640" / "weights" / "best.pt"
        cls.val_image = cls.root_dir / "backend" / "dataset" / "images" / "val" / "IMG-20260315-WA0004.jpg"

    def test_model_file_exists(self):
        self.assertTrue(self.model_path.exists(), f"Experiment 01 model missing at {self.model_path}")

    def test_missing_model_handling(self):
        non_existent = Path("non_existent_model_weights.pt")
        self.assertFalse(non_existent.exists())

    def test_zero_detections_pipeline(self):
        # When an image has 0 detections
        empty_points = []
        metrics = analyze_target_impacts(empty_points, target_center=(300.0, 300.0))
        self.assertEqual(metrics.count, 0)
        self.assertEqual(metrics.max_spread, 0.0)

        # Visuals should still render gracefully with 'No Impacts' indicators
        fig_polar = plot_radial_polar_chart(metrics)
        self.assertIsNotNone(fig_polar)
        fig_offset = plot_offset_scatter_chart(metrics)
        self.assertIsNotNone(fig_offset)

    def test_different_image_dimensions(self):
        # Test images with varying aspect ratios
        dims = [(640, 640), (1280, 720), (800, 1200), (320, 240)]
        for w, h in dims:
            center = (w / 2.0, h / 2.0)
            dummy_pts = [(w / 2.0 + 10, h / 2.0 - 15), (w / 2.0 - 20, h / 2.0 + 5)]
            metrics = analyze_target_impacts(dummy_pts, target_center=center)
            self.assertEqual(metrics.count, 2)
            self.assertEqual(metrics.target_center, center)

    def test_live_inference_with_actual_model(self):
        if not self.model_path.exists() or not self.val_image.exists():
            self.skipTest("Model or sample validation image not available.")

        model = YOLO(str(self.model_path))
        results = model.predict(source=str(self.val_image), imgsz=640, verbose=False)[0]

        pts = []
        confs = []
        for box in results.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            pts.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
            confs.append(float(box.conf[0]))

        # Assume optical center
        img = cv2.imread(str(self.val_image))
        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)

        metrics = analyze_target_impacts(pts, target_center=center, confidences=confs)
        self.assertGreater(metrics.count, 0)
        self.assertGreater(metrics.radial_rms, 0.0)
        self.assertGreater(metrics.max_spread, 0.0)

        summary = metrics.to_summary_dict()
        self.assertEqual(summary["Total Impacts"], metrics.count)


if __name__ == "__main__":
    unittest.main()
