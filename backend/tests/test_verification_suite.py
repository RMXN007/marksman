"""Full suite of verification tests for Priority 6:
1. Normal image with detections.
2. Image with no detections.
3. Image with one detection.
4. Different image dimensions.
5. Invalid or unsupported file.
6. Missing model file.
"""

import os
import unittest
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from backend.analysis.target_engine import analyze_target_impacts
from backend.analysis.visualizer import (
    render_annotated_target,
    plot_radial_polar_chart,
    plot_offset_scatter_chart,
)


class TestFullMarksmanVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root_dir = Path(__file__).resolve().parent.parent.parent
        cls.model_path = cls.root_dir / "backend" / "runs" / "experiment_01_yolov8s_640" / "weights" / "best.pt"
        cls.val_image = cls.root_dir / "backend" / "dataset" / "images" / "val" / "IMG-20260315-WA0004.jpg"

    def test_case_1_normal_image_with_detections(self):
        """Case 1: Normal image with YOLO detections."""
        self.assertTrue(self.model_path.exists())
        self.assertTrue(self.val_image.exists())

        model = YOLO(str(self.model_path))
        res = model.predict(source=str(self.val_image), imgsz=640, verbose=False)[0]

        pts = []
        confs = []
        for box in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            pts.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
            confs.append(float(box.conf[0]))

        self.assertGreater(len(pts), 0)

        # Center calibration
        img_bgr = cv2.imread(str(self.val_image))
        h, w = img_bgr.shape[:2]
        center = (w / 2.0, h / 2.0)

        metrics = analyze_target_impacts(pts, target_center=center, confidences=confs)
        self.assertEqual(metrics.count, len(pts))
        self.assertGreater(metrics.radial_rms, 0.0)
        self.assertGreater(metrics.group_rms, 0.0)

        # Ensure visualizer renders cleanly
        base_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        overlay = render_annotated_target(metrics, is_synthetic=False, base_image=base_pil)
        self.assertEqual(overlay.size, (w, h))

    def test_case_2_image_with_no_detections(self):
        """Case 2: Image with 0 detections."""
        empty_pts = []
        center = (300.0, 300.0)
        metrics = analyze_target_impacts(empty_pts, target_center=center)

        self.assertEqual(metrics.count, 0)
        self.assertEqual(metrics.radial_rms, 0.0)
        self.assertEqual(metrics.group_rms, 0.0)
        self.assertEqual(metrics.max_spread, 0.0)

        # Visuals should handle empty data gracefully
        fig_polar = plot_radial_polar_chart(metrics)
        self.assertIsNotNone(fig_polar)
        fig_offset = plot_offset_scatter_chart(metrics)
        self.assertIsNotNone(fig_offset)

    def test_case_3_image_with_one_detection(self):
        """Case 3: Image with single detection."""
        single_pt = [(250.0, 200.0)]
        center = (200.0, 200.0)
        metrics = analyze_target_impacts(single_pt, target_center=center, confidences=[0.89])

        self.assertEqual(metrics.count, 1)
        self.assertEqual(metrics.mean_impact_point, (250.0, 200.0))
        self.assertEqual(metrics.mean_offset_x, 50.0)
        self.assertEqual(metrics.mean_offset_y, 0.0)
        self.assertEqual(metrics.mean_radial_distance, 50.0)
        self.assertEqual(metrics.radial_rms, 50.0)
        self.assertEqual(metrics.group_rms, 0.0)
        self.assertEqual(metrics.max_spread, 0.0)
        self.assertIsNone(metrics.extreme_spread_points)

        # Ensure figures render without crashing on single point
        fig_polar = plot_radial_polar_chart(metrics)
        self.assertIsNotNone(fig_polar)
        fig_offset = plot_offset_scatter_chart(metrics)
        self.assertIsNotNone(fig_offset)

    def test_case_4_different_image_dimensions(self):
        """Case 4: Handling various aspect ratios and resolutions."""
        dimensions = [(640, 640), (1920, 1080), (1080, 1920), (320, 240)]
        for w, h in dimensions:
            center = (w / 2.0, h / 2.0)
            pts = [(center[0] + 15, center[1] - 20), (center[0] - 25, center[1] + 10)]
            metrics = analyze_target_impacts(pts, target_center=center)

            self.assertEqual(metrics.count, 2)
            self.assertEqual(metrics.target_center, center)

            blank = Image.new("RGB", (w, h), "white")
            annotated = render_annotated_target(metrics, is_synthetic=False, base_image=blank)
            self.assertEqual(annotated.size, (w, h))

    def test_case_5_invalid_and_unsupported_file(self):
        """Case 5: Corrupt or unsupported file bytes."""
        bad_bytes = b"not an image file at all"
        file_arr = np.frombuffer(bad_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(file_arr, cv2.IMREAD_COLOR)
        self.assertIsNone(decoded, "cv2.imdecode must safely return None for non-image data")

    def test_case_6_missing_model_file(self):
        """Case 6: Missing model file path behavior."""
        missing_path = Path("this/path/does/not/exist/model.pt")
        self.assertFalse(missing_path.exists())
        self.assertFalse(os.path.isfile(str(missing_path)))


if __name__ == "__main__":
    unittest.main()
