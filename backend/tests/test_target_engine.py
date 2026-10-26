"""Unit tests for Marksman Target Analysis Engine with precise ground-truth validation."""

import math
import unittest
import numpy as np

from backend.analysis.target_engine import (
    analyze_target_impacts,
    calculate_illustrative_score,
    GroupMetrics,
)


class TestTargetAnalysisCalculations(unittest.TestCase):
    """Rigorous hand-verified unit tests for target analysis calculations."""

    def test_known_4_point_manual_grid(self):
        """Test with 4 manually calculated points forming an asymmetrical rectangle.
        
        Target Center: (100, 100)
        Points:
          P1: (100, 80)   -> dx = 0,   dy = -20, r1 = 20
          P2: (120, 100)  -> dx = +20, dy = 0,   r2 = 20
          P3: (100, 130)  -> dx = 0,   dy = +30, r3 = 30
          P4: (60, 100)   -> dx = -40, dy = 0,   r4 = 40
        """
        center = (100.0, 100.0)
        points = [(100.0, 80.0), (120.0, 100.0), (100.0, 130.0), (60.0, 100.0)]
        confs = [0.95, 0.88, 0.75, 0.92]

        metrics = analyze_target_impacts(points, target_center=center, confidences=confs)

        # 1. Count
        self.assertEqual(metrics.count, 4)

        # 2. Mean Impact Point (MIP):
        # x_mean = (100 + 120 + 100 + 60) / 4 = 380 / 4 = 95.0
        # y_mean = (80 + 100 + 130 + 100) / 4 = 410 / 4 = 102.5
        self.assertAlmostEqual(metrics.mean_impact_point[0], 95.0, places=2)
        self.assertAlmostEqual(metrics.mean_impact_point[1], 102.5, places=2)

        # 3. Mean offsets from target center:
        # mean_dx = 95.0 - 100.0 = -5.0
        # mean_dy = 102.5 - 100.0 = +2.5
        self.assertAlmostEqual(metrics.mean_offset_x, -5.0, places=2)
        self.assertAlmostEqual(metrics.mean_offset_y, 2.5, places=2)

        # 4. Individual radial distances to center:
        # r = [20, 20, 30, 40] -> Mean = (20 + 20 + 30 + 40) / 4 = 110 / 4 = 27.5
        self.assertAlmostEqual(metrics.mean_radial_distance, 27.5, places=2)

        # 5. Radial RMS:
        # sum(r^2) = 20^2 + 20^2 + 30^2 + 40^2 = 400 + 400 + 900 + 1600 = 3300
        # mean(r^2) = 3300 / 4 = 825.0
        # sqrt(825.0) = 28.722813
        expected_radial_rms = math.sqrt(825.0)
        self.assertAlmostEqual(metrics.radial_rms, expected_radial_rms, places=2)

        # 6. Group RMS Spread (dispersion around MIP):
        # Offsets from MIP (95.0, 102.5):
        # P1 - MIP = (5, -22.5)  -> dist^2 = 25 + 506.25 = 531.25
        # P2 - MIP = (25, -2.5)  -> dist^2 = 625 + 6.25 = 631.25
        # P3 - MIP = (5, 27.5)   -> dist^2 = 25 + 756.25 = 781.25
        # P4 - MIP = (-35, -2.5) -> dist^2 = 1225 + 6.25 = 1231.25
        # sum = 531.25 + 631.25 + 781.25 + 1231.25 = 3175.0
        # mean = 3175 / 4 = 793.75
        # group_rms = sqrt(793.75) = 28.173569
        expected_group_rms = math.sqrt(793.75)
        self.assertAlmostEqual(metrics.group_rms, expected_group_rms, places=2)

        # 7. Maximum Pairwise Distance (Extreme Spread):
        # Distance between P2 (120, 100) and P4 (60, 100) = 60.0
        # Distance between P1 (100, 80) and P3 (100, 130) = 50.0
        # Distance between P3 (100, 130) and P4 (60, 100) = sqrt(40^2 + 30^2) = 50.0
        # Distance between P1 (100, 80) and P4 (60, 100) = sqrt(40^2 + 20^2) = sqrt(2000) = 44.72
        # Max pairwise is between P3 (100, 130) and P2 (120, 100)?
        # P3 to P2 = sqrt(20^2 + 30^2) = sqrt(1300) = 36.05
        # So maximum distance is between P2 and P4 = 60.0
        self.assertAlmostEqual(metrics.max_spread, 60.0, places=2)
        self.assertIsNotNone(metrics.extreme_spread_points)
        pt_a, pt_b = metrics.extreme_spread_points
        self.assertIn((120.0, 100.0), [pt_a, pt_b])
        self.assertIn((60.0, 100.0), [pt_a, pt_b])

        # 8. Check individual impact records
        self.assertEqual(len(metrics.impacts), 4)
        self.assertEqual(metrics.impacts[0].confidence, 0.95)
        self.assertEqual(metrics.impacts[3].offset_x, -40.0)
        self.assertEqual(metrics.impacts[3].offset_y, 0.0)

    def test_empty_detections(self):
        metrics = analyze_target_impacts([], target_center=(200.0, 200.0))
        self.assertEqual(metrics.count, 0)
        self.assertEqual(metrics.target_center, (200.0, 200.0))
        self.assertEqual(metrics.mean_impact_point, (200.0, 200.0))
        self.assertEqual(metrics.mean_radial_distance, 0.0)
        self.assertEqual(metrics.radial_rms, 0.0)
        self.assertEqual(metrics.group_rms, 0.0)
        self.assertEqual(metrics.max_spread, 0.0)
        self.assertIsNone(metrics.extreme_spread_points)

    def test_single_impact(self):
        metrics = analyze_target_impacts([(150.0, 220.0)], target_center=(100.0, 100.0))
        self.assertEqual(metrics.count, 1)
        self.assertEqual(metrics.mean_impact_point, (150.0, 220.0))
        self.assertEqual(metrics.mean_offset_x, 50.0)
        self.assertEqual(metrics.mean_offset_y, 120.0)
        expected_r = math.hypot(50.0, 120.0) # 130.0
        self.assertAlmostEqual(metrics.mean_radial_distance, 130.0, places=2)
        self.assertAlmostEqual(metrics.radial_rms, 130.0, places=2)
        self.assertEqual(metrics.group_rms, 0.0) # 0 dispersion for 1 point
        self.assertEqual(metrics.max_spread, 0.0)
        self.assertIsNone(metrics.extreme_spread_points)

    def test_custom_calibrated_center(self):
        # Impacts at (50, 50) and (50, 70)
        # Default center (0, 0) vs Calibrated center (50, 60)
        pts = [(50.0, 50.0), (50.0, 70.0)]
        m_default = analyze_target_impacts(pts, target_center=(0.0, 0.0))
        m_calib = analyze_target_impacts(pts, target_center=(50.0, 60.0))

        # Mean impact point is invariant to center calibration
        self.assertEqual(m_default.mean_impact_point, (50.0, 60.0))
        self.assertEqual(m_calib.mean_impact_point, (50.0, 60.0))

        # Group RMS and max spread are invariant to center calibration
        self.assertAlmostEqual(m_default.group_rms, m_calib.group_rms, places=2)
        self.assertAlmostEqual(m_default.max_spread, m_calib.max_spread, places=2)

        # Offsets and radial distances MUST change based on center
        self.assertEqual(m_calib.mean_offset_x, 0.0)
        self.assertEqual(m_calib.mean_offset_y, 0.0)
        self.assertEqual(m_calib.mean_radial_distance, 10.0)


if __name__ == "__main__":
    unittest.main()
