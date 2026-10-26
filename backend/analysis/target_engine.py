"""Marksman Target Analysis Engine.

A clean, reusable module for calculating geometric grouping metrics,
impact offsets, radial statistics, and illustrative target scores.
Independent of UI frameworks and directly testable.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ImpactRecord:
    """Detailed geometric data for an individual projectile impact."""
    index: int
    x: float
    y: float
    offset_x: float
    offset_y: float
    radial_distance: float
    illustrative_score: Optional[int] = None
    confidence: Optional[float] = None


@dataclass(frozen=True)
class GroupMetrics:
    """Statistical and geometric summary of an impact grouping."""
    count: int
    target_center: Tuple[float, float]
    mean_impact_point: Tuple[float, float]
    mean_offset_x: float
    mean_offset_y: float
    mean_radial_distance: float
    radial_rms: float
    group_rms: float
    max_spread: float
    extreme_spread_points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]
    total_illustrative_score: Optional[int]
    max_possible_score: Optional[int]
    impacts: List[ImpactRecord]

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a clean dictionary representation of summary metrics."""
        return {
            "Total Impacts": self.count,
            "Target Center X (px)": round(self.target_center[0], 2),
            "Target Center Y (px)": round(self.target_center[1], 2),
            "Mean Impact Point X (px)": round(self.mean_impact_point[0], 2),
            "Mean Impact Point Y (px)": round(self.mean_impact_point[1], 2),
            "Mean Offset X (px)": round(self.mean_offset_x, 2),
            "Mean Offset Y (px)": round(self.mean_offset_y, 2),
            "Mean Radial Distance (px)": round(self.mean_radial_distance, 2),
            "Radial RMS (D_RMS) (px)": round(self.radial_rms, 2),
            "Group RMS Spread (S_RMS) (px)": round(self.group_rms, 2),
            "Max Spread (Extreme Spread) (px)": round(self.max_spread, 2),
            "Total Illustrative Score": self.total_illustrative_score,
            "Max Possible Illustrative Score": self.max_possible_score,
        }


def calculate_illustrative_score(
    radial_distance: float,
    ring_radii: Sequence[float] = (60.0, 120.0, 180.0, 240.0, 300.0),
    ring_values: Sequence[int] = (10, 9, 8, 7, 6),
) -> int:
    """Calculate an illustrative radial ring score.

    Note: This is strictly an illustrative heuristic for demonstration
    and educational comparison. It is NOT an official ISSF scoring system.
    """
    for radius, score in zip(ring_radii, ring_values):
        if radial_distance <= radius:
            return score
    return 0


def analyze_target_impacts(
    impact_points: Sequence[Sequence[float] | Tuple[float, float]],
    target_center: Tuple[float, float] = (400.0, 400.0),
    confidences: Optional[Sequence[float]] = None,
    ring_radii: Optional[Sequence[float]] = (60.0, 120.0, 180.0, 240.0, 300.0),
    ring_values: Optional[Sequence[int]] = (10, 9, 8, 7, 6),
) -> GroupMetrics:
    """Calculate comprehensive grouping metrics from impact coordinates.

    Args:
        impact_points: Sequence of (x, y) coordinates.
        target_center: Calibrated or assumed (cx, cy) target center.
        confidences: Optional detection confidences for each point.
        ring_radii: Radii for illustrative scoring rings (ascending).
        ring_values: Integer scores associated with each ring radius.

    Returns:
        GroupMetrics dataclass containing all computed metrics.
    """
    cx, cy = float(target_center[0]), float(target_center[1])

    # Handle empty input safely
    if not impact_points or len(impact_points) == 0:
        return GroupMetrics(
            count=0,
            target_center=(cx, cy),
            mean_impact_point=(cx, cy),
            mean_offset_x=0.0,
            mean_offset_y=0.0,
            mean_radial_distance=0.0,
            radial_rms=0.0,
            group_rms=0.0,
            max_spread=0.0,
            extreme_spread_points=None,
            total_illustrative_score=0,
            max_possible_score=0,
            impacts=[],
        )

    # Convert coordinates safely
    cleaned_points: List[Tuple[float, float]] = []
    for pt in impact_points:
        if len(pt) < 2:
            continue
        try:
            x_val = float(pt[0])
            y_val = float(pt[1])
            if math.isfinite(x_val) and math.isfinite(y_val):
                cleaned_points.append((x_val, y_val))
        except (ValueError, TypeError):
            continue

    count = len(cleaned_points)
    if count == 0:
        return analyze_target_impacts([], target_center=target_center)

    pts = np.asarray(cleaned_points, dtype=np.float64)
    center = np.array([cx, cy], dtype=np.float64)

    # Offsets and radial distances relative to target center
    offsets = pts - center
    radial_distances = np.linalg.norm(offsets, axis=1)

    # Mean impact point (group center of mass)
    mean_point = pts.mean(axis=0)
    mean_x, mean_y = float(mean_point[0]), float(mean_point[1])
    mean_offset_x = float(mean_x - cx)
    mean_offset_y = float(mean_y - cy)

    # Radial statistics relative to target center
    mean_radial_dist = float(np.mean(radial_distances))
    radial_rms = float(np.sqrt(np.mean(radial_distances ** 2)))

    # Dispersion relative to Mean Impact Point (Group RMS) & Max Spread
    if count > 1:
        group_offsets = pts - mean_point
        group_rms = float(np.sqrt(np.mean(np.sum(group_offsets ** 2, axis=1))))

        # Pairwise distances
        max_dist = 0.0
        pt_a_best: Tuple[float, float] = cleaned_points[0]
        pt_b_best: Tuple[float, float] = cleaned_points[1]

        for i in range(count):
            for j in range(i + 1, count):
                dist = float(np.linalg.norm(pts[i] - pts[j]))
                if dist > max_dist:
                    max_dist = dist
                    pt_a_best = cleaned_points[i]
                    pt_b_best = cleaned_points[j]

        max_spread = max_dist
        extreme_points = (pt_a_best, pt_b_best)
    else:
        group_rms = 0.0
        max_spread = 0.0
        extreme_points = None

    # Individual impact records
    impact_records: List[ImpactRecord] = []
    total_score = 0
    calculate_scores = ring_radii is not None and ring_values is not None

    for idx, (pt, r_dist) in enumerate(zip(cleaned_points, radial_distances), start=1):
        score: Optional[int] = None
        if calculate_scores:
            score = calculate_illustrative_score(r_dist, ring_radii, ring_values)
            total_score += score

        conf: Optional[float] = None
        if confidences is not None and idx - 1 < len(confidences):
            try:
                c_val = float(confidences[idx - 1])
                conf = round(c_val, 4) if math.isfinite(c_val) else None
            except (ValueError, TypeError):
                conf = None

        impact_records.append(
            ImpactRecord(
                index=idx,
                x=round(pt[0], 2),
                y=round(pt[1], 2),
                offset_x=round(pt[0] - cx, 2),
                offset_y=round(pt[1] - cy, 2),
                radial_distance=round(float(r_dist), 2),
                illustrative_score=score,
                confidence=conf,
            )
        )

    max_possible = count * ring_values[0] if (calculate_scores and len(ring_values) > 0) else None

    return GroupMetrics(
        count=count,
        target_center=(round(cx, 2), round(cy, 2)),
        mean_impact_point=(round(mean_x, 2), round(mean_y, 2)),
        mean_offset_x=round(mean_offset_x, 2),
        mean_offset_y=round(mean_offset_y, 2),
        mean_radial_distance=round(mean_radial_dist, 2),
        radial_rms=round(radial_rms, 2),
        group_rms=round(group_rms, 2),
        max_spread=round(max_spread, 2),
        extreme_spread_points=extreme_points,
        total_illustrative_score=total_score if calculate_scores else None,
        max_possible_score=max_possible,
        impacts=impact_records,
    )
