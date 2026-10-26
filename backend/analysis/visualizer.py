"""Visualization and target rendering utilities for Marksman.

Provides clean target diagram rendering with annotated impact markers,
calibrated center markers, mean impact point (MIP), dispersion circles,
and radial polar / offset charts using matplotlib and PIL.
"""

from __future__ import annotations

import io
import math
from typing import Sequence, Tuple, Optional, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend.analysis.target_engine import GroupMetrics, ImpactRecord

# Shared standard palette across all visual components
COLOR_IMPACT = "#dc2626"       # Crimson red for impacts
COLOR_CENTER = "#0284c7"       # Vibrant blue for target center
COLOR_MIP = "#d97706"          # Amber / gold for Mean Impact Point
COLOR_EXTREME = "#64748b"      # Muted slate for extreme spread span
COLOR_GRID = "#cbd5e1"         # Subtle gray gridlines

CANVAS_SIZE = 800
DEFAULT_CENTER = (400.0, 400.0)
DEFAULT_RINGS = [300.0, 240.0, 180.0, 120.0, 60.0]


def render_annotated_target(
    metrics: GroupMetrics,
    canvas_size: int = CANVAS_SIZE,
    ring_radii: Sequence[float] = DEFAULT_RINGS,
    is_synthetic: bool = True,
    base_image: Optional[Image.Image] = None,
) -> Image.Image:
    """Render a clean target diagram with unobtrusive markers.

    Keeps the target face unobstructed by moving heavy text legends outside
    the drawing and using crisp, compact markers.
    """
    if base_image is not None:
        image = base_image.convert("RGB").copy()
    else:
        image = Image.new("RGB", (canvas_size, canvas_size), "#f7f5f0")
        draw = ImageDraw.Draw(image)
        cx, cy = metrics.target_center

        # Target boundary
        draw.ellipse(
            (cx - 330, cy - 330, cx + 330, cy + 330),
            fill="#ffffff",
            outline="#222222",
            width=3,
        )

        # Concentric score rings
        for idx, radius in enumerate(ring_radii):
            color = "#1f2421" if idx == 0 else "#ffffff"
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=color,
                outline="#333333",
                width=2,
            )

        # Re-outline inner rings
        for radius in ring_radii[1:]:
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                outline="#333333",
                width=2,
            )

        # Subtle ring indicator digits
        for idx, radius in enumerate(ring_radii):
            val = str(6 + idx) if idx < 5 else ""
            draw.text((cx + 6, cy - radius + 4), val, fill="#777777")

    draw = ImageDraw.Draw(image)
    cx, cy = metrics.target_center

    # 1. Extreme Spread Span Line (between farthest two impacts)
    if metrics.extreme_spread_points is not None:
        pt_a, pt_b = metrics.extreme_spread_points
        draw.line([pt_a, pt_b], fill=COLOR_EXTREME, width=2)

    # 2. Mean Impact Point (MIP) Diamond & Group RMS dispersion circle
    if metrics.count > 0:
        mx, my = metrics.mean_impact_point
        # Group RMS circle around MIP
        if metrics.group_rms > 0:
            s_r = metrics.group_rms
            draw.ellipse(
                (mx - s_r, my - s_r, mx + s_r, my + s_r),
                outline=COLOR_MIP,
                width=2,
            )

        # MIP Diamond marker
        d = 7
        diamond = [(mx, my - d), (mx + d, my), (mx, my + d), (mx - d, my)]
        draw.polygon(diamond, fill=COLOR_MIP, outline="#78350f")

    # 3. Calibrated Target Center (Clean Blue Crosshair + Circle)
    c_r = 10
    draw.ellipse((cx - c_r, cy - c_r, cx + c_r, cy + c_r), outline=COLOR_CENTER, width=2)
    draw.line((cx - c_r - 5, cy, cx + c_r + 5, cy), fill=COLOR_CENTER, width=2)
    draw.line((cx, cy - c_r - 5, cx, cy + c_r + 5), fill=COLOR_CENTER, width=2)

    # 4. Impacts (Numbered circles)
    for record in metrics.impacts:
        x, y = record.x, record.y
        r = 5
        # Impact circle
        draw.ellipse((x - r, y - r, x + r, y + r), fill=COLOR_IMPACT, outline="#111827", width=2)

        # Clean compact badge next to impact
        lbl = str(record.index)
        draw.text((x + 7, y - 7), lbl, fill="#0f172a")

    return image


def plot_radial_polar_chart(metrics: GroupMetrics) -> plt.Figure:
    """Generate a clean polar chart showing direction and radial distance from center."""
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    if metrics.count == 0:
        ax.text(0, 0, "No Impacts", horizontalalignment="center", verticalalignment="center", color="#64748b", fontweight="bold")
        ax.set_yticklabels([])
        ax.set_title("Polar Impact Distribution (pixels)\nCenter = (0, 0)", pad=16, fontsize=10, fontweight="bold", color="#1e293b")
        fig.tight_layout()
        return fig

    thetas = []
    radii = []
    labels = []

    for rec in metrics.impacts:
        # Standard Cartesian to Polar:
        # Screen Y grows downward, so inverted ΔY (-offset_y) represents Upwards
        dx = rec.offset_x
        dy = -rec.offset_y
        theta = math.atan2(dy, dx)
        r = rec.radial_distance
        thetas.append(theta)
        radii.append(r)
        labels.append(str(rec.index))

    # Scatter of individual impacts
    ax.scatter(thetas, radii, c=COLOR_IMPACT, s=60, edgecolors="#0f172a", zorder=5, label="Impacts")

    for th, r, lbl in zip(thetas, radii, labels):
        ax.annotate(
            lbl,
            (th, r),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=8,
            fontweight="bold",
            color="#1e293b",
        )

    # Mean Impact Point on polar
    if metrics.count > 0:
        mdx = metrics.mean_offset_x
        mdy = -metrics.mean_offset_y
        m_theta = math.atan2(mdy, mdx)
        m_r = math.hypot(mdx, mdy)
        ax.scatter([m_theta], [m_r], c=COLOR_MIP, marker="D", s=80, edgecolors="#78350f", zorder=6, label="Mean Impact (MIP)")

    # Radial RMS Ring
    max_r = max(radii) if radii else 50
    limit_r = max(max_r * 1.25, 40)
    ax.set_ylim(0, limit_r)

    theta_grid = np.linspace(0, 2 * np.pi, 200)
    ax.plot(theta_grid, [metrics.radial_rms] * len(theta_grid), "--", color=COLOR_CENTER, lw=1.5, label=f"Radial RMS ({metrics.radial_rms:.1f} px)")

    ax.set_theta_zero_location("E")
    ax.set_title("Polar Impact Distribution (pixels)\nCenter = (0, 0)", pad=16, fontsize=10, fontweight="bold", color="#1e293b")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def plot_offset_scatter_chart(metrics: GroupMetrics) -> plt.Figure:
    """Generate a Cartesian offset chart showing X/Y offsets relative to target center (0, 0)."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    # Center axes lines
    ax.axhline(0, color=COLOR_GRID, linestyle="--", linewidth=1.2)
    ax.axvline(0, color=COLOR_GRID, linestyle="--", linewidth=1.2)

    # Center (0,0) Cross marker
    ax.scatter([0], [0], color=COLOR_CENTER, marker="+", s=130, linewidths=2.5, zorder=6, label="Center (0,0)")

    if metrics.count == 0:
        ax.text(0, 0, "No Impacts", ha="center", va="center", color="#64748b", fontweight="bold")
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_xlabel("Horizontal Offset ΔX (px)", fontsize=9, fontweight="bold", color="#334155")
        ax.set_ylabel("Vertical Offset ΔY (px, Up=+)", fontsize=9, fontweight="bold", color="#334155")
        ax.set_title("Cartesian Impact Offsets from Center (0, 0)", fontsize=10, fontweight="bold", color="#1e293b", pad=12)
        ax.legend(loc="upper right", fontsize=8, frameon=True)
        fig.tight_layout()
        return fig

    xs = [rec.offset_x for rec in metrics.impacts]
    # Invert offset_y so standard up is positive in graph representation
    ys = [-rec.offset_y for rec in metrics.impacts]
    labels = [str(rec.index) for rec in metrics.impacts]

    # Plot points
    ax.scatter(xs, ys, color=COLOR_IMPACT, s=65, edgecolors="#0f172a", zorder=4, label="Impacts")
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(
            f" #{lbl}",
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            fontweight="bold",
            color="#0f172a",
        )

    # Plot Mean Impact Point
    mx = metrics.mean_offset_x
    my = -metrics.mean_offset_y
    ax.scatter([mx], [my], color=COLOR_MIP, marker="D", s=90, edgecolors="#78350f", zorder=5, label=f"MIP ({mx:+.1f}, {my:+.1f} px)")

    # Plot Group RMS dispersion circle around MIP
    if metrics.group_rms > 0:
        circle = plt.Circle((mx, my), metrics.group_rms, color=COLOR_MIP, fill=False, linestyle=":", linewidth=1.8, label=f"Group RMS ({metrics.group_rms:.1f} px)")
        ax.add_patch(circle)

    # Extreme spread span line if applicable
    if metrics.extreme_spread_points is not None and metrics.count > 1:
        pt_a, pt_b = metrics.extreme_spread_points
        dx_a = pt_a[0] - metrics.target_center[0]
        dy_a = -(pt_a[1] - metrics.target_center[1])
        dx_b = pt_b[0] - metrics.target_center[0]
        dy_b = -(pt_b[1] - metrics.target_center[1])
        ax.plot([dx_a, dx_b], [dy_a, dy_b], color=COLOR_EXTREME, linestyle="-", linewidth=1.5, alpha=0.7, label=f"Max Spread ({metrics.max_spread:.1f} px)")

    # Adjust limits symmetrically
    all_x = xs + [mx, 0]
    all_y = ys + [my, 0]
    max_bound = max(max(map(abs, all_x)), max(map(abs, all_y)), 30) * 1.35
    ax.set_xlim(-max_bound, max_bound)
    ax.set_ylim(-max_bound, max_bound)

    ax.set_xlabel("Horizontal Offset ΔX (px)", fontsize=9, fontweight="bold", color="#334155")
    ax.set_ylabel("Vertical Offset ΔY (px, Up=+)", fontsize=9, fontweight="bold", color="#334155")
    ax.set_title("Cartesian Impact Offsets from Center (0, 0)", fontsize=10, fontweight="bold", color="#1e293b", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    return fig
