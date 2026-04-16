"""
Heat diffusion visualisation demo.

Generates a side-by-side comparison showing thermal dots deposited on a
surface and their diffusion over time, highlighting the key physical
phenomenon that makes thermal pattern tracking challenging.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermal_tracker.heat_diffusion import HeatDiffusionSimulator

OUTPUT_DIR = Path("output") / "diffusion"


def run() -> None:
    print("Heat Diffusion Demo")
    print("=" * 40)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    W, H = 320, 240
    sim = HeatDiffusionSimulator(
        width=W,
        height=H,
        fade_rate=0.3,
        erode_rate=0.2,
        min_intensity=1.0,
        min_radius=1.0,
        ambient_temperature=293.15,
    )

    # Deposit a pattern of dots
    dot_positions = [
        (80, 60),
        (160, 60),
        (240, 60),
        (80, 120),
        (160, 120),
        (240, 120),
        (80, 180),
        (160, 180),
        (240, 180),
    ]
    for x, y in dot_positions:
        sim.deposit_heat(x, y, intensity=20.0, radius=3)

    dt = 0.1  # seconds per step

    # Capture snapshots at different diffusion stages
    snapshots: list[tuple[str, np.ndarray]] = []
    stages = [0, 50, 150, 400, 800]
    step_count = 0

    for target in stages:
        while step_count < target:
            sim.step(dt)
            step_count += 1
        field = sim.get_relative_field()
        mx = field.max() if field.max() > 0 else 1
        img8 = (field / mx * 255).astype(np.uint8)
        colored = cv2.applyColorMap(img8, cv2.COLORMAP_INFERNO)
        label = f"t = {step_count * dt:.3f}s  (step {step_count})"
        cv2.putText(
            colored, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )
        snapshots.append((label, colored))
        cv2.imwrite(str(OUTPUT_DIR / f"diffusion_step_{step_count:04d}.png"), colored)
        print(f"  Saved step {step_count}: peak ΔT = {field.max():.2f} K")

    # Composite side-by-side
    row1 = np.hstack([snapshots[0][1], snapshots[1][1], snapshots[2][1]])
    # Pad the bottom row if fewer than 3
    bottom = [snapshots[3][1], snapshots[4][1]]
    blank = np.zeros_like(snapshots[0][1])
    row2 = np.hstack(bottom + [blank])
    composite = np.vstack([row1, row2])
    cv2.imwrite(str(OUTPUT_DIR / "diffusion_composite.png"), composite)
    print(f"\n  Composite saved to: {OUTPUT_DIR / 'diffusion_composite.png'}")


if __name__ == "__main__":
    run()
