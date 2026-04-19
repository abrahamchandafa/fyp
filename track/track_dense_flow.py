"""
Tracker 3 — Dense Optical Flow (Farnebäck).

Computes a full per-pixel motion field between consecutive frames
using the Farnebäck algorithm.  Motion is then sampled only at
bright "hot" dot locations (thresholded blobs) to estimate the
dominant direction and velocity.

Usage:
    python track/track_dense_flow.py
    python track/track_dense_flow.py translate_x
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    BRIGHTNESS_THRESHOLD,
    GROUND_TRUTHS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MIN_BLOB_AREA,
    AnnotatedVideoWriter,
    FrameResult,
    TrackedPoint,
    TrackingResult,
    load_frames,
    print_summary,
    thermal_mask,
    to_gray,
)

# ── Farnebäck parameters ─────────────────────────────────────────────
FARNEBACK_PYR_SCALE: float = 0.5
FARNEBACK_LEVELS: int = 3
FARNEBACK_WINSIZE: int = 15
FARNEBACK_ITERATIONS: int = 3
FARNEBACK_POLY_N: int = 5
FARNEBACK_POLY_SIGMA: float = 1.2

FLOW_MAG_THRESHOLD: float = 0.5  # ignore flow below this (noise)

GEN_OUTPUT = Path(__file__).resolve().parent.parent / "pdr" / "output"
TRACK_OUTPUT = Path(__file__).resolve().parent / "output" / "dense_flow"


def sample_flow_at_blobs(
    flow: np.ndarray, gray: np.ndarray
) -> tuple[list[tuple[float, float, float, float]], np.ndarray]:
    """Sample the flow field at blob centroids and return (x, y, dx, dy) + mask."""
    mask = thermal_mask(gray, BRIGHTNESS_THRESHOLD)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    samples: list[tuple[float, float, float, float]] = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_BLOB_AREA:
            continue
        cx, cy = centroids[i]
        ix, iy = int(round(cx)), int(round(cy))
        if 0 <= iy < flow.shape[0] and 0 <= ix < flow.shape[1]:
            dx, dy = flow[iy, ix]
            mag = np.hypot(dx, dy)
            if mag > FLOW_MAG_THRESHOLD:
                samples.append((cx, cy, float(dx), float(dy)))
    return samples, mask


def run_dof(dof_name: str) -> TrackingResult:
    frames_dir = GEN_OUTPUT / dof_name / "frames"
    frames = load_frames(frames_dir)
    result = TrackingResult(algorithm="DenseFlow", dof_name=dof_name)

    out_dir = TRACK_OUTPUT / dof_name
    out_dir.mkdir(parents=True, exist_ok=True)

    prev_gray = to_gray(frames[0])

    with AnnotatedVideoWriter(out_dir / f"{dof_name}_dense.mp4") as vw:
        for fi in range(len(frames)):
            curr_gray = to_gray(frames[fi])

            if fi == 0:
                fr = FrameResult(fi, [], 0.0, 0.0, 0.0, 0)
                result.frame_results.append(fr)
                vw.write(frames[fi], fr, algo="DenseFlow", dof=dof_name)
                prev_gray = curr_gray
                continue

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                curr_gray,
                None,
                FARNEBACK_PYR_SCALE,
                FARNEBACK_LEVELS,
                FARNEBACK_WINSIZE,
                FARNEBACK_ITERATIONS,
                FARNEBACK_POLY_N,
                FARNEBACK_POLY_SIGMA,
                0,
            )

            samples, _ = sample_flow_at_blobs(flow, curr_gray)

            dxs = [s[2] for s in samples]
            dys = [s[3] for s in samples]

            tracked_pts = [TrackedPoint(i, s[0], s[1]) for i, s in enumerate(samples)]
            fr = FrameResult(
                frame_index=fi,
                points=tracked_pts,
                mean_dx=float(np.mean(dxs)) if dxs else 0.0,
                mean_dy=float(np.mean(dys)) if dys else 0.0,
                mean_speed=float(
                    np.mean([np.hypot(dx, dy) for dx, dy in zip(dxs, dys)])
                )
                if dxs
                else 0.0,
                n_tracked=len(samples),
            )
            result.frame_results.append(fr)

            # Draw flow arrows on the visualisation
            vis = frames[fi].copy()
            for s in samples:
                pt1 = (int(s[0]), int(s[1]))
                pt2 = (int(s[0] + s[2] * 3), int(s[1] + s[3] * 3))
                cv2.arrowedLine(
                    vis, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA, tipLength=0.3
                )
            vw.write(vis, fr, algo="DenseFlow", dof=dof_name)

            prev_gray = curr_gray

    print_summary(result)
    return result


def main() -> None:
    dofs = sys.argv[1:] if len(sys.argv) > 1 else list(GROUND_TRUTHS.keys())
    for dof in dofs:
        run_dof(dof)


if __name__ == "__main__":
    main()
