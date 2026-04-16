"""
Tracker 5 — Frame Differencing + Contour Centroid Tracking.

Instead of looking at absolute brightness, this tracker computes the
absolute difference between consecutive frames to isolate *moving*
regions, then extracts contours and tracks their centroids.

This is analogous to a simple background-subtraction approach tuned
for the thermal-dot scenario where the "background" is the previous
frame.

Usage:
    python track/track_framediff.py
    python track/track_framediff.py translate_x
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    GROUND_TRUTHS,
    MAX_BLOB_AREA,
    MIN_BLOB_AREA,
    AnnotatedVideoWriter,
    FrameResult,
    TrackedPoint,
    TrackingResult,
    load_frames,
    print_summary,
    to_gray,
)

# ── Algorithm-specific constants ──────────────────────────────────────
DIFF_THRESHOLD: int = 25  # binary threshold on |frame_diff|
GATE_DISTANCE: float = 60.0  # max match distance (px)
MORPH_KERNEL: int = 5

GEN_OUTPUT = Path(__file__).resolve().parent.parent / "generation" / "output"
TRACK_OUTPUT = Path(__file__).resolve().parent / "output" / "framediff"


def diff_centroids(
    prev_gray: np.ndarray, curr_gray: np.ndarray
) -> list[tuple[float, float]]:
    """Detect centroids in the absolute frame difference."""
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    pts: list[tuple[float, float]] = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if MIN_BLOB_AREA <= area <= MAX_BLOB_AREA:
            pts.append((float(centroids[i, 0]), float(centroids[i, 1])))
    return pts


def match_nearest(
    prev: list[tuple[int, float, float]],
    curr_pts: list[tuple[float, float]],
    next_id: int,
) -> tuple[list[tuple[int, float, float, float, float]], int]:
    used: set[int] = set()
    matched: list[tuple[int, float, float, float, float]] = []
    for pid, px, py in prev:
        best_j, best_d = -1, GATE_DISTANCE
        for j, (cx, cy) in enumerate(curr_pts):
            if j in used:
                continue
            d = np.hypot(cx - px, cy - py)
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            cx, cy = curr_pts[best_j]
            matched.append((pid, cx, cy, cx - px, cy - py))
            used.add(best_j)
    for j, (cx, cy) in enumerate(curr_pts):
        if j not in used:
            matched.append((next_id, cx, cy, 0.0, 0.0))
            next_id += 1
    return matched, next_id


def run_dof(dof_name: str) -> TrackingResult:
    frames_dir = GEN_OUTPUT / dof_name / "frames"
    frames = load_frames(frames_dir)
    result = TrackingResult(algorithm="FrameDiff", dof_name=dof_name)

    out_dir = TRACK_OUTPUT / dof_name
    out_dir.mkdir(parents=True, exist_ok=True)

    prev_gray = to_gray(frames[0])
    prev_pts: list[tuple[int, float, float]] = []
    next_id = 0

    with AnnotatedVideoWriter(out_dir / f"{dof_name}_framediff.mp4") as vw:
        for fi, frame in enumerate(frames):
            curr_gray = to_gray(frame)

            if fi == 0:
                fr = FrameResult(fi, [], 0.0, 0.0, 0.0, 0)
                result.frame_results.append(fr)
                vw.write(frame, fr, algo="FrameDiff", dof=dof_name)
                prev_gray = curr_gray
                continue

            centroids = diff_centroids(prev_gray, curr_gray)
            matched, next_id = match_nearest(prev_pts, centroids, next_id)

            dxs = [dx for (_, _, _, dx, dy) in matched if dx != 0 or dy != 0]
            dys = [dy for (_, _, _, dx, dy) in matched if dx != 0 or dy != 0]

            fr = FrameResult(
                frame_index=fi,
                points=[TrackedPoint(m[0], m[1], m[2]) for m in matched],
                mean_dx=float(np.mean(dxs)) if dxs else 0.0,
                mean_dy=float(np.mean(dys)) if dys else 0.0,
                mean_speed=float(
                    np.mean([np.hypot(dx, dy) for dx, dy in zip(dxs, dys)])
                )
                if dxs
                else 0.0,
                n_tracked=len(dxs),
            )
            result.frame_results.append(fr)
            vw.write(frame, fr, algo="FrameDiff", dof=dof_name)

            prev_pts = [(m[0], m[1], m[2]) for m in matched]
            prev_gray = curr_gray

    print_summary(result)
    return result


def main() -> None:
    dofs = sys.argv[1:] if len(sys.argv) > 1 else list(GROUND_TRUTHS.keys())
    for dof in dofs:
        run_dof(dof)


if __name__ == "__main__":
    main()
