"""
Tracker 1 — Blob Centroid + Nearest-Neighbour Matching.

Detects bright blobs via adaptive thresholding + connected components,
then matches them frame-to-frame using greedy nearest-neighbour assignment.
The simplest possible thermal-dot tracker.

Usage:
    python track/track_blob.py                           # all DoFs
    python track/track_blob.py translate_x               # single DoF
    python track/track_blob.py translate_x rotate_roll   # subset
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
    MAX_BLOB_AREA,
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

# ── Algorithm-specific constants ──────────────────────────────────────
GATE_DISTANCE: float = 60.0  # max px a point can travel between frames
MORPH_KERNEL: int = 3  # morphological cleanup kernel size

# I/O
GEN_OUTPUT = Path(__file__).resolve().parent.parent / "pdr" / "output"
TRACK_OUTPUT = Path(__file__).resolve().parent / "output" / "blob"


def detect_centroids(gray: np.ndarray) -> list[tuple[float, float]]:
    """Return (cx, cy) for each bright blob in *gray*."""
    mask = thermal_mask(gray, BRIGHTNESS_THRESHOLD)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
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
    """Greedy nearest-neighbour matching.

    Returns list of (id, cx, cy, dx, dy) and the updated next_id.
    """
    used_curr: set[int] = set()
    matched: list[tuple[int, float, float, float, float]] = []

    for pid, px, py in prev:
        best_j, best_d = -1, GATE_DISTANCE
        for j, (cx, cy) in enumerate(curr_pts):
            if j in used_curr:
                continue
            d = np.hypot(cx - px, cy - py)
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            cx, cy = curr_pts[best_j]
            matched.append((pid, cx, cy, cx - px, cy - py))
            used_curr.add(best_j)

    # New points for unmatched current detections
    for j, (cx, cy) in enumerate(curr_pts):
        if j not in used_curr:
            matched.append((next_id, cx, cy, 0.0, 0.0))
            next_id += 1

    return matched, next_id


def run_dof(dof_name: str) -> TrackingResult:
    frames_dir = GEN_OUTPUT / dof_name / "frames"
    frames = load_frames(frames_dir)
    result = TrackingResult(algorithm="BlobCentroid", dof_name=dof_name)

    out_dir = TRACK_OUTPUT / dof_name
    out_dir.mkdir(parents=True, exist_ok=True)

    prev_pts: list[tuple[int, float, float]] = []
    next_id = 0

    with AnnotatedVideoWriter(out_dir / f"{dof_name}_blob.mp4") as vw:
        for fi, frame in enumerate(frames):
            gray = to_gray(frame)
            curr_centroids = detect_centroids(gray)

            matched, next_id = match_nearest(prev_pts, curr_centroids, next_id)

            # Build FrameResult from matched pairs that had a previous position
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

            vw.write(frame, fr, algo="BlobCentroid", dof=dof_name)

            # Carry forward
            prev_pts = [(m[0], m[1], m[2]) for m in matched]

    print_summary(result)
    return result


def main() -> None:
    dofs = sys.argv[1:] if len(sys.argv) > 1 else list(GROUND_TRUTHS.keys())
    for dof in dofs:
        run_dof(dof)


if __name__ == "__main__":
    main()
