"""
Tracker 2 — Sparse Optical Flow (Lucas-Kanade).

Uses cv2.goodFeaturesToTrack to pick initial "hot" keypoints, then
tracks them with the pyramidal Lucas-Kanade method.  When points
are lost (or their count drops), new features are detected and
added to the active set — mirroring the "re-projection" concept
from the FYP.

Usage:
    python track/track_lk.py                   # all DoFs
    python track/track_lk.py translate_x       # single DoF
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
    AnnotatedVideoWriter,
    FrameResult,
    TrackedPoint,
    TrackingResult,
    load_frames,
    print_summary,
    to_gray,
)

# ── Algorithm-specific constants ──────────────────────────────────────
MAX_CORNERS: int = 50
QUALITY_LEVEL: float = 0.15
MIN_DISTANCE: float = 15.0
REDETECT_THRESHOLD: int = 4  # re-detect if fewer points remain
LK_WIN_SIZE: tuple[int, int] = (21, 21)
LK_MAX_LEVEL: int = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

GEN_OUTPUT = Path(__file__).resolve().parent.parent / "pdr" / "output"
TRACK_OUTPUT = Path(__file__).resolve().parent / "output" / "lucas_kanade"


def detect_features(gray: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Detect good-feature-to-track points inside bright regions only."""
    hot_mask = np.zeros_like(gray)
    hot_mask[gray > BRIGHTNESS_THRESHOLD] = 255
    if mask is not None:
        hot_mask = cv2.bitwise_and(hot_mask, mask)
    pts = cv2.goodFeaturesToTrack(
        gray, MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE, mask=hot_mask
    )
    return pts if pts is not None else np.empty((0, 1, 2), dtype=np.float32)


def run_dof(dof_name: str) -> TrackingResult:
    frames_dir = GEN_OUTPUT / dof_name / "frames"
    frames = load_frames(frames_dir)
    result = TrackingResult(algorithm="LucasKanade", dof_name=dof_name)

    out_dir = TRACK_OUTPUT / dof_name
    out_dir.mkdir(parents=True, exist_ok=True)

    prev_gray = to_gray(frames[0])
    prev_pts = detect_features(prev_gray)
    next_id = len(prev_pts)
    # Assign IDs
    ids: list[int] = list(range(next_id))

    with AnnotatedVideoWriter(out_dir / f"{dof_name}_lk.mp4") as vw:
        for fi in range(len(frames)):
            curr_gray = to_gray(frames[fi])

            if len(prev_pts) > 0 and fi > 0:
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray,
                    curr_gray,
                    prev_pts,
                    None,
                    winSize=LK_WIN_SIZE,
                    maxLevel=LK_MAX_LEVEL,
                    criteria=LK_CRITERIA,
                )
                status = status.ravel()

                good_mask = status == 1
                good_prev = prev_pts[good_mask]
                good_curr = curr_pts[good_mask]
                good_ids = [ids[i] for i in range(len(ids)) if good_mask[i]]

                dxs = (
                    (good_curr[:, 0, 0] - good_prev[:, 0, 0]).tolist()
                    if len(good_curr) > 0
                    else []
                )
                dys = (
                    (good_curr[:, 0, 1] - good_prev[:, 0, 1]).tolist()
                    if len(good_curr) > 0
                    else []
                )
            else:
                good_curr = prev_pts
                good_ids = ids
                dxs, dys = [], []

            # Build frame result
            tracked_pts = [
                TrackedPoint(
                    good_ids[i], float(good_curr[i, 0, 0]), float(good_curr[i, 0, 1])
                )
                for i in range(len(good_curr))
            ]
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
                n_tracked=len(dxs),
            )
            result.frame_results.append(fr)
            vw.write(frames[fi], fr, algo="LucasKanade", dof=dof_name)

            # Re-detect if points are running low
            if len(good_curr) < REDETECT_THRESHOLD:
                new_pts = detect_features(curr_gray)
                new_ids = list(range(next_id, next_id + len(new_pts)))
                next_id += len(new_pts)
                if len(good_curr) > 0:
                    prev_pts = (
                        np.vstack([good_curr, new_pts])
                        if len(new_pts) > 0
                        else good_curr
                    )
                    ids = good_ids + new_ids
                else:
                    prev_pts = new_pts
                    ids = new_ids
            else:
                prev_pts = good_curr
                ids = good_ids

            prev_gray = curr_gray

    print_summary(result)
    return result


def main() -> None:
    dofs = sys.argv[1:] if len(sys.argv) > 1 else list(GROUND_TRUTHS.keys())
    for dof in dofs:
        run_dof(dof)


if __name__ == "__main__":
    main()
