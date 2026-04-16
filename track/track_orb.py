"""
Tracker 4 — ORB Feature Descriptor Matching.

Detects ORB keypoints restricted to bright thermal regions,
computes descriptors, and matches them frame-to-frame using
a brute-force Hamming matcher with ratio test.

Usage:
    python track/track_orb.py
    python track/track_orb.py translate_x
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
ORB_N_FEATURES: int = 200
ORB_SCALE_FACTOR: float = 1.2
ORB_N_LEVELS: int = 8
RATIO_THRESHOLD: float = 0.75  # Lowe's ratio test
MAX_MATCH_DIST: float = 80.0  # pixel distance sanity check

GEN_OUTPUT = Path(__file__).resolve().parent.parent / "generation" / "output"
TRACK_OUTPUT = Path(__file__).resolve().parent / "output" / "orb"


def create_orb() -> cv2.ORB:
    return cv2.ORB_create(
        nfeatures=ORB_N_FEATURES,
        scaleFactor=ORB_SCALE_FACTOR,
        nlevels=ORB_N_LEVELS,
    )


def hot_mask(gray: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(gray)
    mask[gray > BRIGHTNESS_THRESHOLD] = 255
    return mask


def run_dof(dof_name: str) -> TrackingResult:
    frames_dir = GEN_OUTPUT / dof_name / "frames"
    frames = load_frames(frames_dir)
    result = TrackingResult(algorithm="ORB", dof_name=dof_name)

    out_dir = TRACK_OUTPUT / dof_name
    out_dir.mkdir(parents=True, exist_ok=True)

    orb = create_orb()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    prev_gray = to_gray(frames[0])
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, hot_mask(prev_gray))

    with AnnotatedVideoWriter(out_dir / f"{dof_name}_orb.mp4") as vw:
        for fi in range(len(frames)):
            curr_gray = to_gray(frames[fi])
            curr_kp, curr_des = orb.detectAndCompute(curr_gray, hot_mask(curr_gray))

            dxs: list[float] = []
            dys: list[float] = []
            matched_curr_pts: list[tuple[int, float, float]] = []

            if (
                fi > 0
                and prev_des is not None
                and curr_des is not None
                and len(prev_des) >= 2
                and len(curr_des) >= 2
            ):
                matches = bf.knnMatch(prev_des, curr_des, k=2)
                good: list[cv2.DMatch] = []
                for pair in matches:
                    if len(pair) == 2:
                        m, n = pair
                        if m.distance < RATIO_THRESHOLD * n.distance:
                            good.append(m)

                for m in good:
                    pt_prev = prev_kp[m.queryIdx].pt
                    pt_curr = curr_kp[m.trainIdx].pt
                    d = np.hypot(pt_curr[0] - pt_prev[0], pt_curr[1] - pt_prev[1])
                    if d > MAX_MATCH_DIST:
                        continue
                    dx = pt_curr[0] - pt_prev[0]
                    dy = pt_curr[1] - pt_prev[1]
                    dxs.append(dx)
                    dys.append(dy)
                    matched_curr_pts.append((m.trainIdx, pt_curr[0], pt_curr[1]))

            tracked_pts = [TrackedPoint(idx, x, y) for idx, x, y in matched_curr_pts]
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
            vw.write(frames[fi], fr, algo="ORB", dof=dof_name)

            prev_gray = curr_gray
            prev_kp, prev_des = curr_kp, curr_des

    print_summary(result)
    return result


def main() -> None:
    dofs = sys.argv[1:] if len(sys.argv) > 1 else list(GROUND_TRUTHS.keys())
    for dof in dofs:
        run_dof(dof)


if __name__ == "__main__":
    main()
