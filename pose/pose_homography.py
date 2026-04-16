"""
Pose Estimator 1 -- Homography Decomposition.

Finds a homography from matched 2D points on consecutive frames,
then decomposes it into rotation + translation using the known
camera intrinsics and the plane-at-known-depth assumption.

Usage:
    python pose/pose_homography.py                    # all DoFs
    python pose/pose_homography.py translate_x        # single DoF
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    GEN_OUTPUT,
    GROUND_TRUTHS,
    PLANE_DEPTH,
    FramePoseResult,
    K,
    PoseResult,
    load_frames,
    match_points,
    print_summary,
    to_gray,
)


def estimate_pose(
    pts1: np.ndarray, pts2: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (R, t) from homography decomposition, or (None, None)."""
    if len(pts1) < 4:
        return None, None

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if H is None:
        return None, None

    num, Rs, ts, normals = cv2.decomposeHomographyMat(H, K)

    # Pick the physically plausible solution:
    #  - plane normal should roughly face the camera  n ~ (0, 0, 1)
    #  - rotation angle should be small
    best_idx, best_score = 0, float("inf")
    for i in range(num):
        nz = normals[i][2, 0]
        if nz < 0.5:
            continue  # normal pointing away — wrong solution
        angle = np.linalg.norm(cv2.Rodrigues(Rs[i])[0])
        if angle < best_score:
            best_score = angle
            best_idx = i

    R = Rs[best_idx]
    t = ts[best_idx] * PLANE_DEPTH  # restore metric scale
    return R, t.reshape(3, 1)


def run_dof(dof_name: str) -> PoseResult:
    frames = load_frames(GEN_OUTPUT / dof_name / "frames")
    result = PoseResult(algorithm="Homography", dof_name=dof_name)

    for i in range(1, len(frames)):
        g1, g2 = to_gray(frames[i - 1]), to_gray(frames[i])
        pts1, pts2 = match_points(g1, g2)
        R, t = estimate_pose(pts1, pts2)
        result.frame_results.append(
            FramePoseResult(
                frame_index=i,
                R=R,
                t=t,
                n_points=len(pts1),
            )
        )

    print_summary(result)
    return result


def main() -> None:
    dofs = sys.argv[1:] if len(sys.argv) > 1 else list(GROUND_TRUTHS.keys())
    for dof in dofs:
        run_dof(dof)


if __name__ == "__main__":
    main()
