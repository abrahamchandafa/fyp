"""
Pose Estimator 3 -- Essential Matrix Decomposition.

Computes the essential matrix from matched 2D points and known
camera intrinsics, then decomposes it into rotation + translation
direction.  Translation scale is NOT recoverable (unit vector only).

NOTE: This method CANNOT handle pure-rotation DoFs (yaw, pitch, roll)
because the essential matrix is degenerate when translation is zero.
The benchmark will show NaN for those cases.

Usage:
    python pose/pose_essential.py                    # all DoFs
    python pose/pose_essential.py translate_x        # single DoF
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    CX,
    CY,
    FOCAL_LENGTH,
    GEN_OUTPUT,
    GROUND_TRUTHS,
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
    """Return (R, t) from essential matrix, or (None, None)."""
    if len(pts1) < 5:
        return None, None

    E, mask_e = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None:
        return None, None

    n_inliers, R, t, mask_p = cv2.recoverPose(E, pts1, pts2, K)
    if n_inliers < 4:
        return None, None

    # Sanity: rotation should be small per frame
    rvec, _ = cv2.Rodrigues(R)
    if np.linalg.norm(rvec) > 0.5:
        return None, None

    return R, t.reshape(3, 1)  # t is a unit vector


def run_dof(dof_name: str) -> PoseResult:
    frames = load_frames(GEN_OUTPUT / dof_name / "frames")
    result = PoseResult(algorithm="Essential", dof_name=dof_name)

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
