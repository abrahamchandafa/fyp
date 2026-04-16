"""
Pose Estimator 2 -- Perspective-n-Point (PnP).

Back-projects frame-1 blob centroids to 3D (plane at known depth),
then solves PnP against frame-2 observations to recover the full
6-DoF camera pose change with absolute metric scale.

Usage:
    python pose/pose_pnp.py                    # all DoFs
    python pose/pose_pnp.py translate_x        # single DoF
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
    PLANE_DEPTH,
    FramePoseResult,
    K,
    PoseResult,
    load_frames,
    match_points,
    print_summary,
    to_gray,
)


def back_project(pts2d: np.ndarray, depth: float) -> np.ndarray:
    """Back-project 2D image points to 3D assuming a plane at *depth*."""
    pts3d = np.empty((len(pts2d), 3), dtype=np.float64)
    for i, (u, v) in enumerate(pts2d):
        pts3d[i, 0] = (u - CX) * depth / FOCAL_LENGTH
        pts3d[i, 1] = (v - CY) * depth / FOCAL_LENGTH
        pts3d[i, 2] = depth
    return pts3d


def estimate_pose(
    pts1: np.ndarray, pts2: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Solve PnP: back-project pts1 -> 3D, match with pts2."""
    if len(pts1) < 4:
        return None, None

    pts3d = back_project(pts1, PLANE_DEPTH)
    pts2d = pts2.reshape(-1, 1, 2).astype(np.float64)

    success, rvec, tvec = cv2.solvePnP(
        pts3d, pts2d, K, None, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None, None

    # Sanity: reject if rotation > ~30 deg (per-frame should be tiny)
    if np.linalg.norm(rvec) > 0.5:
        return None, None

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3, 1)


def run_dof(dof_name: str) -> PoseResult:
    frames = load_frames(GEN_OUTPUT / dof_name / "frames")
    result = PoseResult(algorithm="PnP", dof_name=dof_name)

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
