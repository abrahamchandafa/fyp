"""Quick diagnostic for pose estimation."""

import sys

sys.path.insert(0, "pose")
import cv2
import numpy as np
from common import *

frames = load_frames(GEN_OUTPUT / "translate_x" / "frames")
for f_idx in range(1, 4):
    g1, g2 = to_gray(frames[f_idx - 1]), to_gray(frames[f_idx])
    pts1, pts2 = match_points(g1, g2)
    disp = pts2 - pts1
    print(
        f"Frame {f_idx}: {len(pts1)} pts, mean_dx={disp[:, 0].mean():.3f}, mean_dy={disp[:, 1].mean():.3f}"
    )

    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    num, Rs, ts, normals = cv2.decomposeHomographyMat(H, K)
    for j in range(num):
        t = ts[j].flatten() * PLANE_DEPTH
        cm = (-Rs[j].T @ ts[j].flatten()) * PLANE_DEPTH
        nz = normals[j][2, 0]
        print(
            f"  Sol {j}: nz={nz:.3f} t_pnp=[{t[0]:+.5f},{t[1]:+.5f},{t[2]:+.5f}] cam=[{cm[0]:+.5f},{cm[1]:+.5f},{cm[2]:+.5f}]"
        )

    # Also PnP
    from pose_pnp import back_project

    pts3d = back_project(pts1, PLANE_DEPTH)
    ok, rvec, tvec = cv2.solvePnP(
        pts3d, pts2.reshape(-1, 1, 2), K, None, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if ok:
        R, _ = cv2.Rodrigues(rvec)
        cm_pnp = (-R.T @ tvec).flatten()
        print(
            f"  PnP: t_pnp=[{tvec[0, 0]:+.5f},{tvec[1, 0]:+.5f},{tvec[2, 0]:+.5f}] cam=[{cm_pnp[0]:+.5f},{cm_pnp[1]:+.5f},{cm_pnp[2]:+.5f}]"
        )
