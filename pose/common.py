"""
Shared utilities for 6-DoF pose estimation from thermal dot frames.

Point matching uses Farneback dense optical flow (ranked #1 in tracking
benchmark with 0.726 px/frame average error).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# =====================================================================
#  Camera intrinsics  (editable — adjust to match your setup)
# =====================================================================
FOCAL_LENGTH: float = 500.0
CX: float = 320.0
CY: float = 240.0
PLANE_DEPTH: float = 1.0  # metres — assumed distance to projection plane

K = np.array(
    [[FOCAL_LENGTH, 0, CX], [0, FOCAL_LENGTH, CY], [0, 0, 1]],
    dtype=np.float64,
)

# =====================================================================
#  Blob detection thresholds (from track/common.py)
# =====================================================================
BRIGHTNESS_THRESHOLD: int = 40
MIN_BLOB_AREA: int = 30  # above 25 filters overlay-text characters
MAX_BLOB_AREA: int = 2000

# =====================================================================
#  Farneback parameters (from track/track_dense_flow.py — best tracker)
# =====================================================================
FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)

# =====================================================================
#  Paths
# =====================================================================
GEN_OUTPUT = Path(__file__).resolve().parent.parent / "generation" / "output"
POSE_OUTPUT = Path(__file__).resolve().parent / "output"

# =====================================================================
#  Ground truth per-frame pose changes
# =====================================================================
#
# Convention (OpenCV):
#   X = right, Y = down, Z = into screen.
#   rvec = Rodrigues rotation of camera-2 w.r.t. camera-1.
#   tvec = displacement of the camera *centre* in world (camera-1) frame.
#
# Derived from generation constants:
#   SHIFT_RATE   = 5.0 px/frame
#   ROTATION_RATE = 0.025 rad/frame
#   ZOOM_RATE    = 0.035 / frame
#
# translate_x: dots shift LEFT by 5 px  -> camera moved RIGHT  -> tx = +5*Z/f
# translate_y: dots shift DOWN by 5 px  -> camera moved UP     -> ty = -5*Z/f
# translate_z: dots expand by 3.5%      -> camera moved FORWARD-> tz ~ +0.034
# rotate_yaw:  dots shift LEFT by 5 px  -> camera yawed RIGHT  -> ry = -0.01
# rotate_pitch: dots shift DOWN by 5 px -> camera pitched UP   -> rx = -0.01
# rotate_roll:  dots rotate CW by 0.025 -> camera rolled CW    -> rz = -0.025


@dataclass
class PoseGT:
    rvec: tuple[float, float, float]
    tvec: tuple[float, float, float]


GROUND_TRUTHS: dict[str, PoseGT] = {
    "translate_x": PoseGT(rvec=(0, 0, 0), tvec=(0.01, 0, 0)),
    "translate_y": PoseGT(rvec=(0, 0, 0), tvec=(0, -0.01, 0)),
    "translate_z": PoseGT(rvec=(0, 0, 0), tvec=(0, 0, 0.034)),
    "rotate_yaw": PoseGT(rvec=(0, -0.01, 0), tvec=(0, 0, 0)),
    "rotate_pitch": PoseGT(rvec=(-0.01, 0, 0), tvec=(0, 0, 0)),
    "rotate_roll": PoseGT(rvec=(0, 0, -0.025), tvec=(0, 0, 0)),
}


# =====================================================================
#  Data classes
# =====================================================================


@dataclass
class FramePoseResult:
    frame_index: int
    R: np.ndarray | None  # 3x3 rotation (None if estimation failed)
    t: np.ndarray | None  # 3x1 translation
    n_points: int


@dataclass
class PoseResult:
    algorithm: str
    dof_name: str
    frame_results: list[FramePoseResult] = field(default_factory=list)

    @property
    def valid_frames(self) -> list[FramePoseResult]:
        return [f for f in self.frame_results if f.R is not None]

    @property
    def mean_rvec(self) -> np.ndarray:
        vecs = [cv2.Rodrigues(f.R)[0].flatten() for f in self.valid_frames]
        return np.mean(vecs, axis=0) if vecs else np.zeros(3)

    @property
    def mean_camera_motion(self) -> np.ndarray:
        """Mean camera-centre displacement (world frame)."""
        motions = []
        for f in self.valid_frames:
            motions.append((-f.R.T @ f.t).flatten())
        return np.mean(motions, axis=0) if motions else np.zeros(3)


# =====================================================================
#  Frame I/O
# =====================================================================


def load_frames(frames_dir: Path) -> list[np.ndarray]:
    paths = sorted(frames_dir.glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"No frame_*.png in {frames_dir}")
    return [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]


def to_gray(frame: np.ndarray) -> np.ndarray:
    if len(frame.shape) == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


# =====================================================================
#  Blob detection
# =====================================================================


def detect_blobs(gray: np.ndarray) -> np.ndarray:
    """Return Nx2 float32 array of blob centroids."""
    _, mask = cv2.threshold(gray, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    pts = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if MIN_BLOB_AREA <= area <= MAX_BLOB_AREA:
            pts.append(centroids[i])
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array(pts, dtype=np.float32)


# =====================================================================
#  Point matching via Farneback dense flow
# =====================================================================


def match_points(gray1: np.ndarray, gray2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect blobs in frame1, use dense flow to find matches in frame2.

    Returns (pts1, pts2) as Nx2 float64 arrays.
    """
    blobs = detect_blobs(gray1)
    if len(blobs) < 4:
        return np.zeros((0, 2), np.float64), np.zeros((0, 2), np.float64)

    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, **FARNEBACK_PARAMS)

    h, w = gray1.shape[:2]
    pts1, pts2 = [], []
    for cx, cy in blobs:
        ix = int(np.clip(round(cx), 0, w - 1))
        iy = int(np.clip(round(cy), 0, h - 1))
        dx, dy = flow[iy, ix]
        pts1.append((cx, cy))
        pts2.append((cx + dx, cy + dy))

    return np.array(pts1, dtype=np.float64), np.array(pts2, dtype=np.float64)


# =====================================================================
#  Error metrics
# =====================================================================


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Geodesic distance between two rotations, in degrees."""
    R_diff = R_est @ R_gt.T
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    angle = math.acos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(angle)


def translation_dir_error_deg(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    """Angle between two translation directions, in degrees.

    Returns NaN if either vector has negligible magnitude.
    """
    t_est = t_est.flatten()
    t_gt = t_gt.flatten()
    n1, n2 = np.linalg.norm(t_est), np.linalg.norm(t_gt)
    if n1 < 1e-10 or n2 < 1e-10:
        return float("nan")
    cos = np.clip(np.dot(t_est, t_gt) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos))


# =====================================================================
#  Summary printer
# =====================================================================


def print_summary(result: PoseResult) -> None:
    gt = GROUND_TRUTHS[result.dof_name]
    gt_rvec = np.array(gt.rvec, dtype=np.float64)
    gt_tvec = np.array(gt.tvec, dtype=np.float64)
    gt_R, _ = cv2.Rodrigues(gt_rvec)

    est_rvec = result.mean_rvec
    est_tvec = result.mean_camera_motion
    est_R, _ = cv2.Rodrigues(est_rvec)

    rot_err = rotation_error_deg(est_R, gt_R)
    trans_err = translation_dir_error_deg(est_tvec, gt_tvec)

    n_total = len(result.frame_results)
    n_valid = len(result.valid_frames)

    print(f"\n{'-' * 60}")
    print(f"  Algorithm     : {result.algorithm}")
    print(f"  DoF           : {result.dof_name}")
    print(f"  Frames        : {n_total} ({n_valid} valid)")
    print(
        f"  Est rvec      : [{est_rvec[0]:+.6f}, {est_rvec[1]:+.6f}, {est_rvec[2]:+.6f}]"
    )
    print(
        f"  GT  rvec      : [{gt_rvec[0]:+.6f}, {gt_rvec[1]:+.6f}, {gt_rvec[2]:+.6f}]"
    )
    print(f"  Rotation err  : {rot_err:.3f} deg")
    print(
        f"  Est tvec      : [{est_tvec[0]:+.6f}, {est_tvec[1]:+.6f}, {est_tvec[2]:+.6f}]"
    )
    print(
        f"  GT  tvec      : [{gt_tvec[0]:+.6f}, {gt_tvec[1]:+.6f}, {gt_tvec[2]:+.6f}]"
    )
    if math.isnan(trans_err):
        print(f"  Trans dir err : N/A (pure rotation)")
    else:
        print(f"  Trans dir err : {trans_err:.3f} deg")
    print(f"{'-' * 60}")
