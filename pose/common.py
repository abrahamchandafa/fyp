"""
Shared utilities for SLAM pipeline.

Provides:
  - Camera intrinsics
  - Data structures for pose results
  - Error metrics
  - Ground truth for pose benchmarking
  - Track CSV loading for SLAM
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# ============================================================================
#  CAMERA INTRINSICS
# ============================================================================

FOCAL_LENGTH: float = 500.0
CX: float = 320.0
CY: float = 240.0

K = np.array(
    [[FOCAL_LENGTH, 0, CX], [0, FOCAL_LENGTH, CY], [0, 0, 1]],
    dtype=np.float64,
)

# ============================================================================
#  GROUND TRUTH (camera motion)
# ============================================================================


@dataclass
class PoseGT:
    """Ground truth camera motion between frames."""

    rvec: tuple[float, float, float]
    tvec: tuple[float, float, float]


GROUND_TRUTHS: Dict[str, PoseGT] = {
    "translate_x": PoseGT(rvec=(0, 0, 0), tvec=(0.01, 0, 0)),
    "translate_y": PoseGT(rvec=(0, 0, 0), tvec=(0, -0.01, 0)),
    "translate_z": PoseGT(rvec=(0, 0, 0), tvec=(0, 0, 0.034)),
    "rotate_yaw": PoseGT(rvec=(0, -0.01, 0), tvec=(0, 0, 0)),
    "rotate_pitch": PoseGT(rvec=(-0.01, 0, 0), tvec=(0, 0, 0)),
    "rotate_roll": PoseGT(rvec=(0, 0, -0.025), tvec=(0, 0, 0)),
}

# ============================================================================
#  DATA STRUCTURES
# ============================================================================


@dataclass
class FramePoseResult:
    """Pose estimation result for a single frame."""

    frame_index: int
    R: Optional[np.ndarray]
    t: Optional[np.ndarray]
    n_points: int


@dataclass
class PoseResult:
    """Complete pose estimation result for a sequence."""

    algorithm: str
    dof_name: str
    frame_results: List[FramePoseResult] = field(default_factory=list)

    @property
    def valid_frames(self) -> List[FramePoseResult]:
        return [f for f in self.frame_results if f.R is not None]

    @property
    def mean_rvec(self) -> np.ndarray:
        vecs: List[np.ndarray] = []
        for f in self.valid_frames:
            rvec, _ = cv2.Rodrigues(f.R)
            vecs.append(rvec.flatten())
        return np.mean(vecs, axis=0) if vecs else np.zeros(3)

    @property
    def mean_camera_motion(self) -> np.ndarray:
        motions: List[np.ndarray] = []
        for f in self.valid_frames:
            motions.append((-f.R.T @ f.t).flatten())
        return np.mean(motions, axis=0) if motions else np.zeros(3)


# ============================================================================
#  ERROR METRICS
# ============================================================================


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Geodesic distance between two rotations, in degrees."""
    R_diff = R_est @ R_gt.T
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    angle = math.acos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(angle)


def translation_dir_error_deg(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    """Angle between two translation directions, in degrees."""
    t_est = t_est.flatten()
    t_gt = t_gt.flatten()
    n1, n2 = np.linalg.norm(t_est), np.linalg.norm(t_gt)
    if n1 < 1e-10 or n2 < 1e-10:
        return float("nan")
    cos = np.clip(np.dot(t_est, t_gt) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos))


# ============================================================================
#  LOAD TRACKS FROM CSV
# ============================================================================


def load_tracks_from_csv(csv_path: Path) -> Dict[int, List[tuple[int, float, float]]]:
    """Load persistent track data from track_blob CSV."""
    tracks: Dict[int, List[tuple[int, float, float]]] = {}

    with open(csv_path, "r", newline="") as f:
        next(f)
        for line in f:
            if not line.strip():
                continue
            frame_id, point_id, x, y = line.strip().split(",")
            frame_id = int(frame_id)
            point_id = int(point_id)
            x = float(x)
            y = float(y)
            tracks.setdefault(point_id, []).append((frame_id, x, y))

    for observations in tracks.values():
        observations.sort(key=lambda item: item[0])

    return tracks
