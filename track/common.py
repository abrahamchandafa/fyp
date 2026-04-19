"""
Shared utilities for all tracking scripts.

Every tracker script imports this module, which provides:
  - Calibration constants (edit these to match your generation settings)
  - Frame loading from a folder of PNGs
  - Ground-truth motion definitions for each DoF
  - Common data structures for tracking results
  - Video writer helper
  - Console reporting helper
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
#  CALIBRATION CONSTANTS  —  keep in sync with generation/generate.py
# ═══════════════════════════════════════════════════════════════════════

IMAGE_WIDTH: int = 640
IMAGE_HEIGHT: int = 480
FPS: int = 15
DURATION_S: int = 5
N_FRAMES: int = FPS * DURATION_S  # 75

# Generation motion rates (ground truth)
GT_SHIFT_RATE: float = 5.0  # px/frame for translations
GT_ROTATION_RATE: float = 0.025  # rad/frame for roll
GT_ZOOM_RATE: float = 0.035  # expansion factor/frame for translate-Z

# Detection thresholds (tune per algorithm)
BRIGHTNESS_THRESHOLD: int = 40  # min grayscale value for a "hot" pixel
MIN_BLOB_AREA: int = 6  # smallest acceptable blob (px²)
MAX_BLOB_AREA: int = 800  # largest acceptable blob (px²)


# ═══════════════════════════════════════════════════════════════════════
#  Ground truth for each DoF
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class GroundTruth:
    """Known per-frame motion for a single DoF."""

    dof_name: str
    description: str
    mean_dx: float  # expected mean Δx (px/frame), in image coords
    mean_dy: float  # expected mean Δy (px/frame), in image coords
    is_rotation: bool = False
    is_zoom: bool = False


# The generator shifts dots; from the camera's perspective the *estimated*
# motion should be the *opposite* sign (camera moves right → dots slide left,
# so tracker sees negative dx → infers camera moved right = +dx).
# We store the *dot-level* ground-truth here (what the tracker should measure
# on dots directly).
GROUND_TRUTHS: dict[str, GroundTruth] = {
    "translate_x": GroundTruth(
        "translate_x",
        "Dots slide LEFT (camera moves right)",
        mean_dx=-GT_SHIFT_RATE,
        mean_dy=0.0,
    ),
    "translate_y": GroundTruth(
        "translate_y",
        "Dots slide DOWN (camera moves up)",
        mean_dx=0.0,
        mean_dy=GT_SHIFT_RATE,
    ),
    "translate_z": GroundTruth(
        "translate_z",
        "Dots EXPAND from centre (camera moves forward)",
        mean_dx=0.0,
        mean_dy=0.0,
        is_zoom=True,
    ),
    "rotate_yaw": GroundTruth(
        "rotate_yaw",
        "Dots slide LEFT, edge-weighted (camera yaws right)",
        mean_dx=-GT_SHIFT_RATE,
        mean_dy=0.0,
        is_rotation=True,
    ),
    "rotate_pitch": GroundTruth(
        "rotate_pitch",
        "Dots slide DOWN, edge-weighted (camera pitches up)",
        mean_dx=0.0,
        mean_dy=GT_SHIFT_RATE,
        is_rotation=True,
    ),
    "rotate_roll": GroundTruth(
        "rotate_roll",
        "Dots ROTATE ccw around centre (camera rolls cw)",
        mean_dx=0.0,
        mean_dy=0.0,
        is_rotation=True,
    ),
}


# ═══════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class TrackedPoint:
    """A single point observed across frames."""

    point_id: int
    x: float
    y: float


@dataclass
class FrameResult:
    """Tracking results for a single frame pair (frame i → i+1)."""

    frame_index: int
    points: list[TrackedPoint]
    mean_dx: float
    mean_dy: float
    mean_speed: float  # px/frame
    n_tracked: int


@dataclass
class TrackingResult:
    """Complete result from one tracker on one DoF sequence."""

    algorithm: str
    dof_name: str
    frame_results: list[FrameResult] = field(default_factory=list)

    @property
    def overall_mean_dx(self) -> float:
        vals = [f.mean_dx for f in self.frame_results if f.n_tracked > 0]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def overall_mean_dy(self) -> float:
        vals = [f.mean_dy for f in self.frame_results if f.n_tracked > 0]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def overall_mean_speed(self) -> float:
        vals = [f.mean_speed for f in self.frame_results if f.n_tracked > 0]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def overall_std_dx(self) -> float:
        vals = [f.mean_dx for f in self.frame_results if f.n_tracked > 0]
        return float(np.std(vals)) if vals else 0.0

    @property
    def overall_std_dy(self) -> float:
        vals = [f.mean_dy for f in self.frame_results if f.n_tracked > 0]
        return float(np.std(vals)) if vals else 0.0

    @property
    def total_tracked(self) -> int:
        return sum(f.n_tracked for f in self.frame_results)

    @property
    def inferred_direction(self) -> str:
        dx, dy = self.overall_mean_dx, self.overall_mean_dy
        if abs(dx) < 0.3 and abs(dy) < 0.3:
            return "STATIONARY / ZOOM / ROLL"
        angle = math.degrees(math.atan2(-dy, -dx))  # camera direction
        return f"{angle:+.1f}deg"


# ═══════════════════════════════════════════════════════════════════════
#  Frame I/O
# ═══════════════════════════════════════════════════════════════════════


def load_frames(frames_dir: str | Path) -> list[np.ndarray]:
    """Load all frame_XXXX.png files from a directory, in order."""
    p = Path(frames_dir)
    files = sorted(p.glob("frame_*.png"))
    if not files:
        raise FileNotFoundError(f"No frame_*.png found in {p}")
    return [cv2.imread(str(f)) for f in files]


def to_gray(bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR thermal-colormapped image to grayscale."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def thermal_mask(gray: np.ndarray, threshold: int = BRIGHTNESS_THRESHOLD) -> np.ndarray:
    """Binary mask of 'hot' pixels."""
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return mask


# ═══════════════════════════════════════════════════════════════════════
#  Annotated video writer
# ═══════════════════════════════════════════════════════════════════════


class AnnotatedVideoWriter:
    """Context-managed MP4 writer that adds a standard HUD overlay."""

    def __init__(self, path: Path, fps: int = FPS) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (IMAGE_WIDTH, IMAGE_HEIGHT),
        )

    def write(
        self,
        frame: np.ndarray,
        fr: FrameResult | None = None,
        algo: str = "",
        dof: str = "",
    ) -> None:
        vis = frame.copy()
        # Draw tracked-point IDs
        if fr is not None:
            for pt in fr.points:
                cx, cy = int(pt.x), int(pt.y)
                cv2.circle(vis, (cx, cy), 6, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(
                    vis,
                    str(pt.point_id),
                    (cx + 8, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            # HUD line
            hud = (
                f"[{algo}] {dof}  F{fr.frame_index:03d}  "
                f"pts={fr.n_tracked}  dx={fr.mean_dx:+.2f}  "
                f"dy={fr.mean_dy:+.2f}  spd={fr.mean_speed:.2f}"
            )
            cv2.putText(
                vis,
                hud,
                (6, IMAGE_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
        self._writer.write(vis)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.release()


# ═══════════════════════════════════════════════════════════════════════
#  Summary printer
# ═══════════════════════════════════════════════════════════════════════


def print_summary(result: TrackingResult) -> None:
    gt = GROUND_TRUTHS.get(result.dof_name)
    print(f"\n{'-' * 60}")
    print(f"  Algorithm    : {result.algorithm}")
    print(f"  DoF          : {result.dof_name}")
    print(f"  Frames       : {len(result.frame_results)}")
    print(f"  Total tracked: {result.total_tracked}")
    print(
        f"  Mean dx      : {result.overall_mean_dx:+.3f} px/frame "
        f"(std={result.overall_std_dx:.3f})"
    )
    print(
        f"  Mean dy      : {result.overall_mean_dy:+.3f} px/frame "
        f"(std={result.overall_std_dy:.3f})"
    )
    print(f"  Mean speed   : {result.overall_mean_speed:.3f} px/frame")
    print(f"  Direction    : {result.inferred_direction}")
    if gt:
        print(f"  GT dx        : {gt.mean_dx:+.3f} px/frame")
        print(f"  GT dy        : {gt.mean_dy:+.3f} px/frame")
        err_dx = abs(result.overall_mean_dx - gt.mean_dx)
        err_dy = abs(result.overall_mean_dy - gt.mean_dy)
        print(f"  Error dx     : {err_dx:.3f} px/frame")
        print(f"  Error dy     : {err_dy:.3f} px/frame")
    print(f"{'-' * 60}")
