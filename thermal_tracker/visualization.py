"""
Visualisation utilities for the thermal tracking pipeline.

Provides functions for rendering annotated thermal frames, trajectory plots,
and video output.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .dot_tracker import TrackedDot
from .pose_estimation import PoseEstimate


def draw_tracked_dots(
    frame: np.ndarray,
    tracks: list[TrackedDot],
    draw_trails: bool = True,
    trail_length: int = 15,
) -> np.ndarray:
    """Annotate a thermal frame with tracked dot positions and trails.

    Parameters
    ----------
    frame : np.ndarray
        BGR or grayscale image.
    tracks : list[TrackedDot]
        Currently active tracks.
    draw_trails : bool
        Whether to draw motion history trails.
    trail_length : int
        Maximum number of past positions to draw in the trail.

    Returns
    -------
    np.ndarray
        Annotated BGR image.
    """
    if len(frame.shape) == 2:
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        vis = frame.copy()

    for track in tracks:
        x, y = int(round(track.x)), int(round(track.y))

        # Draw trail
        if draw_trails and len(track.history) > 1:
            pts = track.history[-trail_length:]
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                p1 = (int(round(pts[i - 1][1])), int(round(pts[i - 1][2])))
                p2 = (int(round(pts[i][1])), int(round(pts[i][2])))
                cv2.line(vis, p1, p2, color, 1, cv2.LINE_AA)

        # Draw current position
        cv2.circle(vis, (x, y), 5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1, cv2.LINE_AA)

        # Label
        cv2.putText(
            vis,
            f"#{track.track_id}",
            (x + 7, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    return vis


def draw_pose_info(
    frame: np.ndarray,
    estimate: PoseEstimate,
) -> np.ndarray:
    """Overlay pose estimation information on a frame."""
    if len(frame.shape) == 2:
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        vis = frame.copy()

    pos = estimate.pose.position
    lines = [
        f"Frame {estimate.frame_index}",
        f"Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})",
        f"Inliers: {estimate.inlier_count}",
        f"Reproj err: {estimate.reprojection_error:.2f} px",
    ]

    y0 = 20
    for i, line in enumerate(lines):
        cv2.putText(
            vis,
            line,
            (10, y0 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return vis


class VideoWriter:
    """Context-managed OpenCV video writer.

    Parameters
    ----------
    output_path : str | Path
        Destination file path (e.g. ``output/demo.mp4``).
    fps : float
        Frame rate.
    frame_size : tuple[int, int]
        (width, height) of each frame.
    codec : str
        FourCC codec string.
    """

    def __init__(
        self,
        output_path: str | Path,
        fps: float = 15.0,
        frame_size: tuple[int, int] = (640, 480),
        codec: str = "mp4v",
    ) -> None:
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(str(self.path), fourcc, fps, frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *args: object) -> None:
        self.writer.release()

    def release(self) -> None:
        self.writer.release()
