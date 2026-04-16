"""
Thermal dot detection and frame-to-frame tracking.

Detects bright thermal spots in each frame via adaptive thresholding and
connected-component analysis, then tracks them across consecutive frames
using a nearest-neighbour assignment with a gating distance.

This module intentionally avoids deep-learning dependencies so that the
core pipeline can run on constrained hardware.  The Projection-Diffusion
Reversal (PDR) neural network stage described in Sheinin et al. can be
inserted upstream of the tracker when available.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class TrackedDot:
    """A thermal dot tracked across multiple frames.

    Attributes
    ----------
    track_id : int
        Unique identifier for this track.
    x, y : float
        Current 2D position on the image plane.
    history : list[tuple[int, float, float]]
        List of (frame_index, x, y) positions for this track.
    age : int
        Number of frames since the track was initialised.
    lost_count : int
        Consecutive frames in which the dot was not detected.
    """

    track_id: int
    x: float
    y: float
    history: list[tuple[int, float, float]] = field(default_factory=list)
    age: int = 0
    lost_count: int = 0


class DotTracker:
    """Detect and track thermal dots across a sequence of frames.

    Parameters
    ----------
    detection_threshold : float
        Minimum relative brightness (0–1 of frame max) to consider a pixel
        as part of a thermal dot.
    min_dot_area : int
        Minimum connected-component area (pixels) to accept as a dot.
    max_dot_area : int
        Maximum connected-component area (pixels) to accept as a dot.
    gate_distance : float
        Maximum pixel displacement between frames for a track match.
    max_lost_frames : int
        Number of consecutive lost frames before a track is terminated.
    """

    def __init__(
        self,
        detection_threshold: float = 0.3,
        min_dot_area: int = 4,
        max_dot_area: int = 500,
        gate_distance: float = 40.0,
        max_lost_frames: int = 5,
    ) -> None:
        self.det_thresh = detection_threshold
        self.min_area = min_dot_area
        self.max_area = max_dot_area
        self.gate = gate_distance
        self.max_lost = max_lost_frames

        self._tracks: list[TrackedDot] = []
        self._next_id: int = 0
        self._frame_idx: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray) -> list[TrackedDot]:
        """Process a new thermal frame and return the active tracks.

        Parameters
        ----------
        frame : np.ndarray
            Single-channel thermal image (uint8 or float64).

        Returns
        -------
        list[TrackedDot]
            Currently active (not yet terminated) tracks.
        """
        detections = self._detect(frame)
        self._associate(detections)
        self._frame_idx += 1
        return self.active_tracks

    @property
    def active_tracks(self) -> list[TrackedDot]:
        return [t for t in self._tracks if t.lost_count <= self.max_lost]

    @property
    def all_tracks(self) -> list[TrackedDot]:
        return list(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0
        self._frame_idx = 0

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect(self, frame: np.ndarray) -> list[tuple[float, float]]:
        """Detect dot centroids in a single thermal frame."""
        if frame.dtype != np.uint8:
            norm = frame - frame.min()
            mx = norm.max()
            if mx > 0:
                frame8 = (norm / mx * 255).astype(np.uint8)
            else:
                frame8 = np.zeros_like(frame, dtype=np.uint8)
        else:
            frame8 = frame

        # Threshold
        max_val = frame8.max()
        if max_val == 0:
            return []
        thresh_val = int(max_val * self.det_thresh)
        _, binary = cv2.threshold(frame8, thresh_val, 255, cv2.THRESH_BINARY)

        # Connected components
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        detections: list[tuple[float, float]] = []
        for i in range(1, n_labels):  # skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if self.min_area <= area <= self.max_area:
                cx, cy = centroids[i]
                detections.append((float(cx), float(cy)))
        return detections

    # ------------------------------------------------------------------
    # Data association (Hungarian-free nearest-neighbour for speed)
    # ------------------------------------------------------------------

    def _associate(self, detections: list[tuple[float, float]]) -> None:
        """Match detections to existing tracks and manage track lifecycle."""
        used_detections: set[int] = set()
        used_tracks: set[int] = set()

        if self._tracks and detections:
            # Build cost matrix
            track_pts = np.array([[t.x, t.y] for t in self._tracks])
            det_pts = np.array(detections)
            # Distances: (n_tracks, n_dets)
            diff = track_pts[:, np.newaxis, :] - det_pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)

            # Greedy nearest-neighbour assignment
            while True:
                min_val = dists.min()
                if min_val > self.gate:
                    break
                ti, di = np.unravel_index(dists.argmin(), dists.shape)
                ti, di = int(ti), int(di)
                if ti in used_tracks or di in used_detections:
                    dists[ti, di] = np.inf
                    continue

                # Update track
                track = self._tracks[ti]
                track.x, track.y = detections[di]
                track.history.append((self._frame_idx, track.x, track.y))
                track.age += 1
                track.lost_count = 0

                used_tracks.add(ti)
                used_detections.add(di)
                dists[ti, :] = np.inf
                dists[:, di] = np.inf

        # Mark unmatched tracks as lost
        for ti, track in enumerate(self._tracks):
            if ti not in used_tracks:
                track.lost_count += 1
                track.age += 1

        # Create new tracks for unmatched detections
        for di, (dx, dy) in enumerate(detections):
            if di not in used_detections:
                new_track = TrackedDot(
                    track_id=self._next_id,
                    x=dx,
                    y=dy,
                    history=[(self._frame_idx, dx, dy)],
                    age=0,
                    lost_count=0,
                )
                self._tracks.append(new_track)
                self._next_id += 1

        # Prune dead tracks
        self._tracks = [t for t in self._tracks if t.lost_count <= self.max_lost]
