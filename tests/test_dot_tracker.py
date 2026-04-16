"""Tests for the dot tracker."""

import numpy as np
import pytest

from thermal_tracker.dot_tracker import DotTracker, TrackedDot


def _make_frame_with_dots(
    width: int,
    height: int,
    dots: list[tuple[int, int]],
    dot_radius: int = 5,
    intensity: int = 200,
) -> np.ndarray:
    """Create a synthetic uint8 thermal frame with bright dots."""
    frame = np.zeros((height, width), dtype=np.uint8)
    for x, y in dots:
        yy, xx = np.ogrid[
            max(0, y - dot_radius) : min(height, y + dot_radius + 1),
            max(0, x - dot_radius) : min(width, x + dot_radius + 1),
        ]
        dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        mask = dist <= dot_radius
        frame[
            max(0, y - dot_radius) : min(height, y + dot_radius + 1),
            max(0, x - dot_radius) : min(width, x + dot_radius + 1),
        ] = np.where(
            mask,
            intensity,
            frame[
                max(0, y - dot_radius) : min(height, y + dot_radius + 1),
                max(0, x - dot_radius) : min(width, x + dot_radius + 1),
            ],
        )
    return frame


class TestDetection:
    def test_detect_single_dot(self) -> None:
        tracker = DotTracker(detection_threshold=0.3, min_dot_area=3)
        frame = _make_frame_with_dots(200, 200, [(100, 100)])
        tracks = tracker.update(frame)
        assert len(tracks) == 1
        assert abs(tracks[0].x - 100) < 5
        assert abs(tracks[0].y - 100) < 5

    def test_detect_multiple_dots(self) -> None:
        tracker = DotTracker(detection_threshold=0.3, min_dot_area=3)
        dots = [(50, 50), (150, 50), (50, 150), (150, 150)]
        frame = _make_frame_with_dots(200, 200, dots)
        tracks = tracker.update(frame)
        assert len(tracks) == 4

    def test_no_detection_on_empty_frame(self) -> None:
        tracker = DotTracker()
        frame = np.zeros((200, 200), dtype=np.uint8)
        tracks = tracker.update(frame)
        assert len(tracks) == 0


class TestTracking:
    def test_track_stationary_dot(self) -> None:
        tracker = DotTracker(gate_distance=20.0, min_dot_area=3)
        for _ in range(5):
            frame = _make_frame_with_dots(200, 200, [(100, 100)])
            tracks = tracker.update(frame)
        assert len(tracks) == 1
        assert tracks[0].age >= 4

    def test_track_moving_dot(self) -> None:
        tracker = DotTracker(gate_distance=30.0, min_dot_area=3)
        positions = [(100 + i * 5, 100) for i in range(10)]
        for pos in positions:
            frame = _make_frame_with_dots(300, 200, [pos])
            tracks = tracker.update(frame)
        assert len(tracks) == 1
        # Should be near the last position
        assert abs(tracks[0].x - positions[-1][0]) < 5
        assert len(tracks[0].history) == 10

    def test_track_ids_are_unique(self) -> None:
        tracker = DotTracker(min_dot_area=3)
        # Two dots in separate frames
        frame1 = _make_frame_with_dots(200, 200, [(50, 50)])
        tracker.update(frame1)
        # Move dot far away so it creates a new track
        frame2 = _make_frame_with_dots(200, 200, [(180, 180)])
        tracks = tracker.update(frame2)
        ids = [t.track_id for t in tracker.all_tracks]
        assert len(ids) == len(set(ids))

    def test_lost_dot_is_pruned(self) -> None:
        tracker = DotTracker(
            gate_distance=20.0,
            max_lost_frames=2,
            min_dot_area=3,
        )
        # Appear for 3 frames
        for _ in range(3):
            frame = _make_frame_with_dots(200, 200, [(100, 100)])
            tracker.update(frame)
        assert len(tracker.active_tracks) == 1
        # Disappear for 3 frames (exceeds max_lost=2)
        for _ in range(3):
            frame = np.zeros((200, 200), dtype=np.uint8)
            tracker.update(frame)
        assert len(tracker.active_tracks) == 0

    def test_two_dots_tracked_independently(self) -> None:
        tracker = DotTracker(gate_distance=25.0, min_dot_area=3)
        for i in range(8):
            dots = [(50 + i * 3, 50), (150 - i * 3, 150)]
            frame = _make_frame_with_dots(300, 200, dots)
            tracks = tracker.update(frame)
        assert len(tracks) == 2
        # Each should have its own history
        for t in tracks:
            assert len(t.history) >= 7


class TestReset:
    def test_reset_clears_tracks(self) -> None:
        tracker = DotTracker(min_dot_area=3)
        frame = _make_frame_with_dots(200, 200, [(100, 100)])
        tracker.update(frame)
        assert len(tracker.active_tracks) == 1
        tracker.reset()
        assert len(tracker.active_tracks) == 0
        assert tracker._frame_idx == 0
