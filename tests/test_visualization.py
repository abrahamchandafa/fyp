"""Tests for the visualisation module."""

import numpy as np
import pytest

from thermal_tracker.dot_tracker import TrackedDot
from thermal_tracker.pose_estimation import PoseEstimate
from thermal_tracker.scene import CameraPose
from thermal_tracker.visualization import draw_pose_info, draw_tracked_dots


class TestDrawTrackedDots:
    def test_output_shape_from_grayscale(self) -> None:
        frame = np.zeros((100, 200), dtype=np.uint8)
        tracks = [
            TrackedDot(track_id=0, x=50, y=50, history=[(0, 50, 50)]),
        ]
        vis = draw_tracked_dots(frame, tracks)
        assert vis.shape == (100, 200, 3)  # Converted to BGR

    def test_output_shape_from_bgr(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        tracks = [
            TrackedDot(track_id=0, x=50, y=50, history=[(0, 50, 50)]),
        ]
        vis = draw_tracked_dots(frame, tracks)
        assert vis.shape == (100, 200, 3)

    def test_no_tracks_returns_copy(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        vis = draw_tracked_dots(frame, [])
        np.testing.assert_array_equal(vis, frame)
        assert vis is not frame  # It's a copy

    def test_trail_drawn(self) -> None:
        frame = np.zeros((200, 200), dtype=np.uint8)
        history = [(i, 50 + i * 5, 100) for i in range(10)]
        tracks = [
            TrackedDot(track_id=0, x=95, y=100, history=history),
        ]
        vis = draw_tracked_dots(frame, tracks, draw_trails=True)
        # The trail should have drawn some non-zero pixels
        assert vis.sum() > 0


class TestDrawPoseInfo:
    def test_overlay_on_grayscale(self) -> None:
        frame = np.zeros((200, 300), dtype=np.uint8)
        est = PoseEstimate(
            frame_index=5,
            pose=CameraPose.identity(),
            inlier_count=10,
            reprojection_error=1.23,
            success=True,
        )
        vis = draw_pose_info(frame, est)
        assert vis.shape == (200, 300, 3)
        assert vis.sum() > 0  # Text was drawn

    def test_overlay_on_bgr(self) -> None:
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        est = PoseEstimate(
            frame_index=0,
            pose=CameraPose.identity(),
            inlier_count=5,
            reprojection_error=0.5,
            success=True,
        )
        vis = draw_pose_info(frame, est)
        assert vis.shape == (200, 300, 3)
