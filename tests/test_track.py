"""Tests for track module."""

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Load track/common.py by file path to avoid collision with pose/common.py
track_common_path = Path(__file__).parent.parent / "track" / "common.py"
spec = importlib.util.spec_from_file_location(
    "track_module_common", str(track_common_path)
)
track_common = importlib.util.module_from_spec(spec)
sys.modules["track_module_common"] = track_common
spec.loader.exec_module(track_common)

GroundTruth = track_common.GroundTruth
GROUND_TRUTHS = track_common.GROUND_TRUTHS
IMAGE_WIDTH = track_common.IMAGE_WIDTH
IMAGE_HEIGHT = track_common.IMAGE_HEIGHT
FPS = track_common.FPS
DURATION_S = track_common.DURATION_S
N_FRAMES = track_common.N_FRAMES
thermal_mask = track_common.thermal_mask
to_gray = track_common.to_gray
TrackedPoint = track_common.TrackedPoint
FrameResult = track_common.FrameResult
TrackingResult = track_common.TrackingResult
print_summary = track_common.print_summary


class TestGroundTruth:
    """Test GroundTruth dataclass."""

    def test_ground_truth_creation(self):
        """Test creating a GroundTruth instance."""
        gt = GroundTruth("test_dof", "Test description", -5.0, 0.0)
        assert gt.dof_name == "test_dof"
        assert gt.description == "Test description"
        assert gt.mean_dx == -5.0
        assert gt.mean_dy == 0.0
        assert gt.is_rotation is False
        assert gt.is_zoom is False

    def test_ground_truth_with_rotation(self):
        """Test GroundTruth with rotation flag."""
        gt = GroundTruth(
            "rotate_yaw",
            "Yaw rotation",
            -5.0,
            0.0,
            is_rotation=True,
        )
        assert gt.is_rotation is True

    def test_ground_truth_with_zoom(self):
        """Test GroundTruth with zoom flag."""
        gt = GroundTruth(
            "translate_z",
            "Zoom motion",
            0.0,
            0.0,
            is_zoom=True,
        )
        assert gt.is_zoom is True


class TestGroundTruths:
    """Test GROUND_TRUTHS dictionary."""

    def test_all_six_dofs_present(self):
        """All 6 DoF ground truths are defined."""
        expected_dofs = [
            "translate_x",
            "translate_y",
            "translate_z",
            "rotate_yaw",
            "rotate_pitch",
            "rotate_roll",
        ]
        for dof in expected_dofs:
            assert dof in GROUND_TRUTHS

    def test_ground_truths_have_correct_types(self):
        """All ground truth values are correct types."""
        for dof, gt in GROUND_TRUTHS.items():
            assert isinstance(gt, GroundTruth)
            assert isinstance(gt.dof_name, str)
            assert isinstance(gt.mean_dx, float)
            assert isinstance(gt.mean_dy, float)
            assert isinstance(gt.is_rotation, bool)
            assert isinstance(gt.is_zoom, bool)


class TestThermalMask:
    """Test thermal_mask function."""

    def test_thermal_mask_bright_pixels(self):
        """thermal_mask correctly identifies bright pixels."""
        frame = np.zeros((100, 100), dtype=np.uint8)
        frame[10:20, 10:20] = 200  # bright region
        threshold = 100
        mask = thermal_mask(frame, threshold)
        assert mask[15, 15] == 255  # bright region should be white
        assert mask[50, 50] == 0  # dark region should be black

    def test_thermal_mask_threshold_boundary(self):
        """thermal_mask respects threshold value."""
        frame = np.zeros((100, 100), dtype=np.uint8)
        frame[10:20, 10:20] = 100
        threshold = 100
        mask = thermal_mask(frame, threshold)
        # cv2.THRESH_BINARY uses > not >=, so value at threshold is excluded
        assert np.sum(mask) == 0

        # Value strictly greater than threshold should be included
        frame[10:20, 10:20] = 101
        mask = thermal_mask(frame, threshold)
        assert np.sum(mask) > 0

    def test_thermal_mask_output_type(self):
        """thermal_mask returns uint8 array."""
        frame = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        mask = thermal_mask(frame, 128)
        assert mask.dtype == np.uint8
        assert mask.shape == frame.shape


class TestToGray:
    """Test to_gray function."""

    def test_to_gray_from_bgr(self):
        """to_gray converts BGR to grayscale."""
        bgr_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr_frame[10:20, 10:20] = [100, 150, 200]
        gray = to_gray(bgr_frame)
        assert gray.ndim == 2
        assert gray.shape == (100, 100)
        assert gray.dtype == np.uint8

    def test_to_gray_preserves_brightness_order(self):
        """to_gray preserves relative brightness order."""
        bgr_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr_frame[10:20, 10:20] = [50, 50, 50]  # dark
        bgr_frame[30:40, 30:40] = [200, 200, 200]  # bright
        gray = to_gray(bgr_frame)
        assert gray[15, 15] < gray[35, 35]


class TestTrackedPoint:
    """Test TrackedPoint dataclass."""

    def test_tracked_point_creation(self):
        """Test creating a TrackedPoint."""
        tp = TrackedPoint(point_id=1, x=10.5, y=20.5)
        assert tp.point_id == 1
        assert tp.x == 10.5
        assert tp.y == 20.5


class TestFrameResult:
    """Test FrameResult dataclass."""

    def test_frame_result_creation(self):
        """Test creating a FrameResult."""
        points = [TrackedPoint(1, 10.0, 20.0), TrackedPoint(2, 30.0, 40.0)]
        fr = FrameResult(
            frame_index=0,
            points=points,
            mean_dx=1.5,
            mean_dy=0.5,
            mean_speed=1.6,
            n_tracked=2,
        )
        assert fr.frame_index == 0
        assert len(fr.points) == 2
        assert fr.mean_dx == 1.5
        assert fr.n_tracked == 2

    def test_frame_result_empty_points(self):
        """Test FrameResult with empty points list."""
        fr = FrameResult(
            frame_index=0,
            points=[],
            mean_dx=0.0,
            mean_dy=0.0,
            mean_speed=0.0,
            n_tracked=0,
        )
        assert len(fr.points) == 0


class TestTrackingResult:
    """Test TrackingResult dataclass."""

    def test_tracking_result_creation(self):
        """Test creating a TrackingResult."""
        tr = TrackingResult(algorithm="BlobCentroid", dof_name="translate_x")
        assert tr.algorithm == "BlobCentroid"
        assert tr.dof_name == "translate_x"
        assert len(tr.frame_results) == 0

    def test_tracking_result_add_frames(self):
        """Test adding frames to TrackingResult."""
        tr = TrackingResult(algorithm="BlobCentroid", dof_name="translate_x")
        fr = FrameResult(
            frame_index=0,
            points=[],
            mean_dx=0.0,
            mean_dy=0.0,
            mean_speed=0.0,
            n_tracked=0,
        )
        tr.frame_results.append(fr)
        assert len(tr.frame_results) == 1
        assert tr.frame_results[0].frame_index == 0


class TestCalibrationConstants:
    """Test calibration constants."""

    def test_image_dimensions(self):
        """Image dimensions are positive."""
        assert IMAGE_WIDTH > 0
        assert IMAGE_HEIGHT > 0

    def test_timing_constants(self):
        """FPS and duration are positive."""
        assert FPS > 0
        assert DURATION_S > 0
        assert N_FRAMES == FPS * DURATION_S


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
