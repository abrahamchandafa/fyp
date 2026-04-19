"""Tests for pose module."""

import importlib.util
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Load pose/common.py by file path to avoid collision with track/common.py
pose_common_path = Path(__file__).parent.parent / "pose" / "common.py"
spec = importlib.util.spec_from_file_location(
    "pose_module_common", str(pose_common_path)
)
pose_common = importlib.util.module_from_spec(spec)
sys.modules["pose_module_common"] = pose_common
spec.loader.exec_module(pose_common)

PoseGT = pose_common.PoseGT
FramePoseResult = pose_common.FramePoseResult
PoseResult = pose_common.PoseResult
GROUND_TRUTHS = pose_common.GROUND_TRUTHS
K = pose_common.K
rotation_error_deg = pose_common.rotation_error_deg
translation_dir_error_deg = pose_common.translation_dir_error_deg
load_tracks_from_csv = pose_common.load_tracks_from_csv


class TestPoseGT:
    """Test PoseGT dataclass."""

    def test_pose_gt_creation(self):
        """Test creating a PoseGT instance."""
        gt = PoseGT(rvec=(0.1, 0.2, 0.3), tvec=(0.01, 0.02, 0.03))
        assert gt.rvec == (0.1, 0.2, 0.3)
        assert gt.tvec == (0.01, 0.02, 0.03)

    def test_pose_gt_zero_motion(self):
        """Test PoseGT with zero motion."""
        gt = PoseGT(rvec=(0, 0, 0), tvec=(0, 0, 0))
        assert all(r == 0 for r in gt.rvec)
        assert all(t == 0 for t in gt.tvec)


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

    def test_translation_dofs_have_zero_rotation(self):
        """Translation DoFs have zero rotation."""
        for dof in ["translate_x", "translate_y", "translate_z"]:
            gt = GROUND_TRUTHS[dof]
            assert all(r == 0 for r in gt.rvec)

    def test_rotation_dofs_have_zero_translation(self):
        """Rotation DoFs have zero translation."""
        for dof in ["rotate_yaw", "rotate_pitch", "rotate_roll"]:
            gt = GROUND_TRUTHS[dof]
            assert all(t == 0 for t in gt.tvec)


class TestCameraIntrinsics:
    """Test camera intrinsic matrix K."""

    def test_k_is_3x3(self):
        """K matrix is 3x3."""
        assert K.shape == (3, 3)

    def test_k_is_upper_triangular(self):
        """K matrix is upper triangular."""
        assert K[1, 0] == 0
        assert K[2, 0] == 0
        assert K[2, 1] == 0

    def test_k_bottom_right_is_one(self):
        """K matrix bottom-right element is 1."""
        assert K[2, 2] == 1

    def test_k_focal_length_positive(self):
        """Focal lengths in K are positive."""
        assert K[0, 0] > 0
        assert K[1, 1] > 0


class TestFramePoseResult:
    """Test FramePoseResult dataclass."""

    def test_frame_pose_result_with_valid_pose(self):
        """Test FramePoseResult with valid pose."""
        R = np.eye(3)
        t = np.zeros((3, 1))
        fpr = FramePoseResult(frame_index=0, R=R, t=t, n_points=10)
        assert fpr.frame_index == 0
        assert np.array_equal(fpr.R, R)
        assert np.array_equal(fpr.t, t)
        assert fpr.n_points == 10

    def test_frame_pose_result_with_none_pose(self):
        """Test FramePoseResult with None pose."""
        fpr = FramePoseResult(frame_index=0, R=None, t=None, n_points=0)
        assert fpr.R is None
        assert fpr.t is None
        assert fpr.n_points == 0


class TestPoseResult:
    """Test PoseResult dataclass."""

    def test_pose_result_creation(self):
        """Test creating a PoseResult."""
        pr = PoseResult(algorithm="SLAM", dof_name="translate_x")
        assert pr.algorithm == "SLAM"
        assert pr.dof_name == "translate_x"
        assert len(pr.frame_results) == 0

    def test_pose_result_add_frames(self):
        """Test adding frames to PoseResult."""
        pr = PoseResult(algorithm="SLAM", dof_name="translate_x")
        fpr = FramePoseResult(
            frame_index=0, R=np.eye(3), t=np.zeros((3, 1)), n_points=5
        )
        pr.frame_results.append(fpr)
        assert len(pr.frame_results) == 1

    def test_pose_result_valid_frames_property(self):
        """Test valid_frames property filters None poses."""
        pr = PoseResult(algorithm="SLAM", dof_name="translate_x")
        pr.frame_results.append(
            FramePoseResult(frame_index=0, R=np.eye(3), t=np.zeros((3, 1)), n_points=5)
        )
        pr.frame_results.append(
            FramePoseResult(frame_index=1, R=None, t=None, n_points=0)
        )
        pr.frame_results.append(
            FramePoseResult(frame_index=2, R=np.eye(3), t=np.zeros((3, 1)), n_points=3)
        )
        valid = pr.valid_frames
        assert len(valid) == 2
        assert all(f.R is not None for f in valid)

    def test_pose_result_mean_rvec(self):
        """Test mean_rvec property."""
        pr = PoseResult(algorithm="SLAM", dof_name="translate_x")
        # Add two identity rotations
        pr.frame_results.append(
            FramePoseResult(frame_index=0, R=np.eye(3), t=np.zeros((3, 1)), n_points=5)
        )
        pr.frame_results.append(
            FramePoseResult(frame_index=1, R=np.eye(3), t=np.zeros((3, 1)), n_points=5)
        )
        mean_rvec = pr.mean_rvec
        assert mean_rvec.shape == (3,)
        assert np.allclose(mean_rvec, [0, 0, 0], atol=1e-6)

    def test_pose_result_mean_camera_motion(self):
        """Test mean_camera_motion property."""
        pr = PoseResult(algorithm="SLAM", dof_name="translate_x")
        pr.frame_results.append(
            FramePoseResult(frame_index=0, R=np.eye(3), t=np.zeros((3, 1)), n_points=5)
        )
        pr.frame_results.append(
            FramePoseResult(
                frame_index=1,
                R=np.eye(3),
                t=np.array([[0.1], [0.0], [0.0]]),
                n_points=5,
            )
        )
        mean_motion = pr.mean_camera_motion
        assert mean_motion.shape == (3,)

    def test_pose_result_empty_valid_frames(self):
        """Test properties with no valid frames."""
        pr = PoseResult(algorithm="SLAM", dof_name="translate_x")
        assert len(pr.valid_frames) == 0
        assert pr.mean_rvec.shape == (3,)
        assert np.allclose(pr.mean_rvec, [0, 0, 0], atol=1e-6)


class TestRotationErrorDeg:
    """Test rotation_error_deg function."""

    def test_rotation_error_identical_matrices(self):
        """Identical rotation matrices have zero error."""
        R1 = np.eye(3)
        R2 = np.eye(3)
        error = rotation_error_deg(R1, R2)
        assert abs(error) < 1e-6

    def test_rotation_error_180_degrees(self):
        """180 degree rotation gives ~180 degree error."""
        R1 = np.eye(3)
        # 180 degree rotation around z-axis
        R2 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
        error = rotation_error_deg(R1, R2)
        assert 179 < error < 181

    def test_rotation_error_small_rotation(self):
        """Small rotations give small errors."""
        R1 = np.eye(3)
        # Small rotation: ~5 degree rotation around z
        angle = np.radians(5)
        R2 = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        error = rotation_error_deg(R1, R2)
        assert abs(error - 5.0) < 0.1

    def test_rotation_error_symmetry(self):
        """rotation_error_deg(R1, R2) == rotation_error_deg(R2, R1)."""
        R1 = np.eye(3)
        R2 = np.array(
            [
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        error1 = rotation_error_deg(R1, R2)
        error2 = rotation_error_deg(R2, R1)
        assert abs(error1 - error2) < 1e-6


class TestTranslationDirErrorDeg:
    """Test translation_dir_error_deg function."""

    def test_translation_dir_error_same_direction(self):
        """Translations in same direction have zero error."""
        t1 = np.array([1.0, 0.0, 0.0])
        t2 = np.array([2.0, 0.0, 0.0])
        error = translation_dir_error_deg(t1, t2)
        assert abs(error) < 1e-6

    def test_translation_dir_error_opposite_direction(self):
        """Opposite translations give ~180 degree error."""
        t1 = np.array([1.0, 0.0, 0.0])
        t2 = np.array([-1.0, 0.0, 0.0])
        error = translation_dir_error_deg(t1, t2)
        assert 179 < error < 181

    def test_translation_dir_error_perpendicular(self):
        """Perpendicular translations give ~90 degree error."""
        t1 = np.array([1.0, 0.0, 0.0])
        t2 = np.array([0.0, 1.0, 0.0])
        error = translation_dir_error_deg(t1, t2)
        assert 89 < error < 91

    def test_translation_dir_error_zero_motion(self):
        """Zero motion returns nan."""
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([1.0, 0.0, 0.0])
        error = translation_dir_error_deg(t1, t2)
        assert math.isnan(error)

    def test_translation_dir_error_scale_invariant(self):
        """Scaling translations doesn't change direction error."""
        t1 = np.array([1.0, 1.0, 0.0])
        t2 = np.array([1.0, 0.0, 0.0])
        error1 = translation_dir_error_deg(t1, t2)

        t1_scaled = np.array([10.0, 10.0, 0.0])
        t2_scaled = np.array([2.0, 0.0, 0.0])
        error2 = translation_dir_error_deg(t1_scaled, t2_scaled)

        assert abs(error1 - error2) < 0.1


class TestLoadTracksFromCsv:
    """Test load_tracks_from_csv function."""

    def test_load_tracks_from_csv_simple(self, tmp_path):
        """Test loading tracks from CSV file."""
        csv_file = tmp_path / "test_tracks.csv"
        csv_file.write_text(
            "frame,point_id,x,y\n"
            "0,0,10.0,20.0\n"
            "0,1,30.0,40.0\n"
            "1,0,11.0,21.0\n"
            "1,1,31.0,41.0\n"
        )
        tracks = load_tracks_from_csv(csv_file)
        assert len(tracks) == 2  # 2 points
        assert 0 in tracks
        assert 1 in tracks
        assert len(tracks[0]) == 2  # point 0 appears in 2 frames
        assert len(tracks[1]) == 2  # point 1 appears in 2 frames

    def test_load_tracks_from_csv_order(self, tmp_path):
        """Test that observations are sorted by frame."""
        csv_file = tmp_path / "test_tracks.csv"
        csv_file.write_text(
            "frame,point_id,x,y\n2,0,50.0,60.0\n0,0,10.0,20.0\n1,0,30.0,40.0\n"
        )
        tracks = load_tracks_from_csv(csv_file)
        observations = tracks[0]
        # Should be sorted by frame
        assert observations[0][0] == 0  # frame 0
        assert observations[1][0] == 1  # frame 1
        assert observations[2][0] == 2  # frame 2

    def test_load_tracks_from_csv_empty(self, tmp_path):
        """Test loading empty tracks CSV."""
        csv_file = tmp_path / "test_tracks.csv"
        csv_file.write_text("frame,point_id,x,y\n")
        tracks = load_tracks_from_csv(csv_file)
        assert len(tracks) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
