"""Tests for the pose estimation module."""

import numpy as np
import pytest

from thermal_tracker.pose_estimation import PoseEstimate, PoseEstimator
from thermal_tracker.scene import CameraIntrinsics, CameraPose


def _make_test_correspondences(
    K: CameraIntrinsics,
    pose: CameraPose,
    n_points: int = 12,
    noise_sigma: float = 0.0,
    rng_seed: int = 42,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Generate synthetic 2D–3D correspondences from a known pose.

    Returns (track_ids, image_points [N,2], world_points [N,3]).
    """
    rng = np.random.default_rng(rng_seed)
    # Random 3D points on a wall at z=0 (world frame)
    world_pts = np.zeros((n_points, 3))
    world_pts[:, 0] = rng.uniform(-1, 1, n_points)  # x
    world_pts[:, 1] = rng.uniform(-0.5, 0.5, n_points)  # y
    world_pts[:, 2] = 0  # on the wall plane

    # Project to image
    cam_pts = (pose.R @ world_pts.T).T + pose.t
    img_pts = np.zeros((n_points, 2))
    img_pts[:, 0] = K.fx * cam_pts[:, 0] / cam_pts[:, 2] + K.cx
    img_pts[:, 1] = K.fy * cam_pts[:, 1] / cam_pts[:, 2] + K.cy

    if noise_sigma > 0:
        img_pts += rng.normal(0, noise_sigma, img_pts.shape)

    ids = list(range(n_points))
    return ids, img_pts, world_pts


class TestPoseEstimatorInit:
    def test_construction(self) -> None:
        K = CameraIntrinsics.default_thermal()
        est = PoseEstimator(K)
        assert est.num_landmarks == 0

    def test_register_landmark(self) -> None:
        K = CameraIntrinsics.default_thermal()
        est = PoseEstimator(K)
        est.register_landmark(0, np.array([1.0, 2.0, 3.0]))
        assert est.num_landmarks == 1

    def test_register_batch(self) -> None:
        K = CameraIntrinsics.default_thermal()
        est = PoseEstimator(K)
        pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        est.register_landmarks_batch([10, 20], pts)
        assert est.num_landmarks == 2


class TestPoseEstimation:
    @pytest.fixture()
    def _setup(self):
        self.K = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
        # Camera at world position (0,0,-2) looking toward wall at z=0.
        # P_c = R @ P_w + t.  R=I, t=(0,0,2): world z=0 -> cam z=2 (in front).
        self.gt_pose = CameraPose(R=np.eye(3), t=np.array([0.0, 0.0, 2.0]))
        self.ids, self.img_pts, self.world_pts = _make_test_correspondences(
            self.K,
            self.gt_pose,
            n_points=20,
            noise_sigma=0.0,
        )
        self.estimator = PoseEstimator(self.K, use_ransac=True, refine=True)
        self.estimator.register_landmarks_batch(self.ids, self.world_pts)

    def test_pose_recovery_succeeds(self, _setup) -> None:
        result = self.estimator.estimate(self.ids, self.img_pts)
        assert result.success is True

    def test_pose_translation_accuracy(self, _setup) -> None:
        result = self.estimator.estimate(self.ids, self.img_pts)
        assert result.success
        # Estimated position should be close to ground truth
        gt_position = self.gt_pose.position
        est_position = result.pose.position
        error = np.linalg.norm(est_position - gt_position)
        assert error < 0.1  # within 10 cm

    def test_pose_rotation_accuracy(self, _setup) -> None:
        result = self.estimator.estimate(self.ids, self.img_pts)
        assert result.success
        R_err = result.pose.R @ self.gt_pose.R.T
        angle_err = np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1)))
        assert angle_err < 2.0  # within 2 degrees

    def test_reprojection_error_low(self, _setup) -> None:
        result = self.estimator.estimate(self.ids, self.img_pts)
        assert result.success
        assert result.reprojection_error < 2.0  # pixels

    def test_too_few_points_fails(self, _setup) -> None:
        result = self.estimator.estimate(self.ids[:2], self.img_pts[:2])
        assert result.success is False

    def test_noisy_observations(self) -> None:
        K = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
        pose = CameraPose(R=np.eye(3), t=np.array([0.0, 0.0, 2.0]))
        ids, img_pts, world_pts = _make_test_correspondences(
            K,
            pose,
            n_points=30,
            noise_sigma=1.0,
        )
        estimator = PoseEstimator(K, use_ransac=True)
        estimator.register_landmarks_batch(ids, world_pts)
        result = estimator.estimate(ids, img_pts)
        assert result.success
        # With noise we allow larger error
        assert result.reprojection_error < 5.0


class TestTrajectory:
    def test_trajectory_accumulates(self) -> None:
        K = CameraIntrinsics.default_thermal()
        est = PoseEstimator(K)
        # Deliberate fail (no landmarks)
        est.estimate([], np.empty((0, 2)), frame_index=0)
        est.estimate([], np.empty((0, 2)), frame_index=1)
        assert len(est.trajectory) == 2

    def test_clear_trajectory(self) -> None:
        K = CameraIntrinsics.default_thermal()
        est = PoseEstimator(K)
        est.estimate([], np.empty((0, 2)))
        est.clear_trajectory()
        assert len(est.trajectory) == 0


class TestErrorMetrics:
    def test_translation_errors(self) -> None:
        K = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
        pose = CameraPose(R=np.eye(3), t=np.array([0.0, 0.0, 2.0]))
        ids, img_pts, world_pts = _make_test_correspondences(K, pose, n_points=15)
        est = PoseEstimator(K)
        est.register_landmarks_batch(ids, world_pts)
        est.estimate(ids, img_pts, frame_index=0)

        gt_positions = [pose.position]
        errors = est.translation_errors(gt_positions)
        assert len(errors) == 1
        assert errors[0] < 0.1

    def test_rotation_errors(self) -> None:
        K = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
        pose = CameraPose(R=np.eye(3), t=np.array([0.0, 0.0, 2.0]))
        ids, img_pts, world_pts = _make_test_correspondences(K, pose, n_points=15)
        est = PoseEstimator(K)
        est.register_landmarks_batch(ids, world_pts)
        est.estimate(ids, img_pts, frame_index=0)

        gt_rotations = [pose.R]
        errors = est.rotation_errors(gt_rotations)
        assert len(errors) == 1
        assert errors[0] < 2.0  # degrees
