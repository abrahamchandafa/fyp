"""Tests for the scene module (camera model, thermal scene, rendering)."""

import numpy as np
import pytest

from thermal_tracker.pattern_projection import ThermalDot
from thermal_tracker.scene import CameraIntrinsics, CameraPose, ThermalScene

# ======================================================================
# CameraIntrinsics
# ======================================================================


class TestCameraIntrinsics:
    def test_matrix_shape(self) -> None:
        K = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
        assert K.matrix.shape == (3, 3)

    def test_matrix_values(self) -> None:
        K = CameraIntrinsics(fx=500, fy=400, cx=320, cy=240, width=640, height=480)
        mat = K.matrix
        assert mat[0, 0] == 500
        assert mat[1, 1] == 400
        assert mat[0, 2] == 320
        assert mat[1, 2] == 240
        assert mat[2, 2] == 1.0

    def test_default_thermal(self) -> None:
        K = CameraIntrinsics.default_thermal(640, 480)
        assert K.width == 640
        assert K.height == 480
        assert K.fx > 0


# ======================================================================
# CameraPose
# ======================================================================


class TestCameraPose:
    def test_identity(self) -> None:
        pose = CameraPose.identity()
        np.testing.assert_allclose(pose.R, np.eye(3))
        np.testing.assert_allclose(pose.t, np.zeros(3))

    def test_position_at_identity(self) -> None:
        pose = CameraPose.identity()
        np.testing.assert_allclose(pose.position, np.zeros(3))

    def test_position_inverse(self) -> None:
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        pose = CameraPose(R=R, t=t)
        np.testing.assert_allclose(pose.position, -t)

    def test_extrinsic_matrix_shape(self) -> None:
        pose = CameraPose.identity()
        assert pose.extrinsic_matrix.shape == (3, 4)

    def test_from_look_at_faces_target(self) -> None:
        eye = np.array([0.0, 0.0, 5.0])
        target = np.array([0.0, 0.0, 0.0])
        pose = CameraPose.from_look_at(eye, target)
        # The camera Z-axis should point from eye toward target
        cam_z = pose.R[2, :]  # Third row of R is the forward direction
        expected = (target - eye) / np.linalg.norm(target - eye)
        np.testing.assert_allclose(cam_z, expected, atol=1e-10)

    def test_position_from_look_at(self) -> None:
        eye = np.array([1.0, 2.0, 3.0])
        target = np.array([0.0, 0.0, 0.0])
        pose = CameraPose.from_look_at(eye, target)
        np.testing.assert_allclose(pose.position, eye, atol=1e-10)


# ======================================================================
# ThermalScene
# ======================================================================


class TestThermalScene:
    def test_scene_creation(self) -> None:
        scene = ThermalScene()
        assert scene.K.width == 640
        assert scene.K.height == 480

    def test_project_dots_deposits_heat(self) -> None:
        scene = ThermalScene()
        dots = [ThermalDot(x=320, y=240, intensity=10.0)]
        scene.project_dots(dots)
        assert scene.diffusion.field[240, 320] > scene.diffusion.ambient

    def test_render_thermal_image_shape(self) -> None:
        scene = ThermalScene()
        dots = [ThermalDot(x=100, y=100, intensity=15.0)]
        scene.project_dots(dots)
        img = scene.render_thermal_image(normalize=True)
        assert img.shape == (480, 640)
        assert img.dtype == np.uint8

    def test_render_thermal_image_raw(self) -> None:
        scene = ThermalScene(noise_sigma=0.0)
        dots = [ThermalDot(x=100, y=100, intensity=15.0)]
        scene.project_dots(dots)
        raw = scene.render_thermal_image(normalize=False)
        assert raw.dtype == np.float64
        assert raw.max() > 0

    def test_render_colormapped_shape(self) -> None:
        scene = ThermalScene()
        dots = [ThermalDot(x=200, y=200, intensity=10.0)]
        scene.project_dots(dots)
        bgr = scene.render_colormapped()
        assert bgr.shape == (480, 640, 3)
        assert bgr.dtype == np.uint8

    def test_advance_causes_diffusion(self) -> None:
        scene = ThermalScene(noise_sigma=0.0)
        dots = [ThermalDot(x=320, y=240, intensity=20.0, frame_created=0)]
        scene.project_dots(dots)
        peak_before = scene.diffusion.field[240, 320]

        scene.advance(0.2, n_steps=30)
        peak_after = scene.diffusion.field[240, 320]
        assert peak_after < peak_before

    def test_empty_scene_renders_black(self) -> None:
        scene = ThermalScene(noise_sigma=0.0)
        img = scene.render_thermal_image(normalize=True)
        assert img.max() == 0
