"""
Integration test: full pipeline end-to-end.

Verifies that all modules work together correctly by running a short
simulated sequence and checking that dots are deposited, tracked, and
that pose estimation produces reasonable results.
"""

import numpy as np
import pytest

from thermal_tracker.dot_tracker import DotTracker
from thermal_tracker.heat_diffusion import HeatDiffusionSimulator
from thermal_tracker.pattern_projection import PatternProjector
from thermal_tracker.pose_estimation import PoseEstimator
from thermal_tracker.scene import CameraIntrinsics, CameraPose, ThermalScene


class TestFullPipeline:
    """End-to-end integration test."""

    def test_pipeline_produces_tracks(self) -> None:
        """Deposit dots, render, detect, track across 10 frames."""
        W, H = 320, 240
        intrinsics = CameraIntrinsics.default_thermal(W, H)
        diffusion = HeatDiffusionSimulator(W, H, fade_rate=0.3, erode_rate=0.2)
        scene = ThermalScene(
            intrinsics=intrinsics,
            surface_depth=2.0,
            diffusion_sim=diffusion,
            noise_sigma=0.0,
        )
        projector = PatternProjector(W, H, margin=20, rng_seed=100)
        tracker = DotTracker(detection_threshold=0.2, min_dot_area=2)

        dots = projector.generate_grid(rows=3, cols=3, frame_index=0)
        scene.project_dots(dots)

        dt = 0.1

        all_track_counts = []
        for fi in range(10):
            scene.advance(dt, n_steps=2)
            gray = scene.render_thermal_image(normalize=True)
            tracks = tracker.update(gray)
            all_track_counts.append(len(tracks))

        # Should detect dots in at least some frames
        assert max(all_track_counts) >= 5

    def test_pipeline_with_pose_estimation(self) -> None:
        """Full loop: project, track, estimate pose."""
        W, H = 320, 240
        K = CameraIntrinsics(fx=250, fy=250, cx=160, cy=120, width=W, height=H)
        diffusion = HeatDiffusionSimulator(W, H, fade_rate=0.3, erode_rate=0.2)
        scene = ThermalScene(
            intrinsics=K,
            surface_depth=2.0,
            diffusion_sim=diffusion,
            noise_sigma=0.0,
        )
        projector = PatternProjector(W, H, margin=20, rng_seed=55)
        tracker = DotTracker(detection_threshold=0.2, min_dot_area=2, gate_distance=30)
        estimator = PoseEstimator(K, use_ransac=True)

        # Camera at identity looking at z=2 wall
        pose = CameraPose.identity()

        dots = projector.generate_grid(rows=4, cols=4, frame_index=0)
        scene.project_dots(dots, pose=pose)

        # Register landmarks
        for i, dot in enumerate(dots):
            wp = scene._backproject(dot.x, dot.y, pose)
            estimator.register_landmark(i, wp)

        dt = 0.1

        for fi in range(5):
            scene.advance(dt, n_steps=2)
            gray = scene.render_thermal_image(normalize=True)
            tracks = tracker.update(gray)

            if len(tracks) >= 4:
                ids = [t.track_id for t in tracks]
                pts = np.array([[t.x, t.y] for t in tracks])
                result = estimator.estimate(ids, pts, frame_index=fi)

        # At least one successful estimate
        successful = [e for e in estimator.trajectory if e.success]
        assert len(successful) >= 1

    def test_adaptive_reprojection_maintains_tracks(self) -> None:
        """Verify that adaptive dot replacement maintains tracker vitality."""
        W, H = 320, 240
        K = CameraIntrinsics.default_thermal(W, H)
        diffusion = HeatDiffusionSimulator(W, H, fade_rate=0.3, erode_rate=0.2)
        scene = ThermalScene(
            intrinsics=K,
            surface_depth=2.0,
            diffusion_sim=diffusion,
            noise_sigma=0.0,
        )
        projector = PatternProjector(W, H, margin=20, rng_seed=7)
        tracker = DotTracker(detection_threshold=0.15, min_dot_area=2)

        dots = projector.generate_random(n=8, frame_index=0)
        scene.project_dots(dots)

        dt = 0.1

        for fi in range(30):
            scene.advance(dt, n_steps=3)

            if fi % 10 == 0 and fi > 0:
                rel = diffusion.get_relative_field()
                new_dots = projector.generate_adaptive(
                    rel,
                    threshold=0.5,
                    n_new=4,
                    frame_index=fi,
                )
                scene.project_dots(new_dots)

            gray = scene.render_thermal_image(normalize=True)
            tracks = tracker.update(gray)

        # After adaptive re-projection, we should still have active tracks
        assert len(tracker.active_tracks) >= 1
