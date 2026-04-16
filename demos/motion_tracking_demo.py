"""
Camera motion + dot tracking demo.

Simulates an AR headset moving through a room with thermal dots projected
on all visible surfaces, showing the tracker following dots as they shift
due to camera motion and fade due to heat diffusion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermal_tracker.dot_tracker import DotTracker
from thermal_tracker.heat_diffusion import HeatDiffusionSimulator
from thermal_tracker.pattern_projection import PatternProjector, ThermalDot
from thermal_tracker.scene import CameraIntrinsics, CameraPose, ThermalScene
from thermal_tracker.visualization import VideoWriter, draw_tracked_dots

OUTPUT_DIR = Path("output") / "motion"
WIDTH, HEIGHT = 640, 480
N_FRAMES = 200
FPS = 15.0


def linear_sweep_trajectory(n_frames: int) -> list[CameraPose]:
    """Camera sweeps laterally (left to right) while looking at the wall."""
    poses: list[CameraPose] = []
    for i in range(n_frames):
        frac = i / max(n_frames - 1, 1)
        x = -0.5 + frac * 1.0  # Sweep from -0.5 to +0.5 m
        eye = np.array([x, 0.0, 2.0])
        target = np.array([x * 0.3, 0.0, 0.0])  # Slight look-ahead
        poses.append(CameraPose.from_look_at(eye, target))
    return poses


def run() -> None:
    print("Camera Motion + Tracking Demo")
    print("=" * 40)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    intrinsics = CameraIntrinsics.default_thermal(WIDTH, HEIGHT)
    diffusion = HeatDiffusionSimulator(
        WIDTH,
        HEIGHT,
        fade_rate=0.35,
        erode_rate=0.2,
        min_intensity=1.0,
        min_radius=1.0,
    )
    scene = ThermalScene(
        intrinsics=intrinsics,
        surface_depth=2.0,
        diffusion_sim=diffusion,
        noise_sigma=0.06,
    )
    projector = PatternProjector(
        WIDTH,
        HEIGHT,
        margin=50,
        min_separation=40.0,
        max_dots=16,
        default_intensity=20.0,
        rng_seed=123,
    )
    tracker = DotTracker(
        detection_threshold=0.2,
        min_dot_area=3,
        max_dot_area=800,
        gate_distance=60.0,
        max_lost_frames=10,
    )

    trajectory = linear_sweep_trajectory(N_FRAMES)

    dt = 0.15  # seconds per step

    # Initial dots
    dots = projector.generate_grid(rows=4, cols=4, frame_index=0)
    scene.project_dots(dots, pose=trajectory[0])

    video_path = OUTPUT_DIR / "motion_tracking.mp4"
    with VideoWriter(video_path, fps=FPS, frame_size=(WIDTH, HEIGHT)) as vw:
        for fi in range(N_FRAMES):
            pose = trajectory[fi]

            # Diffuse
            scene.advance(dt, n_steps=2)

            # Re-project periodically
            if fi > 0 and fi % 30 == 0:
                rel = diffusion.get_relative_field()
                dots = projector.generate_adaptive(
                    rel,
                    threshold=1.0,
                    n_new=4,
                    frame_index=fi,
                )
                scene.project_dots(dots, pose=pose, deposit_radius=3)

            # Render & track
            gray = scene.render_thermal_image(pose, normalize=True)
            color = scene.render_colormapped(pose)
            tracks = tracker.update(gray)

            vis = draw_tracked_dots(color, tracks, draw_trails=True, trail_length=20)
            cv2.putText(
                vis,
                f"Frame {fi}  |  Tracks: {len(tracks)}",
                (10, HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            vw.write(vis)

            if fi % 40 == 0:
                cv2.imwrite(str(OUTPUT_DIR / f"motion_{fi:04d}.png"), vis)
                print(f"  [{fi:3d}/{N_FRAMES}] tracks={len(tracks)}")

    print(f"\n  Video saved to: {video_path.resolve()}")


if __name__ == "__main__":
    run()
