"""
Full-pipeline demonstration: simulated AR headset motion with thermal dot tracking.

This demo generates a sequence of synthetic thermal camera frames showing:
1. A sparse pattern of laser-projected thermal dots on a wall surface.
2. Realistic heat diffusion causing the dots to blur and fade over time.
3. Simulated headset motion (the camera follows an elliptical trajectory).
4. Adaptive re-projection of new dots as old ones fade.
5. Frame-to-frame dot tracking with motion trails.
6. 6-DoF pose estimation from the tracked dots.

Output:
    - ``output/thermal_tracking_demo.mp4``  — annotated video
    - ``output/frames/``                    — individual PNG frames
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure the package is importable when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermal_tracker.dot_tracker import DotTracker
from thermal_tracker.heat_diffusion import HeatDiffusionSimulator
from thermal_tracker.pattern_projection import PatternProjector
from thermal_tracker.pose_estimation import PoseEstimator
from thermal_tracker.scene import CameraIntrinsics, CameraPose, ThermalScene
from thermal_tracker.visualization import (
    VideoWriter,
    draw_pose_info,
    draw_tracked_dots,
)

# ======================================================================
# Configuration
# ======================================================================

WIDTH, HEIGHT = 640, 480
FPS = 15.0
N_FRAMES = 150  # ~10 seconds of footage
DT = 0.5  # Diffusion time step (seconds per frame)
DIFFUSION_STEPS = 3  # Diffusion sub-steps per frame

# Camera trajectory — elliptical path in XZ plane
TRAJ_RADIUS_X = 0.6  # metres
TRAJ_RADIUS_Z = 0.3
TRAJ_CENTRE = np.array([0.0, 0.0, 2.0])  # metres
TRAJ_HEIGHT_VAR = 0.08  # vertical wobble (metres)

OUTPUT_DIR = Path("output")
FRAMES_DIR = OUTPUT_DIR / "frames"


# ======================================================================
# Helpers
# ======================================================================


def make_camera_trajectory(n_frames: int) -> list[CameraPose]:
    """Generate a smooth elliptical + vertical-wobble camera trajectory."""
    poses: list[CameraPose] = []
    for i in range(n_frames):
        t = 2 * np.pi * i / n_frames
        x = TRAJ_CENTRE[0] + TRAJ_RADIUS_X * np.sin(t)
        y = TRAJ_CENTRE[1] + TRAJ_HEIGHT_VAR * np.sin(3 * t)
        z = TRAJ_CENTRE[2] + TRAJ_RADIUS_Z * np.cos(t)
        eye = np.array([x, y, z])
        target = np.array([0.0, 0.0, 0.0])  # Looking at the wall origin
        poses.append(CameraPose.from_look_at(eye, target))
    return poses


def shift_dots_for_motion(
    dots_2d: list[tuple[float, float]],
    prev_pose: CameraPose,
    curr_pose: CameraPose,
    intrinsics: CameraIntrinsics,
    depth: float,
) -> list[tuple[float, float]]:
    """Approximate how projected dot positions shift when the camera moves.

    We back-project each 2D point to 3D via the previous pose, then
    re-project through the current pose.
    """
    K = intrinsics.matrix
    shifted: list[tuple[float, float]] = []
    for u, v in dots_2d:
        # Back-project
        z = depth
        x_c = (u - intrinsics.cx) * z / intrinsics.fx
        y_c = (v - intrinsics.cy) * z / intrinsics.fy
        p_cam = np.array([x_c, y_c, z])
        p_world = prev_pose.R.T @ (p_cam - prev_pose.t)
        # Re-project
        p_cam_new = curr_pose.R @ p_world + curr_pose.t
        if p_cam_new[2] <= 0:
            continue
        u2 = intrinsics.fx * p_cam_new[0] / p_cam_new[2] + intrinsics.cx
        v2 = intrinsics.fy * p_cam_new[1] / p_cam_new[2] + intrinsics.cy
        shifted.append((float(u2), float(v2)))
    return shifted


# ======================================================================
# Main demo
# ======================================================================


def run_demo() -> None:
    print("=" * 60)
    print("  Thermal Dot Tracking — Full Pipeline Demo")
    print("=" * 60)

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Setup ---
    intrinsics = CameraIntrinsics.default_thermal(WIDTH, HEIGHT)
    diffusion = HeatDiffusionSimulator(
        width=WIDTH,
        height=HEIGHT,
        fade_rate=0.4,
        erode_rate=0.25,
        min_intensity=1.0,
        min_radius=1.0,
        ambient_temperature=293.15,
    )
    scene = ThermalScene(
        intrinsics=intrinsics,
        surface_depth=2.0,
        diffusion_sim=diffusion,
        noise_sigma=0.08,
    )
    projector = PatternProjector(
        image_width=WIDTH,
        image_height=HEIGHT,
        margin=40,
        min_separation=35.0,
        max_dots=20,
        default_intensity=18.0,
        rng_seed=42,
    )
    tracker = DotTracker(
        detection_threshold=0.25,
        min_dot_area=3,
        max_dot_area=600,
        gate_distance=50.0,
        max_lost_frames=8,
    )
    estimator = PoseEstimator(intrinsics=intrinsics, use_ransac=True)

    trajectory = make_camera_trajectory(N_FRAMES)

    # Time step per sub-step
    dt_safe = DT / DIFFUSION_STEPS

    # --- Initial pattern ---
    initial_dots = projector.generate_random(n=16, frame_index=0)
    scene.project_dots(initial_dots, pose=trajectory[0])

    # Pre-compute world points for initial dots
    initial_world_pts: dict[tuple[float, float], np.ndarray] = {}
    for dot in initial_dots:
        wp = scene._backproject(dot.x, dot.y, trajectory[0])
        initial_world_pts[(dot.x, dot.y)] = wp

    # Render & track the first frame so we get tracker IDs, then register
    # landmarks using those IDs (the tracker assigns its own IDs).
    thermal_gray_0 = scene.render_thermal_image(normalize=True)
    first_tracks = tracker.update(thermal_gray_0)
    for track in first_tracks:
        # Find nearest projected dot
        best_dist = float("inf")
        best_wp = None
        for (dx, dy), wp in initial_world_pts.items():
            d = np.hypot(track.x - dx, track.y - dy)
            if d < best_dist:
                best_dist = d
                best_wp = wp
        if best_wp is not None and best_dist < 20:
            estimator.register_landmark(track.track_id, best_wp)

    print(f"  Frames       : {N_FRAMES}")
    print(f"  Resolution   : {WIDTH}×{HEIGHT}")
    print(f"  Initial dots : {len(initial_dots)}")
    print(f"  Diffusion dt : {dt_safe:.4f} s")
    print(f"  Output       : {OUTPUT_DIR.resolve()}")
    print()

    # --- Frame loop ---
    video_path = OUTPUT_DIR / "thermal_tracking_demo.mp4"
    with VideoWriter(video_path, fps=FPS, frame_size=(WIDTH, HEIGHT)) as vw:
        for fi in range(N_FRAMES):
            pose = trajectory[fi]

            # 1. Advance heat diffusion
            scene.advance(dt_safe, n_steps=DIFFUSION_STEPS)

            # 2. Adaptive re-projection every 20 frames
            if fi > 0 and fi % 20 == 0:
                rel_field = diffusion.get_relative_field()
                new_dots = projector.generate_adaptive(
                    current_field=rel_field,
                    threshold=1.5,
                    n_new=5,
                    frame_index=fi,
                )
                scene.project_dots(new_dots, pose=pose)

            # 3. Render thermal image
            thermal_gray = scene.render_thermal_image(pose, normalize=True)
            thermal_color = scene.render_colormapped(pose)

            # 4. Track dots
            tracks = tracker.update(thermal_gray)

            # 5. Register any new tracks as landmarks (back-project to 3D)
            for track in tracks:
                if track.track_id not in estimator._landmarks and track.age == 0:
                    wp = scene._backproject(track.x, track.y, pose)
                    estimator.register_landmark(track.track_id, wp)

            # 7. Pose estimation (when enough tracks with landmarks)
            est = None
            if len(tracks) >= 4:
                track_ids = [t.track_id for t in tracks]
                img_pts = np.array([[t.x, t.y] for t in tracks], dtype=np.float64)
                est = estimator.estimate(track_ids, img_pts, frame_index=fi)

            # 6. Visualise
            vis = draw_tracked_dots(thermal_color, tracks, draw_trails=True)
            if est is not None and est.success:
                vis = draw_pose_info(vis, est)

            # Frame counter
            cv2.putText(
                vis,
                f"Frame {fi}/{N_FRAMES}  |  Tracks: {len(tracks)}",
                (10, HEIGHT - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

            vw.write(vis)

            # Save selected frames as PNGs
            if fi % 25 == 0 or fi == N_FRAMES - 1:
                cv2.imwrite(str(FRAMES_DIR / f"frame_{fi:04d}.png"), vis)

            if fi % 50 == 0:
                n_active = len(tracks)
                status = "OK" if (est and est.success) else "NO POSE"
                print(f"  [{fi:4d}/{N_FRAMES}]  tracks={n_active:2d}  {status}")

    # --- Summary ---
    traj = estimator.trajectory
    successful = [e for e in traj if e.success]
    print()
    print("-" * 60)
    print(f"  Done. Video saved to: {video_path.resolve()}")
    print(f"  Frames saved to    : {FRAMES_DIR.resolve()}")
    print(f"  Pose estimates     : {len(successful)}/{len(traj)} successful")
    if successful:
        avg_err = np.mean([e.reprojection_error for e in successful])
        print(f"  Mean reproj error  : {avg_err:.2f} px")
    print("-" * 60)


if __name__ == "__main__":
    run_demo()
