"""SLAM pose estimation using persistent blob tracks from BlobCentroid."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from common import (
    GROUND_TRUTHS,
    FramePoseResult,
    K,
    PoseResult,
    load_tracks_from_csv,
    rotation_error_deg,
    translation_dir_error_deg,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


GEN_OUTPUT = ROOT.parent / "pdr" / "output"
TRACK_OUTPUT = ROOT.parent / "track" / "output" / "blob"

BOOTSTRAP_MIN_TRACKS = 4
MAX_REPROJ_ERROR = 2.5
MIN_POINTS_FOR_PNP = 4


def build_frame_observations(
    tracks: Dict[int, List[tuple[int, float, float]]],
) -> Dict[int, List[tuple[int, np.ndarray]]]:
    frames: Dict[int, List[tuple[int, np.ndarray]]] = {}
    for point_id, observations in tracks.items():
        for frame_index, x, y in observations:
            frames.setdefault(frame_index, []).append(
                (point_id, np.array([x, y], dtype=np.float64))
            )
    return frames


def bootstrap_pose(
    frame0_obs: List[tuple[int, np.ndarray]],
    frame1_obs: List[tuple[int, np.ndarray]],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    matches = {}
    for pid, pt in frame0_obs:
        matches[pid] = [pt, None]
    for pid, pt in frame1_obs:
        if pid in matches:
            matches[pid][1] = pt

    pts0 = [pair[0] for pair in matches.values() if pair[1] is not None]
    pts1 = [pair[1] for pair in matches.values() if pair[1] is not None]

    if len(pts0) < BOOTSTRAP_MIN_TRACKS:
        return None

    pts0 = np.array(pts0, dtype=np.float64)
    pts1 = np.array(pts1, dtype=np.float64)

    # Try Essential Matrix first
    E, mask = cv2.findEssentialMat(
        pts0,
        pts1,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )

    if E is not None and E.size > 0:
        if E.ndim == 2 and E.shape == (1, 9):
            E = E.reshape(3, 3)
        elif E.ndim == 2 and E.shape == (9, 1):
            E = E.reshape(3, 3)
        elif E.ndim == 3 and E.shape[0] == 1 and E.shape[1:] == (3, 3):
            E = E[0]

        if E.shape == (3, 3):
            try:
                _, R, t, _ = cv2.recoverPose(E, pts0, pts1, K, mask=mask)
                return R, t
            except cv2.error:
                pass

    # Fallback: try Homography-based initialization
    H, h_mask = cv2.findHomography(
        pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=1.0
    )
    if H is not None:
        result = cv2.decomposeHomographyMat(H, K)
        if len(result) >= 3 and result[0] > 0:
            Rs = result[1]
            ts = result[2]
            if len(Rs) > 0:
                R = Rs[0]
                t = ts[0]
                return R, t

    return None


def triangulate_point(
    point_id: int,
    observations: List[tuple[int, float, float]],
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> Optional[np.ndarray]:
    known = [
        (f, np.array([x, y], dtype=np.float64))
        for f, x, y in observations
        if f in poses
    ]
    if len(known) < 2:
        return None

    frame_a, pt_a = known[0]
    frame_b, pt_b = known[1]

    R_a, t_a = poses[frame_a]
    R_b, t_b = poses[frame_b]

    P_a = K @ np.hstack([R_a, t_a])
    P_b = K @ np.hstack([R_b, t_b])

    pts_a = pt_a.reshape(2, 1)
    pts_b = pt_b.reshape(2, 1)
    X_h = cv2.triangulatePoints(P_a, P_b, pts_a, pts_b)

    if abs(X_h[3]) < 1e-10:
        return None

    X = (X_h[:3] / X_h[3]).reshape(3)

    total_error = 0.0
    checks = 0
    for frame_idx, img_pt in known:
        R, t = poses[frame_idx]
        P = K @ np.hstack([R, t])
        proj = P @ np.append(X, 1.0)
        proj = proj[:2] / proj[2]
        total_error += float(np.linalg.norm(proj - img_pt))
        checks += 1

    if checks == 0:
        return None
    if total_error / checks > MAX_REPROJ_ERROR:
        return None
    if X[2] <= 0:
        return None

    return X


def solve_pose_from_map(
    frame_idx: int,
    tracks: Dict[int, List[tuple[int, float, float]]],
    map_points: Dict[int, Optional[np.ndarray]],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    obj_pts = []
    img_pts = []

    for point_id, observations in tracks.items():
        if point_id not in map_points:
            continue
        pos_3d = map_points[point_id]
        if pos_3d is None:
            continue
        for frame, x, y in observations:
            if frame == frame_idx:
                obj_pts.append(pos_3d)
                img_pts.append((x, y))
                break

    if len(obj_pts) < MIN_POINTS_FOR_PNP:
        return None

    obj_pts = np.array(obj_pts, dtype=np.float64)
    img_pts = np.array(img_pts, dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        K,
        None,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        return None

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec


def print_summary(result: PoseResult) -> None:
    gt = GROUND_TRUTHS.get(result.dof_name)
    print(f"\n{'-' * 60}")
    print(f"  Algorithm : {result.algorithm}")
    print(f"  DoF       : {result.dof_name}")
    print(f"  Frames    : {len(result.frame_results)}")
    print(f"  Valid     : {len(result.valid_frames)}")
    if gt:
        gt_R, _ = cv2.Rodrigues(np.array(gt.rvec, dtype=np.float64))
        gt_t = np.array(gt.tvec, dtype=np.float64)
        est_R = cv2.Rodrigues(result.mean_rvec)[0]
        est_t = result.mean_camera_motion
        print(f"  Rot err   : {rotation_error_deg(est_R, gt_R):.3f} deg")
        print(f"  Trans dir : {translation_dir_error_deg(est_t, gt_t):.3f} deg")
    print(f"{'-' * 60}\n")


def run_dof(dof_name: str) -> PoseResult:
    track_csv = TRACK_OUTPUT / dof_name / f"{dof_name}_blob_tracks.csv"
    if not track_csv.exists():
        raise FileNotFoundError(f"Missing track CSV: {track_csv}")

    tracks = load_tracks_from_csv(track_csv)
    frames = build_frame_observations(tracks)
    result = PoseResult(algorithm="SLAM", dof_name=dof_name)

    poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    map_points: Dict[int, Optional[np.ndarray]] = {pid: None for pid in tracks}

    poses[0] = (np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64))

    if 0 not in frames or 1 not in frames:
        raise ValueError("Track CSV must contain observations for frame 0 and frame 1.")

    seed_pose = bootstrap_pose(frames[0], frames[1])
    if seed_pose is None:
        raise RuntimeError("Failed to bootstrap initial pose from frame 0 and frame 1.")
    poses[1] = seed_pose

    for point_id, observations in tracks.items():
        if len(observations) >= 2:
            X = triangulate_point(point_id, observations, poses)
            if X is not None:
                map_points[point_id] = X

    max_frame = max(frames.keys())
    for frame_idx in range(max_frame + 1):
        if frame_idx in poses and frame_idx <= 1:
            pass
        elif frame_idx in frames:
            pose = solve_pose_from_map(frame_idx, tracks, map_points)
            if pose is not None:
                poses[frame_idx] = pose

            for point_id, observations in tracks.items():
                if map_points[point_id] is None:
                    X = triangulate_point(point_id, observations, poses)
                    if X is not None:
                        map_points[point_id] = X

        result.frame_results.append(
            FramePoseResult(
                frame_index=frame_idx,
                R=poses.get(frame_idx, (None, None))[0],
                t=poses.get(frame_idx, (None, None))[1],
                n_points=sum(
                    1
                    for pid, pos in map_points.items()
                    if pos is not None
                    and any(frame == frame_idx for frame, _, _ in tracks[pid])
                ),
            )
        )

    print_summary(result)
    return result


def main() -> None:
    dofs = sys.argv[1:] if len(sys.argv) > 1 else list(GROUND_TRUTHS.keys())
    for dof in dofs:
        run_dof(dof)


if __name__ == "__main__":
    main()
