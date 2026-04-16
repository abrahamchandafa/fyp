"""
6-DoF camera pose estimation from 2D–3D correspondences.

Implements the Perspective-n-Point (PnP) pose recovery used to estimate the
AR headset's position and orientation from the tracked thermal dot pattern.

Pipeline overview
-----------------
1. Maintain a set of **3D landmark points** (world positions of projected
   thermal dots on the wall surface).
2. For each frame, match tracked 2D dot positions to their known 3D
   landmarks.
3. Solve the PnP problem (via ``cv2.solvePnP``) to obtain [R | t].
4. Optionally refine with iterative Levenberg–Marquardt.

The module also provides trajectory logging and error computation for
benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .scene import CameraIntrinsics, CameraPose


@dataclass
class PoseEstimate:
    """Result of a single-frame pose estimation.

    Attributes
    ----------
    frame_index : int
        Frame number.
    pose : CameraPose
        Estimated 6-DoF pose.
    inlier_count : int
        Number of inlier correspondences used by the PnP solver.
    reprojection_error : float
        Mean reprojection error of inlier points (pixels).
    success : bool
        Whether the solver converged.
    """

    frame_index: int
    pose: CameraPose
    inlier_count: int = 0
    reprojection_error: float = float("inf")
    success: bool = False


class PoseEstimator:
    """Estimate 6-DoF headset pose from tracked thermal dots.

    Parameters
    ----------
    intrinsics : CameraIntrinsics
        Camera calibration.
    dist_coeffs : np.ndarray | None
        Distortion coefficients (default: zero distortion).
    use_ransac : bool
        If True, use ``solvePnPRansac`` for robustness to outliers.
    ransac_reproj_threshold : float
        Maximum reprojection error (pixels) to count as an inlier.
    refine : bool
        Apply iterative refinement after initial PnP solve.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        dist_coeffs: np.ndarray | None = None,
        use_ransac: bool = True,
        ransac_reproj_threshold: float = 3.0,
        refine: bool = True,
    ) -> None:
        self.K = intrinsics.matrix
        self.dist = (
            dist_coeffs if dist_coeffs is not None else np.zeros(4, dtype=np.float64)
        )
        self.use_ransac = use_ransac
        self.ransac_thresh = ransac_reproj_threshold
        self.refine = refine

        # Landmark database: id -> 3D world point
        self._landmarks: dict[int, np.ndarray] = {}
        self._trajectory: list[PoseEstimate] = []

    # ------------------------------------------------------------------
    # Landmark management
    # ------------------------------------------------------------------

    def register_landmark(self, landmark_id: int, world_point: np.ndarray) -> None:
        """Register a 3D world point for a given landmark (dot track) id."""
        self._landmarks[landmark_id] = np.asarray(world_point, dtype=np.float64)

    def register_landmarks_batch(self, ids: list[int], points: np.ndarray) -> None:
        """Register multiple landmarks at once.

        Parameters
        ----------
        ids : list[int]
            Landmark identifiers.
        points : np.ndarray
            (N, 3) array of world coordinates.
        """
        for lid, pt in zip(ids, points):
            self._landmarks[lid] = np.asarray(pt, dtype=np.float64)

    @property
    def num_landmarks(self) -> int:
        return len(self._landmarks)

    # ------------------------------------------------------------------
    # Core estimation
    # ------------------------------------------------------------------

    def estimate(
        self,
        track_ids: list[int],
        image_points: np.ndarray,
        frame_index: int = 0,
    ) -> PoseEstimate:
        """Estimate the camera pose from 2D–3D correspondences.

        Parameters
        ----------
        track_ids : list[int]
            Identifiers of the tracked dots (must match registered landmarks).
        image_points : np.ndarray
            (N, 2) array of 2D dot positions in the current frame.
        frame_index : int
            Current frame number.

        Returns
        -------
        PoseEstimate
            The computed pose (or a failed estimate if too few correspondences).
        """
        # Build correspondence arrays
        obj_pts: list[np.ndarray] = []
        img_pts: list[np.ndarray] = []
        for tid, ip in zip(track_ids, image_points):
            if tid in self._landmarks:
                obj_pts.append(self._landmarks[tid])
                img_pts.append(ip)

        fail = PoseEstimate(
            frame_index=frame_index,
            pose=CameraPose.identity(),
            success=False,
        )

        if len(obj_pts) < 4:
            self._trajectory.append(fail)
            return fail

        obj_arr = np.array(obj_pts, dtype=np.float64)
        img_arr = np.array(img_pts, dtype=np.float64)

        try:
            if self.use_ransac:
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_arr,
                    img_arr,
                    self.K,
                    self.dist,
                    reprojectionError=self.ransac_thresh,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                n_inliers = len(inliers) if inliers is not None else 0
            else:
                ok, rvec, tvec = cv2.solvePnP(
                    obj_arr,
                    img_arr,
                    self.K,
                    self.dist,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                n_inliers = len(obj_pts)
        except cv2.error:
            self._trajectory.append(fail)
            return fail

        if not ok:
            self._trajectory.append(fail)
            return fail

        # Optional refinement
        if self.refine:
            rvec, tvec = cv2.solvePnPRefineLM(
                obj_arr, img_arr, self.K, self.dist, rvec, tvec
            )

        R, _ = cv2.Rodrigues(rvec)
        pose = CameraPose(R=R, t=tvec.ravel())

        # Reprojection error
        projected, _ = cv2.projectPoints(obj_arr, rvec, tvec, self.K, self.dist)
        errors = np.linalg.norm(projected.reshape(-1, 2) - img_arr, axis=1)
        mean_err = float(errors.mean())

        result = PoseEstimate(
            frame_index=frame_index,
            pose=pose,
            inlier_count=n_inliers,
            reprojection_error=mean_err,
            success=True,
        )
        self._trajectory.append(result)
        return result

    # ------------------------------------------------------------------
    # Trajectory access & evaluation
    # ------------------------------------------------------------------

    @property
    def trajectory(self) -> list[PoseEstimate]:
        return list(self._trajectory)

    def translation_errors(self, ground_truth: list[np.ndarray]) -> list[float]:
        """Compute per-frame translation errors against ground truth positions."""
        errors: list[float] = []
        for est, gt in zip(self._trajectory, ground_truth):
            if est.success:
                errors.append(float(np.linalg.norm(est.pose.position - gt)))
            else:
                errors.append(float("inf"))
        return errors

    def rotation_errors(self, ground_truth: list[np.ndarray]) -> list[float]:
        """Compute per-frame rotation errors (degrees) against ground truth.

        Ground truth entries are (3, 3) rotation matrices.
        """
        errors: list[float] = []
        for est, gt_R in zip(self._trajectory, ground_truth):
            if est.success:
                R_err = est.pose.R @ gt_R.T
                angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
                errors.append(float(np.degrees(angle)))
            else:
                errors.append(float("inf"))
        return errors

    def clear_trajectory(self) -> None:
        self._trajectory.clear()
