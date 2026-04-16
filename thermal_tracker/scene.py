"""
3D scene representation and synthetic thermal image rendering.

Provides a lightweight scene model containing planar surfaces and a virtual
thermal camera, enabling generation of realistic synthetic thermal frames for
development and testing without requiring physical hardware.

The renderer projects 3D thermal dot positions through a pinhole camera model,
composites the heat-diffusion field, and produces 16-bit (or 8-bit normalised)
thermal images that mirror real LWIR camera output.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .heat_diffusion import HeatDiffusionSimulator
from .pattern_projection import ThermalDot

# ======================================================================
# Camera model
# ======================================================================


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters.

    Attributes
    ----------
    fx, fy : float
        Focal lengths in pixels.
    cx, cy : float
        Principal point in pixels.
    width, height : int
        Image dimensions.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def matrix(self) -> np.ndarray:
        """Return the 3×3 camera intrinsic matrix *K*."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @classmethod
    def default_thermal(cls, width: int = 640, height: int = 480) -> CameraIntrinsics:
        """Reasonable defaults for a typical LWIR thermal camera."""
        fx = fy = 0.8 * width  # ~50° horizontal FOV
        return cls(
            fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0, width=width, height=height
        )


@dataclass
class CameraPose:
    """6-DoF camera pose (extrinsics).

    Stores a 3×3 rotation matrix *R* and a 3-vector translation *t* such that
    a world point *P_w* is transformed to camera coordinates via:

        P_c = R @ P_w + t

    Attributes
    ----------
    R : np.ndarray
        (3, 3) rotation matrix.
    t : np.ndarray
        (3,) translation vector.
    """

    R: np.ndarray
    t: np.ndarray

    def __post_init__(self) -> None:
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float64).ravel()

    @classmethod
    def identity(cls) -> CameraPose:
        return cls(R=np.eye(3), t=np.zeros(3))

    @classmethod
    def from_look_at(
        cls,
        eye: np.ndarray,
        target: np.ndarray,
        up: np.ndarray = np.array([0.0, -1.0, 0.0]),
    ) -> CameraPose:
        """Construct a camera pose looking from *eye* towards *target*."""
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        down = np.cross(forward, right)

        R = np.stack([right, down, forward], axis=0)
        t = -R @ eye
        return cls(R=R, t=t)

    @property
    def position(self) -> np.ndarray:
        """Camera position in world coordinates."""
        return -self.R.T @ self.t

    @property
    def extrinsic_matrix(self) -> np.ndarray:
        """Return the 3×4 extrinsic matrix [R | t]."""
        return np.hstack([self.R, self.t.reshape(3, 1)])


# ======================================================================
# Scene and renderer
# ======================================================================


class ThermalScene:
    """A simple environment for thermal tracking simulation.

    The scene models a single planar surface (wall / floor) at z = *depth*
    facing the camera, onto which thermal dots are projected.  A
    ``HeatDiffusionSimulator`` maintains the temperature field on that surface.

    Parameters
    ----------
    intrinsics : CameraIntrinsics
        Camera calibration.
    surface_depth : float
        Distance (metres) from the camera to the wall plane along the
        optical axis.
    diffusion_sim : HeatDiffusionSimulator | None
        Pre-configured heat diffusion simulator.  Created automatically
        if *None*.
    noise_sigma : float
        Standard deviation of additive Gaussian sensor noise (in Kelvin).
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics | None = None,
        surface_depth: float = 2.0,
        diffusion_sim: HeatDiffusionSimulator | None = None,
        noise_sigma: float = 0.05,
    ) -> None:
        self.K = intrinsics or CameraIntrinsics.default_thermal()
        self.depth = surface_depth
        self.noise_sigma = noise_sigma

        self.diffusion = diffusion_sim or HeatDiffusionSimulator(
            width=self.K.width, height=self.K.height
        )

        self._dot_world_points: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Dot projection
    # ------------------------------------------------------------------

    def project_dots(
        self,
        dots: list[ThermalDot],
        pose: CameraPose | None = None,
        deposit_radius: int = 3,
    ) -> None:
        """Deposit heat at each dot's image-plane position.

        If a ``CameraPose`` is given the dots' image positions are back-
        projected to 3D world points for later use in pose estimation.
        """
        pose = pose or CameraPose.identity()

        for dot in dots:
            ix, iy = int(round(dot.x)), int(round(dot.y))
            if 0 <= ix < self.K.width and 0 <= iy < self.K.height:
                self.diffusion.deposit_heat(ix, iy, dot.intensity, deposit_radius)

                # Back-project to 3D
                world_pt = self._backproject(dot.x, dot.y, pose)
                self._dot_world_points[id(dot)] = world_pt

    def _backproject(self, u: float, v: float, pose: CameraPose) -> np.ndarray:
        """Back-project pixel (u, v) to a 3D world point on the surface plane."""
        z = self.depth
        x = (u - self.K.cx) * z / self.K.fx
        y = (v - self.K.cy) * z / self.K.fy
        p_cam = np.array([x, y, z])
        p_world = pose.R.T @ (p_cam - pose.t)
        return p_world

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def advance(self, dt: float, n_steps: int = 1) -> None:
        """Advance the heat diffusion by *n_steps × dt* seconds."""
        self.diffusion.step_n(dt, n_steps)

    def render_thermal_image(
        self,
        pose: CameraPose | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """Render a synthetic thermal camera frame.

        Parameters
        ----------
        pose : CameraPose | None
            Current camera pose (unused for the single-plane case but
            kept for API forward-compatibility with multi-surface scenes).
        normalize : bool
            If True return an 8-bit image (0–255) suitable for display.
            Otherwise return the raw float64 ΔT field.
        """
        delta_t = self.diffusion.get_relative_field()

        # Add realistic sensor noise
        if self.noise_sigma > 0:
            noise = np.random.default_rng().normal(0, self.noise_sigma, delta_t.shape)
            delta_t = delta_t + noise
            np.clip(delta_t, 0, None, out=delta_t)

        if not normalize:
            return delta_t

        # Normalise to 8-bit
        max_val = delta_t.max()
        if max_val > 0:
            img = (delta_t / max_val * 255).astype(np.uint8)
        else:
            img = np.zeros_like(delta_t, dtype=np.uint8)
        return img

    def render_colormapped(
        self,
        pose: CameraPose | None = None,
        colormap: int = cv2.COLORMAP_INFERNO,
    ) -> np.ndarray:
        """Render a colour-mapped thermal frame (BGR, 8-bit, 3 channels)."""
        gray = self.render_thermal_image(pose, normalize=True)
        return cv2.applyColorMap(gray, colormap)
