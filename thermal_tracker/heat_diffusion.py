"""
Heat diffusion simulation on a 2D surface.

Models thermal dots that **fade** (intensity decays) and **erode** (radius
shrinks) over time toward configurable floor values.  This simplified model
assumes the diffusion rate decreases with time — as the thermal gradient
flattens the rate of change slows — which we approximate with exponential
decay toward constant asymptotes for both intensity and radius.

Each deposited dot is tracked individually.  The full temperature field is
reconstructed on demand by rendering every active dot as a Gaussian blob
with its current intensity and radius.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class _HeatDot:
    """Internal bookkeeping for a single deposited thermal dot."""

    x: int
    y: int
    intensity: float  # current peak ΔT above ambient (K)
    radius: float  # current Gaussian σ (pixels)
    initial_intensity: float
    initial_radius: float


class HeatDiffusionSimulator:
    """Simulate thermal dot fade-and-erode on a planar surface.

    Each deposited dot decays independently:

        intensity(t+dt) = min_intensity
                          + (intensity(t) - min_intensity) × e^{-fade_rate × dt}

        radius(t+dt)    = min_radius
                          + (radius(t) - min_radius)       × e^{-erode_rate × dt}

    A dot is removed when its intensity falls within a small tolerance of the
    minimum and it no longer contributes visibly to the field.

    Parameters
    ----------
    width, height : int
        Grid dimensions in pixels.
    fade_rate : float
        Exponential decay rate for dot intensity (1/s).  Larger → faster fade.
    erode_rate : float
        Exponential decay rate for dot radius (1/s).  Larger → faster shrink.
    min_intensity : float
        Asymptotic floor for intensity (K above ambient).
    min_radius : float
        Asymptotic floor for Gaussian σ (pixels).
    ambient_temperature : float
        Background / room temperature (K).
    """

    def __init__(
        self,
        width: int,
        height: int,
        fade_rate: float = 0.5,
        erode_rate: float = 0.3,
        min_intensity: float = 1.0,
        min_radius: float = 1.0,
        ambient_temperature: float = 293.15,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Grid dimensions must be positive integers.")
        if fade_rate < 0:
            raise ValueError("fade_rate must be non-negative.")
        if erode_rate < 0:
            raise ValueError("erode_rate must be non-negative.")
        if min_intensity < 0:
            raise ValueError("min_intensity must be non-negative.")
        if min_radius < 0:
            raise ValueError("min_radius must be non-negative.")

        self.width = width
        self.height = height
        self.fade_rate = fade_rate
        self.erode_rate = erode_rate
        self.min_intensity = min_intensity
        self.min_radius = min_radius
        self.ambient = ambient_temperature

        self._dots: list[_HeatDot] = []

        # Lazily-built cache of the rendered field (invalidated on mutation).
        self._field_cache: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def field(self) -> np.ndarray:
        """The full (height, width) absolute-temperature field."""
        if self._field_cache is None:
            self._field_cache = self._render_field()
        return self._field_cache

    def deposit_heat(self, x: int, y: int, intensity: float, radius: int = 2) -> None:
        """Add a thermal dot centred at pixel (x, y).

        Parameters
        ----------
        x, y : int
            Centre of the deposit in pixel coordinates.
        intensity : float
            Peak temperature increase (K) above ambient.
        radius : int
            Initial Gaussian σ (pixels).
        """
        if intensity < 0:
            raise ValueError("Intensity must be non-negative.")
        self._dots.append(
            _HeatDot(
                x=x,
                y=y,
                intensity=float(intensity),
                radius=float(max(radius, 0.1)),
                initial_intensity=float(intensity),
                initial_radius=float(max(radius, 0.1)),
            )
        )
        self._field_cache = None

    def step(self, dt: float) -> None:
        """Advance every dot's fade & erode by *dt* seconds.

        Parameters
        ----------
        dt : float
            Time step in seconds.  Must be positive.
        """
        if dt < 0:
            raise ValueError("Time step must be non-negative.")

        fade_factor = np.exp(-self.fade_rate * dt)
        erode_factor = np.exp(-self.erode_rate * dt)

        surviving: list[_HeatDot] = []
        for dot in self._dots:
            dot.intensity = (
                self.min_intensity + (dot.intensity - self.min_intensity) * fade_factor
            )
            dot.radius = self.min_radius + (dot.radius - self.min_radius) * erode_factor

            # Prune dots that have fully decayed to essentially zero
            # intensity (below 1% of a Kelvin).
            if dot.intensity > 0.01:
                surviving.append(dot)

        self._dots = surviving
        self._field_cache = None

    def step_n(self, dt: float, n: int) -> None:
        """Advance the simulation by *n* steps of size *dt*."""
        for _ in range(n):
            self.step(dt)

    def get_relative_field(self) -> np.ndarray:
        """Return the temperature field relative to ambient (ΔT)."""
        return self.field - self.ambient

    def reset(self) -> None:
        """Remove all dots and reset the field to ambient."""
        self._dots.clear()
        self._field_cache = None

    @property
    def dot_count(self) -> int:
        """Number of currently active dots."""
        return len(self._dots)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _render_field(self) -> np.ndarray:
        """Reconstruct the 2-D temperature field from all active dots."""
        field = np.full((self.height, self.width), self.ambient, dtype=np.float64)

        for dot in self._dots:
            extent = int(np.ceil(3 * dot.radius))
            y_lo = max(0, dot.y - extent)
            y_hi = min(self.height, dot.y + extent + 1)
            x_lo = max(0, dot.x - extent)
            x_hi = min(self.width, dot.x + extent + 1)

            if y_lo >= y_hi or x_lo >= x_hi:
                continue

            yy, xx = np.ogrid[y_lo:y_hi, x_lo:x_hi]
            dist_sq = (xx - dot.x) ** 2 + (yy - dot.y) ** 2
            sigma_sq = dot.radius**2
            gaussian = dot.intensity * np.exp(-dist_sq / (2 * sigma_sq))
            field[y_lo:y_hi, x_lo:x_hi] += gaussian

        return field
