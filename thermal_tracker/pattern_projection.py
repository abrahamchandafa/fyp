"""
Thermal dot pattern generation and projection control.

Implements strategies for placing sparse trackable dot patterns on surfaces,
following Sheinin et al.'s finding that discrete patterns with only a few dots
per frame are optimal for maintaining trackability under heat diffusion.

Supported strategies
--------------------
- **grid** : Regular grid layout (baseline).
- **random** : Uniformly random placement within a region of interest.
- **adaptive** : Places new dots in areas where existing dots have faded, while
  avoiding clustering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ThermalDot:
    """A single projected thermal dot.

    Attributes
    ----------
    x, y : float
        2D image-plane position in pixels.
    world_point : np.ndarray | None
        Corresponding 3D world coordinate (when known).
    intensity : float
        Laser power / peak temperature rise for this dot (K above ambient).
    frame_created : int
        Frame index at which this dot was first deposited.
    """

    x: float
    y: float
    world_point: np.ndarray | None = field(default=None, repr=False)
    intensity: float = 15.0
    frame_created: int = 0


class PatternProjector:
    """Generate and manage sparse thermal dot patterns.

    Parameters
    ----------
    image_width, image_height : int
        Dimensions of the thermal camera image plane.
    margin : int
        Minimum distance from the image border for dot placement.
    min_separation : float
        Minimum Euclidean distance between any two dots (avoids clustering).
    max_dots : int
        Maximum number of simultaneously active dots.
    default_intensity : float
        Default peak intensity (K above ambient) for new dots.
    rng_seed : int | None
        Seed for the random number generator (reproducibility).
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        margin: int = 20,
        min_separation: float = 30.0,
        max_dots: int = 25,
        default_intensity: float = 15.0,
        rng_seed: int | None = None,
    ) -> None:
        if image_width <= 2 * margin or image_height <= 2 * margin:
            raise ValueError("Image dimensions must be larger than 2 × margin.")

        self.w = image_width
        self.h = image_height
        self.margin = margin
        self.min_sep = min_separation
        self.max_dots = max_dots
        self.default_intensity = default_intensity
        self._rng = np.random.default_rng(rng_seed)

        self._active_dots: list[ThermalDot] = []

    # ------------------------------------------------------------------
    # Pattern generation strategies
    # ------------------------------------------------------------------

    def generate_grid(
        self,
        rows: int = 4,
        cols: int = 4,
        frame_index: int = 0,
    ) -> list[ThermalDot]:
        """Place dots on a regular grid within the image margins."""
        xs = np.linspace(self.margin, self.w - self.margin, cols)
        ys = np.linspace(self.margin, self.h - self.margin, rows)
        dots: list[ThermalDot] = []
        for y in ys:
            for x in xs:
                dots.append(
                    ThermalDot(
                        x=float(x),
                        y=float(y),
                        intensity=self.default_intensity,
                        frame_created=frame_index,
                    )
                )
        self._active_dots = dots[: self.max_dots]
        return list(self._active_dots)

    def generate_random(
        self,
        n: int = 12,
        frame_index: int = 0,
    ) -> list[ThermalDot]:
        """Place *n* dots uniformly at random, respecting min separation."""
        dots: list[ThermalDot] = []
        attempts = 0
        max_attempts = n * 50  # Prevent infinite loops

        while len(dots) < n and attempts < max_attempts:
            attempts += 1
            x = self._rng.uniform(self.margin, self.w - self.margin)
            y = self._rng.uniform(self.margin, self.h - self.margin)
            if self._is_valid_position(x, y, dots):
                dots.append(
                    ThermalDot(
                        x=float(x),
                        y=float(y),
                        intensity=self.default_intensity,
                        frame_created=frame_index,
                    )
                )

        self._active_dots = dots[: self.max_dots]
        return list(self._active_dots)

    def generate_adaptive(
        self,
        current_field: np.ndarray,
        threshold: float = 2.0,
        n_new: int = 4,
        frame_index: int = 0,
    ) -> list[ThermalDot]:
        """Add dots in regions where thermal signal has faded below *threshold*.

        Parameters
        ----------
        current_field : np.ndarray
            Current relative temperature field (ΔT above ambient).
        threshold : float
            Temperature below which a region is considered "faded".
        n_new : int
            Number of new dots to attempt to place.
        frame_index : int
            Current frame index.

        Returns
        -------
        list[ThermalDot]
            Updated list of active dots.
        """
        # Remove dots whose region has faded
        surviving: list[ThermalDot] = []
        for dot in self._active_dots:
            ix, iy = int(round(dot.x)), int(round(dot.y))
            ix = np.clip(ix, 0, current_field.shape[1] - 1)
            iy = np.clip(iy, 0, current_field.shape[0] - 1)
            if current_field[iy, ix] >= threshold:
                surviving.append(dot)
        self._active_dots = surviving

        # Fill vacated slots
        added = 0
        attempts = 0
        max_attempts = n_new * 50
        while (
            added < n_new
            and len(self._active_dots) < self.max_dots
            and attempts < max_attempts
        ):
            attempts += 1
            x = self._rng.uniform(self.margin, self.w - self.margin)
            y = self._rng.uniform(self.margin, self.h - self.margin)
            if self._is_valid_position(x, y, self._active_dots):
                self._active_dots.append(
                    ThermalDot(
                        x=float(x),
                        y=float(y),
                        intensity=self.default_intensity,
                        frame_created=frame_index,
                    )
                )
                added += 1

        return list(self._active_dots)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def active_dots(self) -> list[ThermalDot]:
        return list(self._active_dots)

    def set_active_dots(self, dots: Sequence[ThermalDot]) -> None:
        self._active_dots = list(dots)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_valid_position(
        self, x: float, y: float, existing: Sequence[ThermalDot]
    ) -> bool:
        for dot in existing:
            if (dot.x - x) ** 2 + (dot.y - y) ** 2 < self.min_sep**2:
                return False
        return True
