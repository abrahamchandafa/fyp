"""
Synthetic paired data for PDR training.

Each sample
-----------
  input  :  (2, H, W)  -- two consecutive *diffused* grayscale frames [0, 1]
  target :  (1, H, W)  -- the *clean* (pre-diffusion) second frame    [0, 1]

Dot scenes are randomly generated with the same Gaussian rendering and
fade/erode parameters used in ``generation/generate.py``, so the model
learns the exact inverse of the diffusion that was applied.

Training data is unlimited and deterministic per sample index.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

# ── constants (must match generation/generate.py) ────────────────────
IMAGE_W: int = 640
IMAGE_H: int = 480
INITIAL_INTENSITY: float = 220.0
INITIAL_RADIUS: float = 8.0
FADE_FACTOR: float = 0.85
ERODE_FACTOR: float = 0.90
MIN_INTENSITY: float = 8.0
MIN_RADIUS: float = 0.5
MARGIN: int = 30


# ── rendering helpers ─────────────────────────────────────────────────


def render_dots(
    dots: list[tuple[float, float, float, float]],
    h: int = IMAGE_H,
    w: int = IMAGE_W,
) -> np.ndarray:
    """Render *dots* as 2-D Gaussians on a [0, 1] float32 image.

    Each dot is ``(cx, cy, intensity, radius)`` where *intensity* is on
    the 0-255 scale and *radius* is the Gaussian sigma in pixels.
    """
    img = np.zeros((h, w), dtype=np.float32)
    for cx, cy, intensity, radius in dots:
        r = int(3.0 * radius) + 1
        y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
        x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        ys = np.arange(y0, y1, dtype=np.float32) - cy
        xs = np.arange(x0, x1, dtype=np.float32) - cx
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        g = (intensity / 255.0) * np.exp(-(xx**2 + yy**2) / (2.0 * radius**2))
        np.maximum(img[y0:y1, x0:x1], g, out=img[y0:y1, x0:x1])
    return np.clip(img, 0.0, 1.0)


def random_scene(
    rng: np.random.Generator,
    n_range: tuple[int, int] = (10, 30),
) -> list[tuple[float, float, float, float]]:
    """Return random ``(cx, cy, intensity, radius)`` dots at various ages."""
    n = int(rng.integers(*n_range))
    dots: list[tuple[float, float, float, float]] = []
    for _ in range(n):
        cx = float(rng.uniform(MARGIN, IMAGE_W - MARGIN))
        cy = float(rng.uniform(MARGIN, IMAGE_H - MARGIN))
        age = int(rng.integers(0, 10))
        inten = INITIAL_INTENSITY * (FADE_FACTOR**age)
        rad = INITIAL_RADIUS * (ERODE_FACTOR**age)
        if inten >= MIN_INTENSITY and rad >= MIN_RADIUS:
            dots.append((cx, cy, inten, rad))
    return dots


def diffuse(
    dots: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Apply one step of fade + erode."""
    return [
        (cx, cy, i * FADE_FACTOR, r * ERODE_FACTOR)
        for cx, cy, i, r in dots
        if i * FADE_FACTOR >= MIN_INTENSITY and r * ERODE_FACTOR >= MIN_RADIUS
    ]


def shift_dots(
    dots: list[tuple[float, float, float, float]],
    dx: float,
    dy: float,
) -> list[tuple[float, float, float, float]]:
    """Translate every dot by ``(dx, dy)``."""
    return [(cx + dx, cy + dy, i, r) for cx, cy, i, r in dots]


# ── dataset ───────────────────────────────────────────────────────────


class SyntheticPairDataset(Dataset):
    """Deterministic, on-the-fly synthetic training data.

    For each sample index *idx*:

    1. Generate a random dot scene at time *t*.
    2. Shift dots by a small random (dx, dy) -> positions at *t+1*.
    3. Spawn 1-3 fresh dots at *t+1*.
    4. Render the **clean** *t+1* scene -> ``target``.
    5. Apply one diffusion step to both *t* and *t+1* -> render ->
       ``diffused_t``, ``diffused_t1``.
    6. Return ``(stack(diffused_t, diffused_t1), target)``.
    """

    def __init__(self, length: int = 2000, seed: int = 0) -> None:
        self.length = length
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + idx)

        # 1) scene at time t
        dots_t = random_scene(rng)

        # 2) random motion
        dx = float(rng.uniform(-6, 6))
        dy = float(rng.uniform(-6, 6))

        # 3) dots at t+1 *before* diffusion = CLEAN target
        dots_t1_clean = shift_dots(dots_t, dx, dy)
        for _ in range(int(rng.integers(1, 4))):
            cx = float(rng.uniform(MARGIN, IMAGE_W - MARGIN))
            cy = float(rng.uniform(MARGIN, IMAGE_H - MARGIN))
            dots_t1_clean.append((cx, cy, INITIAL_INTENSITY, INITIAL_RADIUS))

        target = render_dots(dots_t1_clean)  # (H, W) clean

        # 4) diffused versions (what the camera actually records)
        diffused_t = render_dots(diffuse(dots_t))
        diffused_t1 = render_dots(diffuse(dots_t1_clean))

        inp = np.stack([diffused_t, diffused_t1], axis=0)  # (2, H, W)
        tgt = target[np.newaxis]  # (1, H, W)
        return torch.from_numpy(inp), torch.from_numpy(tgt)
