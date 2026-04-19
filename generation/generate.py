"""
6-DoF thermal dot motion — image and video generator.

For each degree of freedom, generates a sequence of frames where:
  1. Existing dots are shifted (simulating camera motion in that DoF).
  2. Existing dots are eroded (fade intensity + shrink radius).
  3. Fresh dots are deposited.
  4. The frame is rendered and saved.

Output layout
─────────────
    generation/output/
        translate_x/frames/frame_0000.png … frame_xx.png
        translate_x/translate_x.mp4
        translate_y/…
        translate_z/…
        rotate_pitch/…
        rotate_yaw/…
        rotate_roll/…

Usage
─────
    python generation/generate.py
"""

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import qmc

# ═══════════════════════════════════════════════════════════════════════
#  GLOBAL CONSTANTS  —  can be edited to control the output
# ═══════════════════════════════════════════════════════════════════════

# Frame dimensions in pixels
IMAGE_WIDTH: int = 640
IMAGE_HEIGHT: int = 480

# Timing
FPS: int = 15
DURATION_S: int = 5
N_FRAMES: int = FPS * DURATION_S

# Dot generation
N_DOTS_PER_FRAME: int = 3
INITIAL_INTENSITY: float = 200.0
INITIAL_RADIUS: float = 7.0
MARGIN: int = 50

# Motion rates
SHIFT_RATE: float = 5.0  # translation shift (pixels / frame)
ROTATION_RATE: float = 0.025  # roll rotation (radians / frame)
ZOOM_RATE: float = 0.035  # radial expansion factor / frame (translate-Z)

# Erosion
FADE_FACTOR: float = 0.85
ERODE_FACTOR: float = 0.90
MIN_INTENSITY: float = 20.0
MIN_RADIUS: float = 4

# Visual
COLORMAP: int = cv2.COLORMAP_HOT

# Output
OUTPUT_DIR: Path = Path(__file__).resolve().parent / "output"
RNG_SEED: int = 42


# ═══════════════════════════════════════════════════════════════════════
#  Data model
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Dot:
    """A single thermal dot with position, intensity, and size."""

    x: float
    y: float
    intensity: float
    radius: float


# ═══════════════════════════════════════════════════════════════════════
#  Per-DoF shift functions
# ═══════════════════════════════════════════════════════════════════════
# Each function modifies dots IN PLACE to simulate the apparent 2-D
# motion caused by the corresponding camera degree of freedom.


def shift_translate_x(dots: list[Dot]) -> None:
    """Camera translates right → dots slide left."""
    for d in dots:
        d.x -= SHIFT_RATE


def shift_translate_y(dots: list[Dot]) -> None:
    """Camera translates up → dots slide down (image-y is top-down)."""
    for d in dots:
        d.y += SHIFT_RATE


def shift_translate_z(dots: list[Dot]) -> None:
    """Camera moves forward → dots expand radially from the centre."""
    cx, cy = IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2
    for d in dots:
        d.x += (d.x - cx) * ZOOM_RATE
        d.y += (d.y - cy) * ZOOM_RATE


def shift_rotate_yaw(dots: list[Dot]) -> None:
    """Camera yaws right → dots slide left (stronger near edges)."""
    cx = IMAGE_WIDTH / 2
    for d in dots:
        d.x -= SHIFT_RATE * (0.6 + 0.4 * abs(d.x - cx) / cx)


def shift_rotate_pitch(dots: list[Dot]) -> None:
    """Camera pitches up → dots slide down (stronger near edges)."""
    cy = IMAGE_HEIGHT / 2
    for d in dots:
        d.y += SHIFT_RATE * (0.6 + 0.4 * abs(d.y - cy) / cy)


def shift_rotate_roll(dots: list[Dot]) -> None:
    """Camera rolls clockwise → dots rotate counter-clockwise."""
    cx, cy = IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2
    cos_r = math.cos(-ROTATION_RATE)
    sin_r = math.sin(-ROTATION_RATE)
    for d in dots:
        dx, dy = d.x - cx, d.y - cy
        d.x = cx + dx * cos_r - dy * sin_r
        d.y = cy + dx * sin_r + dy * cos_r


# Registry: name → shift function
DOF_REGISTRY: dict[str, object] = {
    "translate_x": shift_translate_x,
    "translate_y": shift_translate_y,
    "translate_z": shift_translate_z,
    "rotate_yaw": shift_rotate_yaw,
    "rotate_pitch": shift_rotate_pitch,
    "rotate_roll": shift_rotate_roll,
}


# ═══════════════════════════════════════════════════════════════════════
#  Core routines
# ═══════════════════════════════════════════════════════════════════════


def erode_dots(dots: list[Dot]) -> list[Dot]:
    """Fade intensity, shrink radius, keep dots on-screen at their minimum values."""
    surviving: list[Dot] = []
    for d in dots:
        d.intensity *= FADE_FACTOR
        d.radius *= ERODE_FACTOR
        if d.intensity < MIN_INTENSITY:
            d.intensity = MIN_INTENSITY
        if d.radius < MIN_RADIUS:
            d.radius = MIN_RADIUS
        if 0 <= d.x < IMAGE_WIDTH and 0 <= d.y < IMAGE_HEIGHT:
            surviving.append(d)
    return surviving


def spawn_dots(points: np.ndarray) -> list[Dot]:
    """Create N_DOTS_PER_FRAME fresh dots at Halton-sampled positions."""
    return [
        Dot(
            x=float(x),
            y=float(y),
            intensity=INITIAL_INTENSITY,
            radius=INITIAL_RADIUS,
        )
        for x, y in points
    ]


def render(dots: list[Dot]) -> np.ndarray:
    """Render all dots to a colour-mapped thermal image (BGR, uint8)."""
    field = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float64)

    for dot in dots:
        extent = int(math.ceil(3 * dot.radius))
        y_lo = max(0, int(dot.y) - extent)
        y_hi = min(IMAGE_HEIGHT, int(dot.y) + extent + 1)
        x_lo = max(0, int(dot.x) - extent)
        x_hi = min(IMAGE_WIDTH, int(dot.x) + extent + 1)
        if y_lo >= y_hi or x_lo >= x_hi:
            continue

        yy, xx = np.ogrid[y_lo:y_hi, x_lo:x_hi]
        dist_sq = (xx - dot.x) ** 2 + (yy - dot.y) ** 2
        gaussian = dot.intensity * np.exp(-dist_sq / (2 * dot.radius**2))
        field[y_lo:y_hi, x_lo:x_hi] += gaussian

    np.clip(field, 0, 255, out=field)
    gray = field.astype(np.uint8)
    return cv2.applyColorMap(gray, COLORMAP)


# ═══════════════════════════════════════════════════════════════════════
#  Sequence generator
# ═══════════════════════════════════════════════════════════════════════


def generate_dof_sequence(
    dof_name: str,
    shift_fn,
    rng: np.random.Generator,
) -> None:
    """Generate all frames and video for one degree of freedom."""
    frames_dir = OUTPUT_DIR / dof_name / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    halton = qmc.Halton(d=2, scramble=True, rng=rng)
    dots: list[Dot] = []

    for fi in range(N_FRAMES):
        # 1. Shift existing dots (camera motion simulation)
        shift_fn(dots)

        # 2. Erode existing dots (heat dissipation simulaton)
        dots = erode_dots(dots)

        # 3. Deposit fresh dots
        sample = halton.random(n=N_DOTS_PER_FRAME)
        points = qmc.scale(
            sample,
            [MARGIN, MARGIN],
            [IMAGE_WIDTH - MARGIN, IMAGE_HEIGHT - MARGIN],
        )
        dots.extend(spawn_dots(points))

        image = render(dots)

        # Label overlay
        cv2.putText(
            image,
            f"{dof_name}  {fi + 1}/{N_FRAMES}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        cv2.imwrite(str(frames_dir / f"frame_{fi:04d}.png"), image)

    # Compile video
    video_path = OUTPUT_DIR / dof_name / f"{dof_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, FPS, (IMAGE_WIDTH, IMAGE_HEIGHT))
    for fi in range(N_FRAMES):
        frame = cv2.imread(str(frames_dir / f"frame_{fi:04d}.png"))
        writer.write(frame)
    writer.release()

    print(f"    {dof_name}: {N_FRAMES} frames + video -> {video_path}")


def main() -> None:
    print("=" * 60)
    print("  6-DoF Thermal Dot Video Generator")
    print("=" * 60)
    print(f"  Resolution    : {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
    print(f"  FPS           : {FPS}")
    print(f"  Duration      : {DURATION_S}s  ({N_FRAMES} frames)")
    print(f"  Dots / frame  : {N_DOTS_PER_FRAME}")
    print(f"  Shift rate    : {SHIFT_RATE} px/frame")
    print(f"  Fade factor   : {FADE_FACTOR}")
    print(f"  Erode factor  : {ERODE_FACTOR}")
    print(f"  Output dir    : {OUTPUT_DIR}")
    print()

    for dof_name, shift_fn in DOF_REGISTRY.items():
        rng = np.random.default_rng(RNG_SEED)
        generate_dof_sequence(dof_name, shift_fn, rng)

    print("Done.")


if __name__ == "__main__":
    main()
