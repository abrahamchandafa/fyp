"""
Train the PoseCNN on synthetic frame pairs with known camera motion.

Quick smoke test (local):
    python pose/train_cnn.py --smoke

Full training (Google Colab recommended):
    python pose/train_cnn.py --epochs 60 --samples 10000 --batch 8

Checkpoint saved to:
    pose/checkpoints/pose_cnn.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pose_cnn import PoseCNN

CKPT_DIR = ROOT / "checkpoints"

# =====================================================================
#  Scene rendering (matches generation/generate.py parameters)
# =====================================================================
IMAGE_W, IMAGE_H = 640, 480
INITIAL_INTENSITY = 220.0
INITIAL_RADIUS = 8.0
FADE_FACTOR = 0.85
ERODE_FACTOR = 0.90
MIN_INTENSITY = 8.0
MIN_RADIUS = 0.5
MARGIN = 30
FOCAL_LENGTH = 500.0
CX, CY = 320.0, 240.0
PLANE_DEPTH = 1.0

K = np.array(
    [[FOCAL_LENGTH, 0, CX], [0, FOCAL_LENGTH, CY], [0, 0, 1]],
    dtype=np.float64,
)
K_INV = np.linalg.inv(K)


def render_dots(dots, h=IMAGE_H, w=IMAGE_W):
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


def random_scene(rng, n_range=(10, 30)):
    n = int(rng.integers(*n_range))
    dots = []
    for _ in range(n):
        cx = float(rng.uniform(MARGIN, IMAGE_W - MARGIN))
        cy = float(rng.uniform(MARGIN, IMAGE_H - MARGIN))
        age = int(rng.integers(0, 8))
        inten = INITIAL_INTENSITY * (FADE_FACTOR**age)
        rad = INITIAL_RADIUS * (ERODE_FACTOR**age)
        if inten >= MIN_INTENSITY and rad >= MIN_RADIUS:
            dots.append((cx, cy, inten, rad))
    return dots


def diffuse(dots):
    return [
        (cx, cy, i * FADE_FACTOR, r * ERODE_FACTOR)
        for cx, cy, i, r in dots
        if i * FADE_FACTOR >= MIN_INTENSITY and r * ERODE_FACTOR >= MIN_RADIUS
    ]


# =====================================================================
#  Warp dot positions by camera motion using homography
# =====================================================================


def warp_dots(dots, rvec, tvec_motion):
    """Warp dot image positions given a camera motion (rvec, tvec_motion).

    tvec_motion = camera centre displacement in world frame.
    """
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64))
    # PnP-style t: p_cam2 = R @ p_world + t_pnp
    t_pnp = -R @ np.array(tvec_motion, dtype=np.float64).reshape(3, 1)
    # Homography:  H = K @ (R + t_pnp @ n^T / d) @ K_inv
    n = np.array([[0], [0], [1]], dtype=np.float64)
    H = K @ (R + (t_pnp @ n.T) / PLANE_DEPTH) @ K_INV

    warped = []
    for cx, cy, inten, rad in dots:
        p = np.array([cx, cy, 1.0])
        p2 = H @ p
        if p2[2] < 1e-6:
            continue
        p2 /= p2[2]
        warped.append((float(p2[0]), float(p2[1]), inten, rad))
    return warped


# =====================================================================
#  Dataset
# =====================================================================

# Ranges for random camera motion (per frame, realistic for our scenes)
MAX_TRANS = 0.015  # metres
MAX_ROT = 0.03  # radians


class PosePairDataset(Dataset):
    def __init__(self, length: int = 5000, seed: int = 0) -> None:
        self.length = length
        self.seed = seed

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + idx)

        # Random dot scene
        dots = random_scene(rng)

        # Random camera motion
        tx = float(rng.uniform(-MAX_TRANS, MAX_TRANS))
        ty = float(rng.uniform(-MAX_TRANS, MAX_TRANS))
        tz = float(rng.uniform(-MAX_TRANS * 3, MAX_TRANS * 3))
        rx = float(rng.uniform(-MAX_ROT, MAX_ROT))
        ry = float(rng.uniform(-MAX_ROT, MAX_ROT))
        rz = float(rng.uniform(-MAX_ROT, MAX_ROT))

        rvec = (rx, ry, rz)
        tvec = (tx, ty, tz)

        # Frame 1: diffused version of scene
        frame1 = render_dots(diffuse(dots))

        # Frame 2: warp, add fresh dots, then diffuse
        warped = warp_dots(dots, rvec, tvec)
        for _ in range(int(rng.integers(1, 4))):
            cx = float(rng.uniform(MARGIN, IMAGE_W - MARGIN))
            cy = float(rng.uniform(MARGIN, IMAGE_H - MARGIN))
            warped.append((cx, cy, INITIAL_INTENSITY, INITIAL_RADIUS))
        frame2 = render_dots(diffuse(warped))

        inp = np.stack([frame1, frame2], axis=0)  # (2, H, W)
        target = np.array([tx, ty, tz, rx, ry, rz], dtype=np.float32)
        return torch.from_numpy(inp), torch.from_numpy(target)


# =====================================================================
#  Training
# =====================================================================


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    ds = PosePairDataset(length=args.samples, seed=args.seed)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = PoseCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params : {n_params:,}")

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        t0 = time.time()

        for inp, tgt in dl:
            inp, tgt = inp.to(device), tgt.to(device)
            pred = model(inp)
            loss = criterion(pred, tgt)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running += loss.item() * inp.size(0)

        avg = running / len(ds)
        elapsed = time.time() - t0

        tag = ""
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), CKPT_DIR / "pose_cnn.pt")
            tag = "  [saved]"

        print(
            f"  Epoch {epoch:3d}/{args.epochs}  loss={avg:.8f}  ({elapsed:.1f}s){tag}"
        )

    print(f"\n  Best loss : {best_loss:.8f}")
    print(f"  Checkpoint: {CKPT_DIR / 'pose_cnn.pt'}")


def main():
    p = argparse.ArgumentParser(description="Train PoseCNN")
    p.add_argument(
        "--smoke", action="store_true", help="Quick test: 2 epochs, 20 samples"
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.smoke:
        args.epochs = 2
        args.samples = 20
        args.batch = 2

    print("  PoseCNN Training")
    print(
        f"  Epochs={args.epochs}  Samples={args.samples}  "
        f"Batch={args.batch}  LR={args.lr}"
    )
    if args.smoke:
        print("  ** SMOKE TEST mode **")

    train(args)


if __name__ == "__main__":
    main()
