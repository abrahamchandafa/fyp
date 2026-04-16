"""
Pose Estimator 4 -- CNN direct pose regression.

A small convolutional network that takes two consecutive grayscale
frames (2-channel input, 640x480) and directly outputs the 6-DoF
camera motion: (tx, ty, tz, rx, ry, rz).

Usage:
    python pose/pose_cnn.py                    # all DoFs
    python pose/pose_cnn.py translate_x        # single DoF

Requires a trained checkpoint at:
    pose/checkpoints/pose_cnn.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    GEN_OUTPUT,
    GROUND_TRUTHS,
    FramePoseResult,
    PoseResult,
    load_frames,
    print_summary,
    to_gray,
)

CKPT_PATH = Path(__file__).resolve().parent / "checkpoints" / "pose_cnn.pt"


# =====================================================================
#  Model
# =====================================================================


class PoseCNN(nn.Module):
    """Lightweight CNN for 6-DoF pose regression from frame pairs."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6),  # tx, ty, tz, rx, ry, rz
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(x))


# =====================================================================
#  Inference
# =====================================================================


def estimate_pose(
    gray1: np.ndarray,
    gray2: np.ndarray,
    model: PoseCNN,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, t) predicted by the CNN."""
    f1 = gray1.astype(np.float32) / 255.0
    f2 = gray2.astype(np.float32) / 255.0
    inp = np.stack([f1, f2], axis=0)[np.newaxis]  # (1, 2, H, W)

    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(inp).to(device))
    pred = out[0].cpu().numpy()  # (6,)

    tx, ty, tz, rx, ry, rz = pred
    rvec = np.array([rx, ry, rz], dtype=np.float64)
    tvec = np.array([tx, ty, tz], dtype=np.float64)

    R, _ = cv2.Rodrigues(rvec)
    # CNN outputs camera motion directly; convert to PnP-style t
    # so common.py mean_camera_motion = -R^T @ t gives back the motion.
    t_pnp = -R @ tvec.reshape(3, 1)
    return R, t_pnp


def run_dof(dof_name: str) -> PoseResult:
    if not CKPT_PATH.exists():
        print(f"  ERROR: checkpoint not found at {CKPT_PATH}")
        print("  Run  python pose/train_cnn.py --smoke  first")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseCNN()
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    model.to(device).eval()

    frames = load_frames(GEN_OUTPUT / dof_name / "frames")
    result = PoseResult(algorithm="PoseCNN", dof_name=dof_name)

    for i in range(1, len(frames)):
        g1, g2 = to_gray(frames[i - 1]), to_gray(frames[i])
        R, t = estimate_pose(g1, g2, model, device)
        result.frame_results.append(
            FramePoseResult(frame_index=i, R=R, t=t, n_points=0)
        )

    print_summary(result)
    return result


def main() -> None:
    dofs = sys.argv[1:] if len(sys.argv) > 1 else list(GROUND_TRUTHS.keys())
    for dof in dofs:
        run_dof(dof)


if __name__ == "__main__":
    main()
