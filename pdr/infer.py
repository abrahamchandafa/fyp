"""
Apply a trained PDR model to real thermal frames from the generation output.

Usage:
    python pdr/infer.py                          # all 6 DoFs
    python pdr/infer.py translate_x              # single DoF
    python pdr/infer.py translate_x rotate_yaw   # subset

Input frames  :  generation/output/{dof}/frames/frame_XXXX.png
Output frames :  pdr/output/{dof}/frames/frame_XXXX.png
Output video  :  pdr/output/{dof}/{dof}_corrected.mp4
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from model import PDRNet

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# -- paths -----------------------------------------------------------------
GEN_OUTPUT = ROOT.parent / "generation" / "output"
PDR_OUTPUT = ROOT / "output"
CKPT_PATH = ROOT / "checkpoints" / "pdr_model.pt"

# -- video settings (match generation/generate.py) -------------------------
FPS = 15
COLORMAP = cv2.COLORMAP_HOT

ALL_DOFS = [
    "translate_x",
    "translate_y",
    "translate_z",
    "rotate_yaw",
    "rotate_pitch",
    "rotate_roll",
]


def load_grayscale_frames(frames_dir: Path) -> list[np.ndarray]:
    """Load PNGs as grayscale float32 images in [0, 1]."""
    paths = sorted(frames_dir.glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"No frame_*.png files in {frames_dir}")
    frames: list[np.ndarray] = []
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        frames.append(gray)
    return frames


def infer_dof(dof_name: str, model: PDRNet, device: torch.device) -> None:
    frames_dir = GEN_OUTPUT / dof_name / "frames"
    out_frames_dir = PDR_OUTPUT / dof_name / "frames"
    out_frames_dir.mkdir(parents=True, exist_ok=True)

    frames = load_grayscale_frames(frames_dir)
    print(f"    Loaded {len(frames)} frames from {frames_dir}")

    corrected_bgr: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for i, curr in enumerate(frames):
            prev = frames[i - 1] if i > 0 else curr
            inp = np.stack([prev, curr], axis=0)[np.newaxis]  # (1, 2, H, W)
            inp_t = torch.from_numpy(inp).to(device)

            out_t = model(inp_t)
            out_np = out_t[0, 0].cpu().numpy()  # (H, W) [0, 1]

            # Convert to colormapped uint8 for saving
            gray8 = np.clip(out_np * 255, 0, 255).astype(np.uint8)
            colour = cv2.applyColorMap(gray8, COLORMAP)

            cv2.imwrite(str(out_frames_dir / f"frame_{i:04d}.png"), colour)
            corrected_bgr.append(colour)

    # -- compile video ----------------------------------------------------
    h, w = corrected_bgr[0].shape[:2]
    video_path = PDR_OUTPUT / dof_name / f"{dof_name}_corrected.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(video_path), fourcc, FPS, (w, h))
    for frame_bgr in corrected_bgr:
        vw.write(frame_bgr)
    vw.release()

    print(f"    Saved {len(corrected_bgr)} corrected frames -> {out_frames_dir}")
    print(f"    Video -> {video_path}")


def main() -> None:
    if not CKPT_PATH.exists():
        print(f"  ERROR: checkpoint not found at {CKPT_PATH}")
        print("  Run  python pdr/train.py  first (or --smoke for quick test)")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device     : {device}")

    model = PDRNet()
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"  Checkpoint : {CKPT_PATH}")

    dofs = sys.argv[1:] if len(sys.argv) > 1 else ALL_DOFS
    for dof in dofs:
        print(f"\n  Processing: {dof}")
        infer_dof(dof, model, device)

    print("\n  Done.")


if __name__ == "__main__":
    main()
