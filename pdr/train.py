"""
Train the Projection-Diffusion Reversal (PDR) model on synthetic diffusion-reversal pairs.

Quick smoke test (local laptop):
    python pdr/train.py --smoke

Full trainin:
python pdr/train.py --epochs 50 --samples 10000 --batch 8

The trained checkpoint is saved to:
    pdr/checkpoints/pdr_model.pt
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from dataset import SyntheticPairDataset
from model import PDRNet
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


CKPT_DIR = ROOT / "checkpoints"


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    # Data
    ds = SyntheticPairDataset(length=args.samples, seed=args.seed)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = PDRNet().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params : {n_params:,}")

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

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

        avg_loss = running / len(ds)
        elapsed = time.time() - t0

        tag = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), CKPT_DIR / "pdr_model.pt")
            tag = "  [saved]"

        print(
            f"  Epoch {epoch:3d}/{args.epochs}  "
            f"loss={avg_loss:.6f}  ({elapsed:.1f}s){tag}"
        )

    print(f"\n  Best loss : {best_loss:.6f}")
    print(f"  Checkpoint: {CKPT_DIR / 'pdr_model.pt'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train PDR model")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke test: 2 epochs, 20 samples",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.smoke:
        args.epochs = 2
        args.samples = 20
        args.batch = 2

    print("  PDR Training")
    print(
        f"  Epochs={args.epochs}  Samples={args.samples}  "
        f"Batch={args.batch}  LR={args.lr}"
    )
    if args.smoke:
        print("  ** SMOKE TEST mode **")

    train(args)


if __name__ == "__main__":
    main()
