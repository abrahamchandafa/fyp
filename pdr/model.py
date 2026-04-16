"""
PDRNet -- small U-Net that reverses thermal diffusion.

Input : 2-channel float32 tensor  (prev_frame, curr_frame), values in [0, 1]
Output: 1-channel float32 tensor  (corrected curr_frame),   values in [0, 1]

Architecture
------------
Encoder  : 32 -> 64 -> 128 -> 256   (4 levels, MaxPool2d stride 2)
Bottleneck: 512
Decoder  : 256 -> 128 -> 64 -> 32   (skip connections from encoder)
Final    : 1x1 conv -> sigmoid

Compatible with 640x480 input (divisible by 16).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PDRNet(nn.Module):
    """Projection-Diffusion-Reversal U-Net."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        feats = list(features)

        # -- Encoder --
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for f in feats:
            self.encoders.append(_ConvBlock(prev_ch, f))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_ch = f

        # -- Bottleneck --
        self.bottleneck = _ConvBlock(feats[-1], feats[-1] * 2)

        # -- Decoder --
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev_ch = feats[-1] * 2
        for f in reversed(feats):
            self.upconvs.append(nn.ConvTranspose2d(prev_ch, f, kernel_size=2, stride=2))
            self.decoders.append(_ConvBlock(f * 2, f))  # *2 for skip-cat
            prev_ch = f

        self.head = nn.Conv2d(feats[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            # Handle size mismatch from rounding on odd dimensions
            dh = skip.shape[2] - x.shape[2]
            dw = skip.shape[3] - x.shape[3]
            if dh or dw:
                x = nn.functional.pad(x, [0, dw, 0, dh])
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return torch.sigmoid(self.head(x))
