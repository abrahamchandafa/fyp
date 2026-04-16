"""Diagnostic: check flow at blob locations."""

import sys

sys.path.insert(0, "pose")
import cv2
import numpy as np
from common import *

frames = load_frames(GEN_OUTPUT / "translate_x" / "frames")
g1, g2 = to_gray(frames[0]), to_gray(frames[1])
print("shape:", g1.shape, "dtype:", g1.dtype, "max:", g1.max())

flow = cv2.calcOpticalFlowFarneback(g1, g2, None, **FARNEBACK_PARAMS)
print("flow at center:", flow[240, 320])

blobs = detect_blobs(g1)
print(f"{len(blobs)} blobs detected")
for x, y in blobs[:8]:
    ix, iy = int(round(x)), int(round(y))
    iy = min(iy, g1.shape[0] - 1)
    ix = min(ix, g1.shape[1] - 1)
    dx, dy = flow[iy, ix]
    print(f"  blob({x:.1f},{y:.1f}) -> flow=({dx:.3f},{dy:.3f})")
