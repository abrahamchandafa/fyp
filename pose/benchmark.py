"""Benchmark the SLAM estimator against ground truth."""

from common import (
    GROUND_TRUTHS,
    PoseResult,
    rotation_error_deg,
    translation_dir_error_deg,
)
from pose_slam import run_dof

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))



def print_table(rows: list[dict]) -> None:
    hdr = (
        f"{'Algorithm':<14} {'DoF':<14} {'Rot err':>8} {'Trans dir':>10} {'Points':>8}"
    )
    print("\n" + "=" * len(hdr))
    print("  POSE BENCHMARK")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['algorithm']:<14} {r['dof']:<14} {r['rot_err']:>8.3f} {r['trans_dir_err']:>10.3f} {r['points']:>8}"
        )


def score(result: PoseResult) -> dict:
    gt = GROUND_TRUTHS[result.dof_name]
    gt_R, _ = cv2.Rodrigues(np.array(gt.rvec, dtype=np.float64))
    gt_t = np.array(gt.tvec, dtype=np.float64)
    est_R, _ = cv2.Rodrigues(result.mean_rvec)
    est_t = result.mean_camera_motion
    return {
        "algorithm": result.algorithm,
        "dof": result.dof_name,
        "rot_err": rotation_error_deg(est_R, gt_R),
        "trans_dir_err": translation_dir_error_deg(est_t, gt_t),
        "points": sum(1 for f in result.frame_results if f.n_points > 0),
    }


def main() -> None:
    dofs = sys.argv[1:] if len(sys.argv) > 1 else list(GROUND_TRUTHS.keys())
    rows = []
    for dof in dofs:
        result = run_dof(dof)
        rows.append(score(result))
    print_table(rows)


if __name__ == "__main__":
    main()
