"""
Benchmark -- compare all pose estimation methods against ground truth.

Runs every estimator on every DoF and produces:
  1. A console table with rotation error and translation direction error.
  2. A CSV file (pose/output/benchmark.csv).

Usage:
    python pose/benchmark.py                     # all DoFs
    python pose/benchmark.py translate_x         # single DoF
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    GROUND_TRUTHS,
    PoseResult,
    rotation_error_deg,
    translation_dir_error_deg,
)
from pose_cnn import run_dof as cnn_run
from pose_essential import run_dof as essential_run
from pose_homography import run_dof as homography_run
from pose_pnp import run_dof as pnp_run

POSE_OUTPUT = Path(__file__).resolve().parent / "output"

ESTIMATORS: list[tuple[str, object]] = [
    ("Homography", homography_run),
    ("PnP", pnp_run),
    ("Essential", essential_run),
    ("PoseCNN", cnn_run),
]


# =====================================================================
#  Scoring
# =====================================================================


def score(result: PoseResult) -> dict:
    gt = GROUND_TRUTHS[result.dof_name]
    gt_rvec = np.array(gt.rvec, dtype=np.float64)
    gt_tvec = np.array(gt.tvec, dtype=np.float64)
    gt_R, _ = cv2.Rodrigues(gt_rvec)

    est_rvec = result.mean_rvec
    est_tvec = result.mean_camera_motion
    est_R, _ = cv2.Rodrigues(est_rvec)

    rot_err = rotation_error_deg(est_R, gt_R)
    trans_err = translation_dir_error_deg(est_tvec, gt_tvec)

    n_valid = len(result.valid_frames)
    n_total = len(result.frame_results)

    return {
        "algorithm": result.algorithm,
        "dof": result.dof_name,
        "rot_err": rot_err,
        "trans_dir_err": trans_err,
        "n_valid": n_valid,
        "n_total": n_total,
        "est_rvec": f"[{est_rvec[0]:+.5f}, {est_rvec[1]:+.5f}, {est_rvec[2]:+.5f}]",
        "gt_rvec": f"[{gt_rvec[0]:+.5f}, {gt_rvec[1]:+.5f}, {gt_rvec[2]:+.5f}]",
        "est_tvec": f"[{est_tvec[0]:+.5f}, {est_tvec[1]:+.5f}, {est_tvec[2]:+.5f}]",
        "gt_tvec": f"[{gt_tvec[0]:+.5f}, {gt_tvec[1]:+.5f}, {gt_tvec[2]:+.5f}]",
    }


# =====================================================================
#  Console table
# =====================================================================


def fmt_err(val: float) -> str:
    if math.isnan(val):
        return "   N/A"
    return f"{val:7.3f}"


def print_table(rows: list[dict]) -> None:
    hdr = (
        f"{'Algorithm':<14} {'DoF':<16} "
        f"{'Rot err':>8} {'Trans err':>10} "
        f"{'Valid':>6} {'Total':>6}"
    )

    print("\n" + "=" * len(hdr))
    print("  POSE ESTIMATION BENCHMARK")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        print(
            f"{r['algorithm']:<14} {r['dof']:<16} "
            f"{fmt_err(r['rot_err']):>8} {fmt_err(r['trans_dir_err']):>10} "
            f"{r['n_valid']:>6} {r['n_total']:>6}"
        )

    # Mean per algorithm
    algos = list(dict.fromkeys(r["algorithm"] for r in rows))
    print("\n" + "-" * len(hdr))
    print(
        f"{'MEAN ERROR':<14} {'(across DoFs)':<16} "
        f"{'Rot err':>8} {'Trans err':>10} "
        f"{'Valid':>6} {'Rank':>6}"
    )
    print("-" * len(hdr))

    rankings: list[tuple[float, str]] = []
    for algo in algos:
        ar = [r for r in rows if r["algorithm"] == algo]
        avg_rot = np.mean([r["rot_err"] for r in ar])
        trans_vals = [
            r["trans_dir_err"] for r in ar if not math.isnan(r["trans_dir_err"])
        ]
        avg_trans = np.mean(trans_vals) if trans_vals else float("nan")
        tot_valid = sum(r["n_valid"] for r in ar)
        # Ranking score: average rotation error (always available)
        rankings.append((avg_rot, algo))
        print(
            f"{algo:<14} {'':>16} "
            f"{fmt_err(avg_rot):>8} {fmt_err(avg_trans):>10} "
            f"{tot_valid:>6}"
        )

    rankings.sort()
    print("\n  RANKING (lowest avg rotation error):")
    for rank, (err, algo) in enumerate(rankings, 1):
        print(f"    #{rank}  {algo:<14}  avg rot error = {err:.3f} deg")
    print("=" * len(hdr) + "\n")


# =====================================================================
#  CSV
# =====================================================================


def export_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV saved to: {path}")


# =====================================================================
#  Main
# =====================================================================


def main() -> None:
    dofs = sys.argv[1:] if len(sys.argv) > 1 else list(GROUND_TRUTHS.keys())
    all_scores: list[dict] = []

    for dof_name in dofs:
        print(f"\n{'#' * 60}")
        print(f"  DoF: {dof_name}")
        print(f"{'#' * 60}")
        for algo_name, run_fn in ESTIMATORS:
            print(f"\n  Running {algo_name} on {dof_name} ...")
            result = run_fn(dof_name)
            all_scores.append(score(result))

    print_table(all_scores)
    export_csv(all_scores, POSE_OUTPUT / "benchmark.csv")


if __name__ == "__main__":
    main()
