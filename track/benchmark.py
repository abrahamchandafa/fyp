"""
Benchmark — compare all tracking algorithms against ground truth.

Runs every tracker script on every DoF, collects their per-frame
motion estimates, and produces:
  1. A console table summarising mean Δx, Δy, error, and ranking.
  2. A CSV file (track/output/benchmark.csv) for further analysis.

Usage:
    python track/benchmark.py                    # run everything
    python track/benchmark.py translate_x        # single DoF only
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import GROUND_TRUTHS, TrackingResult

#  each tracker's run_dof function
from track_blob import run_dof as blob_run
from track_dense_flow import run_dof as dense_run
from track_framediff import run_dof as framediff_run
from track_lk import run_dof as lk_run
from track_orb import run_dof as orb_run

TRACK_OUTPUT = Path(__file__).resolve().parent / "output"


# ═══════════════════════════════════════════════════════════════════════
#  Registry of all trackers
# ═══════════════════════════════════════════════════════════════════════

TRACKERS: list[tuple[str, object]] = [
    ("BlobCentroid", blob_run),
    ("LucasKanade", lk_run),
    ("DenseFlow", dense_run),
    ("ORB", orb_run),
    ("FrameDiff", framediff_run),
]


# ═══════════════════════════════════════════════════════════════════════
#  Scoring
# ═══════════════════════════════════════════════════════════════════════


def score(result: TrackingResult) -> dict:
    """Compute error metrics for one tracker × one DoF."""
    gt = GROUND_TRUTHS[result.dof_name]
    err_dx = abs(result.overall_mean_dx - gt.mean_dx)
    err_dy = abs(result.overall_mean_dy - gt.mean_dy)
    err_total = np.hypot(err_dx, err_dy)

    return {
        "algorithm": result.algorithm,
        "dof": result.dof_name,
        "gt_dx": gt.mean_dx,
        "gt_dy": gt.mean_dy,
        "est_dx": result.overall_mean_dx,
        "est_dy": result.overall_mean_dy,
        "err_dx": err_dx,
        "err_dy": err_dy,
        "err_total": err_total,
        "std_dx": result.overall_std_dx,
        "std_dy": result.overall_std_dy,
        "mean_speed": result.overall_mean_speed,
        "total_tracked": result.total_tracked,
        "direction": result.inferred_direction,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Console table
# ═══════════════════════════════════════════════════════════════════════


def print_table(rows: list[dict]) -> None:
    hdr = (
        f"{'Algorithm':<16} {'DoF':<16} "
        f"{'GT dx':>7} {'GT dy':>7} "
        f"{'Est dx':>8} {'Est dy':>8} "
        f"{'Err dx':>8} {'Err dy':>8} {'Err tot':>8} "
        f"{'Pts':>6} {'Dir':>14}"
    )

    print("\n" + "=" * len(hdr))
    print("  BENCHMARK RESULTS")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        print(
            f"{r['algorithm']:<16} {r['dof']:<16} "
            f"{r['gt_dx']:>+7.2f} {r['gt_dy']:>+7.2f} "
            f"{r['est_dx']:>+8.3f} {r['est_dy']:>+8.3f} "
            f"{r['err_dx']:>8.3f} {r['err_dy']:>8.3f} {r['err_total']:>8.3f} "
            f"{r['total_tracked']:>6} {r['direction']:>14}"
        )

    # Summary: mean error per algorithm
    algos = list(dict.fromkeys(r["algorithm"] for r in rows))

    print("\n" + "-" * len(hdr))
    print(
        f"{'MEAN ERROR':<16} {'(across DoFs)':<16} "
        f"{'':>7} {'':>7} {'':>8} {'':>8} "
        f"{'Err dx':>8} {'Err dy':>8} {'Err tot':>8} "
        f"{'Pts':>6} {'Rank':>14}"
    )
    print("-" * len(hdr))

    rankings: list[tuple[float, str]] = []
    for algo in algos:
        algo_rows = [r for r in rows if r["algorithm"] == algo]
        avg_err_dx = np.mean([r["err_dx"] for r in algo_rows])
        avg_err_dy = np.mean([r["err_dy"] for r in algo_rows])
        avg_err_tot = np.mean([r["err_total"] for r in algo_rows])
        tot_pts = sum(r["total_tracked"] for r in algo_rows)
        rankings.append((avg_err_tot, algo))
        print(
            f"{algo:<16} {'':>16} "
            f"{'':>7} {'':>7} {'':>8} {'':>8} "
            f"{avg_err_dx:>8.3f} {avg_err_dy:>8.3f} {avg_err_tot:>8.3f} "
            f"{tot_pts:>6}"
        )

    rankings.sort()
    print("\n  RANKING (lowest total error):")
    for rank, (err, algo) in enumerate(rankings, 1):
        print(f"    #{rank}  {algo:<16}  avg error = {err:.3f} px/frame")
    print("=" * len(hdr) + "\n")


# ═══════════════════════════════════════════════════════════════════════
#  CSV export
# ═══════════════════════════════════════════════════════════════════════


def export_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV saved to: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    dofs = sys.argv[1:] if len(sys.argv) > 1 else list(GROUND_TRUTHS.keys())

    all_scores: list[dict] = []

    for dof_name in dofs:
        print(f"\n{'#' * 60}")
        print(f"  DoF: {dof_name}")
        print(f"{'#' * 60}")
        for algo_name, run_fn in TRACKERS:
            print(f"\n  Running {algo_name} on {dof_name} ...")
            result = run_fn(dof_name)
            all_scores.append(score(result))

    print_table(all_scores)
    export_csv(all_scores, TRACK_OUTPUT / "benchmark.csv")


if __name__ == "__main__":
    main()
