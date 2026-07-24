"""
analysis/parameter_sweep.py
============================
Sweeps over a grid of (ambulances, bed_scale_factor, scoring_mode) for both
BASELINE and LOCKED modes, using a small number of seeds per cell to keep
runtime reasonable.

Outputs:
  - results/parameter_sweep.csv         (full raw results)
  - results/sweep_overcommit.png        (overcommit vs ambulance count, paper motivation)
  - results/sweep_lock_swaps.png        (lock swaps vs ambulance count, normalized mode)

Run:
    PYTHONPATH=. py analysis/parameter_sweep.py --seeds-per-cell 5
"""

import argparse
import os
import sys
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.multi_ambulance_sim import generate_calls, run_simulation

# Default sweep grid
AMBULANCE_COUNTS = [5, 15, 30, 50]
BED_SCALES       = [0.02, 0.05, 0.10]
SCORING_MODES    = ["original", "normalized"]
MODES            = ["BASELINE", "LOCKED"]

METRIC_KEYS = [
    "overcommit_events",
    "mean_time_to_admission",
    "p95_time_to_admission",
    "mean_waiting_time",
    "lock_swaps",
]


def run_sweep(
    ambulance_counts: list,
    bed_scales: list,
    scoring_modes: list,
    seeds_per_cell: int,
    re_eval: float,
) -> pd.DataFrame:
    """
    Runs the full parameter grid and returns a DataFrame with one row
    per (ambulances, bed_scale, scoring_mode, mode, seed).
    """
    records = []
    grid = list(itertools.product(ambulance_counts, bed_scales, scoring_modes))
    total_cells = len(grid)

    for cell_idx, (n_amb, bed_scale, scoring_mode) in enumerate(grid, 1):
        print(
            f"[Cell {cell_idx}/{total_cells}] "
            f"ambulances={n_amb}, bed_scale={bed_scale}, scoring_mode='{scoring_mode}'"
        )
        for seed in range(seeds_per_cell):
            calls = generate_calls(n_amb, seed=seed)
            for mode in MODES:
                result = run_simulation(
                    calls=calls,
                    mode=mode,
                    bed_scale_factor=bed_scale,
                    re_eval_interval=re_eval,
                    scoring_mode=scoring_mode,
                )
                row = {
                    "ambulances":    n_amb,
                    "bed_scale":     bed_scale,
                    "scoring_mode":  scoring_mode,
                    "mode":          mode,
                    "seed":          seed,
                }
                for m in METRIC_KEYS:
                    row[m] = result[m]
                records.append(row)
            print(f"  seed {seed} done")

    return pd.DataFrame(records)


def plot_overcommit(df: pd.DataFrame, out_path: str) -> None:
    """
    Line plot: BASELINE overcommit_events vs ambulance count,
    one line per bed_scale_factor (original scoring only).
    This is the paper-motivation plot showing the problem grows with volume.
    """
    sub = df[(df["mode"] == "BASELINE") & (df["scoring_mode"] == "original")]
    grouped = sub.groupby(["ambulances", "bed_scale"])["overcommit_events"]
    means = grouped.mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = cm.plasma(np.linspace(0.15, 0.85, len(BED_SCALES)))

    for color, bed_scale in zip(colors, sorted(means["bed_scale"].unique())):
        subset = means[means["bed_scale"] == bed_scale].sort_values("ambulances")
        ax.plot(subset["ambulances"], subset["overcommit_events"],
                marker="o", linewidth=2.2, markersize=7,
                color=color, label=f"Bed scale = {bed_scale}")

    ax.set_xlabel("Number of Ambulances", fontsize=12)
    ax.set_ylabel("Over-Commitment Events (mean across seeds)", fontsize=12)
    ax.set_title(
        "Baseline Over-Commitment Events vs. Ambulance Volume\n"
        "(motivation: locking prevents these entirely)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(title="Bed Capacity Scale")
    ax.grid(linestyle="--", alpha=0.5)
    ax.set_xticks(sorted(means["ambulances"].unique()))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Overcommit motivation plot saved to: {out_path}")


def plot_lock_swaps(df: pd.DataFrame, out_path: str) -> None:
    """
    Line plot: LOCKED lock_swaps vs ambulance count under normalized scoring,
    one line per bed_scale_factor - confirms swaps increase once more hospitals
    are viable candidates.
    """
    sub = df[(df["mode"] == "LOCKED") & (df["scoring_mode"] == "normalized")]
    grouped = sub.groupby(["ambulances", "bed_scale"])["lock_swaps"]
    means = grouped.mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = cm.viridis(np.linspace(0.15, 0.85, len(BED_SCALES)))

    for color, bed_scale in zip(colors, sorted(means["bed_scale"].unique())):
        subset = means[means["bed_scale"] == bed_scale].sort_values("ambulances")
        ax.plot(subset["ambulances"], subset["lock_swaps"],
                marker="s", linewidth=2.2, markersize=7,
                color=color, label=f"Bed scale = {bed_scale}")

    ax.set_xlabel("Number of Ambulances", fontsize=12)
    ax.set_ylabel("Lock Swaps (mean across seeds)", fontsize=12)
    ax.set_title(
        "Lock Swaps vs. Ambulance Volume - Normalized Scoring\n"
        "(more hospitals in fallback chain -> more dynamic reroutes)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(title="Bed Capacity Scale")
    ax.grid(linestyle="--", alpha=0.5)
    ax.set_xticks(sorted(means["ambulances"].unique()))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Lock-swaps plot saved to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parameter sweep over ambulances × bed_scale × scoring_mode."
    )
    parser.add_argument("--seeds-per-cell", type=int, default=5,
                        help="Seeds per grid cell (default: 5)")
    parser.add_argument("--re-eval", type=float, default=2.0,
                        help="Route re-evaluation interval in minutes (default: 2.0)")
    parser.add_argument("--ambulances", type=int, nargs="+",
                        default=AMBULANCE_COUNTS,
                        help=f"Ambulance counts to sweep (default: {AMBULANCE_COUNTS})")
    parser.add_argument("--bed-scales", type=float, nargs="+",
                        default=BED_SCALES,
                        help=f"Bed scale factors to sweep (default: {BED_SCALES})")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Parameter Sweep")
    print(f"  ambulances={args.ambulances}")
    print(f"  bed_scales={args.bed_scales}")
    print(f"  scoring_modes={SCORING_MODES}")
    print(f"  seeds_per_cell={args.seeds_per_cell}")
    total_runs = (len(args.ambulances) * len(args.bed_scales)
                  * len(SCORING_MODES) * len(MODES) * args.seeds_per_cell)
    print(f"  Total simulation runs: {total_runs}")
    print(f"{'='*65}\n")

    df = run_sweep(
        ambulance_counts=args.ambulances,
        bed_scales=args.bed_scales,
        scoring_modes=SCORING_MODES,
        seeds_per_cell=args.seeds_per_cell,
        re_eval=args.re_eval,
    )

    # Save full results
    csv_path = os.path.join(results_dir, "parameter_sweep.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nFull sweep results saved to: {csv_path}")

    # Quick aggregate summary print
    print(f"\n{'='*65}")
    print("  AGGREGATE MEAN per (scoring_mode, mode):")
    print(f"{'='*65}")
    agg = df.groupby(["scoring_mode", "mode"])[METRIC_KEYS].mean().round(2)
    print(agg.to_string())
    print(f"{'='*65}\n")

    # Plots
    plot_overcommit(df, os.path.join(results_dir, "sweep_overcommit.png"))
    plot_lock_swaps(df, os.path.join(results_dir, "sweep_lock_swaps.png"))


if __name__ == "__main__":
    main()
