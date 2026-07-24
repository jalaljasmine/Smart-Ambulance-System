"""
analysis/multi_seed_runner.py
==============================
Runs BASELINE and LOCKED simulations across multiple random seeds and
reports aggregate statistics (mean ± std, min, max) per mode.

Outputs:
  - Printed summary table
  - results/multi_seed_summary.csv
  - results/multi_seed_comparison.png  (error-bar + box plot grid)

Run:
    PYTHONPATH=. py analysis/multi_seed_runner.py --seeds 20 --ambulances 15 --bed-scale 0.05
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.multi_ambulance_sim import generate_calls, run_simulation

# Metrics collected per run
METRICS = [
    "overcommit_events",
    "mean_time_to_admission",
    "p95_time_to_admission",
    "mean_waiting_time",
    "lock_swaps",
]

METRIC_LABELS = {
    "overcommit_events":       "Over-Commitment Events",
    "mean_time_to_admission":  "Mean Time-to-Admission (min)",
    "p95_time_to_admission":   "p95 Time-to-Admission (min)",
    "mean_waiting_time":       "Mean Waiting Time (min)",
    "lock_swaps":              "Lock Swaps (Dynamic Reroutes)",
}


def run_multi_seed(
    n_seeds: int,
    ambulances: int,
    bed_scale: float,
    re_eval: float,
    scoring_mode: str,
    seed_start: int = 0,
) -> pd.DataFrame:
    """
    Runs both modes across ``n_seeds`` seeds starting from ``seed_start``.

    Returns:
        pd.DataFrame with columns: seed, mode, + each metric in METRICS.
    """
    records = []
    for i in range(n_seeds):
        seed = seed_start + i
        print(f"  Seed {seed:>3d} ({i+1}/{n_seeds}) ...", end=" ", flush=True)
        calls = generate_calls(ambulances, seed=seed)

        for mode in ("BASELINE", "LOCKED"):
            result = run_simulation(
                calls=calls,
                mode=mode,
                bed_scale_factor=bed_scale,
                re_eval_interval=re_eval,
                scoring_mode=scoring_mode,
            )
            row = {"seed": seed, "mode": mode}
            for m in METRICS:
                row[m] = result[m]
            records.append(row)
        print("done")

    return pd.DataFrame(records)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Returns mean ± std and min/max per mode per metric."""
    rows = []
    for mode in ("BASELINE", "LOCKED"):
        sub = df[df["mode"] == mode]
        row = {"Mode": mode}
        for m in METRICS:
            vals = sub[m].values
            row[METRIC_LABELS[m] + " mean"] = f"{np.mean(vals):.2f}"
            row[METRIC_LABELS[m] + " +/-std"]  = f"{np.std(vals):.2f}"
            row[METRIC_LABELS[m] + " min"]   = f"{np.min(vals):.2f}"
            row[METRIC_LABELS[m] + " max"]   = f"{np.max(vals):.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def plot_multi_seed(df: pd.DataFrame, out_path: str, scoring_mode: str) -> None:
    """
    Saves a figure with one subplot per metric, showing error bars
    (mean ± std) for BASELINE vs LOCKED, plus strip/jitter dots.
    """
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    fig.suptitle(
        f"Multi-Seed Simulation Results  |  scoring_mode='{scoring_mode}'\n"
        f"({df['seed'].nunique()} seeds, {df['mode'].nunique()} modes)",
        fontsize=13, fontweight="bold"
    )

    colors = {"BASELINE": "#e03131", "LOCKED": "#2f9e44"}

    for ax, metric in zip(axes, METRICS):
        label = METRIC_LABELS[metric]
        for x_pos, mode in enumerate(["BASELINE", "LOCKED"]):
            vals = df[df["mode"] == mode][metric].values
            mean, std = np.mean(vals), np.std(vals)
            color = colors[mode]

            # Error bar
            ax.errorbar(x_pos, mean, yerr=std, fmt="o", color=color,
                        markersize=9, capsize=6, linewidth=2, zorder=3)

            # Jitter dots
            jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(x_pos + jitter, vals, color=color, alpha=0.35,
                       s=20, zorder=2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Baseline", "Locked"], fontsize=9)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_ylabel("Value")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"\nPlot saved to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed simulation runner for Smart Ambulance System."
    )
    parser.add_argument("--seeds",       type=int,   default=20,
                        help="Number of random seeds to evaluate (default: 20)")
    parser.add_argument("--ambulances",  type=int,   default=15,
                        help="Number of ambulance calls per seed (default: 15)")
    parser.add_argument("--bed-scale",   type=float, default=0.05,
                        help="Hospital bed capacity scale factor (default: 0.05)")
    parser.add_argument("--re-eval",     type=float, default=2.0,
                        help="Route re-evaluation interval in minutes (default: 2.0)")
    parser.add_argument("--scoring-mode", type=str,  default="original",
                        choices=["original", "normalized"],
                        help="Hospital scoring mode (default: original)")
    parser.add_argument("--seed-start",  type=int,   default=0,
                        help="Starting seed value (default: 0)")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Multi-Seed Runner")
    print(f"  seeds={args.seeds}, ambulances={args.ambulances}, "
          f"bed_scale={args.bed_scale}, scoring_mode='{args.scoring_mode}'")
    print(f"{'='*60}\n")

    df = run_multi_seed(
        n_seeds=args.seeds,
        ambulances=args.ambulances,
        bed_scale=args.bed_scale,
        re_eval=args.re_eval,
        scoring_mode=args.scoring_mode,
        seed_start=args.seed_start,
    )

    # Save raw results
    csv_path = os.path.join(results_dir, "multi_seed_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nRaw results saved to: {csv_path}")

    # Print summary statistics
    summary = compute_summary(df)
    print(f"\n{'='*60}")
    print("  SUMMARY STATISTICS (mean +/- std  |  min - max)")
    print(f"{'='*60}")
    for mode in ("BASELINE", "LOCKED"):
        row = summary[summary["Mode"] == mode].iloc[0]
        print(f"\n  [{mode}]")
        for m in METRICS:
            label = METRIC_LABELS[m]
            mean = row[label + " mean"]
            std  = row[label + " +/-std"]
            lo   = row[label + " min"]
            hi   = row[label + " max"]
            print(f"    {label:<40s}: {mean} +/- {std}  [{lo} - {hi}]")
    print(f"\n{'='*60}")

    # Plot
    plot_path = os.path.join(results_dir, "multi_seed_comparison.png")
    plot_multi_seed(df, plot_path, scoring_mode=args.scoring_mode)


if __name__ == "__main__":
    main()
