"""
analysis/scoring_diagnostics.py
================================
Quantifies the hospital-ranking bias in the two scoring modes by running
get_sorted_hospitals() across a batch of generated calls and reporting:

  - How many times each hospital appears in top-1 / top-3 / top-5
  - Mean rank position across all calls
  - Side-by-side comparison between "original" and "normalized" modes

Run:
    PYTHONPATH=. py analysis/scoring_diagnostics.py --calls 50 --seed 42
"""

import argparse
import os
import sys

import pandas as pd

# Allow running from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.multi_ambulance_sim import generate_calls, _cached_graph  # triggers graph load & cache
from models.predict_severity import predict
from models.hospital_score import get_sorted_hospitals


def run_ranking_diagnostic(
    calls_list: list,
    scoring_mode: str = "original",
) -> pd.DataFrame:
    """
    Runs get_sorted_hospitals() for each call and records where each
    hospital lands in the ranked list.

    Args:
        calls_list: List of call dicts from generate_calls().
        scoring_mode: ``"original"`` or ``"normalized"``.

    Returns:
        pd.DataFrame with per-hospital appearance counts and mean rank.
    """
    from models.hospital_score import get_sorted_hospitals

    # Collect hospital names from first call to initialise counters
    first_sev = predict(calls_list[0]["features"])
    hospital_names = list(
        get_sorted_hospitals(first_sev, calls_list[0]["start_coords"],
                             scoring_mode=scoring_mode)["Hospital"]
    )

    # Counters
    top1: dict = {h: 0 for h in hospital_names}
    top3: dict = {h: 0 for h in hospital_names}
    top5: dict = {h: 0 for h in hospital_names}
    rank_sum: dict = {h: 0 for h in hospital_names}
    total_calls = len(calls_list)

    for call in calls_list:
        severity = predict(call["features"])
        ranked = get_sorted_hospitals(
            severity, call["start_coords"], scoring_mode=scoring_mode
        )
        ranked_names = list(ranked["Hospital"])
        for pos, h in enumerate(ranked_names):
            rank_sum[h] = rank_sum.get(h, 0) + (pos + 1)  # 1-indexed rank
            if pos == 0:
                top1[h] = top1.get(h, 0) + 1
            if pos < 3:
                top3[h] = top3.get(h, 0) + 1
            if pos < 5:
                top5[h] = top5.get(h, 0) + 1

    rows = []
    for h in hospital_names:
        rows.append({
            "Hospital": h,
            "Top-1 Appearances": top1.get(h, 0),
            "Top-3 Appearances": top3.get(h, 0),
            "Top-5 Appearances": top5.get(h, 0),
            "Top-1 %": f"{top1.get(h, 0) / total_calls * 100:.1f}%",
            "Top-3 %": f"{top3.get(h, 0) / total_calls * 100:.1f}%",
            "Mean Rank": f"{rank_sum.get(h, 0) / total_calls:.2f}",
        })

    df = pd.DataFrame(rows).sort_values("Top-3 Appearances", ascending=False)
    return df


def print_table(df: pd.DataFrame, title: str) -> None:
    """Pretty-prints a DataFrame as a bordered ASCII table."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    col_widths = {col: max(len(str(col)), df[col].astype(str).map(len).max()) + 2
                  for col in df.columns}
    header = "".join(f"{col:<{col_widths[col]}}" for col in df.columns)
    sep = "-" * len(header)
    print(header)
    print(sep)
    for _, row in df.iterrows():
        print("".join(f"{str(row[col]):<{col_widths[col]}}" for col in df.columns))
    print(f"{'='*70}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose hospital ranking bias across scoring modes."
    )
    parser.add_argument("--calls", type=int, default=50,
                        help="Number of calls to simulate (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for call generation (default: 42)")
    args = parser.parse_args()

    print(f"\nGenerating {args.calls} calls with seed={args.seed}...")
    calls_list = generate_calls(args.calls, seed=args.seed)
    print(f"Generated {len(calls_list)} calls across the road network.\n")

    # --- Original scoring ---
    print("Running ranking diagnostic: ORIGINAL scoring mode...")
    df_orig = run_ranking_diagnostic(calls_list, scoring_mode="original")
    print_table(df_orig, f"ORIGINAL Scoring - Hospital Ranking Distribution ({args.calls} calls)")

    # Count hospitals appearing in top-3 at least once
    orig_top3_active = (df_orig["Top-3 Appearances"].astype(int) > 0).sum()
    print(f"\n  Hospitals appearing in top-3 at least once: {orig_top3_active} / {len(df_orig)}")

    # --- Normalized scoring ---
    print("\nRunning ranking diagnostic: NORMALIZED scoring mode...")
    df_norm = run_ranking_diagnostic(calls_list, scoring_mode="normalized")
    print_table(df_norm, f"NORMALIZED Scoring - Hospital Ranking Distribution ({args.calls} calls)")

    norm_top3_active = (df_norm["Top-3 Appearances"].astype(int) > 0).sum()
    print(f"\n  Hospitals appearing in top-3 at least once: {norm_top3_active} / {len(df_norm)}")

    # --- Summary comparison ---
    print(f"\n{'='*70}")
    print("  BIAS REDUCTION SUMMARY")
    print(f"{'='*70}")
    print(f"  Original mode   - {orig_top3_active}/10 hospitals ever reach top-3")
    print(f"  Normalized mode - {norm_top3_active}/10 hospitals ever reach top-3")
    improvement = norm_top3_active - orig_top3_active
    if improvement > 0:
        print(f"  [+] Normalized mode activates {improvement} more hospital(s) in the fallback chain.")
    else:
        print(f"  [i] No net gain in top-3 coverage (try --calls 100 for more varied locations).")
        print(f"      This may indicate geographic clustering: central hospitals are genuinely")
        print(f"      closer to most call locations regardless of scoring formula.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
