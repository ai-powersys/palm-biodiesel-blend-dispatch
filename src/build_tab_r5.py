"""Compute per-configuration solver runtime.

Aggregates the wall_clock_s column already logged in every run_summary_*.csv.
No re-solve.

Hardware (CPU cores / RAM / GPU) was not recorded during the runs and is
reported as not recorded rather than guessed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CODING = Path(__file__).resolve().parent.parent
TABLES = CODING / "tables"

SWEEPS = {
    "main": ("run_summary_main.csv", " -- main sweep (RP, 9 blends x 2 bases)"),
    "mustrun": ("run_summary_mustrun.csv", " addendum -- must-run sensitivity (18 cases)"),
    "panel_b": ("run_summary_panel_b.csv", " -- min up/down time sweep, T=2,3 (36 new cases)"),
    "ablation": ("run_summary_ablation.csv", " -- WS + EEV ablation (36 cases)"),
}


def main():
    TABLES.mkdir(exist_ok=True)
    rows = []
    for key, (fname, desc) in SWEEPS.items():
        path = CODING / fname
        if not path.exists():
            print(f"SKIP {key}: {fname} not found")
            continue
        df = pd.read_csv(path)
        rows.append({
            "sweep": key,
            "description": desc,
            "n_cases": len(df),
            "n_optimal": int((df.termination_condition == "optimal").sum()),
            "n_terminated_early": int(df.terminated_early.sum()) if "terminated_early" in df else None,
            "wall_clock_total_s": float(df.wall_clock_s.sum()),
            "wall_clock_mean_s": float(df.wall_clock_s.mean()),
            "wall_clock_median_s": float(df.wall_clock_s.median()),
            "wall_clock_max_s": float(df.wall_clock_s.max()),
            "solver": "HiGHS (via highspy 1.15.1, linopy 0.8.0)",
            "hardware": "NOT RECORDED -- never logged for any Colab run in this project (see module docstring)",
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(TABLES / "tab_r5_compute_cost.csv", index=False)

    print("=== Tab-R5: compute cost per configuration ===")
    show = df_out[["sweep", "n_cases", "n_optimal", "wall_clock_total_s",
                   "wall_clock_mean_s", "wall_clock_max_s"]].copy()
    for c in ["wall_clock_total_s", "wall_clock_mean_s", "wall_clock_max_s"]:
        show[c] = show[c].map(lambda v: f"{v:,.1f}")
    print(show.to_string(index=False))

    total_all = df_out.wall_clock_total_s.sum()
    print(f"\nTotal solver wall-clock across all sweeps: {total_all:,.1f} s "
          f"({total_all/60:.1f} min)")
    print("Hardware: NOT RECORDED for any run (flagged, not fabricated).")
    print("\nWritten: tables/tab_r5_compute_cost.csv")


if __name__ == "__main__":
    main()
