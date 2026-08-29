"""Compare the must-run sensitivity sweep against the free-commitment baseline.

Rebuilds from results/*_mustrun.json plus the free-commitment costs already in
metrics.json. Answers: does forcing the documented base-load unit set
committed for the full day change the B0-to-B100 cost ranking?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CODING = Path(__file__).resolve().parent.parent
RESULTS = CODING / "results"
TABLES = CODING / "tables"

BLENDS = ["B0", "B10", "B20", "B30", "B35", "B40", "B50", "B70", "B100"]
PCT = {b: int(b[1:]) for b in BLENDS}
BASES = ["epa", "generator"]
SCENARIOS = ["clear", "partly_cloudy", "overcast"]


def load_mustrun() -> dict:
    recs = {}
    for basis in BASES:
        for blend in BLENDS:
            for s in SCENARIOS:
                p = RESULTS / f"solve_{basis}_{blend}_{s}_mustrun.json"
                if not p.exists():
                    raise FileNotFoundError(f"missing checkpoint: {p}")
                with open(p) as f:
                    recs[(basis, blend, s)] = json.load(f)
    return recs


def main():
    TABLES.mkdir(exist_ok=True)
    with open(CODING / "metrics.json") as f:
        metrics = json.load(f)
    free_cost = metrics["expected_cost_rp_per_blend"]

    recs = load_mustrun()
    rows = []
    for basis in BASES:
        for blend in BLENDS:
            expected = recs[(basis, blend, SCENARIOS[0])]["expected_objective_all_scenarios_rp"]
            free = free_cost[blend][basis]
            delta_rp = expected - free
            delta_pct = 100 * delta_rp / free
            achieved_gaps = [recs[(basis, blend, s)]["solver"]["achieved_mip_gap"] for s in SCENARIOS]
            all_optimal = all(
                recs[(basis, blend, s)]["solver"]["termination_condition"] == "optimal"
                for s in SCENARIOS
            )
            rows.append({
                "basis": basis,
                "blend": blend,
                "pct_b": PCT[blend],
                "expected_cost_free_rp": free,
                "expected_cost_mustrun_rp": expected,
                "delta_rp": delta_rp,
                "delta_pct": delta_pct,
                "max_achieved_mip_gap": max(achieved_gaps),
                "all_optimal": all_optimal,
            })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "tab_mustrun_sensitivity.csv", index=False)

    print("=== Must-run sensitivity: free-commitment vs the case-study reference-documented base-load must-run ===\n")
    pd.set_option("display.width", 160)
    show = df.copy()
    for c in ["expected_cost_free_rp", "expected_cost_mustrun_rp", "delta_rp"]:
        show[c] = show[c].map(lambda v: f"{v:,.0f}")
    show["delta_pct"] = show["delta_pct"].map(lambda v: f"{v:+.2f}%")
    print(show[["basis", "blend", "expected_cost_free_rp", "expected_cost_mustrun_rp",
                "delta_rp", "delta_pct", "all_optimal"]].to_string(index=False))

    print("\n=== Ranking check (monotonic decreasing B0->B100?) ===")
    for basis in BASES:
        sub = df[df.basis == basis].sort_values("pct_b")
        vals = sub.expected_cost_mustrun_rp.values
        monotonic = bool(all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)))
        cheapest = sub.sort_values("expected_cost_mustrun_rp").blend.iloc[0]
        print(f"  {basis:10s}: monotonic B0->B100 = {monotonic}, cheapest blend = {cheapest}")

    print("\n=== Delta% trend across blend (expect: shrinking at higher blends, since idle litres")
    print("    are blend-invariant in VOLUME but cheaper per litre at high blend) ===")
    for basis in BASES:
        sub = df[df.basis == basis].sort_values("pct_b")
        shrinking = bool(np.all(np.diff(sub.delta_pct.values) <= 1e-9))
        print(f"  {basis:10s}: delta% shrinks monotonically B0->B100 = {shrinking} "
              f"(B0={sub.delta_pct.iloc[0]:+.2f}%, B100={sub.delta_pct.iloc[-1]:+.2f}%)")

    all_optimal_overall = bool(df.all_optimal.all())
    conclusion = {
        "all_18_cases_optimal": all_optimal_overall,
        "ranking_monotonic_epa": bool(
            all(df[df.basis == "epa"].sort_values("pct_b").expected_cost_mustrun_rp.diff().dropna() < 0)
        ),
        "ranking_monotonic_generator": bool(
            all(df[df.basis == "generator"].sort_values("pct_b").expected_cost_mustrun_rp.diff().dropna() < 0)
        ),
        "cheapest_blend_epa": df[df.basis == "epa"].sort_values("expected_cost_mustrun_rp").blend.iloc[0],
        "cheapest_blend_generator": df[df.basis == "generator"].sort_values("expected_cost_mustrun_rp").blend.iloc[0],
        "cost_penalty_range_pct": [float(df.delta_pct.min()), float(df.delta_pct.max())],
    }
    with open(CODING / "metrics_mustrun_sensitivity.json", "w") as f:
        json.dump(conclusion, f, indent=2)
    print(f"\nConclusion written to metrics_mustrun_sensitivity.json: {conclusion}")


if __name__ == "__main__":
    main()
