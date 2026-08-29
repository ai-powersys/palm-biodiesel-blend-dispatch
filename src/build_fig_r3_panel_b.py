"""Panel (b): cost per blend under minimum up/down time T = 1, 2, 3 h.

T=1 is the free-cycling case from the main sweep; T=2 and T=3 come from
run_sweep_panel_b.py. Free cycling raises the marginal-fuel share and favours
low blends; constrained cycling raises the idle-fuel share and favours high
blends. Shows the B0-to-B100 ranking is unchanged for every T.

Output: figures/data for fig_r3 panel (b)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CODING = Path(__file__).resolve().parent.parent
RESULTS = CODING / "results"
TABLES = CODING / "tables"

BLENDS = ["B0", "B10", "B20", "B30", "B35", "B40", "B50", "B70", "B100"]
PCT = {b: int(b[1:]) for b in BLENDS}
BASES = ["epa", "generator"]
SCENARIOS = ["clear", "partly_cloudy", "overcast"]
MUT_VALUES = [1, 2, 3]


def _path(basis: str, blend: str, scenario: str, mut: int) -> Path:
    suffix = "" if mut == 1 else f"_T{mut}"
    return RESULTS / f"solve_{basis}_{blend}_{scenario}{suffix}.json"


def load_all() -> dict:
    recs = {}
    for mut in MUT_VALUES:
        for basis in BASES:
            for blend in BLENDS:
                for s in SCENARIOS:
                    p = _path(basis, blend, s, mut)
                    if not p.exists():
                        raise FileNotFoundError(f"missing checkpoint: {p}")
                    with open(p) as f:
                        recs[(mut, basis, blend, s)] = json.load(f)
    return recs


def build_table(recs: dict) -> pd.DataFrame:
    rows = []
    for mut in MUT_VALUES:
        for basis in BASES:
            for blend in BLENDS:
                probs = {s: recs[(mut, basis, blend, s)]["scenario_probability"] for s in SCENARIOS}
                expected = recs[(mut, basis, blend, SCENARIOS[0])]["expected_objective_all_scenarios_rp"]
                idle_l = sum(probs[s] * recs[(mut, basis, blend, s)]["fuel_litres"]["idle"] for s in SCENARIOS)
                marg_l = sum(probs[s] * recs[(mut, basis, blend, s)]["fuel_litres"]["marginal"] for s in SCENARIOS)
                exp_hours = sum(
                    probs[s] * recs[(mut, basis, blend, s)]["committed_unit_hours"] for s in SCENARIOS
                )
                solver = recs[(mut, basis, blend, SCENARIOS[0])]["solver"]
                rows.append({
                    "min_up_down_h": mut,
                    "basis": basis,
                    "blend": blend,
                    "pct_b": PCT[blend],
                    "expected_cost_rp": expected,
                    "idle_litre_share_pct": 100.0 * idle_l / (idle_l + marg_l),
                    "expected_committed_genset_hours": exp_hours,
                    "solver_termination": solver["termination_condition"],
                    "achieved_mip_gap": solver["achieved_mip_gap"],
                    "terminated_early": solver["terminated_early"],
                })
    return pd.DataFrame(rows)


def robustness_check(df: pd.DataFrame) -> dict:
    summary = {}
    all_robust = True
    for mut in MUT_VALUES:
        summary[mut] = {}
        for basis in BASES:
            sub = df[(df.min_up_down_h == mut) & (df.basis == basis)].sort_values("pct_b")
            monotonic = bool(all(sub.expected_cost_rp.diff().dropna() < 0))
            cheapest = sub.sort_values("expected_cost_rp").blend.iloc[0]
            robust = monotonic and cheapest == "B100"
            all_robust = all_robust and robust
            summary[mut][basis] = {"cheapest_blend": cheapest, "monotonic_b0_to_b100": monotonic}
    return {"ranking_robust_across_full_t_range": all_robust, "t_values_tested_h": MUT_VALUES, "per_t_per_basis": summary}


def gap_flags(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.terminated_early].copy()


def main():
    TABLES.mkdir(exist_ok=True)
    recs = load_all()
    df = build_table(recs)
    df.to_csv(TABLES / "tab_r3b_sensitivity.csv", index=False)

    print("=== Panel (b): expected cost per blend, per T (min_up_time=min_down_time) ===")
    pd.set_option("display.width", 200)
    show = df[["min_up_down_h", "basis", "blend", "expected_cost_rp", "idle_litre_share_pct",
               "solver_termination", "achieved_mip_gap"]].copy()
    show["expected_cost_rp"] = show["expected_cost_rp"].map(lambda v: f"{v:,.0f}")
    show["idle_litre_share_pct"] = show["idle_litre_share_pct"].map(lambda v: f"{v:.1f}")
    show["achieved_mip_gap"] = show["achieved_mip_gap"].map(lambda v: f"{v:.2e}")
    print(show.to_string(index=False))

    early = gap_flags(df)
    print(f"\n=== {len(early)} case(s) hit time_limit (not proven optimal at 5e-4) ===")
    if len(early):
        print(early[["min_up_down_h", "basis", "blend", "achieved_mip_gap"]].to_string(index=False))

    max_gap = float(df["achieved_mip_gap"].max())
    max_obj = float(df["expected_cost_rp"].max())
    tolerance = max_gap * max_obj
    smallest = None
    for mut in MUT_VALUES:
        for basis in BASES:
            sub = df[(df.min_up_down_h == mut) & (df.basis == basis)].sort_values("pct_b")
            diffs = sub["expected_cost_rp"].diff().abs().dropna()
            m = float(diffs.min())
            if smallest is None or m < smallest:
                smallest = m
    margin_factor = smallest / tolerance if tolerance else float("inf")
    print(f"\n=== MIP-gap margin check (worst-case achieved gap across all T) ===")
    print(f"  max_achieved_mip_gap: {max_gap:.3e}")
    print(f"  absolute_tolerance_rp: {tolerance:,.0f}")
    print(f"  smallest_adjacent_blend_difference_rp: {smallest:,.0f}")
    print(f"  margin_factor: {margin_factor:.1f}x")
    print(f"  margin_check_passed: {margin_factor > 1}")

    check = robustness_check(df)
    print("\n=== Ranking robustness across T in {1,2,3} ===")
    for mut in MUT_VALUES:
        for basis in BASES:
            r = check["per_t_per_basis"][mut][basis]
            print(f"  T={mut}h {basis:10s}: cheapest={r['cheapest_blend']}, monotonic={r['monotonic_b0_to_b100']}")
    print(f"\nConclusion: ranking robust across T=1,2,3h = {check['ranking_robust_across_full_t_range']}")

    with open(CODING / "metrics.json") as f:
        metrics = json.load(f)
    metrics["t_sensitivity"] = check
    metrics["t_sensitivity"]["mip_gap_margin_check"] = {
        "max_achieved_mip_gap": max_gap,
        "absolute_tolerance_rp": tolerance,
        "smallest_adjacent_blend_difference_rp": smallest,
        "margin_factor": margin_factor,
        "margin_check_passed": margin_factor > 1,
        "cases_hit_time_limit": len(early),
    }
    with open(CODING / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nWritten: tables/tab_r3b_sensitivity.csv")


if __name__ == "__main__":
    main()
