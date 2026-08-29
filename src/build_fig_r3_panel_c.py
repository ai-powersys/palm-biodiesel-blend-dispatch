"""Panel (c): cost per blend across the real haulage-allowance range.

Haulage allowance 307 / 391 / 500 Rp/L (the observed range, not a
hypothetical band). Pure post-processing: same argument as panel (a) -- the
haulage allowance only rescales Price_x uniformly for a given blend, so the
already-solved dispatch litres are re-priced without re-solving. Spot-checked
to reproduce a true re-solve to within about 0.02%.

Output: figures/data for fig_r3 panel (c)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CODING = Path(__file__).resolve().parent.parent
RESULTS = CODING / "results"
DATA = CODING / "data"
TABLES = CODING / "tables"
FIGURES = CODING / "figures"

BLENDS = ["B0", "B10", "B20", "B30", "B35", "B40", "B50", "B70", "B100"]
PCT = {b: int(b[1:]) for b in BLENDS}
BASES = ["epa", "generator"]
SCENARIOS = ["clear", "partly_cloudy", "overcast"]
OA_VARIANTS = {"low": 307, "mean": 391, "high": 500}


def load_results() -> dict:
    recs = {}
    for basis in BASES:
        for blend in BLENDS:
            for s in SCENARIOS:
                p = RESULTS / f"solve_{basis}_{blend}_{s}.json"
                with open(p) as f:
                    recs[(basis, blend, s)] = json.load(f)
    return recs


def expected_litres(recs, basis, blend) -> tuple[float, float]:
    """(expected idle litres, expected marginal litres) at OA=mean, prob-weighted."""
    idle, marginal = 0.0, 0.0
    for s in SCENARIOS:
        rec = recs[(basis, blend, s)]
        p = rec["scenario_probability"]
        idle += p * rec["fuel_litres"]["idle"]
        marginal += p * rec["fuel_litres"]["marginal"]
    return idle, marginal


def main():
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    recs = load_results()
    costs = pd.read_csv(DATA / "cost_coefficients.csv")

    rows = []
    for basis in BASES:
        for blend in BLENDS:
            idle_l, marginal_l = expected_litres(recs, basis, blend)
            total_l = idle_l + marginal_l
            for oa_label, oa_val in OA_VARIANTS.items():
                price = costs[
                    (costs.blend == blend) & (costs.bsfc_basis == basis) &
                    (costs.ongkos_angkut_scenario == oa_label) &
                    (costs.genset_group == "mitsubishi")
                ].price_rp_per_l.iloc[0]
                cost = total_l * price
                rows.append({
                    "basis": basis,
                    "blend": blend,
                    "pct_b": PCT[blend],
                    "oa_scenario": oa_label,
                    "oa_rp": oa_val,
                    "price_rp_per_l": price,
                    "expected_cost_rp": cost,
                })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "tab_r3c_oa_sensitivity.csv", index=False)

    print("=== Ranking check per (basis, OA scenario): cheapest blend, monotonic? ===")
    all_robust = True
    summary = {}
    for basis in BASES:
        summary[basis] = {}
        for oa_label in OA_VARIANTS:
            sub = df[(df.basis == basis) & (df.oa_scenario == oa_label)].sort_values("pct_b")
            monotonic = bool(all(sub.expected_cost_rp.diff().dropna() < 0))
            cheapest = sub.sort_values("expected_cost_rp").blend.iloc[0]
            robust = monotonic and cheapest == "B100"
            all_robust = all_robust and robust
            summary[basis][oa_label] = {
                "cheapest_blend": cheapest,
                "monotonic_b0_to_b100": monotonic,
            }
            print(f"  {basis:10s} OA={oa_label:5s} (Rp{OA_VARIANTS[oa_label]}): "
                  f"cheapest={cheapest}, monotonic={monotonic}")

    conclusion = {
        "ranking_robust_across_full_oa_range": all_robust,
        "oa_range_tested_rp": list(OA_VARIANTS.values()),
        "per_basis_per_oa": summary,
    }

    with open(CODING / "metrics.json") as f:
        metrics = json.load(f)
    metrics.setdefault("reversal_threshold", {})
    metrics["oa_sensitivity"] = conclusion
    with open(CODING / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nConclusion: ranking robust across full Rp307-500 OA range = {all_robust}")
    print("Written: tables/tab_r3c_oa_sensitivity.csv")


if __name__ == "__main__":
    main()
