"""Panel (a): B100 cost vs a hypothetical additional consumption penalty.

Pure post-processing, no re-solve. Because the marginal cost coefficient
b_i(x) = A_x * Price_x is identical across all diesel units for a fixed blend,
the committed-unit schedule that minimises cost is invariant to A_x: only the
marginal litres respond. So B100's cost at any hypothetical A_x can be
recomputed directly from its already-solved idle-litre / marginal-kWh split,
with the other blends held at their solved cost. Swept through the
marginal-only reversal threshold.

Output: figures/data for fig_r3 panel (a)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
DAILY_DEMAND_KWH = 102450.0
PENALTY_SWEEP_PCT = np.linspace(0, 100, 201)  # 0% to 100% BSFC penalty at B100, crosses 65.4%


def load_results() -> dict:
    recs = {}
    for basis in BASES:
        for blend in BLENDS:
            for s in SCENARIOS:
                p = RESULTS / f"solve_{basis}_{blend}_{s}.json"
                with open(p) as f:
                    recs[(basis, blend, s)] = json.load(f)
    return recs


def expected_over_scenarios(recs, basis, blend, key_path):
    total = 0.0
    for s in SCENARIOS:
        rec = recs[(basis, blend, s)]
        v = rec
        for k in key_path:
            v = v[k]
        total += rec["scenario_probability"] * v
    return total


def main():
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    recs = load_results()
    costs = pd.read_csv(DATA / "cost_coefficients.csv")
    with open(CODING / "metrics.json") as f:
        metrics = json.load(f)
    fixed_cost = metrics["expected_cost_rp_per_blend"]  # B0-B70 held fixed here

    rows = []
    for basis in BASES:
        price_b0 = costs[(costs.blend == "B0") & (costs.bsfc_basis == basis) &
                          (costs.ongkos_angkut_scenario == "mean") &
                          (costs.genset_group == "mitsubishi")].price_rp_per_l.iloc[0]
        price_b100 = costs[(costs.blend == "B100") & (costs.bsfc_basis == basis) &
                            (costs.ongkos_angkut_scenario == "mean") &
                            (costs.genset_group == "mitsubishi")].price_rp_per_l.iloc[0]
        a_x_b0 = costs[(costs.blend == "B0") & (costs.bsfc_basis == basis) &
                        (costs.ongkos_angkut_scenario == "mean") &
                        (costs.genset_group == "mitsubishi")].b_i_marginal_rp_per_kwh.iloc[0] / price_b0

        exp_idle_b100 = expected_over_scenarios(recs, basis, "B100", ["fuel_litres", "idle"])
        exp_marginal_litres_b100_actual = expected_over_scenarios(recs, basis, "B100", ["fuel_litres", "marginal"])
        a_x_b100_actual = expected_over_scenarios(recs, basis, "B100", ["fuel_litres", "A_x_L_per_kWh"])
        # marginal kWh is A_x-invariant (energy balance fixes it) -- recover it
        # by dividing the actual solved marginal litres by the actual solved A_x.
        exp_marginal_kwh_b100 = exp_marginal_litres_b100_actual / a_x_b100_actual

        other_costs = {b: fixed_cost[b][basis] for b in BLENDS if b != "B100"}
        cheapest_other_blend = min(other_costs, key=other_costs.get)
        cheapest_other_cost = other_costs[cheapest_other_blend]

        for penalty_pct in PENALTY_SWEEP_PCT:
            a_x_sweep = a_x_b0 * (1 + penalty_pct / 100.0)
            marginal_litres_sweep = a_x_sweep * exp_marginal_kwh_b100
            total_litres_sweep = exp_idle_b100 + marginal_litres_sweep
            cost_b100_sweep = total_litres_sweep * price_b100

            all_costs = dict(other_costs)
            all_costs["B100"] = cost_b100_sweep
            optimal_blend = min(all_costs, key=all_costs.get)

            rows.append({
                "basis": basis,
                "penalty_pct_at_b100": penalty_pct,
                "cost_b100_rp": cost_b100_sweep,
                "cost_b100_rp_per_kwh": cost_b100_sweep / DAILY_DEMAND_KWH,
                "cheapest_alternative_blend": cheapest_other_blend,
                "cheapest_alternative_cost_rp": cheapest_other_cost,
                "optimal_blend": optimal_blend,
                "b100_still_optimal": optimal_blend == "B100",
            })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "tab_r3a_penalty_sweep.csv", index=False)

    # Find the actual crossing threshold per basis (load-factor-corrected, not full-load approx)
    thresholds = {}
    for basis in BASES:
        sub = df[df.basis == basis].sort_values("penalty_pct_at_b100")
        flips = sub[~sub.b100_still_optimal]
        threshold = float(flips.penalty_pct_at_b100.iloc[0]) if len(flips) else None
        thresholds[basis] = threshold
        print(f"{basis:10s}: B100 stops being optimal at penalty = "
              f"{threshold if threshold is not None else '>100% (not reached in sweep)'}%, "
              f"flips to {flips.optimal_blend.iloc[0] if len(flips) else 'n/a'}")

    with open(CODING / "metrics.json") as f:
        metrics = json.load(f)
    metrics.setdefault("reversal_threshold", {})
    metrics["reversal_threshold"]["idle_corrected_pct_by_basis"] = thresholds
    metrics["reversal_threshold"]["idle_corrected_pct_note"] = (
        "load-factor-corrected via sweep, using the ACTUAL realised"
        "idle/marginal split at each blend's solved dispatch -- supersedes "
        "the full-load approximation in tab_r0_thresholds.csv"
    )
    with open(CODING / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nWritten: tables/tab_r3a_penalty_sweep.csv ({len(df)} rows)")
    print("Reference: marginal-only threshold (Tab-R0, B0-vs-B100 only) = 32.5%, "
          "full-load idle-corrected = 40.2%")


if __name__ == "__main__":
    main()
