"""Aggregate the stochastic-layer ablation table.

Post-processing from results/, run after run_sweep_ablation.py. Two separate
comparisons, kept apart:

1. Cost, between the two fully-reliable policies: EVPI = RP - WS >= 0, the
   value of a perfect weather forecast. Both serve all load, so their costs
   are comparable.
2. Reliability: RP and WS serve all load; the expected-weather commitment
   (EEV) can leave energy unserved once evaluated against the true
   scenarios. EEV's diesel cost is reported but is not comparable to RP/WS
   when its unserved energy is positive.

Output: tables/tab_r4_ablation_stochastic.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CODING = Path(__file__).resolve().parent.parent
RESULTS = CODING / "results"
DATA = CODING / "data"
TABLES = CODING / "tables"

BLENDS = ["B0", "B10", "B20", "B30", "B35", "B40", "B50", "B70", "B100"]
PCT = {b: int(b[1:]) for b in BLENDS}
BASES = ["epa", "generator"]
SCENARIOS = ["clear", "partly_cloudy", "overcast"]
DAILY_DEMAND_KWH = 102450.0


class MissingResult(Exception):
    pass


def _load(path: Path) -> dict:
    if not path.exists():
        raise MissingResult(str(path))
    with open(path) as f:
        return json.load(f)


def _weighted(records: dict, probs: dict, field: str) -> float:
    total = 0.0
    for s in SCENARIOS:
        rec = records[s]
        if rec.get("infeasible"):
            raise RuntimeError(
                f"Unexpected infeasible record with no slack coverage: {rec['case']} "
                "-- the unserved-load slack should make this impossible; investigate "
                "before trusting any number here."
            )
        total += probs[s] * rec[field]
    return total


def _rp_records(basis: str, blend: str) -> dict:
    return {s: _load(RESULTS / f"solve_{basis}_{blend}_{s}.json") for s in SCENARIOS}


def _variant_records(prefix: str, basis: str, blend: str) -> dict:
    return {s: _load(RESULTS / f"solve_{prefix}_{basis}_{blend}_{s}.json") for s in SCENARIOS}


def main():
    TABLES.mkdir(exist_ok=True)
    with open(DATA / "scenario_probabilities.json") as f:
        probs = json.load(f)

    rows = []
    missing = []
    for basis in BASES:
        for blend in BLENDS:
            try:
                rp_recs = _rp_records(basis, blend)
                rp_cost = sum(
                    rp_recs[s]["scenario_probability"] * rp_recs[s]["scenario_cost_rp"]
                    for s in SCENARIOS
                )
                ws_recs = _variant_records("pf", basis, blend)
                ws_cost = _weighted(ws_recs, probs, "scenario_cost_rp")
                ws_unserved = _weighted(ws_recs, probs, "unserved_energy_kwh")

                eev_recs = _variant_records("eev", basis, blend)
                eev_cost = _weighted(eev_recs, probs, "scenario_cost_rp")
                eev_unserved = _weighted(eev_recs, probs, "unserved_energy_kwh")
                eev_unserved_by_scenario = {
                    s: eev_recs[s]["unserved_energy_kwh"] for s in SCENARIOS
                }
            except MissingResult as e:
                missing.append(f"{basis}/{blend}: {e}")
                continue

            rows.append({
                "basis": basis,
                "blend": blend,
                "pct_b": PCT[blend],
                "rp_expected_cost_rp": rp_cost,
                "ws_expected_cost_rp": ws_cost,
                "ws_expected_unserved_kwh": ws_unserved,
                "evpi_rp_minus_ws_rp": rp_cost - ws_cost,
                "eev_expected_cost_rp": eev_cost,
                "eev_expected_unserved_kwh": eev_unserved,
                "eev_unserved_pct_of_daily_demand": 100.0 * eev_unserved / DAILY_DEMAND_KWH,
                "eev_unserved_clear_kwh": eev_unserved_by_scenario["clear"],
                "eev_unserved_partly_cloudy_kwh": eev_unserved_by_scenario["partly_cloudy"],
                "eev_unserved_overcast_kwh": eev_unserved_by_scenario["overcast"],
            })

    if missing:
        print(f"=== {len(missing)} case(s) missing -- run run_sweep_ablation.py in Colab first ===")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    if not rows:
        print("No cases available yet. Nothing written.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "tab_r4_ablation_stochastic.csv", index=False)

    print("\n=== Tab-R4: stochastic-layer ablation ===")
    show = df[["basis", "blend", "rp_expected_cost_rp", "ws_expected_cost_rp",
               "evpi_rp_minus_ws_rp", "eev_expected_unserved_kwh",
               "eev_unserved_pct_of_daily_demand"]].copy()
    show["rp_expected_cost_rp"] = show["rp_expected_cost_rp"].map(lambda v: f"{v:,.0f}")
    show["ws_expected_cost_rp"] = show["ws_expected_cost_rp"].map(lambda v: f"{v:,.0f}")
    show["evpi_rp_minus_ws_rp"] = show["evpi_rp_minus_ws_rp"].map(lambda v: f"{v:,.0f}")
    show["eev_expected_unserved_kwh"] = show["eev_expected_unserved_kwh"].map(lambda v: f"{v:,.1f}")
    show["eev_unserved_pct_of_daily_demand"] = show["eev_unserved_pct_of_daily_demand"].map(
        lambda v: f"{v:.3f}%"
    )
    print(show.to_string(index=False))

    print(f"\nMean EVPI (RP vs WS, both 100% reliable): "
          f"{df.evpi_rp_minus_ws_rp.mean():,.0f} Rp/day "
          f"({100.0*df.evpi_rp_minus_ws_rp.mean()/df.rp_expected_cost_rp.mean():.3f}% of RP cost)")
    print(f"Mean EEV unserved energy (naive expected-weather planner): "
          f"{df.eev_expected_unserved_kwh.mean():,.1f} kWh/day "
          f"({df.eev_unserved_pct_of_daily_demand.mean():.3f}% of daily demand)")
    print(f"RP and WS: 0 kWh unserved in all {len(df)} cases (by construction -- "
          f"full commitment freedom in both).")
    n_eev_unreliable = int((df.eev_expected_unserved_kwh > 1e-6).sum())
    print(f"EEV cases with nonzero unserved energy: {n_eev_unreliable} of {len(df)}")

    with open(CODING / "metrics.json") as f:
        metrics = json.load(f)
    metrics["stochastic_ablation"] = {
        "mean_evpi_rp": float(df.evpi_rp_minus_ws_rp.mean()),
        "mean_evpi_pct_of_rp": float(100.0 * df.evpi_rp_minus_ws_rp.mean() / df.rp_expected_cost_rp.mean()),
        "mean_eev_unserved_kwh": float(df.eev_expected_unserved_kwh.mean()),
        "mean_eev_unserved_pct_of_demand": float(df.eev_unserved_pct_of_daily_demand.mean()),
        "n_cases_eev_unreliable": n_eev_unreliable,
        "n_cases_total": len(df),
        "n_cases_missing": len(missing),
        "note": "No VOLL-based VSS-in-Rupiah figure computed -- no such reference "
                "exists in this paper's ledger. Reliability gap reported in kWh instead.",
    }
    with open(CODING / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nWritten: tables/tab_r4_ablation_stochastic.csv")


if __name__ == "__main__":
    main()
