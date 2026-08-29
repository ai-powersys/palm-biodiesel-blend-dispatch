"""Expected daily emissions per blend (five species, two bases, tailpipe).

Rebuilds from results/ (fuel litres per solve) combined with
data/emission_factors.csv. Emissions are reported, never optimised: every
solve minimises cost only, and this script translates the solved fuel volumes
into emission mass. Includes the cost-emission pairs for the trade-off
discussion.

Output: tables/tab_r2_emissions.csv
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

BLENDS = ["B0", "B10", "B20", "B30", "B35", "B40", "B50", "B70", "B100"]
PCT = {b: int(b[1:]) for b in BLENDS}
BASES = ["epa", "generator"]
SCENARIOS = ["clear", "partly_cloudy", "overcast"]
SPECIES = ["CO2", "NOx", "CO", "THC", "PM"]


def load_results() -> dict:
    recs = {}
    for basis in BASES:
        for blend in BLENDS:
            for s in SCENARIOS:
                p = RESULTS / f"solve_{basis}_{blend}_{s}.json"
                if not p.exists():
                    raise FileNotFoundError(f"missing checkpoint: {p}")
                with open(p) as f:
                    recs[(basis, blend, s)] = json.load(f)
    return recs


def main():
    TABLES.mkdir(exist_ok=True)
    recs = load_results()
    ef = pd.read_csv(DATA / "emission_factors.csv").set_index(["blend", "bsfc_basis"])
    with open(CODING / "metrics.json") as f:
        metrics = json.load(f)
    cost_by_blend = metrics["expected_cost_rp_per_blend"]

    rows = []
    for basis in BASES:
        for blend in BLENDS:
            probs = {s: recs[(basis, blend, s)]["scenario_probability"] for s in SCENARIOS}
            litres_per_scen = {s: recs[(basis, blend, s)]["fuel_litres"]["total"] for s in SCENARIOS}
            exp_litres = sum(probs[s] * litres_per_scen[s] for s in SCENARIOS)

            factors = ef.loc[(blend, basis)]
            row = {
                "basis": basis,
                "blend": blend,
                "pct_b": PCT[blend],
                "expected_fuel_l_per_day": exp_litres,
                "expected_cost_rp_per_day": cost_by_blend[blend][basis],
            }
            for sp in SPECIES:
                g_per_l = factors[f"{sp}_g_per_L"]
                for s in SCENARIOS:
                    row[f"{sp.lower()}_{s}_kg_per_day"] = litres_per_scen[s] * g_per_l / 1000.0
                row[f"{sp.lower()}_expected_kg_per_day"] = exp_litres * g_per_l / 1000.0
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "tab_r2_emissions.csv", index=False)

    # Cost-vs-emission pairs (CO2 as the headline trade-off pair; full species in the CSV)
    metrics["emissions_per_blend"] = {
        basis: {
            row["blend"]: {
                f"{sp.lower()}_kg": float(row[f"{sp.lower()}_expected_kg_per_day"]) for sp in SPECIES
            }
            for row in df[df.basis == basis].to_dict("records")
        }
        for basis in BASES
    }
    metrics["task"] = metrics.get("task", "") + ",T14"
    with open(CODING / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.set_option("display.width", 200)
    show = df[["basis", "blend", "expected_cost_rp_per_day"] +
              [f"{sp.lower()}_expected_kg_per_day" for sp in SPECIES]].copy()
    show["expected_cost_rp_per_day"] = show["expected_cost_rp_per_day"].map(lambda v: f"{v:,.0f}")
    for sp in SPECIES:
        col = f"{sp.lower()}_expected_kg_per_day"
        show[col] = show[col].map(lambda v: f"{v:,.1f}")
    print(show.to_string(index=False))

    print("\n=== CO2 trend check (expect: rises with blend -- more litres burned, higher LHV-adjusted CO2) ===")
    for basis in BASES:
        sub = df[df.basis == basis].sort_values("pct_b")
        print(f"  {basis:10s}: B0={sub.co2_expected_kg_per_day.iloc[0]:,.1f} kg -> "
              f"B100={sub.co2_expected_kg_per_day.iloc[-1]:,.1f} kg")

    print("\n=== Cost-vs-CO2 trade-off (both fall or cost falls while CO2 direction varies?) ===")
    for basis in BASES:
        sub = df[df.basis == basis].sort_values("pct_b")
        cost_dir = "falls" if sub.expected_cost_rp_per_day.iloc[-1] < sub.expected_cost_rp_per_day.iloc[0] else "rises"
        co2_dir = "falls" if sub.co2_expected_kg_per_day.iloc[-1] < sub.co2_expected_kg_per_day.iloc[0] else "rises"
        print(f"  {basis:10s}: cost {cost_dir} B0->B100, CO2 {co2_dir} B0->B100")


if __name__ == "__main__":
    main()
