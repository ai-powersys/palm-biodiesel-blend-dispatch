"""Minimum up/down time sweep, T in {1, 2, 3} hours.

The parameter that moves realised load factor: free cycling (T=1) pushes
units toward full load and favours low blends; constrained cycling (T=2, 3)
leaves units idling and favours high blends. This is the study's main
robustness check on the ranking.

T=1 already exists from the main sweep (untagged filenames). This script
solves T=2 and T=3 only (nine blends x two bases x two T-values = 36 solves),
writing T-tagged files:

    results/solve_{basis}_{blend}_{scenario}_T{2,3}.json

Resumable per (basis, blend, T).
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from build_network import build_network, SCENARIOS, SCENARIO_OF, PERIOD_OF, GENSETS
from stochastic_layer import apply_layer

CODING = Path(__file__).resolve().parent.parent
RESULTS = CODING / "results"
DATA = CODING / "data"

BLENDS = ["B0", "B10", "B20", "B30", "B35", "B40", "B50", "B70", "B100"]
BASES = ["epa", "generator"]
OA = "mean"
GAP = 5e-4
TIME_CAP_S = 900  # 
# cap without reaching optimal) via 1800 first, then reduced to 900 once
# the re-solve showed the worst actual convergence time was 546s (39%
# margin at 900s); 1800s left too much unused headroom. See the run log
# "Panel (b) time_limit re-run" and "TIME_CAP_S settled at 900s" entries.
BETA = 0.056
MIN_UP_DOWN_VALUES = [2, 3]  # T=1 already exists untagged in results/


def _case_files(basis: str, blend: str, mut: int) -> list[Path]:
    return [RESULTS / f"solve_{basis}_{blend}_{s}_T{mut}.json" for s in SCENARIOS]


def _inputs() -> dict:
    bsfc = pd.read_csv(DATA / "bsfc_bases.csv")
    costs = pd.read_csv(DATA / "cost_coefficients.csv")
    with open(DATA / "scenario_probabilities.json") as f:
        probs = json.load(f)
    a_x = {
        "epa": dict(zip(bsfc.blend, bsfc.A_x_basis_a_L_per_kWh)),
        "generator": dict(zip(bsfc.blend, bsfc.A_x_basis_b_L_per_kWh)),
    }
    return {"a_x": a_x, "costs": costs, "probs": probs}


def solve_case(basis: str, blend: str, mut: int, inp: dict) -> dict:
    n = build_network(blend=blend, basis=basis, oa_scenario=OA)
    apply_layer(n, min_up_time=mut, min_down_time=mut)

    t0 = time.time()
    status, condition = n.optimize.solve_model(
        solver_name="highs",
        solver_options={"mip_rel_gap": GAP, "time_limit": TIME_CAP_S, "output_flag": False},
    )
    wall = time.time() - t0

    solver_model = n.model.solver_model
    try:
        info = solver_model.getInfo()
        achieved_gap = float(info.mip_gap)
    except Exception:
        achieved_gap = float("nan")

    a_x = inp["a_x"][basis][blend]
    case_costs = inp["costs"][
        (inp["costs"].blend == blend)
        & (inp["costs"].bsfc_basis == basis)
        & (inp["costs"].ongkos_angkut_scenario == OA)
    ].set_index("genset_group")

    gens = [f"{g}_{i}" for g, i, _ in GENSETS]
    rated = {f"{g}_{i}": p for g, i, p in GENSETS}
    group_of = {f"{g}_{i}": g for g, i, _ in GENSETS}

    status_df = n.generators_t.status[gens]
    p_df = n.generators_t.p[gens]
    pv_p = n.generators_t.p["pv"]
    pv_avail = n.generators_t.p_max_pu["pv"] * n.generators.at["pv", "p_nom"]
    soc = n.storage_units_t.state_of_charge["bess"]
    charge = n.storage_units_t.p_store["bess"]
    discharge = n.storage_units_t.p_dispatch["bess"]

    records = {}
    for scenario in SCENARIOS:
        period = PERIOD_OF[scenario]
        u = status_df.xs(period, level=0)
        p = p_df.xs(period, level=0)

        idle_l = sum(BETA * rated[g] * float(u[g].sum()) for g in gens)
        marginal_l = float(a_x * p.values.sum())

        cost = 0.0
        for g in gens:
            grp = group_of[g]
            cost += float(case_costs.at[grp, "a_i_idle_rp_per_hr"] * u[g].sum())
            cost += float(case_costs.at[grp, "b_i_marginal_rp_per_kwh"] * p[g].sum())

        load_factors = {
            g: [
                (float(p[g].iloc[t]) / rated[g]) if float(u[g].iloc[t]) > 0.5 else None
                for t in range(24)
            ]
            for g in gens
        }

        records[scenario] = {
            "case": {
                "basis": basis, "blend": blend, "ongkos_angkut": OA, "scenario": scenario,
                "min_up_time_h": mut, "min_down_time_h": mut,
            },
            "expected_objective_all_scenarios_rp": float(n.objective),
            "scenario_probability": inp["probs"][scenario],
            "scenario_cost_rp": cost,
            "scenario_energy_kwh": float(p.values.sum()),
            "commitment_u": {g: [int(round(float(v))) for v in u[g].values] for g in gens},
            "dispatch_p_kw": {g: [float(v) for v in p[g].values] for g in gens},
            "bess_soc_kwh": [float(v) for v in soc.xs(period, level=0).values],
            "bess_charge_kw": [float(v) for v in charge.xs(period, level=0).values],
            "bess_discharge_kw": [float(v) for v in discharge.xs(period, level=0).values],
            "pv_dispatch_kw": [float(v) for v in pv_p.xs(period, level=0).values],
            "pv_available_kw": [float(v) for v in pv_avail.xs(period, level=0).values],
            "pv_curtailed_kw": [
                float(a - b)
                for a, b in zip(pv_avail.xs(period, level=0).values, pv_p.xs(period, level=0).values)
            ],
            "fuel_litres": {
                "idle": idle_l,
                "marginal": marginal_l,
                "total": idle_l + marginal_l,
                "A_x_L_per_kWh": float(a_x),
                "beta_L_per_h_per_kW": BETA,
            },
            "load_factors": load_factors,
            "committed_unit_hours": float(u.values.sum()),
            "solver": {
                "status": status,
                "termination_condition": condition,
                "requested_mip_gap": GAP,
                "achieved_mip_gap": achieved_gap,
                "terminated_early": condition != "optimal",
                "wall_clock_s": wall,
                "time_cap_s": TIME_CAP_S,
            },
        }
    return records


SUMMARY_CSV = CODING / "run_summary_panel_b.csv"
SUMMARY_FIELDS = [
    "min_up_down_h", "basis", "blend", "expected_cost_rp",
    "termination_condition", "achieved_mip_gap", "wall_clock_s", "terminated_early",
]


def _append_summary_row(row: dict) -> None:
    import csv

    is_new = not SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def main(only_missing: bool = True):
    RESULTS.mkdir(exist_ok=True)
    inp = _inputs()
    total_t0 = time.time()

    for mut in MIN_UP_DOWN_VALUES:
        for basis in BASES:
            for blend in BLENDS:
                files = _case_files(basis, blend, mut)
                if only_missing and all(f.exists() for f in files):
                    print(f"SKIP T={mut} {basis:9s} {blend:5s} (all 3 scenario files exist)", flush=True)
                    continue

                recs = solve_case(basis, blend, mut, inp)
                for scenario, rec in recs.items():
                    path = RESULTS / f"solve_{basis}_{blend}_{scenario}_T{mut}.json"
                    with open(path, "w") as f:
                        json.dump(rec, f, indent=2)

                s = recs[SCENARIOS[0]]["solver"]
                flag = "" if not s["terminated_early"] else "  <-- HIT CAP, not proven optimal"
                _append_summary_row({
                    "min_up_down_h": mut,
                    "basis": basis,
                    "blend": blend,
                    "expected_cost_rp": recs[SCENARIOS[0]]["expected_objective_all_scenarios_rp"],
                    "termination_condition": s["termination_condition"],
                    "achieved_mip_gap": s["achieved_mip_gap"],
                    "wall_clock_s": s["wall_clock_s"],
                    "terminated_early": s["terminated_early"],
                })
                print(
                    f"DONE T={mut} {basis:9s} {blend:5s} wall={s['wall_clock_s']:7.1f}s "
                    f"cond={s['termination_condition']:12s} achieved_gap={s['achieved_mip_gap']:.3e} "
                    f"expected=Rp {recs[SCENARIOS[0]]['expected_objective_all_scenarios_rp']:,.0f}{flag}",
                    flush=True,
                )

    print(f"SWEEP TOTAL wall={time.time() - total_t0:.1f}s", flush=True)
    print(f"Summary CSV: {SUMMARY_CSV}", flush=True)


if __name__ == "__main__":
    main()
