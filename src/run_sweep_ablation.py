"""Stochastic-layer ablation: what the two-stage recourse layer buys.

Two deterministic alternatives are compared against the recourse solution (RP)
from run_sweep.py:

1. Perfect foresight (WS): each weather scenario solved independently, with
   non-anticipativity off. EVPI = RP - WS is the value of a perfect forecast.
   -> results/solve_pf_{basis}_{blend}_{scenario}.json
2. Expected weather (EEV): commit once on the probability-weighted average PV
   profile, then evaluate that fixed commitment under the three real scenarios
   with only dispatch adapting. A diagnostic unserved-load slack is added so
   an infeasible commitment reports how much energy it fails to serve rather
   than failing outright.
   -> results/solve_eev_{basis}_{blend}_{scenario}.json

RP and WS serve all load by construction; EEV can leave energy unserved, and
its diesel cost is then not directly comparable to RP/WS.
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

from build_network import (
    build_network,
    build_network_expected_weather,
    SCENARIOS,
    PERIOD_OF,
    GENSETS,
)
from stochastic_layer import apply_layer, add_fixed_commitment

CODING = Path(__file__).resolve().parent.parent
RESULTS = CODING / "results"
DATA = CODING / "data"

BLENDS = ["B0", "B10", "B20", "B30", "B35", "B40", "B50", "B70", "B100"]
BASES = ["epa", "generator"]
OA = "mean"
GAP = 5e-4
TIME_CAP_S = 900
BETA = 0.056

GENS = [f"{g}_{i}" for g, i, _ in GENSETS]
RATED = {f"{g}_{i}": p for g, i, p in GENSETS}
GROUP_OF = {f"{g}_{i}": g for g, i, _ in GENSETS}


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


def _solve(n, gap: float = GAP, time_cap: int = TIME_CAP_S):
    t0 = time.time()
    status, condition = n.optimize.solve_model(
        solver_name="highs",
        solver_options={"mip_rel_gap": gap, "time_limit": time_cap, "output_flag": False},
    )
    wall = time.time() - t0
    try:
        achieved_gap = float(n.model.solver_model.getInfo().mip_gap)
    except Exception:
        achieved_gap = float("nan")
    return {
        "status": status,
        "termination_condition": condition,
        "requested_mip_gap": gap,
        "achieved_mip_gap": achieved_gap,
        "terminated_early": condition != "optimal",
        "wall_clock_s": wall,
        "time_cap_s": time_cap,
    }


def _extract_record(n, basis, blend, scenario, period, a_x, coef, variant: str,
                     solver: dict) -> dict:
    if solver["status"] != "ok":
        # Should not happen with the unserved-load slack in place (the MILP
        # is always well-posed) -- kept as a defensive fallback only, e.g.
        # for an unrelated numerical/solver failure. Record honestly rather
        # than crash or fabricate a number.
        return {
            "case": {"variant": variant, "basis": basis, "blend": blend, "scenario": scenario},
            "infeasible": True,
            "scenario_cost_rp": None,
            "unserved_energy_kwh": None,
            "solver": solver,
        }

    status_df = n.generators_t.status[GENS]
    p_df = n.generators_t.p[GENS]
    u = status_df.xs(period, level=0)
    p = p_df.xs(period, level=0)

    unserved_kwh = 0.0
    if "unserved_load" in n.generators.index:
        unserved_kwh = float(n.generators_t.p["unserved_load"].xs(period, level=0).sum())

    idle_l = sum(BETA * RATED[g] * float(u[g].sum()) for g in GENS)
    marginal_l = float(a_x * p.values.sum())

    cost = 0.0
    for g in GENS:
        grp = GROUP_OF[g]
        cost += float(coef.at[grp, "a_i_idle_rp_per_hr"] * u[g].sum())
        cost += float(coef.at[grp, "b_i_marginal_rp_per_kwh"] * p[g].sum())

    return {
        "case": {"variant": variant, "basis": basis, "blend": blend, "scenario": scenario},
        "infeasible": False,
        "scenario_cost_rp": cost,  # real diesel fuel cost only -- excludes the slack penalty
        "unserved_energy_kwh": unserved_kwh,
        "scenario_energy_kwh": float(p.values.sum()),
        "commitment_u": {g: [int(round(float(v))) for v in u[g].values] for g in GENS},
        "fuel_litres": {"idle": idle_l, "marginal": marginal_l, "total": idle_l + marginal_l},
        "committed_unit_hours": float(u.values.sum()),
        "solver": solver,
    }


def solve_perfect_foresight(basis: str, blend: str, inp: dict) -> dict:
    """Each scenario solved independently (non-anticipativity OFF).
    Unserved-load slack included for consistency/well-posedness, though WS
    should always find 0 unserved energy (full commitment freedom, and
    installed capacity 10.8 MW comfortably exceeds peak load 5.7 MW)."""
    n = build_network(blend=blend, basis=basis, oa_scenario=OA, add_unserved_slack=True)
    apply_layer(n, non_anticipative=False, min_up_time=1, min_down_time=1)
    solver = _solve(n)

    a_x = inp["a_x"][basis][blend]
    coef = inp["costs"][
        (inp["costs"].blend == blend) & (inp["costs"].bsfc_basis == basis)
        & (inp["costs"].ongkos_angkut_scenario == OA)
    ].set_index("genset_group")

    records = {}
    for scenario in SCENARIOS:
        period = PERIOD_OF[scenario]
        rec = _extract_record(n, basis, blend, scenario, period, a_x, coef,
                               "perfect_foresight", solver=solver)
        records[scenario] = rec
    return records


def solve_expected_weather(basis: str, blend: str, inp: dict) -> tuple[dict, dict]:
    """Stage 1: commit on the expected/average PV profile (no slack -- this
    solve was never infeasible in practice, always found optimal).
    Stage 2: fix that commitment, evaluate dispatch under the 3 real
    scenarios, WITH the unserved-load slack -- this is where the fixed,
    expected-weather-derived commitment can genuinely fall short (see
    module docstring)."""
    n1 = build_network_expected_weather(blend=blend, basis=basis, oa_scenario=OA)
    apply_layer(n1, non_anticipative=False, min_up_time=1, min_down_time=1)
    solver1 = _solve(n1)

    status_df = n1.generators_t.status[GENS]
    u_expected = status_df.xs(0, level=0)
    commitment = {g: [int(round(float(v))) for v in u_expected[g].values] for g in GENS}
    stage1 = {
        "case": {"variant": "expected_weather_stage1", "basis": basis, "blend": blend},
        "infeasible": solver1["status"] != "ok",
        "expected_objective_rp": float(n1.objective) if solver1["status"] == "ok" else None,
        "commitment_u": commitment,
        "solver": solver1,
    }

    n2 = build_network(blend=blend, basis=basis, oa_scenario=OA, add_unserved_slack=True)
    apply_layer(n2, non_anticipative=True, min_up_time=1, min_down_time=1)
    add_fixed_commitment(n2, commitment)
    solver2 = _solve(n2)

    a_x = inp["a_x"][basis][blend]
    coef = inp["costs"][
        (inp["costs"].blend == blend) & (inp["costs"].bsfc_basis == basis)
        & (inp["costs"].ongkos_angkut_scenario == OA)
    ].set_index("genset_group")

    records = {}
    for scenario in SCENARIOS:
        period = PERIOD_OF[scenario]
        rec = _extract_record(n2, basis, blend, scenario, period, a_x, coef,
                               "expected_weather_stage2", solver=solver2)
        records[scenario] = rec
    return stage1, records


def _case_done(basis: str, blend: str) -> bool:
    pf_files = [RESULTS / f"solve_pf_{basis}_{blend}_{s}.json" for s in SCENARIOS]
    eev_files = [RESULTS / f"solve_eev_{basis}_{blend}_{s}.json" for s in SCENARIOS]
    eev_stage1 = RESULTS / f"solve_eev_stage1_{basis}_{blend}.json"
    return all(f.exists() for f in pf_files + eev_files) and eev_stage1.exists()


SUMMARY_CSV = CODING / "run_summary_ablation.csv"
SUMMARY_FIELDS = [
    "variant", "basis", "blend", "expected_cost_rp", "expected_unserved_kwh",
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


def _weighted(recs: dict, probs: dict, field: str):
    if any(recs[s].get("infeasible") for s in SCENARIOS):
        return None
    return sum(probs[s] * recs[s][field] for s in SCENARIOS)


def main(only_missing: bool = True):
    RESULTS.mkdir(exist_ok=True)
    inp = _inputs()
    probs = inp["probs"]
    total_t0 = time.time()

    for basis in BASES:
        for blend in BLENDS:
            if only_missing and _case_done(basis, blend):
                print(f"SKIP {basis:9s} {blend:5s} (ablation files exist)", flush=True)
                continue

            # --- Perfect foresight (WS) ---
            pf_recs = solve_perfect_foresight(basis, blend, inp)
            for scenario, rec in pf_recs.items():
                with open(RESULTS / f"solve_pf_{basis}_{blend}_{scenario}.json", "w") as f:
                    json.dump(rec, f, indent=2)
            ws_cost = _weighted(pf_recs, probs, "scenario_cost_rp")
            ws_unserved = _weighted(pf_recs, probs, "unserved_energy_kwh")
            s = pf_recs[SCENARIOS[0]]["solver"]
            print(f"DONE pf  {basis:9s} {blend:5s} WS_cost=Rp {ws_cost:,.0f}  "
                  f"unserved={ws_unserved:.2f} kWh  cond={s['termination_condition']}"
                  if ws_cost is not None else
                  f"DONE pf  {basis:9s} {blend:5s} UNEXPECTED FAILURE (see json)", flush=True)
            _append_summary_row({
                "variant": "perfect_foresight", "basis": basis, "blend": blend,
                "expected_cost_rp": ws_cost, "expected_unserved_kwh": ws_unserved,
                "termination_condition": s["termination_condition"],
                "achieved_mip_gap": s["achieved_mip_gap"], "wall_clock_s": s["wall_clock_s"],
                "terminated_early": s["terminated_early"],
            })

            # --- Expected weather (EEV) ---
            stage1, eev_recs = solve_expected_weather(basis, blend, inp)
            with open(RESULTS / f"solve_eev_stage1_{basis}_{blend}.json", "w") as f:
                json.dump(stage1, f, indent=2)
            for scenario, rec in eev_recs.items():
                with open(RESULTS / f"solve_eev_{basis}_{blend}_{scenario}.json", "w") as f:
                    json.dump(rec, f, indent=2)
            eev_cost = _weighted(eev_recs, probs, "scenario_cost_rp")
            eev_unserved = _weighted(eev_recs, probs, "unserved_energy_kwh")
            s2 = eev_recs[SCENARIOS[0]]["solver"]
            per_scenario_unserved = {s_: eev_recs[s_].get("unserved_energy_kwh") for s_ in SCENARIOS}
            print(f"DONE eev {basis:9s} {blend:5s} EEV_cost=Rp {eev_cost:,.0f}  "
                  f"unserved={eev_unserved:.2f} kWh {per_scenario_unserved}  "
                  f"cond={s2['termination_condition']}"
                  if eev_cost is not None else
                  f"DONE eev {basis:9s} {blend:5s} UNEXPECTED FAILURE (see json)", flush=True)
            _append_summary_row({
                "variant": "expected_weather", "basis": basis, "blend": blend,
                "expected_cost_rp": eev_cost, "expected_unserved_kwh": eev_unserved,
                "termination_condition": s2["termination_condition"],
                "achieved_mip_gap": s2["achieved_mip_gap"], "wall_clock_s": s2["wall_clock_s"],
                "terminated_early": s2["terminated_early"],
            })

    print(f"SWEEP TOTAL wall={time.time() - total_t0:.1f}s", flush=True)
    print(f"Summary CSV: {SUMMARY_CSV}", flush=True)


if __name__ == "__main__":
    main()
