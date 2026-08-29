"""Price-anchor sensitivity: verification sweep.

Re-solves the two extreme corners of the A1 +/-15% x A2 +/-15% grid for the
decision-relevant blend subset, then checks each solved cost against the
scaling identity

    cost(blend, cfg) = cost(blend, base) * Price_x(cfg) / Price_x(base)

which holds because changing a config multiplies every cost coefficient of a
given blend by one factor, leaving the optimal dispatch unchanged. If every
corner matches its prediction to within the MIP gap, build_tab_r3d.py fills
the interior of the grid by that identity with no further solves.

Output: results/solve_pa_{cfg}_{basis}_{blend}_{scenario}.json
"""

from __future__ import annotations

import csv
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from build_network import (
    build_network, _load_inputs, SCENARIOS, PERIOD_OF, GENSETS,
)
from stochastic_layer import apply_layer

CODING = Path(__file__).resolve().parent.parent
RESULTS = CODING / "results"
DATA = CODING / "data"
TABLES = CODING / "tables"

# Decision-relevant subset: endpoints + mandated + true nearest competitor.
BLENDS = ["B0", "B40", "B70", "B100"]
BASES = ["epa", "generator"]
# The two extreme corners. a1lo_a2hi is the only config that inverts the
# ladder (diesel cheaper per litre than B100); a1hi_a2lo is the opposite.
CONFIGS = ["a1lo_a2hi", "a1hi_a2lo"]

GAP = 5e-4                # matches the main sweep
TIME_CAP_S = 900
BETA = 0.056

SUMMARY_CSV = CODING / "run_summary_price_anchor.csv"
SUMMARY_FIELDS = [
    "config", "basis", "blend", "expected_cost_rp",
    "predicted_cost_rp", "rel_delta_vs_identity",
    "termination_condition", "achieved_mip_gap", "wall_clock_s", "terminated_early",
]


def _pa_inputs() -> dict:
    """base build_network inputs, with costs swapped for the price-anchor CSV."""
    inp = _load_inputs()
    inp["costs"] = pd.read_csv(DATA / "cost_coefficients_price_anchor.csv")
    return inp


def _price_of(costs: pd.DataFrame, cfg: str, blend: str) -> float:
    row = costs[(costs.ongkos_angkut_scenario == cfg) & (costs.blend == blend)]
    return float(row.price_rp_per_l.iloc[0])


def _baseline_cost(basis: str, blend: str) -> float:
    """ expected cost for this (basis, blend) from tab_r1_blend_cost.csv."""
    t1 = pd.read_csv(TABLES / "tab_r1_blend_cost.csv")
    r = t1[(t1.basis == basis) & (t1.blend == blend)]
    return float(r.expected_cost_rp.iloc[0])


def _case_files(cfg: str, basis: str, blend: str) -> list[Path]:
    return [RESULTS / f"solve_pa_{cfg}_{basis}_{blend}_{s}.json" for s in SCENARIOS]


def solve_case(cfg: str, basis: str, blend: str, inp: dict) -> dict:
    n = build_network(blend=blend, basis=basis, oa_scenario=cfg, inputs=inp)
    apply_layer(n, min_up_time=1, min_down_time=1)

    t0 = time.time()
    status, condition = n.optimize.solve_model(
        solver_name="highs",
        solver_options={"mip_rel_gap": GAP, "time_limit": TIME_CAP_S, "output_flag": False},
    )
    wall = time.time() - t0
    try:
        achieved_gap = float(n.model.solver_model.getInfo().mip_gap)
    except Exception:
        achieved_gap = float("nan")

    gens = [f"{g}_{i}" for g, i, _ in GENSETS]
    rated = {f"{g}_{i}": p for g, i, p in GENSETS}
    group_of = {f"{g}_{i}": g for g, i, _ in GENSETS}
    costs = inp["costs"]
    case_costs = costs[
        (costs.blend == blend) & (costs.bsfc_basis == basis)
        & (costs.ongkos_angkut_scenario == cfg)
    ].set_index("genset_group")

    status_df = n.generators_t.status[gens]
    p_df = n.generators_t.p[gens]

    records = {}
    for scenario in SCENARIOS:
        period = PERIOD_OF[scenario]
        u = status_df.xs(period, level=0)
        p = p_df.xs(period, level=0)
        cost = 0.0
        for g in gens:
            grp = group_of[g]
            cost += float(case_costs.at[grp, "a_i_idle_rp_per_hr"] * u[g].sum())
            cost += float(case_costs.at[grp, "b_i_marginal_rp_per_kwh"] * p[g].sum())
        records[scenario] = {
            "case": {"config": cfg, "basis": basis, "blend": blend, "scenario": scenario},
            "expected_objective_all_scenarios_rp": float(n.objective),
            "scenario_cost_rp": cost,
            "scenario_energy_kwh": float(p.values.sum()),
            "committed_unit_hours": float(u.values.sum()),
            "commitment_u": {g: [int(round(float(v))) for v in u[g].values] for g in gens},
            "price_rp_per_l": _price_of(costs, cfg, blend),
            "solver": {
                "status": status, "termination_condition": condition,
                "requested_mip_gap": GAP, "achieved_mip_gap": achieved_gap,
                "terminated_early": condition != "optimal",
                "wall_clock_s": wall, "time_cap_s": TIME_CAP_S,
            },
        }
    return records


def _append_summary(row: dict) -> None:
    is_new = not SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def main(only_missing: bool = True):
    RESULTS.mkdir(exist_ok=True)
    inp = _pa_inputs()
    costs = inp["costs"]
    t_all = time.time()
    worst = 0.0

    for cfg in CONFIGS:
        for basis in BASES:
            for blend in BLENDS:
                files = _case_files(cfg, basis, blend)
                if only_missing and all(f.exists() for f in files):
                    print(f"SKIP {cfg:11s} {basis:9s} {blend:5s}", flush=True)
                    continue

                recs = solve_case(cfg, basis, blend, inp)
                for scenario, rec in recs.items():
                    with open(RESULTS / f"solve_pa_{cfg}_{basis}_{blend}_{scenario}.json", "w") as f:
                        json.dump(rec, f, indent=2)

                solved = recs[SCENARIOS[0]]["expected_objective_all_scenarios_rp"]
                predicted = _baseline_cost(basis, blend) * (
                    _price_of(costs, cfg, blend) / _price_of(costs, "base", blend)
                )
                rel = abs(solved - predicted) / predicted
                worst = max(worst, rel)
                s = recs[SCENARIOS[0]]["solver"]
                _append_summary({
                    "config": cfg, "basis": basis, "blend": blend,
                    "expected_cost_rp": solved, "predicted_cost_rp": predicted,
                    "rel_delta_vs_identity": rel,
                    "termination_condition": s["termination_condition"],
                    "achieved_mip_gap": s["achieved_mip_gap"],
                    "wall_clock_s": s["wall_clock_s"],
                    "terminated_early": s["terminated_early"],
                })
                flag = "  <-- IDENTITY BROKEN, see spec 3.2" if rel > 2e-3 else ""
                print(
                    f"DONE {cfg:11s} {basis:9s} {blend:5s} wall={s['wall_clock_s']:6.1f}s "
                    f"cond={s['termination_condition']:10s} gap={s['achieved_mip_gap']:.2e} "
                    f"solved=Rp{solved:,.0f} predicted=Rp{predicted:,.0f} "
                    f"rel_delta={rel:.2e}{flag}",
                    flush=True,
                )

    print(f"\nSWEEP TOTAL wall={time.time() - t_all:.1f}s", flush=True)
    print(f"WORST rel_delta vs scaling identity = {worst:.2e} "
          f"({'PASS - interior can be filled by identity' if worst <= 2e-3 else 'FAIL - solve the full grid'})",
          flush=True)
    print(f"Summary CSV: {SUMMARY_CSV}", flush=True)


if __name__ == "__main__":
    main()
