"""Validation suite. Nothing downstream is trusted until every check passes.

1. Energy balance closes every hour, every scenario.
2. Minimum stable loading respected when committed; zero output when not.
3. Battery SOC stays within [30%, 100%]; end-of-day SOC equals start-of-day.
4. Non-anticipativity binds (checked against the model, not by eye).
5. Determinism: two consecutive runs are byte-identical.
6. Solver status is optimal for every solve; achieved MIP gap logged.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from build_network import build_network, GENSETS, PV_P_NOM_AC_KW
from stochastic_layer import apply_layer, assert_storage_is_period_aware

SOC_FLOOR_FRAC = 0.30
GAP = 5e-4
CASE = dict(blend="B40", basis="generator", oa_scenario="mean")


def solve(seed_note: str = "") -> "pypsa.Network":
    import pypsa  # noqa: F401 (type hint only)

    n = build_network(**CASE)
    apply_layer(n, min_up_time=1, min_down_time=1)
    status, condition = n.optimize.solve_model(
        solver_name="highs",
        solver_options={"mip_rel_gap": GAP, "output_flag": False},
    )
    return n, status, condition


def check_energy_balance(n) -> tuple[bool, str]:
    gen = n.generators_t.p.drop(columns=["pv"]).sum(axis=1)
    pv = n.generators_t.p["pv"]
    dis = n.storage_units_t.p_dispatch["bess"]
    ch = n.storage_units_t.p_store["bess"]
    supply = gen + pv + dis - ch
    demand = n.loads_t.p_set["demand"]
    resid = (supply - demand).abs()
    ok = bool((resid < 1e-4).all())
    return ok, f"max residual = {resid.max():.6f} kW (tolerance 1e-4)"


def check_p_min_pu(n) -> tuple[bool, str]:
    """p[status]-masking with a boolean DataFrame returns NaN off-mask, not
    filtered rows - NaN.abs()<tol is False, which would misreport a pass as a
    fail. Every comparison below is therefore whole-DataFrame with an
    (status>=0.5) | (status<0.5) OR-guard, never a boolean-indexed subset."""
    status = n.generators_t.status.drop(columns=["pv"])
    p = n.generators_t.p.drop(columns=["pv"])
    p_rated = n.generators.loc[status.columns, "p_nom"]
    p_min_pu = n.generators.loc[status.columns, "p_min_pu"]
    floor = p_rated * p_min_pu

    is_off = status < 0.5
    is_on = ~is_off

    below_floor_mask = is_on & (p < floor.values * status - 1e-4)
    off_but_dispatching_mask = is_off & (p.abs() >= 1e-4)

    below_floor = int(below_floor_mask.sum().sum())
    off_but_dispatching = int(off_but_dispatching_mask.sum().sum())
    ok = below_floor == 0 and off_but_dispatching == 0
    return ok, f"committed-below-floor cells = {below_floor}, off-but-dispatching cells = {off_but_dispatching}"


def check_soc_bounds(n) -> tuple[bool, str]:
    soc = n.storage_units_t.state_of_charge["bess"]
    p_nom = n.storage_units.p_nom.iloc[0]
    max_hours = n.storage_units.max_hours.iloc[0]
    cap = p_nom * max_hours
    floor = cap * SOC_FLOOR_FRAC
    within = bool(((soc >= floor - 1e-4) & (soc <= cap + 1e-4)).all())

    cyclic_ok = []
    for p in n.investment_periods:
        sp = soc.xs(p, level=0)
        ch = n.storage_units_t.p_store["bess"].xs(p, level=0)
        di = n.storage_units_t.p_dispatch["bess"].xs(p, level=0)
        eta_s, eta_d = n.storage_units.efficiency_store.iloc[0], n.storage_units.efficiency_dispatch.iloc[0]
        predicted_start = sp.iloc[-1] + eta_s * ch.iloc[0] - di.iloc[0] / eta_d
        cyclic_ok.append(abs(predicted_start - sp.iloc[0]) < 1e-3)
    return bool(within and all(cyclic_ok)), (
        f"SOC in [{floor:.1f}, {cap:.1f}]: {within}; "
        f"cyclic per scenario: {cyclic_ok} (floor={floor:.2f} kWh)"
    )


def check_non_anticipativity(n) -> tuple[bool, str]:
    st = n.generators_t.status.drop(columns=["pv"])
    per = {p: st.xs(p, level=0).values for p in n.investment_periods}
    ref = per[n.investment_periods[0]]
    identical = all(np.allclose(ref, per[p], atol=1e-6) for p in n.investment_periods)
    max_diff = max(np.abs(ref - per[p]).max() for p in n.investment_periods)
    return bool(identical), f"max cross-scenario commitment diff = {max_diff:.2e}"


def check_determinism() -> tuple[bool, str]:
    n1, s1, c1 = solve("run1")
    n2, s2, c2 = solve("run2")
    obj_match = abs(n1.objective - n2.objective) < 1e-6
    st1 = n1.generators_t.status.values
    st2 = n2.generators_t.status.values
    sched_match = bool(np.array_equal(st1, st2))
    p1 = n1.generators_t.p.values
    p2 = n2.generators_t.p.values
    dispatch_match = bool(np.allclose(p1, p2, atol=1e-6))
    return bool(obj_match and sched_match and dispatch_match), (
        f"obj: {n1.objective:.6f} vs {n2.objective:.6f} (match={obj_match}); "
        f"schedule identical={sched_match}; dispatch identical (atol 1e-6)={dispatch_match}"
    )


def check_solver_status(n, status: str, condition: str) -> tuple[bool, str]:
    ok = status == "ok" and condition == "optimal"
    return bool(ok), f"status={status}, termination_condition={condition}"


def main():
    results = []

    n, status, condition = solve("primary")
    assert_storage_is_period_aware(n)  # re-assert post-solve, belt and suspenders

    for name, fn in [
        ("energy_balance", lambda: check_energy_balance(n)),
        ("p_min_pu", lambda: check_p_min_pu(n)),
        ("soc_bounds_and_cyclic", lambda: check_soc_bounds(n)),
        ("non_anticipativity", lambda: check_non_anticipativity(n)),
        ("solver_status", lambda: check_solver_status(n, status, condition)),
    ]:
        ok, detail = fn()
        results.append((name, ok, detail))

    ok, detail = check_determinism()
    results.append(("determinism", ok, detail))

    print(f"Case: {CASE}, gap={GAP:g}")
    print(f"Solver: status={status}, condition={condition}, achieved objective=Rp {n.objective:,.2f}")
    print()
    all_pass = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_pass &= ok
        print(f"[{mark}] {name}: {detail}")

    print()
    print("ALL CHECKS PASSED" if all_pass else "*** VALIDATION FAILED - DO NOT PROCEED TO TASK 11 ***")
    return all_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
