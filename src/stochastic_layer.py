"""Custom linopy constraints layered onto the PyPSA model before solving.

Apply after n.optimize.create_model() and before solving.

1. Non-anticipativity (Eq. 2): forces the commitment variable u[i,t] to be
   identical across the three weather scenarios, while dispatch p[i,t,s] stays
   scenario-specific. This is what makes the model two-stage.
2. SOC floor (Eq. 6): soc_t >= frac * max_hours * p_nom. Added as an explicit
   constraint rather than by shrinking max_hours, which would redefine the
   reported SOC.
3. Period-aware minimum up/down time: PyPSA's own implementation rolls a sum
   over the flat snapshot axis and crosses scenario boundaries, so it is
   disabled in build_network.py and rebuilt here in terms of the status
   variable, per scenario block.

Also provides add_must_run (force a unit set committed for the whole day).
"""

from __future__ import annotations

import pandas as pd

from build_network import GENSETS, SCENARIOS, PERIOD_OF


def _periods(n) -> list:
    return list(n.investment_periods)


def _snaps_for(n, period: int) -> list:
    return [s for s in n.snapshots if s[0] == period]


def _aligned(var, snaps, ref):
    """Select `snaps` from `var` and re-label the snapshot coord to `ref`.

    linopy aligns on coordinates, so slices taken from different scenarios (or
    different hours) must be re-labelled onto a shared index before they can be
    combined in one expression.
    """
    return var.sel(snapshot=snaps).assign_coords(snapshot=ref)


def add_non_anticipativity(n) -> int:
    """u[i,t] identical across all scenarios. Returns constraints added."""
    m = n.model
    status = m.variables["Generator-status"]
    periods = _periods(n)
    ref_period, other_periods = periods[0], periods[1:]

    ref_snaps = _snaps_for(n, ref_period)
    ref_coord = pd.RangeIndex(len(ref_snaps), name="snapshot")
    base = _aligned(status, ref_snaps, ref_coord)

    added = 0
    for p in other_periods:
        this = _aligned(status, _snaps_for(n, p), ref_coord)
        m.add_constraints(this - base, "=", 0, name=f"nonanticipativity-p{p}")
        added += 1
    return added


def add_soc_floor(n, floor_frac: float = 0.30) -> int:
    """soc_t >= floor_frac * max_hours * p_nom, every snapshot."""
    m = n.model
    soc = m.variables["StorageUnit-state_of_charge"]
    su = n.storage_units
    floor = (su.p_nom * su.max_hours * floor_frac).rename_axis("name")
    m.add_constraints(soc, ">=", floor.to_xarray(), name="StorageUnit-soc-floor")
    return 1


def add_min_up_down_time(n, min_up_time: int = 1, min_down_time: int = 1) -> int:
    """Scenario-aware minimum up/down time. T=1 is vacuous and adds nothing."""
    m = n.model
    status = m.variables["Generator-status"]
    added = 0

    for p in _periods(n):
        snaps = _snaps_for(n, p)
        n_h = len(snaps)

        for T, sign, label in (
            (min_up_time, +1, "up"),
            (min_down_time, -1, "down"),
        ):
            if T is None or T <= 1:
                continue
            for k in range(1, T):
                # valid turn-on/turn-off hours t with t+k still inside the day
                ts = [t for t in range(1, n_h) if t + k < n_h]
                if not ts:
                    continue
                ref = pd.RangeIndex(len(ts), name="snapshot")
                s_tk = _aligned(status, [snaps[t + k] for t in ts], ref)
                s_t = _aligned(status, [snaps[t] for t in ts], ref)
                s_tm1 = _aligned(status, [snaps[t - 1] for t in ts], ref)

                if sign > 0:
                    # status[t+k] - status[t] + status[t-1] >= 0
                    m.add_constraints(
                        s_tk - s_t + s_tm1, ">=", 0,
                        name=f"min-up-time-p{p}-k{k}",
                    )
                else:
                    # (1 - status[t+k]) - (status[t-1] - status[t]) >= 0
                    m.add_constraints(
                        -s_tk - s_tm1 + s_t, ">=", -1,
                        name=f"min-down-time-p{p}-k{k}",
                    )
                added += 1
    return added


def add_symmetry_breaking(n) -> int:
    """OPTIONAL, off by default - a solver aid, not a modelling choice.

    Units within a capacity group are modelled identically (same p_nom, same
    marginal_cost, same stand_by_cost), so any solution has 3!*3!*5! = 4,320
    equivalent relabellings and branch-and-bound explores them all. Ordering
    them (unit k+1 may run only if unit k runs) removes the relabellings while
    preserving at least one optimal solution, so the optimal objective is
    unchanged. It does make *which physical unit* runs an artefact - but that
    was already arbitrary, since the units are indistinguishable in the model.

    Off by default; if enabled, check that the objective matches an unbroken
    run.
    """
    m = n.model
    status = m.variables["Generator-status"]
    groups: dict[str, list[str]] = {}
    for group, idx, _ in GENSETS:
        groups.setdefault(group, []).append(f"{group}_{idx}")

    added = 0
    for group, units in groups.items():
        for lower, upper in zip(units, units[1:]):
            m.add_constraints(
                status.sel(name=upper) - status.sel(name=lower), "<=", 0,
                name=f"symmetry-{upper}",
            )
            added += 1
    return added


def add_must_run(n, units: list[str]) -> int:
    """Force status[i,t] = 1 for the given units, every snapshot (all scenarios).

    Used by the must-run sensitivity sweep. Replicates the documented
    base-load unit set from the case-study reference (three Mitsubishi and
    two Cummins units); makes no claim about why those units are base-load,
    only that this is the documented operational split worth testing.
    """
    m = n.model
    status = m.variables["Generator-status"]
    for u in units:
        m.add_constraints(status.sel(name=u), "=", 1, name=f"must-run-{u}")
    return len(units)


def add_fixed_commitment(n, commitment: dict[str, list[int]]) -> int:
    """Force status[i, p, t] = commitment[i][t] for EVERY period p (all 3
    weather scenarios share the same fixed schedule).

    Used by the expected-weather (EEV) ablation: replicates a schedule
    decided without seeing the true weather (optimised on the averaged PV
    profile) across all three real scenarios, so only dispatch can adapt.
    The standard EEV construction in two-stage stochastic programming: commit
    once on
    the expected-value problem, then evaluate that fixed commitment's true
    expected cost under the actual scenario set. `commitment[i]` must have
    one 0/1 entry per hour (24), in the same hour order as the network's
    24-hour axis within each period. Not called by `apply_layer` — invoked
    explicitly by `run_sweep_ablation.py`'s expected-weather stage 2.
    """
    m = n.model
    status = m.variables["Generator-status"]
    added = 0
    for p in _periods(n):
        snaps = _snaps_for(n, p)
        ref_coord = pd.RangeIndex(len(snaps), name="snapshot")
        for gen, pattern in commitment.items():
            if len(pattern) != len(snaps):
                raise ValueError(
                    f"commitment[{gen!r}] has {len(pattern)} entries, "
                    f"expected {len(snaps)} (one per hour)"
                )
            var = _aligned(status.sel(name=gen), snaps, ref_coord)
            rhs = pd.Series([int(v) for v in pattern], index=ref_coord).to_xarray()
            m.add_constraints(var, "=", rhs, name=f"fixed-commitment-{gen}-p{p}")
            added += 1
    return added


def assert_storage_is_period_aware(n) -> None:
    """Fail loudly if the storage balance chains across scenarios.

    This guard exists because the failure is silent and wrong. Setting
    `cyclic_state_of_charge_per_period=True` on the component is not enough:
    PyPSA only reads it inside `if n._multi_invest:`, and that flag is set by
    `create_model(multi_investment_periods=True)`, NOT by assigning
    `n.investment_periods`. Without the argument the model still builds, still
    solves, and still reports plausible numbers - but the battery carries
    charge from one weather scenario into the next, which is meaningless when
    the scenarios are alternative versions of the same day. Caught
    by inspecting the constraint matrix; this assertion makes a regression
    impossible to miss.
    """
    if not getattr(n, "_multi_invest", 0):
        raise RuntimeError(
            "Storage balance is NOT period-aware: n._multi_invest is falsy. "
            "The model must be created with "
            "create_model(multi_investment_periods=True)."
        )
    soc = n.model.variables["StorageUnit-state_of_charge"]
    lbl2snap = {int(soc.labels.sel(snapshot=s).item()): s for s in n.snapshots}
    con = n.model.constraints["StorageUnit-energy_balance"]
    flat = con.flat
    for period in _periods(n):
        first = (period, 0)
        row = int(con.labels.sel(snapshot=first).item())
        refs = [
            lbl2snap[int(r.vars)]
            for r in flat[flat["labels"] == row].itertuples()
            if int(r.vars) in lbl2snap
        ]
        wrapped = [s for s in refs if s != first]
        if not wrapped or any(s[0] != period for s in wrapped):
            raise RuntimeError(
                f"Storage balance at {first} references {wrapped}; expected the "
                f"last hour of period {period} only (per-scenario cycling)."
            )


def apply_layer(
    n,
    non_anticipative: bool = True,
    soc_floor_frac: float = 0.30,
    min_up_time: int = 1,
    min_down_time: int = 1,
    symmetry_breaking: bool = False,
    must_run_units: list[str] | None = None,
) -> dict:
    """Create the model and add the custom constraints. Returns a summary."""
    if n.model is None:
        # multi_investment_periods=True is REQUIRED - see
        # assert_storage_is_period_aware for why omitting it fails silently.
        n.optimize.create_model(multi_investment_periods=True)
    assert_storage_is_period_aware(n)

    summary = {
        "non_anticipativity": add_non_anticipativity(n) if non_anticipative else 0,
        "soc_floor": add_soc_floor(n, soc_floor_frac),
        "min_up_down_time": add_min_up_down_time(n, min_up_time, min_down_time),
        "symmetry_breaking": add_symmetry_breaking(n) if symmetry_breaking else 0,
        "must_run": add_must_run(n, must_run_units) if must_run_units else 0,
        "settings": {
            "non_anticipative": non_anticipative,
            "soc_floor_frac": soc_floor_frac,
            "min_up_time": min_up_time,
            "min_down_time": min_down_time,
            "symmetry_breaking": symmetry_breaking,
            "must_run_units": must_run_units or [],
        },
    }
    return summary


def main():
    import warnings

    warnings.filterwarnings("ignore")
    from build_network import build_network

    for mut in (1, 3):
        n = build_network()
        s = apply_layer(n, min_up_time=mut, min_down_time=mut)
        soc_floor = n.storage_units.p_nom.iloc[0] * n.storage_units.max_hours.iloc[0] * 0.30
        print(f"min_up/down_time = {mut} h")
        print(f"  non-anticipativity blocks : {s['non_anticipativity']}"
              f"  (expect {len(SCENARIOS) - 1})")
        print(f"  soc-floor blocks          : {s['soc_floor']}  (floor = {soc_floor:.1f} kWh)")
        print(f"  min-up/down blocks        : {s['min_up_down_time']}"
              f"  (expect {2 * (mut - 1) * len(SCENARIOS)})")
        print(f"  total model constraints   : {len(n.model.constraints)}")
        print("  storage period-aware      : PASS (asserted, per-scenario cycling)")
        print()


if __name__ == "__main__":
    main()
