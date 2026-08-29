"""Build the PyPSA network for one blend and one consumption basis.

The three weather scenarios are carried as PyPSA investment periods on a
(scenario, hour) MultiIndex snapshot axis (72 snapshots). This is not a
capacity-expansion use; it is the only built-in mechanism that keeps the
storage balance and the cyclic SOC condition period-aware, so hour 0 of one
scenario does not inherit the SOC of hour 23 of the previous one.

Cost mapping:
  b_i(x) = A_x * Price_x        -> marginal_cost   [Rp/kWh]
  a_i(x) = beta*P_rated*Price_x -> stand_by_cost   [Rp/h], charged whenever a
           unit is committed, including at zero dispatch (the no-load term).

Three things are deliberately left out here and added in stochastic_layer.py,
because PyPSA cannot express them correctly on this snapshot axis:
non-anticipativity, the SOC floor, and period-aware minimum up/down time.
"""

import json
from pathlib import Path

import pandas as pd
import pypsa

CODING_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CODING_DIR / "data"

SCENARIOS = ["clear", "partly_cloudy", "overcast"]
# PyPSA requires investment periods to be strictly increasing integers, so the
# scenarios are carried as integer codes and mapped back for reporting.
PERIOD_OF = {s: i for i, s in enumerate(SCENARIOS)}
SCENARIO_OF = {i: s for s, i in PERIOD_OF.items()}

# the case-study reference Table I: 11 units in 3 capacity groups
GENSETS = (
    [("mitsubishi", i + 1, 1100.0) for i in range(3)]
    + [("cummins", i + 1, 1000.0) for i in range(3)]
    + [("deutz", i + 1, 900.0) for i in range(5)]
)

P_MIN_PU = 0.30          # minimum stable loading
PV_P_NOM_AC_KW = 1100.0  # inverter-limited AC capacity
BESS_P_NOM_KW = 800.0
BESS_E_NOM_KWH = 873.0
BESS_SOC_FLOOR_FRAC = 0.30  # applied in , recorded here for provenance
ETA_STORE = 0.85
ETA_DISPATCH = 0.95

# -- unserved-load slack, OFF by default (see add_unserved_load_slack).
UNSERVED_LOAD_PENALTY_RP_PER_KWH = 1_000_000.0
UNSERVED_LOAD_P_NOM_KW = 6_000.0


def add_unserved_load_slack(n: pypsa.Network) -> None:
    """Add a slack "unserved_load" generator at the load bus with a large
    technical penalty (standard big-M practice to keep the unit-commitment
    model well-posed). Its dispatch, if any, is the unserved-energy metric
    for the expected-weather ablation.

    Used only by the ablation; the main sweep never adds it. Callers opt in
    explicitly; build_network() defaults to off.
    """
    n.add(
        "Generator",
        "unserved_load",
        bus="selayar",
        carrier="AC",
        p_nom=UNSERVED_LOAD_P_NOM_KW,
        marginal_cost=UNSERVED_LOAD_PENALTY_RP_PER_KWH,
    )


def _load_inputs() -> dict:
    load = pd.read_csv(DATA_DIR / "load_selayar_24h.csv", comment="#")
    pv = pd.read_csv(DATA_DIR / "pv_profiles.csv")
    costs = pd.read_csv(DATA_DIR / "cost_coefficients.csv")
    with open(DATA_DIR / "scenario_probabilities.json") as f:
        probs = json.load(f)
    return {"load": load, "pv": pv, "costs": costs, "probs": probs}


def build_network(
    blend: str = "B40",
    basis: str = "generator",
    oa_scenario: str = "mean",
    inputs: dict | None = None,
    add_unserved_slack: bool = False,
) -> pypsa.Network:
    """Build the network for one (blend, consumption basis, haulage) case.

    Returns a network with 72 snapshots (3 scenarios x 24 h) whose objective,
    once solved, is the probability-weighted expected daily fuel cost in Rp.
    apply_layer() from stochastic_layer.py must run before solving; without
    it, commitment is scenario-dependent and the SOC floor is absent.

    add_unserved_slack=True adds the diagnostic slack generator (used only by
    the ablation).
    """
    inputs = inputs or _load_inputs()
    load, pv, costs, probs = (inputs[k] for k in ("load", "pv", "costs", "probs"))

    case = costs[
        (costs.blend == blend)
        & (costs.bsfc_basis == basis)
        & (costs.ongkos_angkut_scenario == oa_scenario)
    ]
    if case.empty:
        raise ValueError(f"No cost coefficients for {blend}/{basis}/{oa_scenario}")
    coef = case.set_index("genset_group")[
        ["a_i_idle_rp_per_hr", "b_i_marginal_rp_per_kwh"]
    ]

    n = pypsa.Network()
    snapshots = pd.MultiIndex.from_product(
        [[PERIOD_OF[s] for s in SCENARIOS], range(24)], names=["period", "timestep"]
    )
    n.set_snapshots(snapshots)
    n.investment_periods = [PERIOD_OF[s] for s in SCENARIOS]

    # pi_s enters the objective through the snapshot objective weighting.
    # The `stores`/`generators` weightings stay at 1 so the physics remains
    # hourly; only the cost side is probability-weighted.
    n.snapshot_weightings["objective"] = [
        probs[SCENARIO_OF[p]] for p, _ in snapshots
    ]
    n.investment_period_weightings["objective"] = 1.0

    n.add("Carrier", "diesel")
    n.add("Carrier", "solar")
    n.add("Carrier", "battery")
    n.add("Carrier", "AC")
    n.add("Bus", "selayar", carrier="AC")

    for group, idx, p_rated in GENSETS:
        n.add(
            "Generator",
            f"{group}_{idx}",
            bus="selayar",
            carrier="diesel",
            p_nom=p_rated,
            committable=True,
            p_min_pu=P_MIN_PU,
            marginal_cost=float(coef.loc[group, "b_i_marginal_rp_per_kwh"]),
            stand_by_cost=float(coef.loc[group, "a_i_idle_rp_per_hr"]),
            # start_up_cost / shut_down_cost stay at their 0 default: the
            # exclusion IS the default, not a special setting.
            # min_up_time / min_down_time stay 0 - see module docstring.
        )

    pv_wide = pv.pivot(index="hour", columns="scenario", values="pv_ac_kw")
    pv_series = pd.Series(
        [pv_wide.loc[h, SCENARIO_OF[p]] / PV_P_NOM_AC_KW for p, h in snapshots],
        index=snapshots,
    )
    n.add(
        "Generator",
        "pv",
        bus="selayar",
        carrier="solar",
        p_nom=PV_P_NOM_AC_KW,
        marginal_cost=0.0,
        p_max_pu=pv_series,  # p_min_pu stays 0, so curtailment is allowed
    )

    n.add(
        "StorageUnit",
        "bess",
        bus="selayar",
        carrier="battery",
        p_nom=BESS_P_NOM_KW,
        max_hours=BESS_E_NOM_KWH / BESS_P_NOM_KW,
        efficiency_store=ETA_STORE,
        efficiency_dispatch=ETA_DISPATCH,
        cyclic_state_of_charge=False,
        cyclic_state_of_charge_per_period=True,  # per-scenario, period-aware
    )

    load_series = pd.Series(
        [load.set_index("hour").loc[h, "load_kw"] for _, h in snapshots], index=snapshots
    )
    n.add("Load", "demand", bus="selayar", carrier="AC", p_set=load_series)

    if add_unserved_slack:
        add_unserved_load_slack(n)

    return n


def build_network_expected_weather(
    blend: str = "B40",
    basis: str = "generator",
    oa_scenario: str = "mean",
    inputs: dict | None = None,
    add_unserved_slack: bool = False,
) -> pypsa.Network:
    """Single-period network on the probability-weighted average PV profile.

    The expected-weather (EEV) deterministic alternative: commit and dispatch
    as if the day were the probability-weighted average of the three weather
    scenarios. One 24-hour period, so PyPSA's flat-axis min-up/down-time
    handling is safe here without the period-aware rebuild. Still routed
    through apply_layer(non_anticipative=False, ...) for the SOC floor and to
    keep the storage-cycling machinery identical to the main model.
    """
    inputs = inputs or _load_inputs()
    load, pv, costs, probs = (inputs[k] for k in ("load", "pv", "costs", "probs"))

    case = costs[
        (costs.blend == blend)
        & (costs.bsfc_basis == basis)
        & (costs.ongkos_angkut_scenario == oa_scenario)
    ]
    if case.empty:
        raise ValueError(f"No cost coefficients for {blend}/{basis}/{oa_scenario}")
    coef = case.set_index("genset_group")[
        ["a_i_idle_rp_per_hr", "b_i_marginal_rp_per_kwh"]
    ]

    n = pypsa.Network()
    period = 0
    snapshots = pd.MultiIndex.from_product([[period], range(24)], names=["period", "timestep"])
    n.set_snapshots(snapshots)
    n.investment_periods = [period]
    n.snapshot_weightings["objective"] = 1.0
    n.investment_period_weightings["objective"] = 1.0

    n.add("Carrier", "diesel")
    n.add("Carrier", "solar")
    n.add("Carrier", "battery")
    n.add("Carrier", "AC")
    n.add("Bus", "selayar", carrier="AC")

    for group, idx, p_rated in GENSETS:
        n.add(
            "Generator",
            f"{group}_{idx}",
            bus="selayar",
            carrier="diesel",
            p_nom=p_rated,
            committable=True,
            p_min_pu=P_MIN_PU,
            marginal_cost=float(coef.loc[group, "b_i_marginal_rp_per_kwh"]),
            stand_by_cost=float(coef.loc[group, "a_i_idle_rp_per_hr"]),
        )

    # Probability-weighted average PV profile across the 3 real scenarios.
    pv_wide = pv.pivot(index="hour", columns="scenario", values="pv_ac_kw")
    expected_pv_kw = sum(probs[s] * pv_wide[s] for s in SCENARIOS) / sum(probs.values())
    pv_series = pd.Series(
        [expected_pv_kw.loc[h] / PV_P_NOM_AC_KW for _, h in snapshots], index=snapshots,
    )
    n.add(
        "Generator", "pv", bus="selayar", carrier="solar",
        p_nom=PV_P_NOM_AC_KW, marginal_cost=0.0, p_max_pu=pv_series,
    )

    n.add(
        "StorageUnit", "bess", bus="selayar", carrier="battery",
        p_nom=BESS_P_NOM_KW, max_hours=BESS_E_NOM_KWH / BESS_P_NOM_KW,
        efficiency_store=ETA_STORE, efficiency_dispatch=ETA_DISPATCH,
        cyclic_state_of_charge=False, cyclic_state_of_charge_per_period=True,
    )

    load_series = pd.Series(
        [load.set_index("hour").loc[h, "load_kw"] for _, h in snapshots], index=snapshots
    )
    n.add("Load", "demand", bus="selayar", carrier="AC", p_set=load_series)

    if add_unserved_slack:
        add_unserved_load_slack(n)

    return n


def main():
    n = build_network()
    print(f"Snapshots: {len(n.snapshots)} ({len(SCENARIOS)} scenarios x 24 h)")
    print(f"Investment periods (scenarios): {list(n.investment_periods)}")
    print(f"Objective weightings sum: {n.snapshot_weightings['objective'].sum():.4f}"
          f"  (= 24 h x sum(pi_s) = 24.0 expected)")
    print(f"\nGenerators: {len(n.generators)} "
          f"({int(n.generators.committable.sum())} committable + PV)")
    print(n.generators[["p_nom", "committable", "p_min_pu", "marginal_cost",
                        "stand_by_cost", "min_up_time", "min_down_time"]].to_string())
    print(f"\nStorage: p_nom={n.storage_units.p_nom.iloc[0]:.0f} kW, "
          f"max_hours={n.storage_units.max_hours.iloc[0]:.5f} "
          f"(= {n.storage_units.p_nom.iloc[0]*n.storage_units.max_hours.iloc[0]:.0f} kWh), "
          f"cyclic_per_period={bool(n.storage_units.cyclic_state_of_charge_per_period.iloc[0])}")
    print(f"Peak demand: {n.loads_t.p_set.max().iloc[0]:.0f} kW, "
          f"daily energy per scenario: {n.loads_t.p_set.groupby(level=0).sum().iloc[0,0]/1000:.2f} MWh")
    print("\nPV availability (kW) by scenario:")
    pv_kw = (n.generators_t.p_max_pu["pv"] * PV_P_NOM_AC_KW).groupby(level=0)
    print(pv_kw.agg(["max", "sum"]).rename(columns={"max": "peak_kW", "sum": "daily_kWh"}).to_string())


if __name__ == "__main__":
    main()
