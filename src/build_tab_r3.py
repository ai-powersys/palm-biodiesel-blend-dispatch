"""Idle-vs-marginal fuel split and the schedule-invariance check.

Rebuilds from results/. Measures, per basis/blend/scenario:
1. the idle (no-load) vs marginal litre split and the load-factor
   distribution behind it, plus the probability-weighted expected split;
2. whether the commitment schedule is identical across blends within a
   basis and scenario, unit by unit and hour by hour, against the B0
   schedule as reference.

Output: tables/tab_r3_loadfactor_split.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CODING = Path(__file__).resolve().parent.parent
RESULTS = CODING / "results"
TABLES = CODING / "tables"

BLENDS = ["B0", "B10", "B20", "B30", "B35", "B40", "B50", "B70", "B100"]
PCT = {b: int(b[1:]) for b in BLENDS}
BASES = ["epa", "generator"]
SCENARIOS = ["clear", "partly_cloudy", "overcast"]
UNITS = [
    "mitsubishi_1", "mitsubishi_2", "mitsubishi_3",
    "cummins_1", "cummins_2", "cummins_3",
    "deutz_1", "deutz_2", "deutz_3", "deutz_4", "deutz_5",
]


def load_all() -> dict:
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


def flatten_load_factors(rec: dict) -> np.ndarray:
    vals = []
    for u in UNITS:
        for lf in rec["load_factors"][u]:
            if lf is not None:
                vals.append(lf)
    return np.array(vals, dtype=float)


def commitment_diff(rec_a: dict, rec_b: dict) -> int:
    """Count of differing unit-hour entries between two commitment schedules."""
    diff = 0
    ua, ub = rec_a["commitment_u"], rec_b["commitment_u"]
    for u in UNITS:
        a = np.array(ua[u])
        b = np.array(ub[u])
        diff += int(np.sum(a != b))
    return diff


def build_per_scenario_rows(recs: dict) -> pd.DataFrame:
    rows = []
    for basis in BASES:
        for scen in SCENARIOS:
            ref = recs[(basis, "B0", scen)]
            for blend in BLENDS:
                rec = recs[(basis, blend, scen)]
                litres = rec["fuel_litres"]
                lf = flatten_load_factors(rec)
                diff = commitment_diff(ref, rec)
                rows.append(
                    {
                        "basis": basis,
                        "blend": blend,
                        "pct_b": PCT[blend],
                        "scenario": scen,
                        "scenario_probability": rec["scenario_probability"],
                        "idle_litres": litres["idle"],
                        "marginal_litres": litres["marginal"],
                        "total_litres": litres["total"],
                        "idle_share_pct": 100.0 * litres["idle"] / litres["total"],
                        "mean_load_factor": float(lf.mean()),
                        "std_load_factor": float(lf.std(ddof=1)) if len(lf) > 1 else 0.0,
                        "min_load_factor": float(lf.min()),
                        "max_load_factor": float(lf.max()),
                        "n_committed_unit_hours": int(len(lf)),
                        "committed_unit_hours": rec["committed_unit_hours"],
                        "differing_unit_hours_vs_b0": diff,
                        "commitment_identical_to_b0": bool(diff == 0),
                    }
                )
    return pd.DataFrame(rows)


def build_expected_rows(df_scen: pd.DataFrame) -> pd.DataFrame:
    """Probability-weighted expected split per (basis, blend), across scenarios."""
    rows = []
    for basis in BASES:
        for blend in BLENDS:
            sub = df_scen[(df_scen.basis == basis) & (df_scen.blend == blend)]
            probs = sub.scenario_probability.values
            probs = probs / probs.sum()
            exp_idle = float((sub.idle_litres.values * probs).sum())
            exp_marg = float((sub.marginal_litres.values * probs).sum())
            exp_total = exp_idle + exp_marg
            exp_mean_lf = float((sub.mean_load_factor.values * probs).sum())
            max_diff = int(sub.differing_unit_hours_vs_b0.max())
            rows.append(
                {
                    "basis": basis,
                    "blend": blend,
                    "pct_b": PCT[blend],
                    "scenario": "expected",
                    "scenario_probability": 1.0,
                    "idle_litres": exp_idle,
                    "marginal_litres": exp_marg,
                    "total_litres": exp_total,
                    "idle_share_pct": 100.0 * exp_idle / exp_total,
                    "mean_load_factor": exp_mean_lf,
                    "std_load_factor": None,
                    "min_load_factor": float(sub.min_load_factor.min()),
                    "max_load_factor": float(sub.max_load_factor.max()),
                    "n_committed_unit_hours": int(sub.n_committed_unit_hours.sum()),
                    "committed_unit_hours": float(
                        (sub.committed_unit_hours.values * probs).sum()
                    ),
                    "differing_unit_hours_vs_b0": max_diff,
                    "commitment_identical_to_b0": bool(max_diff == 0),
                }
            )
    return pd.DataFrame(rows)


def schedule_invariance_summary(df_scen: pd.DataFrame) -> dict:
    max_diff = int(df_scen.differing_unit_hours_vs_b0.max())
    all_identical = bool((df_scen.differing_unit_hours_vs_b0 == 0).all())
    per_basis = {}
    for basis in BASES:
        sub = df_scen[df_scen.basis == basis]
        per_basis[basis] = {
            "identical_across_blends": bool((sub.differing_unit_hours_vs_b0 == 0).all()),
            "max_differing_unit_hours": int(sub.differing_unit_hours_vs_b0.max()),
        }
    worst = df_scen.loc[df_scen.differing_unit_hours_vs_b0.idxmax()]
    return {
        "commitment_identical_across_blends": all_identical,
        "max_differing_unit_hours": max_diff,
        "total_unit_hours_per_run": len(UNITS) * 24,
        "per_basis": per_basis,
        "worst_case": {
            "basis": worst.basis,
            "blend": worst.blend,
            "scenario": worst.scenario,
            "differing_unit_hours": int(worst.differing_unit_hours_vs_b0),
        }
        if max_diff > 0
        else None,
    }


def main():
    TABLES.mkdir(exist_ok=True)
    recs = load_all()
    df_scen = build_per_scenario_rows(recs)
    df_exp = build_expected_rows(df_scen)
    df = pd.concat([df_scen, df_exp], ignore_index=True)
    df = df.sort_values(["basis", "pct_b", "scenario"])
    df.to_csv(TABLES / "tab_r3_loadfactor_split.csv", index=False)

    inv = schedule_invariance_summary(df_scen)

    metrics_path = CODING / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)
    metrics["coupling"] = {
        basis: {
            blend: {
                "idle_litre_share_pct": float(
                    df_exp[(df_exp.basis == basis) & (df_exp.blend == blend)].idle_share_pct.iloc[0]
                ),
                "mean_load_factor": float(
                    df_exp[(df_exp.basis == basis) & (df_exp.blend == blend)].mean_load_factor.iloc[0]
                ),
            }
            for blend in BLENDS
        }
        for basis in BASES
    }
    metrics["schedule_invariance"] = inv
    metrics["task"] = "T11,T12"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    stats_path = CODING / "stats.json"
    stats = {}
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
    for basis in BASES:
        for blend in BLENDS:
            sub = df_scen[(df_scen.basis == basis) & (df_scen.blend == blend)]
            key = f"idle_litre_share_pct__{basis}__{blend}"
            stats[key] = {
                "per_scenario_values": {
                    s: float(sub[sub.scenario == s].idle_share_pct.iloc[0]) for s in SCENARIOS
                },
                "scenario_probabilities": {
                    s: float(sub[sub.scenario == s].scenario_probability.iloc[0]) for s in SCENARIOS
                },
                "expected_value": float(
                    df_exp[(df_exp.basis == basis) & (df_exp.blend == blend)].idle_share_pct.iloc[0]
                ),
                "mean": float(sub.idle_share_pct.mean()),
                "std": float(sub.idle_share_pct.std(ddof=1)),
                "test": "n/a - deterministic model, no sampling noise",
                "statistic": "n/a",
                "p_value": "n/a",
                "effect_size": "n/a",
                "sensitivity_support": {
                    "holds_under_epa_basis": None,
                    "holds_under_generator_basis": None,
                    "holds_across_ongkos_angkut_range": None,
                    "conditional_note": "pending sensitivity sweep",
                },
                "task": "T12",
            }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    pd.set_option("display.width", 200)
    show = df_exp[
        ["basis", "blend", "idle_share_pct", "mean_load_factor",
         "differing_unit_hours_vs_b0", "commitment_identical_to_b0"]
    ].copy()
    show["idle_share_pct"] = show["idle_share_pct"].map(lambda v: f"{v:.1f}")
    show["mean_load_factor"] = show["mean_load_factor"].map(lambda v: f"{v:.3f}")
    print(show.to_string(index=False))
    print()
    print("=== Schedule-invariance test (the novelty check) ===")
    for k, v in inv.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
