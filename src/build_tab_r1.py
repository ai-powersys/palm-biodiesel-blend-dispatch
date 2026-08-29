"""Aggregate the main sweep into the cost table and the cost-ladder figure.

Rebuilds from results/ (never from a cached solve). Reports all three
per-scenario values alongside the mean and standard deviation, since three
scenarios are too few for a summary statistic to stand alone. Also runs the
MIP-gap margin check (smallest adjacent-blend cost gap vs the solver
tolerance).

Output: tables/tab_r1_blend_cost.csv, figures/fig_r1_cost_ladder.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CODING = Path(__file__).resolve().parent.parent
RESULTS = CODING / "results"
TABLES = CODING / "tables"
FIGURES = CODING / "figures"

BLENDS = ["B0", "B10", "B20", "B30", "B35", "B40", "B50", "B70", "B100"]
PCT = {b: int(b[1:]) for b in BLENDS}
BASES = ["epa", "generator"]
# Legend keys stay short/descriptive, no citation -- standard figure convention.
# Full attribution (Basis A = Lapuerta, Armas, Rodriguez-Fernandez 2008 (Lapuerta et al. (2008)),
# reporting the US EPA predictive equation; Basis B = Lin, Lee, Hou 2006 (Lin et al. (2006)))
# belongs in the figure CAPTION and Section 3.4 body text at S5 drafting, not here --
# putting "(EPA)" in the legend alone risks misreading it as a direct EPA citation,
# when Lapuerta et al. (2008) is a secondary review paper we actually read, not the EPA source itself.
BASIS_LABEL = {"epa": "Automotive-pooled", "generator": "Generator-scale"}
SCENARIOS = ["clear", "partly_cloudy", "overcast"]
DAILY_DEMAND_KWH = 102450.0
REVERSAL_THRESHOLD_PCT = 65.4  # marginal-only, Tab-R0


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


def build_table(recs: dict) -> pd.DataFrame:
    rows = []
    for basis in BASES:
        for blend in BLENDS:
            per_scen_cost = {s: recs[(basis, blend, s)]["scenario_cost_rp"] for s in SCENARIOS}
            probs = {s: recs[(basis, blend, s)]["scenario_probability"] for s in SCENARIOS}
            expected = recs[(basis, blend, SCENARIOS[0])]["expected_objective_all_scenarios_rp"]
            vals = np.array([per_scen_cost[s] for s in SCENARIOS])
            litres = {s: recs[(basis, blend, s)]["fuel_litres"] for s in SCENARIOS}
            exp_litres = sum(probs[s] * litres[s]["total"] for s in SCENARIOS)
            exp_idle = sum(probs[s] * litres[s]["idle"] for s in SCENARIOS)
            exp_marg = sum(probs[s] * litres[s]["marginal"] for s in SCENARIOS)
            exp_hours = sum(
                probs[s] * recs[(basis, blend, s)]["committed_unit_hours"] for s in SCENARIOS
            )
            solver = recs[(basis, blend, SCENARIOS[0])]["solver"]
            rows.append(
                {
                    "basis": basis,
                    "blend": blend,
                    "pct_b": PCT[blend],
                    "cost_clear_rp": per_scen_cost["clear"],
                    "cost_partly_cloudy_rp": per_scen_cost["partly_cloudy"],
                    "cost_overcast_rp": per_scen_cost["overcast"],
                    "cost_mean_rp": float(vals.mean()),
                    "cost_std_rp": float(vals.std(ddof=1)),
                    "expected_cost_rp": expected,
                    "expected_cost_rp_per_kwh": expected / DAILY_DEMAND_KWH,
                    "expected_fuel_l_per_day": exp_litres,
                    "expected_idle_l_per_day": exp_idle,
                    "expected_marginal_l_per_day": exp_marg,
                    "expected_committed_genset_hours": exp_hours,
                    "solver_termination": solver["termination_condition"],
                    "achieved_mip_gap": solver["achieved_mip_gap"],
                    "wall_clock_s": solver["wall_clock_s"],
                }
            )
    return pd.DataFrame(rows)


def gap_margin_check(df: pd.DataFrame) -> dict:
    max_gap = float(df["achieved_mip_gap"].max())
    max_obj = float(df["expected_cost_rp"].max())
    tolerance = max_gap * max_obj

    smallest = None
    detail = {}
    for basis in BASES:
        sub = df[df.basis == basis].sort_values("pct_b")
        diffs = sub["expected_cost_rp"].diff().abs().dropna()
        pairs = list(zip(sub.blend[:-1], sub.blend[1:], diffs))
        lo = min(pairs, key=lambda t: t[2])
        detail[basis] = {"pair": f"{lo[0]}-{lo[1]}", "diff_rp": float(lo[2])}
        if smallest is None or lo[2] < smallest:
            smallest = float(lo[2])

    ties = []
    for basis in BASES:
        sub = df[df.basis == basis].sort_values("pct_b").reset_index(drop=True)
        for i in range(len(sub) - 1):
            d = abs(sub.expected_cost_rp[i + 1] - sub.expected_cost_rp[i])
            if d <= tolerance:
                ties.append(f"{basis}:{sub.blend[i]}~{sub.blend[i+1]}")

    return {
        "max_achieved_mip_gap": max_gap,
        "largest_objective_rp": max_obj,
        "absolute_tolerance_rp": tolerance,
        "smallest_adjacent_blend_difference_rp": smallest,
        "smallest_per_basis": detail,
        "margin_factor": smallest / tolerance if tolerance else float("inf"),
        "margin_check_passed": bool(smallest > tolerance),
        "blends_within_gap_reported_tied": ties,
        "all_cases_optimal": bool((df.solver_termination == "optimal").all()),
    }


def make_figure(df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    colours = {"epa": "#0072B2", "generator": "#D55E00"}
    markers = {"epa": "o", "generator": "s"}
    styles = {"epa": "-", "generator": "--"}

    for basis in BASES:
        sub = df[df.basis == basis].sort_values("pct_b")
        ax.plot(
            sub.pct_b,
            sub.expected_cost_rp_per_kwh,
            marker=markers[basis],
            linestyle=styles[basis],
            color=colours[basis],
            linewidth=2,
            markersize=6,
            label=BASIS_LABEL[basis],
        )

    ax.axvline(40, color="#555555", linewidth=1, linestyle=":", zorder=0)
    ax.annotate(
        "B40 (mandated)",
        xy=(40, ax.get_ylim()[1]),
        xytext=(41, ax.get_ylim()[1] - 0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
        fontsize=9,
        color="#555555",
    )

    ax.set_xlabel("Palm biodiesel content (% v/v)")
    ax.set_ylabel("Expected operating cost (Rp/kWh of demand)")
    ax.set_xticks([PCT[b] for b in BLENDS])
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_r1_cost_ladder.pdf")
    fig.savefig(FIGURES / "fig_r1_cost_ladder.png", dpi=300)
    plt.close(fig)


def main():
    TABLES.mkdir(exist_ok=True)
    recs = load_all()
    df = build_table(recs)
    df.to_csv(TABLES / "tab_r1_blend_cost.csv", index=False)

    check = gap_margin_check(df)
    # Load-modify-save, NOT a full overwrite: metrics.json is shared by
    # every Tab/Fig build script ( coupling/schedule_invariance,
    # emissions_per_blend, reversal_threshold/
    # t_sensitivity/oa_sensitivity, stochastic_ablation, ...).
    # A bare `json.dump({...}, f)` here silently wiped all of those keys
    # every time this script re-ran (
    # triggered by an unrelated Fig-R1 resize re-run -- see the run log).
    metrics_path = CODING / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics.update({
        "task": "T11",
        "expected_cost_rp_per_blend": {
            b: {
                r.basis: r.expected_cost_rp
                for r in df[df.blend == b].itertuples()
            }
            for b in BLENDS
        },
        "expected_cost_rp_per_kwh": {
            b: {
                r.basis: r.expected_cost_rp_per_kwh
                for r in df[df.blend == b].itertuples()
            }
            for b in BLENDS
        },
        "cheapest_blend": {
            basis: df[df.basis == basis].sort_values("expected_cost_rp").blend.iloc[0]
            for basis in BASES
        },
        "mip_gap_check": check,
    })
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    make_figure(df)

    pd.set_option("display.width", 200)
    show = df[
        ["basis", "blend", "expected_cost_rp", "expected_cost_rp_per_kwh",
         "cost_clear_rp", "cost_partly_cloudy_rp", "cost_overcast_rp",
         "expected_fuel_l_per_day", "expected_committed_genset_hours", "achieved_mip_gap"]
    ].copy()
    for c in ["expected_cost_rp", "cost_clear_rp", "cost_partly_cloudy_rp", "cost_overcast_rp"]:
        show[c] = show[c].map(lambda v: f"{v:,.0f}")
    show["expected_cost_rp_per_kwh"] = show["expected_cost_rp_per_kwh"].map(lambda v: f"{v:,.1f}")
    show["expected_fuel_l_per_day"] = show["expected_fuel_l_per_day"].map(lambda v: f"{v:,.0f}")
    show["expected_committed_genset_hours"] = show["expected_committed_genset_hours"].map(lambda v: f"{v:.1f}")
    show["achieved_mip_gap"] = show["achieved_mip_gap"].map(lambda v: f"{v:.2e}")
    print(show.to_string(index=False))

    print()
    print("=== MIP-gap margin check (the project notes requirement) ===")
    for k, v in check.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
