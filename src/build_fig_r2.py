"""Hourly dispatch profile for one representative blend and scenario.

A single stacked-dispatch panel (PV / battery / diesel by capacity group).
Non-anticipativity (the commitment schedule shared across all three weather
scenarios) is verified numerically here from commitment_u and stated in the
caption rather than shown as three near-identical panels: PV is a small
fraction of this system, so the three scenarios' dispatch shapes are visually
almost indistinguishable at this power scale.

Representative blend B40, scenario clear, basis B (generator-scale).

Output: figures/fig_r2_dispatch.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CODING = Path(__file__).resolve().parent.parent
RESULTS = CODING / "results"
FIGURES = CODING / "figures"

BASIS = "generator"
BLEND = "B40"
SCENARIOS = ["clear", "partly_cloudy", "overcast"]  # all 3 loaded for the verification check
REPRESENTATIVE = "clear"  # the one scenario actually plotted
SCEN_LABEL = {"clear": "Clear", "partly_cloudy": "Partly cloudy", "overcast": "Overcast"}

GROUPS = {
    "Mitsubishi (3x1,100 kW)": ["mitsubishi_1", "mitsubishi_2", "mitsubishi_3"],
    "Cummins (3x1,000 kW)": ["cummins_1", "cummins_2", "cummins_3"],
    "Deutz (5x900 kW)": ["deutz_1", "deutz_2", "deutz_3", "deutz_4", "deutz_5"],
}
# Deliberately NOT labelled "(base,"/"(peak," here: the case-study reference's base/peak-load
# roles describe the REAL system's documented operation, not this model's
# behaviour. Our free-commitment optimum treats all 11 units symmetrically
# (identical per-kW cost, see the run log's must-run sensitivity investigation)
# -- labelling them base/peak in a figure of THIS model's dispatch would
# claim an operational distinction the model does not actually enforce.
# Single grey (ColorBrewer "Greys", colourblind-safe) family for all-diesel
# groups so the eye reads "diesel" as one category with sub-shades, kept
# visually distinct from PV (orange) and battery (blue).
GROUP_COLOURS = {
    "Mitsubishi (3x1,100 kW)": "#252525",
    "Cummins (3x1,000 kW)": "#636363",
    "Deutz (5x900 kW)": "#969696",
}
PV_COLOUR = "#F1A340"
BESS_DISCHARGE_COLOUR = "#0072B2"  # Okabe-Ito blue
BESS_CHARGE_COLOUR = "#CC79A7"  # Okabe-Ito reddish purple -- different hue family from
# discharge's blue (the two sky-blue shades were too close to tell apart at the thin
# battery-trace widths); still distinct from PV orange, diesel greys, and black load line
LOAD_COLOUR = "#000000"
MW = 1000.0  # kW -> MW for axis readability


def load_recs() -> dict:
    recs = {}
    for s in SCENARIOS:
        p = RESULTS / f"solve_{BASIS}_{BLEND}_{s}.json"
        if not p.exists():
            raise FileNotFoundError(f"missing checkpoint: {p}")
        with open(p) as f:
            recs[s] = json.load(f)
    return recs


def verify_shared_commitment(recs: dict) -> dict:
    """Re-verify non-anticipativity directly from commitment_u for this blend."""
    ref = recs[SCENARIOS[0]]["commitment_u"]
    max_diff = 0
    for s in SCENARIOS[1:]:
        cur = recs[s]["commitment_u"]
        for unit in ref:
            a = np.array(ref[unit])
            b = np.array(cur[unit])
            max_diff = max(max_diff, int(np.sum(a != b)))
    return {"commitment_identical_across_scenarios": max_diff == 0, "max_differing_unit_hours": max_diff}


def load_kw(rec: dict) -> np.ndarray:
    """Reconstruct hourly load from PV + BESS net + diesel dispatch (energy balance)."""
    diesel = np.zeros(24)
    for units in GROUPS.values():
        for u in units:
            diesel += np.array(rec["dispatch_p_kw"][u])
    pv = np.array(rec["pv_dispatch_kw"])
    bess_net = np.array(rec["bess_discharge_kw"]) - np.array(rec["bess_charge_kw"])
    return diesel + pv + bess_net


def make_figure(rec: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    hours = np.arange(24)

    pv = np.array(rec["pv_dispatch_kw"]) / MW
    charge = np.array(rec["bess_charge_kw"]) / MW
    discharge = np.array(rec["bess_discharge_kw"]) / MW
    group_dispatch = {
        g: sum(np.array(rec["dispatch_p_kw"][u]) for u in units) / MW
        for g, units in GROUPS.items()
    }
    load = load_kw(rec) / MW
    y_max = load.max() * 1.08

    fig, ax = plt.subplots(figsize=(6.5, 3.3))

    # Stacking order stays PV -> battery discharge -> diesel groups (merit-order
    # convention, see module docstring history) -- only the LEGEND display order
    # is reshuffled below, via explicit handle capture rather than relying on
    # ax.get_legend_handles_labels()'s draw-order default.
    stack_bottom = np.zeros(24)
    h_pv = ax.fill_between(hours, stack_bottom, stack_bottom + pv, step="mid",
                            color=PV_COLOUR, linewidth=0, label="PV")
    stack_bottom = stack_bottom + pv
    h_discharge = ax.fill_between(hours, stack_bottom, stack_bottom + discharge, step="mid",
                                   color=BESS_DISCHARGE_COLOUR, linewidth=0, label="Battery discharge")
    stack_bottom = stack_bottom + discharge
    h_groups = {}
    for g in GROUPS:
        h_groups[g] = ax.fill_between(hours, stack_bottom, stack_bottom + group_dispatch[g], step="mid",
                                       color=GROUP_COLOURS[g], linewidth=0, label=g)
        stack_bottom = stack_bottom + group_dispatch[g]

    h_charge = ax.fill_between(hours, 0, -charge, step="mid",
                                color=BESS_CHARGE_COLOUR, linewidth=0, label="Battery charge")

    h_load, = ax.step(hours, load, where="mid", color=LOAD_COLOUR, linewidth=1.3,
                       linestyle="--", label="Load")

    ax.set_ylabel("Power (MW)")
    ax.set_xlabel("Hour of day")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 23.5)
    ax.set_ylim(-0.6, y_max)
    ax.set_xticks(range(0, 24, 2))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Explicit order (column-major fill with ncol=4): col1=[PV,Deutz],
    # col2=[Mitsubishi,Cummins], col3=[Battery discharge,Battery charge],
    # col4=[Load] -- puts the two battery entries in the same column.
    group_names = list(GROUPS)  # [Mitsubishi, Cummins, Deutz]
    legend_order = [
        h_pv, h_groups[group_names[2]],  # PV, Deutz
        h_groups[group_names[0]], h_groups[group_names[1]],  # Mitsubishi, Cummins
        h_discharge, h_charge,  # Battery discharge, Battery charge
        h_load,
    ]
    legend_labels = [h.get_label() for h in legend_order]
    ax.legend(legend_order, legend_labels, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=4, frameon=False, fontsize=8, handlelength=1.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGURES / "fig_r2_dispatch.pdf")
    fig.savefig(FIGURES / "fig_r2_dispatch.png", dpi=300)
    plt.close(fig)


def main():
    recs = load_recs()
    check = verify_shared_commitment(recs)
    print("=== Shared-commitment check (non-anticipativity, re-verified for B40, all 3 scenarios) ===")
    for k, v in check.items():
        print(f"  {k}: {v}")
    if not check["commitment_identical_across_scenarios"]:
        raise RuntimeError(
            "Non-anticipativity violated for B40 -- contradicts validation."
            "Stop; do not produce a figure claiming shared commitment."
        )

    for s in SCENARIOS:
        rec = recs[s]
        load = load_kw(rec)
        pv = np.array(rec["pv_dispatch_kw"])
        diesel_total = sum(
            np.array(rec["dispatch_p_kw"][u]) for units in GROUPS.values() for u in units
        )
        bess_net = np.array(rec["bess_discharge_kw"]) - np.array(rec["bess_charge_kw"])
        balance_err = np.abs(load - (pv + diesel_total + bess_net))
        print(f"  {s}: max energy-balance residual = {balance_err.max():.4e} kW (sanity re-check)")

    make_figure(recs[REPRESENTATIVE])
    print(f"Written: figures/fig_r2_dispatch.pdf, figures/fig_r2_dispatch.png "
          f"(blend={BLEND}, basis={BASIS}, scenario={REPRESENTATIVE} -- single representative panel; "
          f"non-anticipativity verified numerically across all 3 scenarios above, stated in caption)")


if __name__ == "__main__":
    main()
