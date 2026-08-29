"""Render the three manuscript figures at final size from the canonical CSVs.

  Figure 1  fig_m1_cost_ladder    cost vs blend, both bases, weather spread shaded
  Figure 2  fig_m2_penalty_sweep  B100 cost vs a hypothetical consumption penalty
  Figure 3  fig_m3_emissions      five species vs blend, normalised to B0

The full pre-cut figure set (dispatch profile, three-panel sensitivity) is
also kept in figures/. Pure post-processing, no re-solve.

    python src/build_manuscript_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402
from pathlib import Path                 # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "tables"
FIG = ROOT / "figures"

BLENDS = ["B0", "B10", "B20", "B30", "B35", "B40", "B50", "B70", "B100"]
PCT = {b: int(b[1:]) for b in BLENDS}
BASES = ["epa", "generator"]
LABEL = {"epa": "Basis A", "generator": "Basis B"}
COLOUR = {"epa": "#0072B2", "generator": "#D55E00"}
MARKER = {"epa": "o", "generator": "s"}
STYLE = {"epa": "-", "generator": "--"}
DAILY_DEMAND_KWH = 102450.0


def save(fig, stem):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}.pdf / .png")


def fig1_cost_ladder():
    """Cost vs blend, both bases, with the clear-to-overcast band shaded.

    The band carries the per-scenario spread visually: it is narrow next to
    the blend-to-blend difference, which is the point the ranking rests on.
    """
    df = pd.read_csv(TAB / "tab_r1_blend_cost.csv")
    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    for basis in BASES:
        sub = df[df.basis == basis].copy()
        sub["order"] = sub.blend.map(PCT)
        sub = sub.sort_values("order")
        lo = sub.cost_clear_rp / DAILY_DEMAND_KWH
        hi = sub.cost_overcast_rp / DAILY_DEMAND_KWH
        ax.fill_between(sub.order, lo, hi, color=COLOUR[basis], alpha=0.16,
                        linewidth=0)
        ax.plot(sub.order, sub.expected_cost_rp_per_kwh,
                marker=MARKER[basis], linestyle=STYLE[basis],
                color=COLOUR[basis], linewidth=2, markersize=5.5,
                label=LABEL[basis])

    ax.axvline(40, color="#555555", linewidth=1, linestyle=":", zorder=0)
    ylo, yhi = ax.get_ylim()
    ax.annotate("B40 (mandated)", xy=(41, yhi - 0.06 * (yhi - ylo)),
                fontsize=8.5, color="#555555")
    ax.annotate("shaded band: clear to overcast scenario",
                xy=(0.035, 0.08), xycoords="axes fraction",
                fontsize=7.5, color="#555555")

    ax.set_xlabel("Palm biodiesel content (% v/v)")
    ax.set_ylabel("Expected operating cost (Rp/kWh of demand)")
    ax.set_xticks([PCT[b] for b in BLENDS])
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "fig_m1_cost_ladder")


def fig2_penalty_sweep():
    """The old panel (a), now standalone and full width."""
    df = pd.read_csv(TAB / "tab_r3a_penalty_sweep.csv")
    fig, ax = plt.subplots(figsize=(6.6, 3.5))

    for basis in BASES:
        sub = df[df.basis == basis].sort_values("penalty_pct_at_b100")
        ax.plot(sub.penalty_pct_at_b100, sub.cost_b100_rp_per_kwh,
                color="#CC79A7", linestyle=STYLE[basis], linewidth=1.8,
                label=f"B100 cost ({LABEL[basis]})")
        alt = sub.cheapest_alternative_cost_rp.iloc[0] / DAILY_DEMAND_KWH
        ax.axhline(alt, color="#888888", linestyle=STYLE[basis],
                   linewidth=1.1, alpha=0.8)
        ax.annotate(f"cheapest alternative ({sub.cheapest_alternative_blend.iloc[0]},"
                    f" {LABEL[basis].lower()})",
                    xy=(99, alt), xytext=(-2, 5 if basis == "epa" else -13),
                    textcoords="offset points", fontsize=7, color="#666666",
                    ha="right")

    epa = df[df.basis == "epa"].sort_values("penalty_pct_at_b100").reset_index(drop=True)
    gen = df[df.basis == "generator"].sort_values("penalty_pct_at_b100").reset_index(drop=True)
    gap = (100.0 * (gen.cost_b100_rp_per_kwh - epa.cost_b100_rp_per_kwh)
           / epa.cost_b100_rp_per_kwh).abs().max()
    ax.annotate(f"the two curves overlap: bases agree within {gap:.2f}%",
                xy=(0.04, 0.60), xycoords="axes fraction", fontsize=7.5,
                color="#555555")

    ax.set_xlabel("Hypothetical additional BSFC penalty at B100 (%)")
    ax.set_ylabel("Expected cost (Rp/kWh)")
    ax.set_xlim(0, 100)
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    save(fig, "fig_m2_penalty_sweep")


def fig3_emissions():
    """Five species vs blend, each normalised to its own B0 value."""
    df = pd.read_csv(TAB / "tab_r2_emissions.csv")
    species = [("co2_expected_kg_per_day", "CO$_2$", "#0072B2"),
               ("nox_expected_kg_per_day", "NO$_x$", "#D55E00"),
               ("co_expected_kg_per_day", "CO", "#009E73"),
               ("thc_expected_kg_per_day", "THC", "#CC79A7"),
               ("pm_expected_kg_per_day", "PM", "#000000")]
    fig, ax = plt.subplots(figsize=(7.1, 3.6))

    for col, name, colour in species:
        for basis in BASES:
            sub = df[df.basis == basis].copy()
            sub["order"] = sub.blend.map(PCT)
            sub = sub.sort_values("order")
            base = sub[sub.blend == "B0"][col].iloc[0]
            ax.plot(sub.order, 100.0 * sub[col] / base,
                    linestyle=STYLE[basis], color=colour, linewidth=1.7,
                    label=name if basis == "epa" else None)

    ax.axhline(100, color="#999999", linewidth=0.9, linestyle=":", zorder=0)
    ax.set_xlabel("Palm biodiesel content (% v/v)")
    ax.set_ylabel("Daily emissions, B0 = 100%")
    ax.set_xticks([PCT[b] for b in BLENDS])
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.annotate("solid: basis A     dashed: basis B",
                xy=(0.035, 0.06), xycoords="axes fraction",
                fontsize=7.5, color="#555555")
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center")
    save(fig, "fig_m3_emissions")


if __name__ == "__main__":
    print("writing manuscript figures:")
    fig1_cost_ladder()
    fig2_penalty_sweep()
    fig3_emissions()
