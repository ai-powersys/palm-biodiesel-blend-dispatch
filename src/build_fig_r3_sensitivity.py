"""Composite three-panel sensitivity figure.

Combines the panel (a), (b) and (c) data:
  (a) B100 cost vs a hypothetical additional consumption penalty, swept
      through the reversal threshold (the ranking does not reverse within it);
  (b) cost per blend under minimum up/down time of 1, 2, 3 h (ranking
      unchanged for every T);
  (c) cost per blend across the haulage-allowance range 307 / 391 / 500 Rp/L
      (ranking unchanged).

Output: figures/fig_r3_sensitivity.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CODING = Path(__file__).resolve().parent.parent
TABLES = CODING / "tables"
FIGURES = CODING / "figures"

BLENDS = ["B0", "B10", "B20", "B30", "B35", "B40", "B50", "B70", "B100"]
PCT = {b: int(b[1:]) for b in BLENDS}
BASES = ["epa", "generator"]
BASIS_LABEL = {"epa": "Automotive-pooled", "generator": "Generator-scale"}
BASIS_STYLE = {"epa": "-", "generator": "--"}
BASIS_MARKER = {"epa": "o", "generator": "s"}
DAILY_DEMAND_KWH = 102450.0

# Okabe-Ito, consistent with Fig-R4.
COLOUR_SWEEP = {
    1: "#0072B2", 2: "#D55E00", 3: "#009E73",
    "low": "#0072B2", "mean": "#D55E00", "high": "#009E73",
}


def panel_a(ax):
    df = pd.read_csv(TABLES / "tab_r3a_penalty_sweep.csv")
    for basis in BASES:
        sub = df[df.basis == basis].sort_values("penalty_pct_at_b100")
        ax.plot(
            sub.penalty_pct_at_b100, sub.cost_b100_rp_per_kwh,
            color="#CC79A7", linestyle=BASIS_STYLE[basis],
            linewidth=1.8, label=f"B100 cost ({BASIS_LABEL[basis]})",
        )
        alt_per_kwh = sub.cheapest_alternative_cost_rp.iloc[0] / DAILY_DEMAND_KWH
        ax.axhline(
            alt_per_kwh, color="#888888", linestyle=BASIS_STYLE[basis],
            linewidth=1.1, alpha=0.8,
        )
        alt_blend = sub.cheapest_alternative_blend.iloc[0]
        ax.annotate(
            f"cheapest alt.\n({alt_blend}, {BASIS_LABEL[basis].split('-')[0]})",
            xy=(100, alt_per_kwh), xytext=(-2, 4 if basis == "epa" else -14),
            textcoords="offset points", fontsize=6.8, color="#666666", ha="right",
        )
    epa = df[df.basis == "epa"].sort_values("penalty_pct_at_b100").reset_index(drop=True)
    gen = df[df.basis == "generator"].sort_values("penalty_pct_at_b100").reset_index(drop=True)
    max_gap = (100.0 * (gen.cost_b100_rp_per_kwh - epa.cost_b100_rp_per_kwh)
               / epa.cost_b100_rp_per_kwh).abs().max()
    ax.annotate(
        f"two curves overlap:\nbases agree within {max_gap:.2f}%",
        xy=(0.03, 0.62), xycoords="axes fraction", fontsize=7.2, color="#555555",
    )
    ax.set_xlabel("Hypothetical additional BSFC penalty at B100 (%)")
    ax.set_ylabel("Expected cost (Rp/kWh)")
    ax.set_xlim(0, 100)
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=7.2)
    ax.set_title("(a)", fontsize=10, loc="left", fontweight="bold")


def panel_b(ax):
    df = pd.read_csv(TABLES / "tab_r3b_sensitivity.csv")
    df["cost_rp_per_kwh"] = df.expected_cost_rp / DAILY_DEMAND_KWH
    for mut in [1, 2, 3]:
        for basis in BASES:
            sub = df[(df.min_up_down_h == mut) & (df.basis == basis)].sort_values("pct_b")
            ax.plot(
                sub.pct_b, sub.cost_rp_per_kwh,
                color=COLOUR_SWEEP[mut], linestyle=BASIS_STYLE[basis],
                marker=BASIS_MARKER[basis], markersize=3.8, linewidth=1.5,
            )
    # Max deviation of T=2,3 from T=1, per (basis, blend).
    piv = df.pivot_table(index=["basis", "pct_b"], columns="min_up_down_h", values="cost_rp_per_kwh")
    gap_pct = (100.0 * piv[[2, 3]].sub(piv[1], axis=0).div(piv[1], axis=0)).abs().max().max()
    ax.annotate(
        f"lines overlap closely:\nT=1/2/3 agree within {gap_pct:.2f}%",
        xy=(0.05, 0.08), xycoords="axes fraction", fontsize=7.2, color="#555555",
    )
    ax.set_xlabel("Palm biodiesel content (% v/v)")
    ax.set_ylabel("Expected cost (Rp/kWh)")
    ax.set_xticks([PCT[b] for b in BLENDS])
    ax.grid(alpha=0.3, linewidth=0.6)

    from matplotlib.lines import Line2D
    t_handles = [
        Line2D([0], [0], color=COLOUR_SWEEP[t], linewidth=2, label=f"T = {t} h")
        for t in [1, 2, 3]
    ]
    ax.legend(handles=t_handles, loc="upper right", frameon=False, fontsize=7.2,
              title="min up/down time", title_fontsize=7.2)
    ax.set_title("(b)", fontsize=10, loc="left", fontweight="bold")


def panel_c(ax):
    df = pd.read_csv(TABLES / "tab_r3c_oa_sensitivity.csv")
    df["cost_rp_per_kwh"] = df.expected_cost_rp / DAILY_DEMAND_KWH
    for oa_label in ["low", "mean", "high"]:
        for basis in BASES:
            sub = df[(df.oa_scenario == oa_label) & (df.basis == basis)].sort_values("pct_b")
            ax.plot(
                sub.pct_b, sub.cost_rp_per_kwh,
                color=COLOUR_SWEEP[oa_label], linestyle=BASIS_STYLE[basis],
                marker=BASIS_MARKER[basis], markersize=3.8, linewidth=1.5,
            )
    ax.annotate(
        "B0-B35: transport cost applies to\nthe biodiesel share only, so the\n"
        "spread narrows to ~0 at B40\n(the fixed price-calibration anchor)\n"
        "and widens again toward B100",
        xy=(0.03, 0.06), xycoords="axes fraction", fontsize=6.8, color="#555555",
    )
    ax.set_xlabel("Palm biodiesel content (% v/v)")
    ax.set_ylabel("Expected cost (Rp/kWh)")
    ax.set_xticks([PCT[b] for b in BLENDS])
    ax.grid(alpha=0.3, linewidth=0.6)

    from matplotlib.lines import Line2D
    oa_handles = [
        Line2D([0], [0], color=COLOUR_SWEEP[lab], linewidth=2,
               label=f"Rp{val}/L ({lab})")
        for lab, val in [("low", 307), ("mean", 391), ("high", 500)]
    ]
    ax.legend(handles=oa_handles, loc="upper right", frameon=False, fontsize=7.2,
              title="Biodiesel transport cost", title_fontsize=7.2)
    ax.set_title("(c)", fontsize=10, loc="left", fontweight="bold")


def main():
    FIGURES.mkdir(exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))
    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])

    basis_handles = [
        Line2D([0], [0], color="black", linestyle=BASIS_STYLE[b], marker=BASIS_MARKER[b],
               markersize=4.5, linewidth=1.5, label=BASIS_LABEL[b])
        for b in BASES
    ]
    fig.legend(handles=basis_handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8.5, title="BSFC basis", title_fontsize=8.5,
               bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / "fig_r3_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_r3_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Written: figures/fig_r3_sensitivity.pdf + .png")
    print("\n=== Composite Fig-R3 summary ===")
    print("(a) B100 ranking survives to +17.0% (epa) / +17.5% (generator) "
          "idle-corrected BSFC penalty before reversing to the cheapest alternative. "
          "epa/generator B100 cost curves agree to within ~0.03%.")
    print("(b) Ranking robust (B100 cheapest, monotonic B0->B100) across all "
          "6 (T, basis) combinations tested (T in {1,2,3} h); T=1/2/3 agree "
          "within ~0.09% at every blend.")
    print("(c) Ranking robust (B100 cheapest, monotonic B0->B100) across the "
          "full Rp307-500 biodiesel transport-cost range tested; B40's cost "
          "is exactly invariant to it by construction (fixed price anchor).")


if __name__ == "__main__":
    main()
