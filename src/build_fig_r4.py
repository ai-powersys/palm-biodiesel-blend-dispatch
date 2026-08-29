"""Emissions ladder: five species vs blend ratio, each normalised to its B0 value.

Post-processing from tables/tab_r2_emissions.csv. Normalising each species to
its own B0 = 100% puts species that differ by orders of magnitude in absolute
mass on one axis and makes the counter-trend visible (NOx and basis-B PM
rising while CO, THC and basis-A PM fall).

Output: figures/fig_r4_emissions.{pdf,png}
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
SPECIES = ["co2", "nox", "co", "thc", "pm"]
SPECIES_LABEL = {
    "co2": "CO$_2$", "nox": "NO$_x$", "co": "CO", "thc": "THC", "pm": "PM",
}
# Colourblind-safe (Okabe-Ito), one colour per species; basis distinguished by line style.
SPECIES_COLOUR = {
    "co2": "#0072B2", "nox": "#D55E00", "co": "#009E73",
    "thc": "#CC79A7", "pm": "#E69F00",
}
BASIS_STYLE = {"epa": "-", "generator": "--"}
BASIS_MARKER = {"epa": "o", "generator": "s"}


def main():
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(TABLES / "tab_r2_emissions.csv")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.7))

    for species in SPECIES:
        col = f"{species}_expected_kg_per_day"
        for basis in BASES:
            sub = df[df.basis == basis].sort_values("pct_b")
            b0_val = sub[sub.blend == "B0"][col].iloc[0]
            normalized = 100.0 * sub[col].values / b0_val
            ax.plot(
                sub.pct_b,
                normalized,
                color=SPECIES_COLOUR[species],
                linestyle=BASIS_STYLE[basis],
                marker=BASIS_MARKER[basis],
                markersize=4.5,
                linewidth=1.8,
                label=f"{SPECIES_LABEL[species]} ({BASIS_LABEL[basis]})",
            )

    ax.axhline(100, color="#888888", linewidth=0.8, linestyle=":", zorder=0)
    ax.set_xlabel("Palm biodiesel content (% v/v)")
    ax.set_ylabel("Expected daily emissions,\nnormalized to B0 = 100%")
    ax.set_xticks([PCT[b] for b in BLENDS])
    ax.grid(alpha=0.3, linewidth=0.6)

    # Two-part legend: species (colour) + basis (line style). Moved below the
    # axes (, same pattern as Fig-R2) -- at the shrunk 1/3-page
    # height, both legends previously sat inside the axes (upper-left,
    # lower-left) and collided with the data (PM-generator dips to ~50% at
    # B10 then climbs to ~135% by B100, sweeping through both legend boxes).
    from matplotlib.lines import Line2D

    species_handles = [
        Line2D([0], [0], color=SPECIES_COLOUR[s], linewidth=2, label=SPECIES_LABEL[s])
        for s in SPECIES
    ]
    basis_handles = [
        Line2D([0], [0], color="black", linestyle=BASIS_STYLE[b], marker=BASIS_MARKER[b],
               markersize=4.5, linewidth=1.5, label=BASIS_LABEL[b])
        for b in BASES
    ]
    # BSFC basis: small 2-entry legend, fits cleanly inside the axes at the
    # top-left corner (data there is flat at ~100%, no line reaches that
    # high that early -- unlike the 5-entry Species legend, which needed to
    # move below the axes). Frees the bottom row for Species to be centred.
    leg1 = ax.legend(handles=basis_handles, loc="upper left", frameon=False,
                      fontsize=7.5, title="BSFC basis", title_fontsize=7.5,
                      handlelength=1.8)
    ax.add_artist(leg1)
    ax.legend(handles=species_handles, loc="lower left", ncol=5, frameon=False,
               fontsize=8, title="Species", title_fontsize=8,
               handlelength=1.4, columnspacing=1.0)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_r4_emissions.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_r4_emissions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Console summary: direction of each species B0->B100, per basis.
    print("=== Fig-R4 emissions ladder: B0 -> B100 direction ===")
    for basis in BASES:
        sub = df[df.basis == basis].sort_values("pct_b")
        print(f"\n{BASIS_LABEL[basis]} ({basis}):")
        for species in SPECIES:
            col = f"{species}_expected_kg_per_day"
            b0_val = sub[sub.blend == "B0"][col].iloc[0]
            b100_val = sub[sub.blend == "B100"][col].iloc[0]
            pct_change = 100.0 * (b100_val - b0_val) / b0_val
            direction = "UP" if pct_change > 0 else "DOWN"
            min_row = sub.loc[sub[col].idxmin()]
            max_row = sub.loc[sub[col].idxmax()]
            monotonic = (sub[col].is_monotonic_increasing or sub[col].is_monotonic_decreasing)
            shape = "monotonic" if monotonic else (
                f"non-monotonic (min at {min_row.blend}, max at {max_row.blend})"
            )
            print(f"  {SPECIES_LABEL[species]:6s} {direction:4s} {pct_change:+6.1f}% at B100  [{shape}]")

    print(f"\nWritten: figures/fig_r4_emissions.pdf + .png")


if __name__ == "__main__":
    main()
