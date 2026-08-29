"""Render the manuscript's data tables as Markdown from the canonical CSVs.

The only path from a solved result to a number in the paper; nothing is typed
by hand.

  Table 2  per-blend fuel properties   <- data/bsfc_bases.csv + cost_coefficients.csv
  Table 4  cost per blend by scenario  <- tables/tab_r1_blend_cost.csv
  Table 5  stochastic-layer ablation   <- tables/tab_r4_ablation_stochastic.csv

Tables 1 and 3 (system spec, assumptions and provenance) are written directly
in the manuscript. Run after build_tab_r1..r4.

    python src/build_manuscript_tables.py   ->  tables/manuscript_tables.md
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "tables"
DATA = ROOT / "data"
OUT = TAB / "manuscript_tables.md"

BLENDS = ["B0", "B10", "B20", "B30", "B35", "B40", "B50", "B70", "B100"]
BASES = [("epa", "A"), ("generator", "B")]

BASIS_NOTE = (
    "Basis A is the pooled-engine fit of Lapuerta et al. [9], basis B the "
    "generator sweep of Lin et al. [15] (Section 2.3)."
)


def fmt(x, nd=0):
    return f"{x:,.{nd}f}"


def by_blend(df):
    df = df.copy()
    df["blend"] = pd.Categorical(df["blend"], categories=BLENDS, ordered=True)
    return df.sort_values("blend")


def table2():
    """Per-blend fuel properties: price, density, and both consumption ladders."""
    bs = by_blend(pd.read_csv(DATA / "bsfc_bases.csv"))

    # One price per blend, at the mean haulage allowance actually used.
    cc = pd.read_csv(DATA / "cost_coefficients.csv")
    mid = sorted(cc.ongkos_angkut_rp.unique())[len(cc.ongkos_angkut_rp.unique()) // 2]
    price = (cc[cc.ongkos_angkut_rp == mid]
             .drop_duplicates("blend")
             .set_index("blend")["price_rp_per_l"])

    lines = [
        "**Table 2.** Per-blend fuel price at the mean haulage allowance of "
        f"Rp {mid}/L (pre-tax, from equation (7)), blend density, and marginal "
        "fuel consumption under both bases. " + BASIS_NOTE + " Basis B points "
        "marked *int.* are linear interpolations between adjacent measured "
        "points, never extrapolations beyond the measured range.",
        "",
        "| Blend | Price (Rp/L) | Density (kg/m³) | $A_x$ basis A (L/kWh) "
        "| $A_x$ basis B (L/kWh) |",
        "|---|---|---|---|---|",
    ]
    for _, r in bs.iterrows():
        mark = " *int.*" if r.basis_b_interpolated else ""
        lines.append(
            f"| {r.blend} | {fmt(price[r.blend], 1)} | {fmt(r.density_g_per_l, 2)} "
            f"| {r.A_x_basis_a_L_per_kWh:.5f} "
            f"| {r.A_x_basis_b_L_per_kWh:.5f}{mark} |"
        )
    return "\n".join(lines)


def table4():
    """Cost per blend, all three scenarios, both bases."""
    df = by_blend(pd.read_csv(TAB / "tab_r1_blend_cost.csv"))
    lines = [
        "**Table 4.** Daily operating cost per blend under each weather "
        "scenario (million Rp), the probability-weighted expected cost per "
        "unit of served demand, and daily fuel volume. All three per-scenario "
        "values are given rather than a summary statistic alone, since the "
        "scenario set has only three members. " + BASIS_NOTE,
        "",
        "| Basis | Blend | Clear | Partly cloudy | Overcast "
        "| Expected (Rp/kWh) | Fuel (L/day) |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, label in BASES:
        for _, r in df[df.basis == key].iterrows():
            lines.append(
                f"| {label} | {r.blend} "
                f"| {fmt(r.cost_clear_rp / 1e6, 1)} "
                f"| {fmt(r.cost_partly_cloudy_rp / 1e6, 1)} "
                f"| {fmt(r.cost_overcast_rp / 1e6, 1)} "
                f"| {fmt(r.expected_cost_rp_per_kwh, 2)} "
                f"| {fmt(r.expected_fuel_l_per_day, 0)} |"
            )
    return "\n".join(lines)


def table5():
    """Stochastic-layer ablation, one row per blend with both bases side by
    side. EEV's objective is deliberately not shown: it is not serving the same
    load, so the gap is reported in energy."""
    df = pd.read_csv(TAB / "tab_r4_ablation_stochastic.csv")
    df["evpi_pct"] = 100.0 * df.evpi_rp_minus_ws_rp / df.rp_expected_cost_rp
    piv = {k: by_blend(df[df.basis == k]).set_index("blend") for k, _ in BASES}

    lines = [
        "**Table 5.** Stochastic-layer ablation per blend, both bases side by "
        "side. The recourse solution is set against a perfect-foresight bound "
        "and against a deterministic plan committed on expected weather. The "
        "recourse and perfect-foresight solutions serve demand in full in every "
        "case, so only the expected-weather plan leaves energy unserved, and "
        "that shortfall falls entirely in the overcast scenario. The "
        "expected-weather objective is not tabulated because it is not serving "
        "the same load, so the gap is reported in energy. " + BASIS_NOTE,
        "",
        "| Blend | EVPI, basis A (%) | EVPI, basis B (%) "
        "| Unserved, basis A (kWh/day) | Unserved, basis B (kWh/day) "
        "| Unserved (% of daily demand) |",
        "|---|---|---|---|---|---|",
    ]
    for bl in BLENDS:
        a, b = piv["epa"].loc[bl], piv["generator"].loc[bl]
        lo = min(a.eev_unserved_pct_of_daily_demand, b.eev_unserved_pct_of_daily_demand)
        hi = max(a.eev_unserved_pct_of_daily_demand, b.eev_unserved_pct_of_daily_demand)
        rng = fmt(lo, 3) if abs(hi - lo) < 5e-4 else f"{fmt(lo, 3)}-{fmt(hi, 3)}"
        lines.append(
            f"| {bl} | {fmt(a.evpi_pct, 3)} | {fmt(b.evpi_pct, 3)} "
            f"| {fmt(a.eev_expected_unserved_kwh, 1)} "
            f"| {fmt(b.eev_expected_unserved_kwh, 1)} | {rng} |"
        )
    return "\n".join(lines)


def main():
    blocks = [
        "<!-- GENERATED by src/build_manuscript_tables.py from data/*.csv and "
        "tables/*.csv. Do not edit by hand. -->",
        table2(), table4(), table5(),
    ]
    OUT.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"wrote {OUT}\n")
    print("\n\n".join(blocks))


if __name__ == "__main__":
    main()
