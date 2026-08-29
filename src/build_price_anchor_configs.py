"""Cost-coefficient tables for the price-anchor sensitivity grid.

The price ladder rests on two external anchors: A1 = the regional B40 market
price (18,300 Rp/L) and A2 = the B100 index price (14,924 Rp/L). This script
moves each by +/-15% on a 3x3 grid (base included) and writes the same schema
as build_cost_coefficients.py, so build_network can consume the rows directly.
Haulage stays fixed at its mean inside every config.

Output: data/cost_coefficients_price_anchor.csv
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BLENDS = [0, 10, 20, 30, 35, 40, 50, 70, 100]
BETA = 0.056  # L/hr per kW rated (idle term), blend-invariant in litres

OA_MEAN = 391  # Rp/L, haulage held fixed at its mean inside every config

A1_BASE = 18300.0   # B40 pump price, the regional market-price source
A2_BASE = 14924.0   # B100 index price, the index-price source
PERTURB = 0.15      # +/-15%, the review's stated band

A1_LEVELS = {"lo": A1_BASE * (1 - PERTURB), "base": A1_BASE, "hi": A1_BASE * (1 + PERTURB)}
A2_LEVELS = {"lo": A2_BASE * (1 - PERTURB), "base": A2_BASE, "hi": A2_BASE * (1 + PERTURB)}

GENSET_GROUPS = {
    "mitsubishi": {"p_rated_kw": 1100, "n_units": 3},
    "cummins": {"p_rated_kw": 1000, "n_units": 3},
    "deutz": {"p_rated_kw": 900, "n_units": 5},
}


def config_label(a1_key: str, a2_key: str) -> str:
    if a1_key == "base" and a2_key == "base":
        return "base"
    return f"a1{a1_key}_a2{a2_key}"


def price_table(a1: float, a2: float) -> dict:
    price_b100 = a2 + OA_MEAN
    price_b0 = (a1 - 0.4 * price_b100) / 0.6
    return {pb: (pb / 100) * price_b100 + (1 - pb / 100) * price_b0 for pb in BLENDS}


def main():
    bsfc = pd.read_csv(DATA_DIR / "bsfc_bases.csv")
    a_x = {
        "epa": dict(zip(bsfc.pct_b, bsfc.A_x_basis_a_L_per_kWh)),
        "generator": dict(zip(bsfc.pct_b, bsfc.A_x_basis_b_L_per_kWh)),
    }

    print("Price-anchor grid (Rp/L, OA held at mean = 391):")
    print(f"{'config':<14s} {'A1':>10s} {'A2':>10s} {'Price(B0)':>11s} {'Price(B100)':>12s}  ladder")
    rows = []
    for a1_key, a1 in A1_LEVELS.items():
        for a2_key, a2 in A2_LEVELS.items():
            label = config_label(a1_key, a2_key)
            prices = price_table(a1, a2)
            slope = "falls" if prices[100] < prices[0] else "RISES (inverted)"
            print(f"{label:<14s} {a1:>10,.0f} {a2:>10,.0f} {prices[0]:>11,.0f} {prices[100]:>12,.0f}  {slope}")

            for group_name, spec in GENSET_GROUPS.items():
                p_rated = spec["p_rated_kw"]
                for pb in BLENDS:
                    price_x = prices[pb]
                    for basis_name, a_x_dict in a_x.items():
                        rows.append({
                            "ongkos_angkut_scenario": label,   # config label, not a haulage level
                            "ongkos_angkut_rp": OA_MEAN,
                            "a1_b40_price_rp": round(a1, 1),
                            "a2_b100_index_rp": round(a2, 1),
                            "genset_group": group_name,
                            "p_rated_kw": p_rated,
                            "n_units": spec["n_units"],
                            "blend": f"B{pb}",
                            "pct_b": pb,
                            "price_rp_per_l": round(price_x, 1),
                            "bsfc_basis": basis_name,
                            "a_i_idle_rp_per_hr": round(BETA * p_rated * price_x, 2),
                            "b_i_marginal_rp_per_kwh": round(a_x_dict[pb] * price_x, 2),
                        })

    df = pd.DataFrame(rows)
    out = DATA_DIR / "cost_coefficients_price_anchor.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows to {out.name}")
    print(f"  (9 configs x 3 genset groups x 9 blends x 2 bases)")

    # cross-check: the 'base' config must reproduce cost_coefficients.csv's
    # mean-OA rows exactly (same anchors, same formula)
    base_here = df[(df.ongkos_angkut_scenario == "base")
                   & (df.genset_group == "mitsubishi") & (df.bsfc_basis == "generator")]
    ref = pd.read_csv(DATA_DIR / "cost_coefficients.csv")
    ref = ref[(ref.ongkos_angkut_scenario == "mean")
              & (ref.genset_group == "mitsubishi") & (ref.bsfc_basis == "generator")]
    merged = base_here.merge(ref, on="blend", suffixes=("_new", "_ref"))
    max_dp = (merged.price_rp_per_l_new - merged.price_rp_per_l_ref).abs().max()
    print(f"\nSanity: 'base' config vs cost_coefficients.csv (OA=mean) max price delta = Rp {max_dp:.2f}")
    assert max_dp < 0.5, "base config does not reproduce the live price ladder"
    print("  OK - base config reproduces the live ladder.")


if __name__ == "__main__":
    main()
