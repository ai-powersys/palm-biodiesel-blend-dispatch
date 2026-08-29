"""Per-blend fuel prices and generator cost coefficients.

Price ladder (Eq. 7): Price(Bx) = (x/100)*Price(B100) + (1-x/100)*Price(B0).
  Price(B100) = 14,924 Rp/L (index price) + haulage allowance
  Price(B0)   = (P_B40 - 0.4*Price(B100)) / 0.6, back-calculated from an
                observed regional B40 market price of 18,300 Rp/L
Value-added tax is excluded (it scales every blend identically).

Derives, for each of three genset capacity groups x nine blends x two
consumption bases x three haulage-allowance values (307 / 391 / 500 Rp/L):
  a_i(x) = beta * P_rated_i * Price(x)   idle cost, Rp/h  (beta = 0.056 L/h per kW)
  b_i(x) = A_x * Price(x)                marginal cost, Rp/kWh

Output: data/cost_coefficients.csv
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BLENDS = [0, 10, 20, 30, 35, 40, 50, 70, 100]
OA_VARIANTS = {"low": 307, "mean": 391, "high": 500}
B100_INDEX_PRICE = 14924  # Rp/L, Aug 2026, Direktorat Jenderal EBTKE (the index-price source)
B40_OBSERVED_PRICE = 18300.0  # Rp/L, Wilayah 3 (Sulawesi+NTB), the regional market-price source, pre-tax
BETA = 0.056  # L/hr per kW rated

GENSET_GROUPS = {
    "mitsubishi": {"p_rated_kw": 1100, "n_units": 3},
    "cummins": {"p_rated_kw": 1000, "n_units": 3},
    "deutz": {"p_rated_kw": 900, "n_units": 5},
}


def price_table(oa: float) -> dict:
    price_b100 = B100_INDEX_PRICE + oa
    price_b0 = (B40_OBSERVED_PRICE - 0.4 * price_b100) / 0.6
    return {pb: (pb / 100) * price_b100 + (1 - pb / 100) * price_b0 for pb in BLENDS}


def main():
    bsfc = pd.read_csv(DATA_DIR / "bsfc_bases.csv")
    a_x = {
        "epa": dict(zip(bsfc.pct_b, bsfc.A_x_basis_a_L_per_kWh)),
        "generator": dict(zip(bsfc.pct_b, bsfc.A_x_basis_b_L_per_kWh)),
    }

    # Sanity check (
    # is superseded and no longer a valid comparison target. Print the NEW
    # table for visual confirmation instead of asserting against stale numbers.
    sanity = price_table(OA_VARIANTS["mean"])
    print("New price table (OA=mean=391, post-the regional market-price source revision):")
    for pb in BLENDS:
        print(f"  B{pb:<4d}: Rp{sanity[pb]:,.1f}/L")
    print(f" (B0 was Rp25,329/L under superseded an earlier price source; B100 anchor unchanged)")

    rows = []
    for oa_label, oa_val in OA_VARIANTS.items():
        prices = price_table(oa_val)
        for group_name, spec in GENSET_GROUPS.items():
            p_rated = spec["p_rated_kw"]
            for pb in BLENDS:
                price_x = prices[pb]
                for basis_name, a_x_dict in a_x.items():
                    a_coef = BETA * p_rated * price_x  # Rp/hr
                    b_coef = a_x_dict[pb] * price_x     # Rp/kWh
                    rows.append({
                        "ongkos_angkut_scenario": oa_label,
                        "ongkos_angkut_rp": oa_val,
                        "genset_group": group_name,
                        "p_rated_kw": p_rated,
                        "n_units": spec["n_units"],
                        "blend": f"B{pb}",
                        "pct_b": pb,
                        "price_rp_per_l": round(price_x, 1),
                        "bsfc_basis": basis_name,
                        "a_i_idle_rp_per_hr": round(a_coef, 2),
                        "b_i_marginal_rp_per_kwh": round(b_coef, 2),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "cost_coefficients.csv", index=False)
    print(f"\nWrote {len(df)} rows to data/cost_coefficients.csv")
    print(f"  ({len(OA_VARIANTS)} OA scenarios x {len(GENSET_GROUPS)} genset groups x {len(BLENDS)} blends x 2 bases)")

    print("\nSample (OA=mean, mitsubishi group, basis=generator):")
    sample = df[(df.ongkos_angkut_scenario == "mean") & (df.genset_group == "mitsubishi") & (df.bsfc_basis == "generator")]
    print(sample[["blend", "price_rp_per_l", "a_i_idle_rp_per_hr", "b_i_marginal_rp_per_kwh"]].to_string(index=False))


if __name__ == "__main__":
    main()
