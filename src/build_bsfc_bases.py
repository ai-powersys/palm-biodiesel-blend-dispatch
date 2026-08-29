"""Marginal fuel consumption A_x (L/kWh) per blend, under two consumption bases.

Basis A (automotive-pooled): the exponential fit bsfc/bsfc_D = exp(0.0008189 * %B)
from Lapuerta et al. (2008), mass-based, converted to volumetric with measured
blend densities (palm 875 kg/m3, diesel 853.36 kg/m3).

Basis B (generator-scale): the seven-point BSFC sweep measured on a diesel
generator by Lin et al. (2006), already in L/kWh. Six of the nine blends fall
on a measured point; B35, B40 and B70 are linearly interpolated between
adjacent measured points (all inside the measured range).

A two-point generator study by Hamdi et al. (2026) is carried only as a
sensitivity comparison; it disagrees sharply with Lin et al. even though both
use generators.

Input:  data/emission_factors.csv is not needed here
Output: data/bsfc_bases.csv
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TABLES_DIR = Path(__file__).resolve().parent.parent / "tables"

BLENDS = [0, 10, 20, 30, 35, 40, 50, 70, 100]
# REVISED : B40 anchor changed an earlier price source->the regional market-price source (Rp21,167->Rp18,300/L
# pre-tax, Wilayah 3/Sulawesi -- see build_cost_coefficients.py's docstring
# and the regional market-price source). Values below match that script's
# OA=mean price_table() output exactly -- keep these two in sync manually,
# there is no shared import between them (build_bsfc_bases.py predates the
# cost_coefficients.csv pipeline and was never refactored to read from it).
PRICES = {  # Rp/L, OA=mean=391, post-the regional market-price source revision
    0: 20290.0, 10: 19792.5, 20: 19295.0, 30: 18797.5, 35: 18548.8,
    40: 18300.0, 50: 17802.5, 70: 16807.5, 100: 15315.0,
}

RHO_DIESEL = 853.36  # g/L
RHO_PALM = 875.0     # g/L
A_B0 = 0.236          # L/kWh, Jeyaprabha & Milanovic (2023)
BETA = 0.056          # L/h per kW rated, no-load consumption


def ratio_a(pct_b: float) -> float:
    return math.exp(0.0008189 * pct_b)


# Basis B: Lin et al. (2006)'s own volumetric BSFC (L/kWh), QC495 generator, ratio to P0
_GEN_BSFC = {0: 0.360, 10: 0.362, 20: 0.364, 30: 0.371, 50: 0.373, 75: 0.377, 100: 0.379}
_GEN_RATIO = {pb: v / _GEN_BSFC[0] for pb, v in _GEN_BSFC.items()}
_GEN_POINTS = sorted(_GEN_RATIO)


def ratio_b(pct_b: float) -> tuple[float, bool]:
    """Returns (ratio, is_interpolated). All points fall inside the measured
    measured range [0,100], so 'interpolated' here never means extrapolated."""
    if pct_b in _GEN_RATIO:
        return _GEN_RATIO[pct_b], False
    # linear interpolation between the two bracketing Lin et al. (2006) points
    lo = max(p for p in _GEN_POINTS if p < pct_b)
    hi = min(p for p in _GEN_POINTS if p > pct_b)
    frac = (pct_b - lo) / (hi - lo)
    r = _GEN_RATIO[lo] + frac * (_GEN_RATIO[hi] - _GEN_RATIO[lo])
    return r, True


def density(pct_b: float) -> float:
    return RHO_DIESEL + (pct_b / 100.0) * (RHO_PALM - RHO_DIESEL)


def main():
    mass_bsfc_diesel = A_B0 * RHO_DIESEL  # g/kWh

    rows = []
    for pct_b in BLENDS:
        rho = density(pct_b)
        ra = ratio_a(pct_b)
        rb, rb_interp = ratio_b(pct_b)

        a_x_a = mass_bsfc_diesel * ra / rho  # Basis A: mass-based, needs density correction
        a_x_b = A_B0 * rb  # Basis B: Lin et al. (2006)'s own ratio is already volumetric (L/kWh), applied directly

        rows.append({
            "blend": f"B{pct_b}",
            "pct_b": pct_b,
            "density_g_per_l": round(rho, 2),
            "mass_ratio_basis_a": round(ra, 5),
            "A_x_basis_a_L_per_kWh": round(a_x_a, 5),
            "volumetric_ratio_basis_b": round(rb, 5),
            "basis_b_interpolated": rb_interp,
            "basis_b_source": "Lin et al. (2006) measured (QC495 generator)" if not rb_interp else "Lin et al. (2006) linear interpolation (inside measured range)",
            "A_x_basis_b_L_per_kWh": round(a_x_b, 5),
        })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "bsfc_bases.csv", index=False)

    # --- Reversal thresholds (Tab-R0, C3) ---
    price_ratio = PRICES[0] / PRICES[100]  # marginal-only threshold, Eq 9

    a_b0_val = A_B0
    a_b100_a = df.loc[df.pct_b == 100, "A_x_basis_a_L_per_kWh"].iloc[0]
    a_b100_b = df.loc[df.pct_b == 100, "A_x_basis_b_L_per_kWh"].iloc[0]

    # Idle-corrected threshold, evaluated at full load (lf=1) as the fixed
    # reference point (β and A both fully weighted); the load-factor-
    # dependent version is characterized in /15b, not here.
    idle_corrected_a100_threshold = price_ratio * (BETA + A_B0) - BETA
    idle_corrected_ratio_threshold = idle_corrected_a100_threshold / A_B0

    threshold_rows = [
        {"metric": "marginal_only_price_ratio_Price0_over_Price100", "value": round(price_ratio, 4)},
        {"metric": "marginal_only_reversal_threshold_A100_over_A0", "value": round(price_ratio, 4)},
        {"metric": "marginal_only_reversal_threshold_pct_penalty", "value": round((price_ratio - 1) * 100, 1)},
        {"metric": "idle_corrected_reversal_threshold_A100_over_A0_at_full_load", "value": round(idle_corrected_ratio_threshold, 4)},
        {"metric": "idle_corrected_reversal_threshold_pct_penalty_at_full_load", "value": round((idle_corrected_ratio_threshold - 1) * 100, 1)},
        {"metric": "actual_A100_over_A0_basis_a_EPA", "value": round(a_b100_a / a_b0_val, 4)},
        {"metric": "actual_A100_over_A0_basis_a_pct_penalty", "value": round((a_b100_a / a_b0_val - 1) * 100, 1)},
        {"metric": "basis_a_crosses_marginal_threshold", "value": bool(a_b100_a / a_b0_val > price_ratio)},
        {"metric": "actual_A100_over_A0_basis_b_generator", "value": round(a_b100_b / a_b0_val, 4)},
        {"metric": "actual_A100_over_A0_basis_b_pct_penalty", "value": round((a_b100_b / a_b0_val - 1) * 100, 1)},
        {"metric": "basis_b_crosses_marginal_threshold", "value": bool(a_b100_b / a_b0_val > price_ratio)},
    ]
    thresh_df = pd.DataFrame(threshold_rows)
    thresh_df.to_csv(TABLES_DIR / "tab_r0_thresholds.csv", index=False)

    print(df.to_string(index=False))
    print()
    print(thresh_df.to_string(index=False))


if __name__ == "__main__":
    main()
