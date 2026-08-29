"""Per-blend tailpipe emission factors: five species, nine blends, two bases.

Diesel-baseline per-litre factors are taken from a microgrid design study
(Jeyaprabha and Milanovic, 2023). NOx, CO and THC are scaled across blends by
the exponential relations in Lapuerta et al. (2008) and do not depend on the
consumption basis except through the fuel volume each basis implies.

CO2 is treated on a tailpipe basis at 3.17 kg CO2 per kg fuel (Matragi et al.,
2024), scaled by blend density.

Particulates carry two curves: an exponential decline under basis A, and the
non-monotonic measured ratio curve of Lin et al. (2006) under basis B (a dip
near B10, then a rise past the B0 baseline from roughly B35 on).

Output: data/emission_factors.csv
"""

import math
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BLENDS = [0, 10, 20, 30, 35, 40, 50, 70, 100]

RHO_DIESEL = 853.36  # g/L
RHO_PALM = 875.0     # g/L

# Jeyaprabha & Milanovic (2023) diesel (B0) baseline emission factors, g/L
EF_B0 = {"CO": 16.5, "THC": 0.72, "PM": 0.1, "NOx": 15.5}

# Lapuerta et al. (2008) EPA exponential relations (exponent per %B) - used for CO/THC/NOx
# always, and for PM only under basis=epa (basis=generator uses Lin et al. (2006) below)
EXPONENT = {"CO": -0.006561, "THC": -0.011195, "PM": -0.006384, "NOx": 0.0009794}

CO2_INTENSITY = 3.17  # kg CO2 / kg fuel (Matragi et al. (2024), tailpipe, no biogenic credit)

# Lin et al. (2006) (QC495 diesel generator) measured PM ratio-to-P0 - non-monotonic,
# crossover ~35% (Lin et al. (2006) "PM emissions")
_GEN_PM_RATIO = {0: 1.000, 10: 0.490, 20: 0.786, 30: 0.954, 50: 1.109, 75: 1.269, 100: 1.293}
_GEN_PM_POINTS = sorted(_GEN_PM_RATIO)


def pm_ratio_generator(pct_b: float) -> tuple[float, bool]:
    if pct_b in _GEN_PM_RATIO:
        return _GEN_PM_RATIO[pct_b], False
    lo = max(p for p in _GEN_PM_POINTS if p < pct_b)
    hi = min(p for p in _GEN_PM_POINTS if p > pct_b)
    frac = (pct_b - lo) / (hi - lo)
    r = _GEN_PM_RATIO[lo] + frac * (_GEN_PM_RATIO[hi] - _GEN_PM_RATIO[lo])
    return r, True


def density(pct_b: float) -> float:
    return RHO_DIESEL + (pct_b / 100.0) * (RHO_PALM - RHO_DIESEL)


def per_litre_ef(pct_b: float, basis: str) -> dict:
    ef = {sp: EF_B0[sp] * math.exp(EXPONENT[sp] * pct_b) for sp in ("CO", "THC", "NOx")}
    if basis == "generator":
        pm_ratio, _ = pm_ratio_generator(pct_b)
        ef["PM"] = EF_B0["PM"] * pm_ratio
    else:
        ef["PM"] = EF_B0["PM"] * math.exp(EXPONENT["PM"] * pct_b)
    ef["CO2"] = CO2_INTENSITY * density(pct_b)  # g/L (3.17 kg/kg x density g/L)
    return ef


def main():
    bsfc = pd.read_csv(DATA_DIR / "bsfc_bases.csv")
    a_x = {
        "epa": dict(zip(bsfc.pct_b, bsfc.A_x_basis_a_L_per_kWh)),
        "generator": dict(zip(bsfc.pct_b, bsfc.A_x_basis_b_L_per_kWh)),
    }

    rows = []
    for pct_b in BLENDS:
        for basis_name, a_x_dict in a_x.items():
            ef_per_l = per_litre_ef(pct_b, basis_name)
            a_val = a_x_dict[pct_b]
            row = {"blend": f"B{pct_b}", "pct_b": pct_b, "bsfc_basis": basis_name,
                   "A_x_L_per_kWh": round(a_val, 5)}
            for sp, ef_l in ef_per_l.items():
                row[f"{sp}_g_per_L"] = round(ef_l, 4)
                row[f"{sp}_g_per_kWh"] = round(ef_l * a_val, 4)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "emission_factors.csv", index=False)

    print(f"Wrote {len(df)} rows to data/emission_factors.csv (5 species x 9 blends x 2 bases)")
    print("\nSample (basis=generator):")
    sample = df[df.bsfc_basis == "generator"]
    cols = ["blend"] + [f"{sp}_g_per_kWh" for sp in ["CO2", "NOx", "CO", "THC", "PM"]]
    print(sample[cols].to_string(index=False))

    print("\nPM comparison, EPA (monotonic) vs generator/Lin et al. (2006) (non-monotonic, crossover ~B35):")
    pm_compare = df.pivot(index="blend", columns="bsfc_basis", values="PM_g_per_kWh").reindex(
        [f"B{b}" for b in BLENDS])
    print(pm_compare.to_string())

    print("\nHamdi et al. (2026) measured NOx DECREASING at every load for B25 (Hamdi, Yahya,")
    print("Ennetta 2026) - opposite of the Lapuerta et al. (2008) majority direction used here (85%")
    print("of literature reports NOx rising, exp(+0.0009794*%B)). Not corrected")
    print("away - documented counter-example for the Discussion.")


if __name__ == "__main__":
    main()
