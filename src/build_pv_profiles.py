"""Convert each scenario's irradiance to AC PV output for the case-study array.

1,302 kWp DC, a system derating factor on the DC side, then a hard clip at
the 1,100 kW AC inverter limit.

Input:  data/scenarios_irradiance.csv
Output: data/pv_profiles.csv
"""

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DC_CAPACITY_KWP = 1302.0
AC_LIMIT_KW = 1100.0
# System derating factor (temperature, soiling, wiring, mismatch, DC-side
# availability - excludes inverter clipping, modeled separately below via
# AC_LIMIT_KW). Not sourced from the case-study reference (which gives no PR/derate figure);
# standard PVWatts-style assumption. [AUTHOR: confirm or replace with a
# sourced derate figure for Selayar/similar tropical PV installations.]
DERATE_FACTOR = 0.85


def main():
    irr = pd.read_csv(DATA_DIR / "scenarios_irradiance.csv")
    load = pd.read_csv(DATA_DIR / "load_selayar_24h.csv", comment="#")

    irr["pv_dc_kw"] = DC_CAPACITY_KWP * (irr["allsky"] / 1000.0) * DERATE_FACTOR
    irr["pv_ac_kw"] = irr["pv_dc_kw"].clip(upper=AC_LIMIT_KW)
    irr["clipped"] = irr["pv_dc_kw"] > AC_LIMIT_KW

    out = irr[["scenario", "representative_date", "hour", "pv_ac_kw"]].copy()
    out.to_csv(DATA_DIR / "pv_profiles.csv", index=False)

    n_clipped = int(irr["clipped"].sum())
    max_dc = irr["pv_dc_kw"].max()
    print(f"Derate factor: {DERATE_FACTOR}, AC limit: {AC_LIMIT_KW} kW")
    print(f"Max theoretical DC output across all scenarios: {max_dc:.1f} kW")
    print(f"Hours clipped by inverter limit: {n_clipped} / {len(irr)}")

    merged = out.merge(load, on="hour")
    merged["pv_load_ratio"] = merged["pv_ac_kw"] / merged["load_kw"]
    max_ratio = merged["pv_load_ratio"].max()
    max_row = merged.loc[merged["pv_load_ratio"].idxmax()]
    print(f"\nSanity check - max instantaneous PV/load ratio: {max_ratio:.1%}")
    print(f"  at scenario={max_row['scenario']}, hour={int(max_row['hour'])}, "
          f"PV={max_row['pv_ac_kw']:.0f} kW, load={max_row['load_kw']:.0f} kW")
    if max_ratio > 0.30:
        print(" WARNING: exceeds the ~30% sanity-check ceiling from the plan - flag, don't proceed silently.")
    else:
        print("  OK: within the ~30% sanity-check ceiling.")

    per_scenario_peak = out.groupby("scenario")["pv_ac_kw"].max()
    print("\nPeak AC output per scenario (kW):")
    print(per_scenario_peak.to_string())


if __name__ == "__main__":
    main()
