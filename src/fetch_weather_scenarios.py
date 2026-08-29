"""Build the three weather scenarios from NASA POWER hourly irradiance.

Queries NASA POWER for a multi-year hourly record at the site coordinate,
computes a daily clear-sky index (ALLSKY_SFC_SW_DWN / CLRSKY_SFC_SW_DWN over
daylight hours), splits days into clear / partly cloudy / overcast terciles,
and picks one representative day per tercile (index closest to the tercile
median). Each tercile's historical frequency is its scenario probability.

Output: data/scenarios_irradiance.csv, data/scenario_probabilities.json
"""

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

LAT, LON = -6.117, 120.458
START, END = "20210101", "20251231"  # 5-year window, more recent climate pattern
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

POWER_URL = (
    "https://power.larc.nasa.gov/api/temporal/hourly/point"
    "?parameters=ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN"
    f"&community=RE&longitude={LON}&latitude={LAT}"
    f"&start={START}&end={END}&format=JSON"
)


def fetch_power_data() -> pd.DataFrame:
    with urllib.request.urlopen(POWER_URL, timeout=120) as r:
        payload = json.load(r)
    params = payload["properties"]["parameter"]
    allsky = params["ALLSKY_SFC_SW_DWN"]
    clrsky = params["CLRSKY_SFC_SW_DWN"]
    idx = pd.to_datetime(list(allsky.keys()), format="%Y%m%d%H")
    df = pd.DataFrame(
        {
            "allsky": [allsky[k] for k in allsky],
            "clrsky": [clrsky[k] for k in allsky],
        },
        index=idx,
    )
    # NASA POWER uses -999 as a fill/missing-value sentinel.
    df = df.replace(-999.0, np.nan)
    return df


def compute_daily_clearness(df: pd.DataFrame) -> pd.Series:
    daylight = df[df["clrsky"] > 1.0].copy()  # exclude night (clrsky ~0)
    daily = daylight.groupby(daylight.index.date).agg({"allsky": "sum", "clrsky": "sum"})
    daily = daily.dropna()
    daily["kc"] = daily["allsky"] / daily["clrsky"]
    return daily["kc"]


def pick_representative_days(df: pd.DataFrame, kc: pd.Series) -> tuple[pd.DataFrame, dict]:
    terciles = kc.quantile([1 / 3, 2 / 3]).values
    labels = pd.cut(
        kc, bins=[-np.inf, terciles[0], terciles[1], np.inf],
        labels=["overcast", "partly_cloudy", "clear"],
    )
    scenario_rows = []
    probabilities = {}
    for label in ["clear", "partly_cloudy", "overcast"]:
        group = kc[labels == label]
        probabilities[label] = len(group) / len(kc)
        median_val = group.median()
        rep_date = (group - median_val).abs().idxmin()
        day_hours = df.loc[pd.Timestamp(rep_date).strftime("%Y-%m-%d")].copy()
        day_hours["scenario"] = label
        day_hours["representative_date"] = str(rep_date)
        day_hours["hour"] = day_hours.index.hour
        scenario_rows.append(day_hours.reset_index(drop=True))
    scenarios_df = pd.concat(scenario_rows, ignore_index=True)
    return scenarios_df, probabilities


def main():
    df = fetch_power_data()
    n_missing = df["allsky"].isna().sum()
    kc = compute_daily_clearness(df)
    scenarios_df, probabilities = pick_representative_days(df, kc)

    out_cols = ["scenario", "representative_date", "hour", "allsky", "clrsky"]
    scenarios_df[out_cols].to_csv(DATA_DIR / "scenarios_irradiance.csv", index=False)

    assert abs(sum(probabilities.values()) - 1.0) < 1e-9, "probabilities must sum to 1"
    with open(DATA_DIR / "scenario_probabilities.json", "w") as f:
        json.dump(probabilities, f, indent=2)

    print(f"Queried {len(df)} hourly points, {n_missing} missing (-999) sentinel values")
    print(f"Computed clear-sky index for {len(kc)} days")
    print("Representative days:")
    for label in ["clear", "partly_cloudy", "overcast"]:
        rep_date = scenarios_df.loc[scenarios_df["scenario"] == label, "representative_date"].iloc[0]
        print(f"  {label}: {rep_date}  pi = {probabilities[label]:.4f}")


if __name__ == "__main__":
    main()
