"""Price-anchor sensitivity: verify the corners, then fill the 3x3 grid.

1. Read the 16 corner solves (solve_pa_*) and compare each cost to the
   scaling identity cost(blend, cfg) = cost_main(blend, base) *
   Price(cfg, blend) / Price(base, blend). Stop if the worst relative
   deviation exceeds 2e-3.
2. For every (basis, config) in the full grid, recompute the nine-blend
   cost ladder by that identity and report the cheapest blend, whether the
   ladder is monotonic, and the B40-vs-B0, B100-vs-B40 and B100-vs-B0
   percentages.

Output: tables/tab_r3d_price_anchor_sensitivity.csv, metrics.json key
'price_anchor_sensitivity'
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CODING = Path(__file__).resolve().parent.parent
RESULTS = CODING / "results"
DATA = CODING / "data"
TABLES = CODING / "tables"
METRICS = CODING / "metrics.json"

BLENDS = ["B0", "B10", "B20", "B30", "B35", "B40", "B50", "B70", "B100"]
PCT = {b: int(b[1:]) for b in BLENDS}
BASES = ["epa", "generator"]
SCENARIOS = ["clear", "partly_cloudy", "overcast"]
CORNER_BLENDS = ["B0", "B40", "B70", "B100"]
CORNER_CONFIGS = ["a1lo_a2hi", "a1hi_a2lo"]
IDENTITY_TOL = 2e-3


def _prices() -> pd.DataFrame:
    df = pd.read_csv(DATA / "cost_coefficients_price_anchor.csv")
    # one price per (config, blend) - genset group / basis do not change it
    return (df.groupby(["ongkos_angkut_scenario", "blend"], as_index=False)
              .agg(price_rp_per_l=("price_rp_per_l", "first"),
                   a1=("a1_b40_price_rp", "first"),
                   a2=("a2_b100_index_rp", "first")))


def _baseline() -> pd.DataFrame:
    t1 = pd.read_csv(TABLES / "tab_r1_blend_cost.csv")
    return t1[["basis", "blend", "expected_cost_rp"]].copy()


def verify(prices: pd.DataFrame, base: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    pr = prices.set_index(["ongkos_angkut_scenario", "blend"]).price_rp_per_l
    bc = base.set_index(["basis", "blend"]).expected_cost_rp
    rows, worst = [], 0.0
    for cfg in CORNER_CONFIGS:
        for basis in BASES:
            for blend in CORNER_BLENDS:
                paths = [RESULTS / f"solve_pa_{cfg}_{basis}_{blend}_{s}.json" for s in SCENARIOS]
                if not all(p.exists() for p in paths):
                    raise FileNotFoundError(f"missing corner solve: {cfg}/{basis}/{blend} - run the Colab sweep first")
                with open(paths[0]) as f:
                    solved = json.load(f)["expected_objective_all_scenarios_rp"]
                predicted = bc[(basis, blend)] * pr[(cfg, blend)] / pr[("base", blend)]
                rel = abs(solved - predicted) / predicted
                worst = max(worst, rel)
                rows.append(dict(config=cfg, basis=basis, blend=blend,
                                 solved_rp=solved, predicted_rp=predicted, rel_delta=rel))
    return pd.DataFrame(rows), worst


def fill_grid(prices: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    pr = prices.set_index(["ongkos_angkut_scenario", "blend"]).price_rp_per_l
    anchors = prices.set_index("ongkos_angkut_scenario")[["a1", "a2"]].groupby(level=0).first()
    bc = base.set_index(["basis", "blend"]).expected_cost_rp
    configs = [c for c in prices.ongkos_angkut_scenario.unique()]
    out = []
    for cfg in configs:
        for basis in BASES:
            ladder = {b: bc[(basis, b)] * pr[(cfg, b)] / pr[("base", b)] for b in BLENDS}
            order = sorted(BLENDS, key=lambda b: ladder[b])
            cheapest = order[0]
            costs_by_pct = [ladder[b] for b in BLENDS]  # B0..B100 order
            monotonic_dec = all(x > y for x, y in zip(costs_by_pct, costs_by_pct[1:]))
            monotonic_inc = all(x < y for x, y in zip(costs_by_pct, costs_by_pct[1:]))
            shape = ("monotonic_decreasing" if monotonic_dec
                     else "monotonic_increasing" if monotonic_inc
                     else "interior_optimum" if cheapest not in ("B0", "B100")
                     else "non_monotonic")
            out.append(dict(
                config=cfg,
                a1_b40_rp=float(anchors.at[cfg, "a1"]),
                a2_b100_index_rp=float(anchors.at[cfg, "a2"]),
                basis=basis,
                cheapest_blend=cheapest,
                second_cheapest=order[1],
                curve_shape=shape,
                ranking_matches_base=(cheapest == "B100" and monotonic_dec),
                cost_B0_rp_per_kwh=ladder["B0"] / 102.45,
                cost_B40_rp_per_kwh=ladder["B40"] / 102.45,
                cost_B100_rp_per_kwh=ladder["B100"] / 102.45,
                B40_vs_B0_pct=100 * (ladder["B0"] - ladder["B40"]) / ladder["B0"],
                B100_vs_B40_pct=100 * (ladder["B40"] - ladder["B100"]) / ladder["B40"],
                B100_vs_B0_pct=100 * (ladder["B0"] - ladder["B100"]) / ladder["B0"],
            ))
    return pd.DataFrame(out)


def main():
    prices = _prices()
    base = _baseline()

    vdf, worst = verify(prices, base)
    print("=== Step 1: scaling-identity verification (16 corner solves) ===")
    print(vdf.to_string(index=False,
          formatters={"solved_rp": "{:,.0f}".format, "predicted_rp": "{:,.0f}".format,
                      "rel_delta": "{:.2e}".format}))
    print(f"\nworst rel_delta = {worst:.2e}  tol = {IDENTITY_TOL:.0e}")
    if worst > IDENTITY_TOL:
        raise SystemExit("IDENTITY BROKEN - solve the full 9x2x8 grid, do not trust the fill.")
    print("PASS - interior filled by the scaling identity.\n")

    grid = fill_grid(prices, base)
    grid.to_csv(TABLES / "tab_r3d_price_anchor_sensitivity.csv", index=False)
    print("=== Step 2: full 3x3 grid (both bases) ===")
    show = grid[["config", "basis", "cheapest_blend", "curve_shape",
                 "ranking_matches_base", "B100_vs_B0_pct", "B40_vs_B0_pct"]]
    print(show.to_string(index=False, formatters={
        "B100_vs_B0_pct": "{:+.1f}".format, "B40_vs_B0_pct": "{:+.1f}".format}))
    print(f"\nWrote tables/tab_r3d_price_anchor_sensitivity.csv ({len(grid)} rows)")

    n_match = int((grid.ranking_matches_base).sum())
    flips = grid[~grid.ranking_matches_base][["config", "basis", "cheapest_blend", "curve_shape"]]
    block = {
        "band": "A1 (B40 pump price) +/-15%, A2 (B100 index price) +/-15%, OA held at mean",
        "grid": "3x3 configs x 2 BSFC bases = 18 (base, epa) reproduces",
        "method": "16 corner solves verified against the exact scaling identity "
                  "(worst rel_delta {:.2e}); interior by identity".format(worst),
        "cells_ranking_matches_base_B100_monotonic": n_match,
        "cells_total": int(len(grid)),
        "flips": flips.to_dict(orient="records"),
        "base_case": {
            "cheapest": "B100", "B100_vs_B0_pct": float(
                grid[(grid.config == "base") & (grid.basis == "epa")].B100_vs_B0_pct.iloc[0]),
        },
        "most_adverse_corner_a1lo_a2hi": grid[grid.config == "a1lo_a2hi"][
            ["basis", "cheapest_blend", "curve_shape", "B100_vs_B0_pct"]].to_dict(orient="records"),
        "source_table": "tables/tab_r3d_price_anchor_sensitivity.csv",
    }
    metrics = json.loads(METRICS.read_text()) if METRICS.exists() else {}
    metrics["price_anchor_sensitivity"] = block
    METRICS.write_text(json.dumps(metrics, indent=2))
    print(f"\nmetrics.json['price_anchor_sensitivity'] written. "
          f"{n_match}/{len(grid)} cells keep the base ranking (B100, monotonic).")
    if len(flips):
        print("Flips:")
        print(flips.to_string(index=False))


if __name__ == "__main__":
    main()
