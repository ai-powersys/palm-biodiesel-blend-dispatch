# Palm biodiesel blend dispatch

Two-stage stochastic unit commitment and economic dispatch for an isolated
diesel-photovoltaic-battery microgrid, run separately for nine palm biodiesel
blends (B0 to B100) under solar-uncertainty scenarios. The blend that minimises
expected daily operating cost is identified by comparison across runs; carbon
and multi-pollutant emissions are computed and reported for every run but do
not enter the objective.

This repository contains the model, the input data, the sweep scripts, and the
scripts that build every table and figure in the associated study. The study
is in preparation; publication details will be added here and to `CITATION.cff`
once available.

## What the model does

- **System**: eleven diesel units (three capacity groups), a 1,302 kWp / 1,100 kW
  photovoltaic array, and an 873 kWh / 800 kW lithium-iron-phosphate battery,
  on an isolated grid with a peak demand of about 5.7 MW.
- **Formulation**: commitment is a first-stage decision fixed across three
  weather scenarios (non-anticipativity); generator output is second-stage
  recourse. Built on PyPSA's `committable` generator components, with
  non-anticipativity, a battery state-of-charge floor, and period-aware
  minimum up/down time added directly in `linopy` (`src/stochastic_layer.py`).
- **Scenarios**: three representative days (clear / partly cloudy / overcast)
  from NASA POWER irradiance, probability-weighted by historical frequency.
- **Consumption bases**: marginal fuel consumption per blend is carried under
  two literature-adapted bases (an automotive-pooled exponential fit, and a
  generator-scale measured sweep); both are run through every calculation.

## Environment

Python 3.10+ with the pinned dependencies:

```
pip install -r requirements.txt
```

The mixed-integer solver is HiGHS, used through `highspy` / `linopy`. The
results in the paper were produced on a standard cloud CPU runtime (two virtual
cores, ~13.6 GB RAM, no GPU).

## Reproducing the results

The solves are the slow part (about one hour for the main sweep). They can be
run either from the notebooks in `notebooks/` (on Google Colab, the runtime
used, or any machine) or by calling the `src/run_sweep*.py` modules directly.

**1. Solves** (run `01` first; the others depend on its output):

| Notebook | Script(s) | Produces |
|---|---|---|
| `01_main_sweep.ipynb` | `run_sweep.py`, `run_sweep_mustrun.py` | main 18 solves + 18 must-run solves |
| `02_cycling_time_sweep.ipynb` | `run_sweep_panel_b.py` | minimum up/down time T = 2, 3 h |
| `03_stochastic_ablation.ipynb` | `run_sweep_ablation.py` | perfect-foresight and expected-weather alternatives |
| `04_price_anchor_sensitivity.ipynb` | `build_price_anchor_configs.py`, `run_sweep_price_anchor.py` | price-anchor grid corners |

Each solve writes a JSON checkpoint under `results/`; reruns skip completed
cases. Precomputed `results/` are included, so the aggregation step below can
be run without re-solving.

**2. Validation** (blocking - nothing downstream is trusted until it passes):

```
python src/validate.py
```

**3. Tables and figures** (pure post-processing from `results/`, no re-solve):

```
python src/build_tab_r1.py          # cost table + cost-ladder figure
python src/build_tab_r3.py          # idle/marginal split + schedule-invariance check
python src/build_tab_r2.py          # emissions per blend
python src/build_tab_mustrun.py     # must-run sensitivity
python src/build_fig_r3_panel_a.py
python src/build_fig_r3_panel_b.py
python src/build_fig_r3_panel_c.py
python src/build_fig_r3_sensitivity.py
python src/build_fig_r4.py
python src/build_tab_r4.py          # stochastic-layer ablation
python src/build_tab_r3d.py         # price-anchor grid (fills the interior analytically)
python src/build_tab_r5.py          # solver runtime
python src/build_manuscript_tables.py
python src/build_manuscript_figures.py
```

Regenerating the input data tables (only needed if the sources change):

```
python src/fetch_weather_scenarios.py
python src/build_pv_profiles.py
python src/build_bsfc_bases.py
python src/build_emission_factors.py
python src/build_cost_coefficients.py
```

## Layout

```
src/         model, sweep scripts, table/figure builders
notebooks/   the four sweeps as runnable notebooks
data/        input tables (weather scenarios, PV profiles, fuel properties, prices)
results/     one JSON per solve (precomputed)
tables/      generated CSV tables and the manuscript table markdown
figures/     generated PDF and PNG figures
run_summary_*.csv   per-case solver log for each sweep
```

## Notes

- The model is a deterministic mixed-integer program over a fixed three-scenario
  set. There are no random seeds; determinism is checked in `src/validate.py`.
- Fuel prices use an August 2026 basis (during the transition from a B40 to a
  B50 national blending mandate). Prices for the intermediate blends are
  interpolated between two observed anchors, not surveyed. See `DATA.md`.
- Fuel-property and emission-factor inputs are adapted from published engine
  studies, not measured on the units modelled here; see `DATA.md` for sources.

## Citing this repository

An archived snapshot is available on Zenodo. Cite the concept DOI (the
"all versions" DOI), which always resolves to the latest release:

> DOI: 10.5281/zenodo.XXXXXXX

Development continues at
https://github.com/USERNAME/palm-biodiesel-blend-dispatch

`CITATION.cff` holds the machine-readable metadata; add the Zenodo DOI to its
`doi:` field after the first release.

## License

Code: MIT (`LICENSE`). Data: CC BY 4.0. See `DATA.md` for third-party data
attribution.
