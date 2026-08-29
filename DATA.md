# Data sources and licensing

The files in `data/` are either public data reprocessed for this study or
values adapted from published literature. The generated tables in `tables/` and
figures in `figures/` are outputs of the scripts in `src/`.

Original data and generated outputs in this repository are released under
**CC BY 4.0**. Third-party sources are attributed below and remain under their
own terms.

## Solar irradiance and weather scenarios

`data/scenarios_irradiance.csv`, `data/pv_profiles.csv`,
`data/scenario_probabilities.json`

Derived from NASA POWER hourly irradiance (`ALLSKY_SFC_SW_DWN`,
`CLRSKY_SFC_SW_DWN`) for the case-study coordinate, a five-year record. NASA
POWER data are free and public; see https://power.larc.nasa.gov and cite the
project per its data-use guidelines. `src/fetch_weather_scenarios.py` documents
the query and the tercile classification.

## Load profile

`data/load_selayar_24h.csv`

Approximate values digitised from Figure 1 of:

> M. F. Ramadhan, R. M. Azmi, E. Supriyadi, W. Agustiawan, F. Sastrowijoyo,
> "PV Intermittent Smoothing with BESS: A Case Study in Selayar Island
> Electrical Grid," 2022 International Conference on Power Engineering and
> Renewable Energy (ICPERE), IEEE, 2022. doi:10.1109/ICPERE56870.2022.10037559

These are values read from a published figure, not the original tabulated data.
Expect an error of a few hundred kW per point. The same reference is the source
for the plant specification (unit capacities, PV, battery).

## Fuel prices

`data/cost_coefficients.csv`, `data/cost_coefficients_price_anchor.csv`

Two anchor prices, August 2026:

- **B100 index price** 14,924 Rp/L, plus a haulage allowance, from the
  Indonesian biodiesel market index price (Directorate General of New,
  Renewable Energy and Energy Conservation) and the maximum haulage-allowance
  schedule (Ministerial Decree No. 290.K/EK.05/MEM.E/2025). Public government
  instruments.
- **B40 market price** 18,300 Rp/L pre-tax, regional (Sulawesi and West Nusa
  Tenggara), from two independent commercial industrial-diesel price listings.

The B0 price is back-calculated from the observed B40 price and the B100 index
price. Prices for all other blends are linear interpolations between the B0 and
B100 endpoints (Eq. 7 in the paper), not surveyed. Value-added tax is excluded.

## Fuel properties and emission factors

`data/bsfc_bases.csv`, `data/emission_factors.csv`

Adapted from published engine studies, not measured on the units modelled here:

- Marginal consumption, basis A: the exponential blend relation in
  M. Lapuerta, O. Armas, J. Rodriguez-Fernandez, "Effect of biodiesel fuels on
  diesel engine emissions," *Progress in Energy and Combustion Science*, 2008.
- Marginal consumption and particulates, basis B: the seven-point generator
  sweep in Y.-C. Lin, W.-J. Lee, H.-C. Hou, "PAH emissions and energy
  efficiency of palm-biodiesel blends fueled on diesel generator,"
  *Atmospheric Environment*, 2006.
- Diesel-baseline per-litre emission factors and the no-load consumption
  intercept: S. B. Jeyaprabha, J. V. Milanovic, "Probabilistic Techno-Economic
  Design of Isolated Microgrid," *IEEE Transactions on Power Systems*, 2023.
- Tailpipe CO2 intensity (3.17 kg CO2/kg fuel): I. Matragi, A. Maiboom,
  X. Tauzia, Y. Thevenoux, "A novel algorithm for optimizing genset operations
  to minimize fuel consumption in remote diesel-RES microgrids," *Energy
  Conversion and Management: X*, 2024.
- A two-point generator study (F. Hamdi, I. Yahya, R. Ennetta, *E3S Web of
  Conferences*, 2026) is carried only as a sensitivity comparison.

Minimum stable loading (30% of rated) follows Matragi et al. (2024).

## Software

The model is built on PyPSA (`pypsa==1.2.4`, MIT License) and `linopy`
(`linopy==0.8.0`), solved with HiGHS via `highspy==1.15.1`.

> T. Brown, J. Horsch, D. Schlachtberger, "PyPSA: Python for Power System
> Analysis," *Journal of Open Research Software*, 2018. doi:10.5334/jors.188
