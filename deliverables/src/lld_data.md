## 8. Data: datasets, acquisition and pipeline

No public dataset links property features, monthly meter readings and sale prices for the same homes. Every part therefore uses **real data where it exists**, joined at the level each dataset allows. **Physics-generated synthetic data** fills only the missing piece (monthly readings per home, and the labels), and is calibrated and validated against real aggregates.

### 8.1 Datasets (verified {{VERIFIED_DATE}})

{{DATASETS_TABLE}}

### 8.2 Start here (day 1, in this order)

{{DAY1}}

### 8.3 How to get each dataset

{{DATASETS_DETAIL}}

### 8.4 EPC live tables (MHCLG): processing and use

**Source.** GOV.UK *Live tables on Energy Performance of Buildings Certificates*: quarterly ODS files (data to June 2026, published 30 July 2026), England and Wales, built from the EPC register since 2008. Each table has sheets for England & Wales, England only, Wales only, by region and **by local authority**. The LA sheet has one row per LA code × quarter (`YYYY/Q`, from 2008/4).

| Table | Columns used | Becomes |
| --- | --- | --- |
| D1 (all dwellings) / EB1 (existing) | LA code, quarter, number of lodgements, total floor area, counts A–G + not recorded | `AreaEPCStats.band_counts`, `lodgements`, `total_floor_area_m2` |
| D3 / EB3 | lodgements, total floor area, habitable rooms, energy consumption current (kWh/m² pa, total), CO2 (t pa), fuel costs | `mean_energy_kwh_m2`, `mean_energy_kwh_per_dwelling`, `mean_co2_t_per_dwelling`, `mean_fuel_cost_gbp`, **recomputed from totals ÷ counts** (the December 2023 correction note) |
| EB4 / NB4 | lodgements and floor area for bungalow, flat & maisonette, house | `property_type_counts` |

**Loader** (`greentwin/data/live_tables.py`):
1. `pandas.read_excel(path, sheet_name="D1_by_LA", engine="odf", header=3)`.
2. Rename columns and coerce numerics; suppressed or blank cells become `None`.
3. Melt bands; join D1 + D3 + EB4 on (LA code, quarter).
4. Write `area_epc_stats.parquet` + `meta.json` (`release`, revision notes).

Repeat for the region and England & Wales sheets with `area_type` set.

| Use | Part | How |
| --- | --- | --- |
| **Population calibration targets** | 0 | Rake (iterative proportional fitting) the sampled EPC records and the synthetic population so band mix (D1), property-type mix (EB4) and mean floor area per LA match the official totals for the same period |
| **Download / mapper validation** | 0 | Counts of our downloaded certificates per LA × quarter × band are reconciled against D1. A gap > 2% flags missing files, opt-outs or mapping errors |
| **Twin validation** | 1 | Twin **standard mode** (SAP-like occupancy) aggregated per LA vs D3 mean modelled kWh/m² and fuel cost: a physics check against official EPC-modelled figures |
| **Score peer context** | 3 | `ScoreResult.peer_context` gives the local-authority band shares and median modelled kWh/m² ("your home vs your area") |
| **Synthetic-data validation** | 0 | Synthetic band, type and kWh/m² distributions vs D1 / EB4 / D3 per LA (chi-square, KS), reported in the evaluation |
| **Dashboard layers** | Orchestrator | `DashboardArea.epc_stats`: share A–C vs D–G, trend over quarters, mean modelled kWh/m², CO2 per dwelling (official LA layer beside our modelled LSOA layer) |

**Caveats** (documented in `DatasetMeta.notes`):
- Counts are **lodgements in a period**, not the dwelling stock; repeat certificates and opt-outs exist.
- "All" tables mix new and existing dwellings; use the EB tables for existing stock.
- A Q1 2026 LA allocation error was revised on 15 and 19 May 2026.
- D3 averages must be recomputed from totals.

### 8.5 Synthetic data generation (`greentwin/twin/generate.py`)

1. **Homes:** real EPC records from the three regions (≈ 50–100k per region), raked to the EPC live-table band and type mixes per LA.
2. **Behaviour:** sample setpoint, heating hours, occupants, baseload and post-retrofit rebound from priors tuned so the generated annual totals match **NEED** distributions per archetype (type × age × floor area × region).
3. **Monthly usage:** run the twin (calibrated mode) on each home with its **real monthly weather**, then add meter noise (StudentT, ν = 5). Inject missing months (≈ 8%), estimated reads (≈ 10%) and conflicting records (≈ 1%, e.g. rating A with 400 kWh/m²).
4. **Labels** (`ReferenceLabel`, every rule documented with `rule_id`):
   - per-retrofit and package per-fuel savings (twin, true parameters)
   - improvement potential and ranking
   - feasibility (rules)
   - rating change (standard mode)
   - synthetic value with a planted premium, **for method checks only**
5. **Validation report** (`eval/reports/synthetic_validation.md`):
   - KS / quantiles vs NEED annual consumption per archetype
   - band / type / kWh/m² vs the EPC live tables per LA
   - the stated caveat: agreement with synthetic labels is not real-world accuracy

### 8.6 Load profiles and area labels

- **London smart meters** → `LoadProfileArchetype`: drop households with < 300 days of data; normalise each day to sum 1; k-means (k ≈ 8) per season × day type; keep archetypes with ≥ 50 households. Used by the electricity layer (Part 1). The data are electricity-only, London 2011–14; this is stated as a limitation.
- **DESNZ sub-national LSOA consumption** → `LSOAConsumption`. Suppressed cells (few meters) stay `None`, never 0. These are the labels for the GraphSAGE area model:
  - **Graph:** LSOA adjacency from the 2021 boundaries.
  - **Features:** EPC stock mix, IMD decile, degree days, off-gas share, mean floor area.
  - **Validation:** spatial-block CV by local authority, against an EBM baseline without the graph.

### 8.7 Commands

```bash
uv run python scripts/download_epc.py --regions NW SW London      # needs EPC service account (One Login)
uv run python scripts/download_public.py --all                    # NEED, DESNZ LSOA, UCL, PPD, HPI, ONSPD, IMD, boundaries, live tables
uv run python scripts/download_weather.py --districts data/interim/districts.csv   # Open-Meteo, cached
uv run python -m greentwin.data.build                             # map → split → DuckDB + meta.json
uv run python -m greentwin.data.live_tables                       # EPC live tables → area_epc_stats
uv run python -m greentwin.twin.generate --n 50000 --seed 42      # synthetic usage + labels
uv run pytest -q                                                  # contracts, splits, leakage, reconciliation
```
