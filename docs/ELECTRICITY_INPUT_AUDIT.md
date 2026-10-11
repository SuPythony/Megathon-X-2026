# Electricity input audit and handoff

Read-only audit of Satyarth commit `7eeb4d651695fa7311509aa746daacbae06b6d79`
and this worktree's existing `thermal-development-v1` synthetic dataset.
No other branch/worktree was changed. Preparation precedes electricity implementation.

| Required input | Already available | Action |
|---|---|---|
| Geometry, fabric, ventilation, heating conversion | TwinParams and synthetic true parameters | Reuse existing thermal model |
| Occupants and presence | Synthetic latent household; Satyarth intake | Reuse; optional user count and out/part-day/home answer |
| Thermostat and heating schedule | Synthetic setpoint/hours; Satyarth thermostat/periods | Reuse duration; generate labelled timing if absent |
| Appliance energy and lighting share | Synthetic baseload/share; Satyarth per-appliance catalogue and hourly profiles | Scale generated or supplied profiles to monthly demand, no double counting |
| DHW equipment and conversion | Current FuelConfig; Satyarth separate hot-water system | Preserve separate efficiency; consume extras explicitly |
| Monthly weather and location | 100 synthetic homes, 24 months each; district coordinates | Reuse monthly means/GHI; generate conditional half-hour variation |
| Actual half-hour temperature | Not downloaded | Generated development profile centred on each supplied monthly mean; externally sourced series required for factual runs |
| Half-hour load shapes | London dataset referenced, not downloaded; Satyarth working appliance shapes | Generated seeded occupancy/day profiles, labelled assumed; accept supplied appliance profiles |
| PV capacity, orientation, tilt | TwinParams / EPC fields, zero PV in current showcase homes | Preserve existing values; use explicit hypothetical PV scenarios to test solar |
| Actual PV generation | Not downloaded; no local PVGIS cache | Generated monthly-GHI/solar-geometry scenario; actual external PVGIS/observed series accepted |
| Battery capacity | TwinParams; zero in existing showcase homes | Preserve; hypothetical batteries only in explicit scenarios |
| Battery power, usable capacity and efficiencies | Missing | Development-only 0.5C power, 90% usable, 95% each-way efficiency; real systems need model/installer specs |
| Initial battery energy | Missing | Explicit empty initial state; carry state continuously and report final stored energy |
| Inverter and export limits | Missing | Development inverter kW=PV kWp; unrestricted-export assumption flagged |
| Standard price | contracts/tariffs.json, dated 2026 Q4 | Only apply within validity dates, or explicitly label a reference-price scenario |
| ToU/export prices | Unsourced/null in tariff contract | Never fabricate prices; energy simulation can run, those bill fields stay unavailable |
| Meter observations | Synthetic usage, with noise/missingness; Satyarth monthly intake | Not needed for forward simulation; Bayesian calibration remains later |

## Easy optional questions for a real home

Reuse Satyarth's existing questions before adding any: number of occupants, daytime
presence, thermostat setting, heating periods, appliance type/size, current heating
and hot-water system, monthly bill kWh when available. Unknown answers remain unknown.

Additional questions are conditional: “Do you have solar panels?”, “Do you have a
home battery?”, their capacity or make/model if known, approximate panel direction,
and tariff name. Equipment model/installer documents should supply power limits and
efficiencies automatically; ordinary users should not be asked to invent COP,
inverter efficiency, or 17,520 half-hour readings. Technical values can be optionally
entered by an installer. A real-home estimate with assumed specifications must be
labelled estimated; generated equipment values are development inputs only.

No additional user answers are needed to prepare reproducible development inputs
from the existing synthetic homes. These are conditional plausible scenarios, not
recovered real routines or empirical London/PVGIS data.

## Input preparation

`contracts/electricity.py` defines input extensions without changing shared models
or generated TypeScript. `greentwin/twin/electricity_inputs.py` prepares UTC intervals
for whole local months, including DST and leap days. Seeded daily load fluctuations
and cloudy-day variation provide randomness. Monthly temperature means are preserved;
solar shape is conditioned on monthly irradiation and astronomical daylight.
The solar proxy uses approximate position, 80% direct / 20% diffuse radiation,
ground reflectance 0.2 and performance ratio 0.80. These are explicit development
assumptions; it does not reproduce temperature-dependent module physics or measured
weather. PV energy remains zero at astronomical night. Installed capacity stays
unchanged unless an explicit hypothetical scenario changes it.

```bash
.venv/bin/python scripts/prepare_electricity_inputs.py --homes 100
```

All packages must validate before this command writes them. Outputs are compressed
interval arrays plus per-home JSON parameters, original monthly weather, latent
household profiles, equipment and input provenance. The manifest records seed,
version, validation and limitations. Existing output directories are refused;
all outputs remain inside this worktree.

Inputs are ready for development once validation passes. Actual PVGIS/weather and
London archetype ingestion remains the data layer's responsibility. Prices, physical
unknowns and missing observations are never presented as real because we generated them.
