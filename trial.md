# GreenTwin — System Architecture (end to end)

Reference: Lloyds PS2 problem statement (`LLOYDS-ps2.pdf`). Every block is tagged with the section it satisfies.
Core rule (PS §1): score, savings, carbon and value come only from documented models; the LLM only explains them. A what-if reruns the models first, then the LLM explains the new outputs.

## 6 core philosophies

| # | Philosophy | In this system |
| --- | --- | --- |
| 1 | **Physics is the source of truth** | Every energy, savings, carbon and rating number comes from the Level-1 twin (block 2). ML only sets priors (EBM), estimates value and checks results. |
| 2 | **Bayesian in thought** | Each home is calibrated on its own meter readings. Every output is a full distribution (mean, median, mode, std dev, percentiles, 90% interval, tail probabilities), plus P(Poor), P(suitable), P(on front). Missing data widens the range, never becomes 0. |
| 3 | **Rerun, never add up** | A retrofit is a twin parameter change. Every what-if and every package is a fresh rerun, reported per fuel. |
| 4 | **Grounded explanation** | The LLM only quotes tool outputs; a validator rejects anything else; Jev only routes and screens feasibility. No backing → "unavailable". |
| 5 | **Explainability first** | Score waterfall, heat loss per building element, EBM contributions, value-model coefficients, a reason on every recommendation or veto. |
| 6 | **See it to believe it** | A living 3D twin: heat loss glows, retrofits appear, numbers update live, all driven by the same calibrated model. |

**In one line:** a Bayesian-calibrated physics twin of every home, which you can see in 3D, explain in plain language and act on, where every number is backed by physics and the range it comes with.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 0. INPUT DATA  (PS §9: public + synthetic only, licences documented)                           │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • EPC certificates + recommendations + certificate history (England & Wales)                   │
│ • NEED 50k: real annual gas + electricity with property attributes                             │
│ • UCL Land Registry–EPC linked sales (price, date, floor area, rooms, EPC at sale) + UK HPI    │
│ • Weather: HadUK-Grid / Open-Meteo degree days (HDD 15.5 °C), irradiance; PVGIS solar yield    │
│ • Geography: ONS postcode→LSOA, IMD, conservation/listed, off-gas grid, OSM roof & party walls │
│ • Sourced factors: DESNZ 2025 emissions · Ofgem tariffs · grants (all with units + source)     │
│ • Retrofit catalogue: applicability, cost (BEIS 2017×CPI), lifetime, twin edits, sources       │
│ • Customer input: EPC lookup / self-serve form / agent visit (photos + tags) + memory          │
│ • Schema keeps current value estimate (baseline) apart from historical sale prices             │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 │  Property + usage + sales, split
                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. DATA PREPARATION + SYNTHETIC GENERATION  (PS §9)                                            │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Map to contracts (Property, EnergyUsageMonth, SaleRecord); location = district/LSOA only     │
│ • Property/customer-level splits 70/15/15, assigned once · leakage audit (epc.potential_*)     │
│ • Annualise metered usage from available months; missing months imputed + flagged, never 0     │
│ • Physics-based synthetic generator: real EPC homes → twin + behaviour noise → monthly usage   │
│   + reference labels (per-fuel savings, improvement potential, feasibility, value: method      │
│   check only); every rule disclosed; validated vs NEED; synthetic fit ≠ real-world accuracy    │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 │  calibrated home (posterior samples)
                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. DIGITAL TWIN + BAYESIAN CALIBRATION + ENERGY PREDICTION  (only source of energy numbers)    │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Level-1 RC thermal twin: heat loss Σ U·A + ventilation, driven by degree days;               │
│   heating efficiency / heat-pump COP → per-fuel kWh; PV cuts grid import, not demand           │
│ • EBM energy-intensity prior (homes without meter data); XGBoost + NEED median baselines       │
│ • Bayesian calibration (NumPyro, hierarchical type × region): UA, setpoint, baseload, COP      │
│   fitted to each home's meter data → POSTERIOR SAMPLES → full distribution on every output     │
│ • Modes: calibrated (real behaviour) · standard (SAP occupancy, for the rating estimator)      │
│ • ENERGY PREDICTION: annual + monthly electricity and gas (kWh/yr) per home, full distribution │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 │  score + key issues
                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. GREEN HOME SCORE + POOR-HOME DETECTOR  (PS §2, §3)                                          │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • S = Σ w_k·s_k, fixed weights; monotonic, documented sub-scores; rounded; 5 bands             │
│ • Energy efficiency = recorded SAP · Intensity = annualised METERED kWh/m² (not normalised)    │
│ • Inputs used by two components (e.g. PV: renewables + intensity) justified in SCORE.md        │
│ • Waterfall of 7 component contributions · score interval · P(each band)                       │
│ • Flag Poor / Very poor + key issues (per m², peer context: large ≠ inefficient)               │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 │  ladder + chosen package
                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. RETROFIT ENGINE  (PS §4) — feasibility first, then simulation                               │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ STEP A  Feasibility                                                                            │
│   • Hard rules veto first (conservation area → no EWI, no roof access → no PV/loft, ...)       │
│   • Jev (System 1) on the survivors: feasible / with conditions / needs survey / infeasible,   │
│     calibrated confidence; low confidence → needs survey · no key → Claude fallback            │
│ STEP B  Digital-twin simulation of every feasible retrofit (posterior samples)                 │
│   • Order: fabric → glazing → controls → heating (sized on reduced demand) → PV/battery        │
│   • Marginal effect of each step = rerun twin with it added to the steps before it             │
│   • Package = one rerun of the whole set (interactions computed, never summed)                 │
│ STEP C  Improvement ladder: 'what do I gain by adding the next retrofit?'                      │
│   • Per step, full distribution: Δgas, Δelec, Δpurchased, Δscore, ΔCO2, Δbill, net cost        │
│   • Verdict per step: Recommended / Optional / Not worth adding (+ reason), per goal           │
│   • Suitability: P(suitable) = P(feasible) × P(net benefit) → High / Medium / Low              │
│ STEP D  Package choice: Pareto front (cost, CO2, bills, score) · P(on front) · knee · budget   │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 │  ScenarioResult
                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. ESTIMATES  — all computed from the twin's posterior samples                                 │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Energy savings (PS §5): per retrofit + package, ELECTRICITY and GAS separately, against      │
│   one baseline (calibrated twin, same weather year before/after); increases = negative         │
│   savings (heat pump: gas ↓, electricity ↑); labelled estimates, not guarantees                │
│ • Carbon (PS §6): ΔC = Σ_f ΔE_f × EF_f with DESNZ 2025 factors (units + source);               │
│   fuel with no sourced factor (oil, LPG, ...) → unavailable + reason                           │
│ • Improvement potential (PS §3): 100 × (E_base − E_after) / E_base on purchased energy         │
│   (grid import + gas) with ALL feasible retrofits; baseline 0/missing → unavailable + reason   │
│ • Rating estimator: SAP 2012 formula on twin (standard mode), anchored to recorded SAP         │
│   → estimated rating after ('not an EPC') → projected score (intensity after = twin            │
│   prediction, labelled projected: no meter readings exist for the future)                      │
│ • PROPERTY VALUE (PS §7): Bayesian hierarchical hedonic model on real UK sales                 │
│   (location, size, type, market via HPI); applied to the current value estimate;               │
│   same home at current vs estimated-after rating → INDICATIVE range (association, not cause)   │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 │  computed JSON only
                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 6. SCENARIO ENGINE / WHAT-IF  (PS §8)                                                          │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • run_scenario(spec): every what-if reruns twin → score → retrofit → estimates → value         │
│ • One ScenarioResult with provenance + intervals; cached by hash(spec + model versions)        │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 7. GROUNDED LLM  (PS §1 core rule)                                                             │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Tool-only: get_home_facts, get_results, run_scenario, search_docs, query_graph               │
│ • Never calculates; quotes tool numbers with unit + interval; states missing/infeasible data   │
│ • Numeric validator checks every number → regenerate once → template fallback                  │
│ • Targets: groundedness ≥ 90%, factual accuracy ≥ 90%, unsupported claims < 5%                 │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 8. EXPERIENCE                                                                                  │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Customer app: score waterfall, improvement ladder, per-fuel savings, value range, chat,      │
│   green loan (illustrative), memory (PS §10: recommendations change as memory builds)          │
│ • 3D digital twin: inside/outside, estimated heat-loss colours, retrofits appear live          │
│ • Lloyds dashboard: LSOA heat map, segments, Green Profits scenarios (synthetic customers)     │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Side services used by every block:**
- Provenance on every number; cached runs (seed + data version).
- Evaluation harness on held-out splits vs simple baselines (PS §11).
- Scenario tests for the judges' hidden cases: missing or conflicting data, infeasible or interacting retrofits, electricity rising after a heating change, too little data for savings or value.
- Performance (PS §11): showcase packages are precomputed, per-home Bayesian updates run in under a second, and LLM/Jev calls are cached, so cost and latency stay low.

---

## Energy prediction (block 2)

**What is predicted:** each home's annual and monthly **electricity and gas use (kWh/yr)**, the baseline that every saving, carbon and bill figure is measured against.

**How:**
1. **Physics:** the twin computes heat loss from walls, roof, floor, windows and ventilation × the local degree days. It adds hot water and appliances, then converts to fuel through boiler efficiency or heat-pump COP. Solar reduces grid import.
2. **Prior for homes without meter data:** an **EBM** (glass-box) model trained on real NEED consumption predicts the expected kWh/m² for homes like this one. That sets the starting point.
3. **Calibration for homes with meter data:** Bayesian calibration (below) fits the twin to the home's own bills, so the prediction reflects *this* home, not an average one.

**Output:** electricity and gas kWh/yr as a full distribution, e.g. gas: mean 12,010 · mode 11,950 · std dev 520 · 90% interval 11,200–12,900 · P(> 13,000) = 0.03. Homes without meter data get the same output with a wider interval and a data-confidence flag.

**Checked on held-out homes:** error (MAE/MAPE) per fuel and interval coverage, vs simple baselines: the NEED median for the home type, and XGBoost.

---

## The Bayesian layer (where uncertainty comes from)

**One idea:** we don't know each home's exact physics, so we keep many plausible versions of it (posterior samples) and run everything on all of them.

| Step | What happens |
| --- | --- |
| **Prior** | Starting beliefs per home type × region: typical U-values (RdSAP), heating setpoint, baseload, heat-pump COP (Electrification of Heat trial), centred on the EBM prediction |
| **Calibration** | NumPyro fits heat loss, setpoint, baseload and COP to the home's monthly meter readings → **posterior samples** (fitted offline; under 1 s per home) |
| **Missing months** | Treated as unknowns, never 0. They simply widen the interval. |
| **Every output** | Computed once per posterior sample, then reported with the standard distribution summary (below) |

What the customer sees as a result:
- **Energy, savings, carbon, bills:** full distributions (mean, median, mode, std dev, percentiles, tails).
- **Score:** mean 38.2 · mode 38 · std dev 3.4 · 90% interval 33–44 · **P(Poor) = 0.81**.
- **Each retrofit:** **P(suitable)** = P(feasible) × P(net benefit) → High / Medium / Low.
- **Each package:** **P(on front)**, how often it is among the best trade-offs.
- **Value range:** value-model uncertainty × uncertainty in the rating after.
- **Memory:** each new bill or completed retrofit updates the posterior, so intervals narrow over time.

**Checked on held-out homes:** do 90% intervals contain the truth ~90% of the time? (Coverage, CRPS.)

---

## Uncertainty in every output (Bayesian, end to end)

**Rule:** every uncertain output is computed **once per posterior sample** and reported with the **standard distribution summary** below (or as a probability). Intervals are never added or multiplied together: the samples carry the correlations (a leaky home saves more from every retrofit). A bare single number appears only where the quantity is fixed by design.

### The standard distribution summary

Every uncertain output carries the same summary, computed from its posterior samples (typically 1,000–4,000):

| Part | Fields | Why |
| --- | --- | --- |
| **Centre** | mean · median (P50) · mode (most likely value: peak of the sample density) | They differ when the distribution is skewed. Showing all three stops the mean hiding a lopsided result. |
| **Spread** | standard deviation · IQR (P25–P75) | How uncertain the number is |
| **Percentiles** | P1 · P5 · P10 · P25 · P50 · P75 · P90 · P95 · P99 | The full shape, including the tails |
| **Credible intervals** | 50% (P25–P75) and 90% (P5–P95) | The 90% interval is the headline range |
| **Tail values** | Tail probabilities against meaningful thresholds, e.g. P(saving < 0), P(bill rises > £100/yr), P(score < 30) · expected value of the worst 5% (for savings and money) · skewness | What could go wrong, and how badly |
| **Shape** | Histogram (≈ 30 bins) for the UI density plot | See it to believe it |
| **Meta** | unit · number of samples · provenance (model + version) · data-confidence flag | Traceability |

**Discrete outputs** (score band, EPC band, verdict) report the **full probability of every category** plus the mode (most probable category). Mean and standard deviation are not meaningful for categories.

**Probabilities** (P(suitable), P(on front), P(net benefit)) are already a summary of the samples, so they are reported as the probability itself.

**Display rule:**
- The headline shows **mean (90% interval)**, or the median when the distribution is clearly skewed.
- Tapping any number opens the **full summary card**: centre, spread, percentiles, tails and the density plot.
- The LLM quotes the headline, the interval and any relevant tail probability, all straight from the tool output.

**Example, energy prediction (SH2 gas baseline, illustrative):**

| Mean | Median | Mode | Std dev | P5 | P25 | P75 | P95 | Tail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12,010 | 12,000 | 11,950 | 520 | 11,200 | 11,650 | 12,350 | 12,900 | P(gas > 13,000 kWh/yr) = 0.03 |

**Implemented in the contracts (schema 0.3.0):** `Distribution` in `contracts/models.py` (mirrored in `types.ts`) carries exactly these fields, with a validator for consistency. `contracts/summarise.py` builds it from samples (`summarise`, `exact` for known values, `category_probabilities` for bands), so every module reports uncertainty the same way. Tests: `tests/test_distribution.py`.

**Where the uncertainty comes from:**

| Source | How it is modelled |
| --- | --- |
| The home's physics (heat loss, setpoint, baseload, COP) | Posterior from Bayesian calibration on its meter readings; prior only (wider) if no meter data |
| Missing months, estimated reads, imputed fields | Latent variables / wider noise, never 0 |
| Weather | Typical, cold and warm years sampled |
| Behaviour after retrofit (rebound: warmer home, higher setpoint) | Prior on setpoint shift after fabric retrofits |
| Retrofit costs | Distribution over the sourced low–high range (BEIS 2017 × CPI); grants per eligibility |
| Rating after retrofit | Twin samples through the SAP estimator → rating distribution + band probabilities |
| Property value | Posterior of the hedonic model (location effects, β) |
| Feasibility | Jev confidence recalibrated (isotonic) into P(feasible); rule veto = 0 |

**What every output looks like** ("full summary" = the standard summary above):

| Output (PS section) | Shown as |
| --- | --- |
| Electricity and gas use (§5) | kWh/yr, full summary |
| Green Home Score (§2) | Score full summary; **P(each band)**; the classification is the most probable band with its probability |
| Score contributions (§2) | Waterfall bars; uncertain components (intensity, imputed fields) carry their full summary |
| Poor home flag + key issues (§3) | **P(Poor or Very poor)**; e.g. "P(intensity above peers) = 0.9" |
| Savings per retrofit + package, per fuel (§5) | kWh/yr full summary; increases shown negative |
| Carbon (§6) | t/yr full summary (factors fixed; uncertainty comes from energy) |
| Improvement potential (§3) | % full summary; P(in top 10%) for ranking |
| Bills, net cost, payback | £ full summary |
| Suitability (§4) | P(suitable) = P(feasible) × P(net benefit) → High / Medium / Low |
| Best package | P(on front) per package; knee chosen on the posterior front |
| Estimated rating and projected score | SAP full summary; P(each EPC band) |
| Value impact (§7) | % and £ range (full summary), indicative; current value with an interval if it is model-estimated |
| Green loan | Loan amount full summary; **P(monthly bill saving ≥ repayment)**, "cash-flow positive?" |
| Lloyds dashboard | Area totals summed over samples → full summary; Green Profits Monte Carlo → full summary incl. downside tail |
| 3D twin | Colours = posterior mean heat-loss share; hover shows the full summary and the "estimated" label |
| LLM answers | Quotes the headline (mean or median), the 90% interval and any relevant tail probability, all from the tool output |

**Fixed by design (single values are correct here):** the PS weights, the score formula applied to fixed inputs, DESNZ emission factors, Ofgem tariffs, grant amounts, and rule vetoes. These are documented constants or rules, not estimates.

**Checked on held-out homes:** do 90% intervals contain the truth ~90% of the time (coverage, CRPS)? Are the probabilities calibrated (a "0.8" happens about 80% of the time)?

---

## Property value (block 5, PS §7)

**Model:** a **Bayesian hierarchical hedonic regression** on real UK sales (UCL Land Registry–EPC linked data):

```
log(price ÷ local HPI) = location effect (LSOA within local authority within region)
                       + floor area + property type + built form + age + tenure
                       + β × energy rating                    ← the green effect, by region × type
```

- **Location, size, type, market:** all four factors the PS names (market = HPI adjustment).
- **Before vs after (as the PS asks):** the same home at its current rating vs the estimated rating after the retrofits, everything else held constant → a **range**, applied to the home's current value estimate (never to an old sale price).
- **Bayesian:** location effects are pooled (sparse areas borrow from their region), and the result carries uncertainty from both the model and the rating estimate.
- **Explainable:** each coefficient is a % effect, so the customer sees what drives their home's value.
- **Honest:** always labelled *"indicative range based on historical market relationships; not a proven or guaranteed increase."* If the range includes 0, we say the data shows no clear effect.
- **Checked on held-out sales:** interval coverage and error vs a simple baseline (the same regression without the energy rating), and β vs published figures (Nationwide, DECC 2013).

---

## Retrofit engine in one line

**Rules + Jev decide *what is possible*; the digital twin decides *what it is worth*.**

- **Feasibility** = hard rule vetoes first, then Jev on the survivors. Jev only ever answers *feasible / with conditions / needs survey / infeasible*, with a confidence.
- **Everything numeric** (kWh per fuel, carbon, bills, score after, improvement potential, value input) comes from rerunning the calibrated twin. Jev never produces a number.

---

## The improvement ladder: what each retrofit adds

The PS asks for savings per retrofit **and** for the package, accounting for interactions (§4, §5). The ladder shows both at once, and tells the customer whether adding the next retrofit is worth it.

**How it is computed**

1. Take the feasible retrofits in the physical order: fabric → glazing → controls → heating system → generation.
2. Step *k*'s marginal effect = twin(steps 1..k) − twin(steps 1..k−1), run on every posterior sample, against **one baseline** (the calibrated twin, same weather year before and after).
3. The marginals **add up exactly** to the package result, because each step is measured on top of the previous ones. Per-retrofit numbers stay consistent with the package and still include interactions (e.g. the heat pump is sized on the already-insulated home).
4. Removing a step changes the steps after it, so any other combination is a **new twin run**, never "package minus one row".

**Example layout (SH2, 1972 cavity semi). ILLUSTRATIVE NUMBERS ONLY; real values come from the twin.**

**How to read it:** each cell shows the headline **mean (90% interval)**. In the product, every cell opens the full summary card (mean, median, mode, std dev, percentiles, tails); one card is shown below the table. Verdicts use probabilities, not means.

- Baseline: gas 12,000 (11,200–12,900) + electricity 3,200 (2,900–3,500) kWh/yr = 15,200 kWh/yr purchased · score 60 (57–63), P(Moderate) = 0.86.
- Carbon uses the repo's DESNZ 2025 factors (gas 0.18296, electricity 0.19553 kgCO2e/kWh).
- Bills use Ofgem Q4 2026 unit rates (gas 7.97p, electricity 26.32p per kWh).

| Step | Add | Gas saving (kWh/yr) | Elec. saving (kWh/yr) | Improvement potential so far | Score after | CO2 saved (t/yr) | Bill saving (£/yr) | P(net benefit) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Cavity wall insulation | +2,600 (2,100–3,100) | 0 | 17% (14–20) | 66 (64–68) | 0.48 (0.38–0.57) | +207 (167–247) | 0.93 | **Recommended** |
| 2 | Loft top-up to 270 mm | +900 (650–1,150) | 0 | 23% (20–27) | 68 (66–70) | 0.16 (0.12–0.21) | +72 (52–92) | 0.88 | **Recommended** |
| 3 | Draught proofing | +400 (200–600) | 0 | 26% (22–30) | 69 (67–71) | 0.07 (0.04–0.11) | +32 (16–48) | 0.71 | Optional |
| 4 | Smart thermostat + TRVs | +500 (300–750) | 0 | 29% (25–33) | 71 (69–73) | 0.09 (0.05–0.14) | +40 (24–60) | 0.84 | **Recommended** |
| 5 | Replace pre-2002 double glazing | +450 (250–650) | 0 | 32% (27–36) | 72 (70–74) | 0.08 (0.05–0.12) | +36 (20–52) | 0.22 | Not worth adding |
| 6 | Air-source heat pump | +7,150 (6,500–7,800) | **−2,300 (−2,900 to −1,800)** rises | 64% (58–69) | 80 (77–83) | 0.86 (0.70–1.01) | **−35 (−180 to +110)** | bills 0.38 · carbon 0.97 | **Depends on goal** |
| 7 | Solar PV 3 kWp | 0 | +1,100 (850–1,300) import cut | 71% (65–76) | 86 (83–88) | 0.22 (0.17–0.25) | +290 (224–342) | 0.81 | **Recommended** |
| | **Package (one twin run)** | **+12,000 (11,200–12,900)** | **−1,200 (−1,900 to −500)** | **71% (65–76)** | **86 (83–88)** | **1.96 (1.75–2.15)** | **+641 (480–790)** | | |

Package intervals come from running the whole package on each posterior sample. They are **not** the sum of the row intervals, because the rows are correlated: a leakier home saves more from every step. The package gas saving equals the whole baseline gas use, since the heat pump removes gas entirely, so it carries exactly the baseline's interval.

**Full summary card, heat-pump bill change (step 6, illustrative).** This shows why tails matter:

| Mean | Median | Mode | Std dev | P5 | P10 | P25 | P75 | P90 | P95 | Skew |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| −£35 | −£42 | −£55 | £88 | −£180 | −£145 | −£95 | +£20 | +£80 | +£110 | +0.3 |

Tails:
- **P(bill saving > 0) = 0.38**.
- **P(bill rises by more than £100/yr) = 0.23**.
- Average of the worst 5% of outcomes: **−£205/yr**.

The mean alone (−£35) would hide that roughly 1 in 4 homes like this could see their bill rise by over £100/yr. The mode (−£55) is the single most likely outcome.

**Heat-pump note.**
- At current UK prices electricity costs ~3.3× gas per kWh, and a typical seasonal COP is ~3. The bill change is therefore centred near zero: **−£35/yr (−180 to +110)**, and **P(bill saving > 0) = 0.38**. Meanwhile carbon falls 0.86 t/yr with P(CO2 cut > 0) > 0.99.
- If the gas supply is removed entirely, the gas standing charge (29.68p/day ≈ £108/yr) also goes, and P(bill saving > 0) rises to about 0.8. This is reported separately and labelled, as `contracts/tariffs.json` requires.
- This is exactly the "electricity rises after a heating change" scenario judges test. The probability makes the trade-off honest instead of hiding it in a net figure.

Each row in the product also shows net cost after grants, £ per kWh saved and payback, all as intervals from the cost and savings samples.

**Suitability vs verdict.**
- **Suitability** (High / Medium / Low) is the field the PS requires for every listed retrofit: P(suitable) = P(feasible) × P(net benefit), with High ≥ 0.8, Medium 0.5–0.8, Low < 0.5.
- **Verdict** adds one question on top: *how likely is the extra gain from this step to be big enough?*
- Both come from the same samples, so they never contradict each other.

**Verdict rules (probabilistic, per step; thresholds to be agreed by the team):**

| Verdict | Rule |
| --- | --- |
| **Recommended** | P(suitable) ≥ 0.8 **and** P(step adds ≥ 3% of baseline purchased energy, or ≥ 2 score points) ≥ 0.8 |
| **Optional** | P(suitable) 0.5–0.8, or the gain probability is below 0.8 at low cost |
| **Not worth adding** | P(suitable) < 0.5, or P(cost per kWh saved below the cut-off) < 0.5 |
| **Needs survey** | Passed the rules but Jev's calibrated P(feasible) is below τ; shown with what the survey must check |
| **Not feasible** | Rule veto (P(feasible) = 0) or Jev veto, always shown with the reason (e.g. "conservation area: external wall insulation not allowed"). Not in the recommended list (PS §4: feasible only) |

**"Net benefit" follows the customer's goal:**
- **Bills** (default): P(discounted lifetime bill saving ≥ net cost after grants), computed per sample with cost uncertainty included.
- **Carbon**: P(CO2 cut per £ above the cut-off).

The goal changes verdicts, never the computed distributions.

**"Should I add another retrofit?" card.** For any current plan, rerun the twin once per remaining feasible retrofit and show the best next addition with its marginal gain (full distribution) and P(gain ≥ threshold). If none is likely to pass, say so: "Each further addition has less than a 20% chance of improving purchased energy by 3% or more; your plan is close to this home's improvement potential."

---

## Improvement potential (PS §3)

- **Definition (as the PS states it):** improvement potential = 100 × (E_baseline − E_after) / E_baseline.
  - E = annual **purchased** energy = grid electricity import + gas (kWh/yr), with the same accounting convention for both totals.
  - E_after uses **all retrofits feasible for that home**, simulated together by the twin.
- **Which retrofits count as feasible:** "feasible" and "feasible with conditions" count. "Needs survey" items are left out of the headline number and reported separately ("up to X% if the survey confirms the heat pump"). "Not feasible" items never count.
- **Potential ≠ the recommended plan.** Potential is the technical maximum: it includes feasible steps marked "Not worth adding", like the glazing above, because the PS defines it by feasibility, not economics. The customer's plan shows its own %, from its own twin run.
- **A distribution, not a single %:** computed per posterior sample → full summary, e.g. mean 71% · mode 72% · std dev 3.4 · 90% interval 65–76%.
- **Ranking homes under uncertainty:** rank by the posterior mean, and show P(in the top 10%) so the Lloyds view can tell a confidently high-potential home from an uncertain one.
- **Always shown with the per-fuel split.** A heat pump can cut gas and raise electricity, so the % never hides the fuel change.
- **Solar** reduces grid import, not demand, so it counts in purchased energy exactly as the PS defines it.
- **Unavailable cases:** baseline 0 or missing → "unavailable" + the reason, never 0%.
- **Uses:**
  - Ranks Poor / Very poor homes by how much they could improve.
  - Feeds the Lloyds heat map (where finance would cut the most energy and carbon).
  - The ladder shows the path from today to that potential.
- **Fairness:** a flat with no roof access has fewer feasible retrofits, so its potential is honestly lower. Rankings use % and kWh/m², so large homes are not penalised for size.

---

## Checks that the recommendations are good (PS §11, ML & recommendations)

**1. Sanity check against the PS illustrative impacts.** On typical homes, each twin marginal is compared with the brief's illustrative ranges (not additive, and our documented catalogue + twin take precedence, PS §4):

| Retrofit | PS illustrative impact | Check |
| --- | --- | --- |
| Roof / loft insulation | 12–18% of space-heating demand | Lower is expected for top-ups (e.g. 100 → 270 mm) |
| Heat pump | 15–25% of purchased energy; electricity may rise | **The twin can exceed this.** Replacing ~1 kWh of gas with ~⅓ kWh of electricity cuts purchased kWh more. If so, document why (Electrification of Heat trial SCOP, gas boiler efficiency). |
| Solar panels | 20–35% of grid imports | Self-consumption share from load shape |
| Double glazing | 8–12% of space-heating demand | Smaller when upgrading existing double glazing |
| Smart thermostat | 5–10% of heating energy | — |

**2. Real-world check against EPC assessors.**
- Compare our ranked recommendations with the **real EPC recommendation lists** for held-out homes (precision@k, ranking agreement).
- **Baselines:** "recommend every feasible retrofit" and "rank by kWh saved".
- **Hard requirement:** 0 infeasible items recommended.

**3. Reference labels.**
- Per-fuel savings and improvement-potential ranking against the synthetic reference labels (Spearman / NDCG vs "rank by total consumption").
- Reported with the caveat that synthetic agreement is not real-world accuracy.

**4. Jev.** Feasibility precision/recall on the infeasible class and calibration error on a labelled home × retrofit set, vs rules only.

---

## What each PS requirement maps to

| PS section | Delivered by |
| --- | --- |
| §1 Core rule + what-if rerun | Blocks 6 + 7 |
| §2 Green Home Score, contributions, justified overlaps | Block 3 (+ rating estimator in Block 5 for projected scores) |
| §3 Poor homes, key issues, improvement potential | Blocks 3 + 5 |
| §4 Feasible, prioritised retrofits with suitability + saving + reason; catalogue documented | Blocks 0 + 4 (rules + Jev, ladder, verdicts) |
| §5 Per-fuel savings, per retrofit + package, interactions, consistent baseline | Blocks 2 + 4 + 5 (twin reruns) |
| §6 Carbon with documented factors | Block 5 |
| §7 Indicative value range, before vs after, location/size/type/market | Block 5 (rating estimator + Bayesian hedonic) |
| §8 Expected outputs, pipeline, what-if flow | Blocks 3–8 |
| §9 Data, schema, splits, leakage, synthetic labels, value baseline vs sale price | Blocks 0 + 1 |
| §10 Customer memory | Blocks 0 + 8 |
| §11 Evaluation vs baselines, scenario tests, LLM targets, performance | Side services + checks section |
| §12 Business relevance, UX | Block 8 (green loan, Lloyds dashboard, 3D twin) |
| §13 F: architecture diagram | This document |
