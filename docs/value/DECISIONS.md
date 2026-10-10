# Property value (Part 4): decisions, proofs and worked examples

Running log for Part 4, same method as the Green Home Score (`docs/score/DECISIONS.md`). Every decision has an id (V-xx), its reason, and where it applies a quantitative proof with the command that reproduces it. Research and benchmarks: [VALUE_RESEARCH.md](VALUE_RESEARCH.md).

Branch: `p4/property-value` (from `p3/green-home-score`, so it reuses the EPC parsers, priors and score code).

**Brief §7 (what is required):** an estimated potential change in property value **as a range**, comparing a consistent baseline with the post-retrofit scenario for the same home (other characteristics held constant), accounting for location, size, property type and market conditions, described as an *indicative estimate based on historical and market relationships*, never as a guaranteed or causal increase. Evaluated on transparency, the held-out range against reference estimates, and treatment of confounding.

## Decision index

| Id | Decision | Status |
| --- | --- | --- |
| V-01 | Sales data: UCL linked Land Registry PPD + EPC (ReShare 857911, CC BY 4.0), streamed remotely | Decided (user, 11 Oct 2026) |
| V-02 | Method: Bayesian hierarchical hedonic (NumPyro) primary; baselines OLS without rating and LightGBM quantile; matched-pairs check vs published premiums | Decided (user) |
| V-03 | Rating enters as continuous SAP points: uplift = exp(β · ΔSAP) − 1, β by region × type | Decided (user) |
| V-04 | Coverage: all England & Wales, property-level sample; hierarchy pools sparse areas | Decided (user) |
| V-05 | Use only sales whose linked EPC was lodged **before** the sale (and within a window): the linked EPC is not the rating at sale | Decided |
| V-06 | Property key = hash of the full address; the address is never stored; postcode used only to join NSPL, then reduced to district + LSOA | Decided |
| V-07 | Market conditions: price adjusted to July 2026 with UK HPI by LA × property type (fallback region × type, then England & Wales) | Decided |
| V-08 | Evaluation: property-level 70/15/15 split + out-of-time test (2024 sales of unseen properties) | Decided |

---

## V-01 Sales data
`857911_bundle_2.zip` holds the linked table `tranall_link_26122024.csv` (106 columns, 23.25 GB uncompressed, 3.31 GB zipped, nested zip) plus NSPL November 2024 (postcode → LSOA, LA, region) and the raw PPD. ReShare serves the bundles with HTTP Range (verified: `206 Partial Content`), so `scripts/ucl_sales_extract.py` reads the nested zip remotely and keeps only the needed columns, like the EPC extractor (score D-14). Disk is tight (D: 6.3 GB, E: 7.7 GB free), so nothing large is stored. Coverage 1 Jan 1995 – 31 Oct 2024, England and Wales, > 22M sales. Attribution: Chi, Dennett, Oléron-Evans, Morphet (UCL); contains HM Land Registry data © Crown copyright, OGL v3.0.

## V-02 Method
Primary: Bayesian hierarchical hedonic regression (LLD §4.4):
`log(price / HPI) = location (LSOA in LA in region, pooled) + f(log floor area) + type × form + age band + tenure + rooms + β_{region×type} · SAP`. Gives a full posterior for β, so the value range is a real distribution. Baselines (brief §11): OLS **without** the rating (does the rating add anything?) and LightGBM quantile (non-linear ceiling for the interval). Check: matched pairs (same type/form/age/LSOA/period, floor area ±10%, ≥ 15-point Green Home Score gap) and published premiums (VALUE_RESEARCH.md).

## V-03 Rating term
Continuous SAP points: `uplift_s = exp(β_s · ΔSAP_s) − 1` per posterior sample s, with ΔSAP from Part 2's `RatingEstimate` (sap_estimator). Keeps sub-band improvements and propagates rating uncertainty. Band dummies are kept as a check, because buyers see bands.

## V-04 Coverage
All England and Wales, sampled by property (10% of properties by address hash ⇒ about 2.2M sales), so every region and type informs its own β with partial pooling. Showcase homes are in the North West, South West and London.

## V-05 The linked EPC is not the rating at sale
The linked table attaches an EPC to each sale, but not necessarily one in force at the sale: the first rows include a **1998 sale linked to an EPC lodged in 2021**. Using a later EPC would let improvements made after the sale explain the price (leakage, and reverse timing). Rule: keep a sale only if its linked EPC was lodged **before** the transfer date and **at most N years earlier** (N chosen from data, see the extraction results; EPCs are valid for 10 years). Sales before EPCs existed (pre-2007) are therefore dropped from the rating model, though they can still inform location effects. Logged so the value model never sees a post-sale rating.

**Proof** (`uv run python scripts/prepare_sales.py`, 10% property sample, 2,241,184 sales): the gap *sale date − EPC lodgement date* has quantiles P5 −19.7, P25 −11.3, P50 −0.87, P75 +0.67, P95 +7.3 years; **51.4% of linked sales have an EPC lodged after the sale**. With N = 10 years, 1,191,184 sales are dropped and **1,047,182 kept (2008–2024)**. Among the kept sales the EPC was lodged a median **0.7 years** before the sale (mean 2.0), consistent with EPCs commissioned for the sale itself, so the rating is close to the rating at sale.

## V-06 Privacy
PPD and the linked table contain full addresses. The extractor builds `prop_key = md5(postcode|PAON|SAON|street|locality)[:16]` for property-level splits and repeat-sales, then drops the address columns before writing. Postcode is kept only until the NSPL join (LSOA, LA, region), then reduced to the outward code. Brief §9: no personal data; location shown as district or LSOA only.

## V-07 Market conditions (UK HPI)
UK HPI full file to July 2026 (HM Land Registry, OGL) gives a monthly index per local authority with separate detached / semi / terraced / flat series. Each sale price is scaled to July 2026: `price_adj = price × HPI(LA, type, Jul 2026) / HPI(LA, type, sale month)`, falling back to the region × type and then the England & Wales all-type index. Result: 98.8% matched at LA × type, 1.2% at region × type, 0% needed the national fallback. This (a) puts 2008 and 2024 sales on one footing and (b) makes the model's predictions directly "today's money", which is what `current_value_estimate_gbp` is in. Residual year effects stay in the model to absorb HPI misfit. Raw prices: median £225,000; adjusted: median £308,636.

## V-08 Evaluation design
- Property-level split by address hash (V-06): 732,252 / 157,991 / 156,939 sales; no property in two splits (brief §9). The fit uses only pre-2024 sales: **train 712,053**, **test 152,630**.
- Out-of-time test: 2024 sales of properties in val/test (**8,602**): unseen homes in an unseen period, the "market conditions" check.
- Metrics (brief §11, LLD §10): 90% interval coverage, mean pinball loss at 5/50/95%, MAE of log price, median absolute % error of the median; rating effect vs the published premiums (VALUE_RESEARCH.md §1).
- Baselines: OLS with LA + type/form + age + year fixed effects, **without** SAP and **with** SAP (does the rating add anything?), and LightGBM quantile (non-linear ceiling for the intervals).

## V-09 First Bayesian result: rating effect larger than published (confounding signal)
v1 fit (`scripts/train_value_model.py --steps 8000 --version v1`; 712,053 train sales; ELBO 6.53M → −102k; posterior saved in `artifacts/value_model/v1/`):
- Pooled rating effect **+0.25% per SAP point** (90% 0.20–0.31%), inside the top of the expected 0.1–0.25%/point range.
- Implied premium vs D: A/B **+7.8%** (6.1–9.4), C +3.4% (2.7–4.0), E −3.7% (−4.5 to −3.0), F/G **−10.0%** (−11.9 to −8.0). Published owner-occupied figures: A/B +1.6% to +3.4%, F/G −1.4% to −4.2%.
- Our extremes are 2–3× the published ones: likely omitted condition/renovation (refurbished homes are better rated *and* worth more). **Next:** within-LSOA × type × age-band × year comparison and a repeat-sales check, before the β is used in the product; until then the range must be presented as an upper-leaning, indicative association.
- Evaluation incomplete: the OLS baseline hit a MemoryError at `lstsq` on the 250k design. Fix pending: smaller OLS sample or sparse solver. Smoke run (50 SVI steps, 2k eval rows): OLS without SAP 90% coverage 0.901, median APE 16.2%; with SAP essentially identical (MAE 0.2077 → 0.2078); LightGBM quantile coverage 0.874, median APE 12.2%.

## V-10 Green twins (code written, not yet run)
`greentwin/value/green_twins.py`: greener comparable sales nearby, as ranges only. Score at sale = Green Home Score without intensity (brief weights renormalised over six components, same EPC parsers as the score); same type, form, age band, floor area ±10%, sold 2019–2024; LSOA → MSOA → LA; tiers +15–25 / +25–35 / +35 points; ≥ 5 sales per tier; HPI-adjusted £/m² as a Distribution.

---

## Data on hand
| Dataset | Path (git-ignored) | Rows | Notes |
| --- | --- | --- | --- |
| UCL linked sales, 10% of properties | `data/interim/sales/tranall_link_26122024_s10.parquet` | 2,241,184 | 41 columns incl. price, date, PPD type/tenure/new-build, EPC rating, floor area, rooms, age, element descriptions; streamed in 51 min (3.25 GB read) |
| Prepared training table | `data/processed/sales_prepared_v1.parquet` | 1,047,182 | After V-05, NSPL join, HPI adjustment, splits |
| NSPL Nov 2024 | `data/interim/geo/NSPL21_NOV_2024_UK.parquet` | 2,709,658 postcodes | LSOA21, MSOA21, LA, region, IMD rank, urban/rural |
| UK HPI full file Jul 2026 | `data/raw/hpi/UK-HPI-full-file-2026-07.csv` | 151,920 | LA × month, by property type |
| EPC register 25% sample, NEED | in the score worktree (`data/interim/epc25`, `data/raw/need`) | 7M certs / 4M | Reused for priors and features |
