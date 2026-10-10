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

## V-06 Privacy
PPD and the linked table contain full addresses. The extractor builds `prop_key = md5(postcode|PAON|SAON|street|locality)[:16]` for property-level splits and repeat-sales, then drops the address columns before writing. Postcode is kept only until the NSPL join (LSOA, LA, region), then reduced to the outward code. Brief §9: no personal data; location shown as district or LSOA only.

---

## Data on hand
| Dataset | Path (git-ignored) | Rows | Notes |
| --- | --- | --- | --- |
| UCL linked sales, 10% of properties | `data/interim/sales/tranall_link_26122024_s10.parquet` | ≈ 2.2M (extracting) | 42 columns incl. price, date, PPD type/tenure/new-build, EPC rating, floor area, rooms, age, element descriptions |
| EPC register 25% sample, NEED | in the score worktree (`data/interim/epc25`, `data/raw/need`) | 7M certs / 4M | Reused for priors and features |
