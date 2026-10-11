# Property value: research, benchmarks and causal limits

What the evidence says about energy ratings and UK house prices, which benchmarks our estimates will be compared with, and what the model can and cannot claim. Status labels as in the score docs: **[verified]** read in the primary source; **[secondary]** press or commentary; **[to verify]**.

## 1. Published premiums (owner-occupied, relative to EPC D)

| Source | Data and method | A/B vs D | Other bands | Status |
| --- | --- | --- | --- | --- |
| Nationwide (2021) | Hedonic on England sales | +1.7% | F/G −3.5% | [secondary: [Estate Agent Today](https://www.estateagenttoday.co.uk/breaking-news/2021/8/top-epc-ratings-have-almost-no-impact-on-house-price--nationwide)] |
| Nationwide (2024) | Hedonic, controls for bedrooms, location, new build | +2.8% | F/G −4.2% | [secondary: [Landlord Today](https://old.landlordtoday.co.uk/breaking-news/2024/8/only-small-premiums-for-homes-with-better-epcs-finds-nationwide)] |
| Nationwide (2026) | Same method, latest | +1.6% (≈ £4,500) | F/G −1.4% | [secondary: [Mortgage Solutions](https://www.mortgagesolutions.co.uk/mortgage-news/2026/06/26/epc-ratings-have-limited-impact-on-owner-occupied-house-prices-despite-push-for-lower-ratings-by-2030/); report: [Nationwide](https://www.nationwide.co.uk/media/hpi/reports/energy-efficiency-ratings-have-limited-impact-on-owner-occupied-house-prices-despite-increased-interest-in-going-green)] |
| Oxford Economics (2024) | ≈ 11M Land Registry sales linked to EPCs | +3.4% | — | [secondary: [Oxford Economics](https://oxfordeconomics.com/resource/the-growing-importance-of-energy-efficiency-in-home-buying-decisions)] |
| The Mortgage Works (2024) | Owner-occupier England | +2.2% | buy-to-let A/B +10.9%, C +3.4% | [secondary: [Mortgage Solutions](https://www.mortgagesolutions.co.uk/mortgage-news/2024/10/17/greener-btls-worth-up-to-20000-more-than-lower-rated-homes/)] |
| Fuerst, McAllister, Nanda, Wyatt (DECC 2013; Energy Economics 2015) | ≈ 333k repeat sales 1995–2012, hedonic + repeat sales | positive premium vs G for better bands | figures not accessed | [secondary: [Reading CentAUR](https://centaur.reading.ac.uk/39371); full text restricted, **to verify**] |
| Systematic review, 68 European studies (Glasgow PhD 2026) | Meta-review | ≈ +1–3% per EPC band | — | [secondary: [thesis](https://theses.gla.ac.uk/86042/4/2026OuPhD.pdf)] |
| Brief §7 example | Illustrative | E → C: +2–5% | — | brief |

**Benchmark range used for validation:** owner-occupied A/B vs D ≈ **+1.6% to +3.4%**; roughly **+1–3% per band**. A SAP point is about 1/12 of a band (bands are 11–17 points wide), so the per-SAP-point effect we expect is in the order of **+0.1% to +0.25% per point**. Our posterior for β will be reported against this, by region and type.

## 2. Why the estimate is indicative, not causal (brief §7, §9)
Confounders the model must name and, where possible, control:
1. **Other renovations.** Homes that improve their EPC often get new kitchens, extensions or redecoration at the same time; the rating change picks up their value.
2. **Selection.** Owners who improve differ from those who don't (income, intention to sell).
3. **Location and neighbourhood quality** correlated with building age and fabric (controlled by LSOA effects).
4. **Market timing.** Prices move by area, type and month (controlled by UK HPI deflation and period effects).
5. **Size and type.** Larger and detached homes differ in both price and rating (controlled; per-m² terms).
6. **Measurement.** An EPC is a modelled rating that can be months or years older than the sale (V-05 window); the rating at sale is unknown when no certificate was in force.
7. **Buy-to-let vs owner-occupied.** Landlords face MEES rules, so premiums differ by tenure (the BTL figure is several times larger in the sources above).

Wording, everywhere (contract `ValueImpact.wording`): "Indicative range based on historical associations in sales data; not a promised or proven increase." If the 90% interval includes 0, say there is no clear effect. Never a reason to borrow (AGENTS.md compliance fix 6).

## 3. Design choices to make next (will become V-xx decisions)
- EPC-before-sale window N (V-05).
- Market adjustment: UK HPI by LA × property type × month, or period fixed effects in the model.
- Outcome: log price (brief: "value") vs log price per m² (`priceper` in the data); log price with log floor area as a regressor is the usual hedonic form.
- Exclusions: PPD category B (non-market), new builds (separate premium), extreme prices.
- Splits: by property (address hash), 70/15/15; plus an out-of-time test (latest year) for "market conditions".
- How `current_value_estimate_gbp` is used: the uplift applies to the current value, never to a historical sale price; if missing, the model's posterior predictive value for the home.
