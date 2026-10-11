# Retrofit engine: decisions, proofs and worked examples

Running log for the Part 2 retrofit engine (`greentwin/retrofit`, `greentwin/optimise`). Every decision has an id, the reason and, where it applies, a proof or worked example that can be rerun.

Branch: `p2/retrofit-engine` (from `main`). Design source: `LLD.pdf` §4.2 (`deliverables/src/lld_content.md`).

## Decision index

| Id | Decision | Status |
| --- | --- | --- |
| R-01 | Branch from `main`, code against `GreenDigitalTwin`; take only `greentwin/retrofit/rules.py` from `satyarth` | Decided (user, 11 Oct 2026) |
| R-02 | Ranged costs → two-piece lognormal: median = central, P5 = low, P95 = high | Decided |
| R-03 | Central-only costs → lognormal, log-sd 0.25, flagged as an assumption | Decided (assumption) |
| R-04 | Only confirmed grants reduce net cost: BUS £7,500. BUS £9,000 off-gas, ECO4, GBIS listed but not applied | Decided |
| R-05 | Package cost = sum of item costs (costs add, savings never do); `None` if any item is unsourced | Decided |
| R-06 | Build order: catalogue → rating → ladder → potential → Pareto | Decided (user, 11 Oct 2026) |

---

## R-01 Base branch
`satyarth` holds the real physics, calibration and rules, but isn't merged. Working from `main` keeps this branch small; `GreenDigitalTwin` loads `greentwin/twin/physics.py` automatically once it exists, so the real physics drops in without code changes here. `rules.py` (feasibility layer 1) is copied unchanged from `origin/satyarth`; it uses only contract fields already on `main`.

**Proof (showcase rules).** `assess(SH*)` on the example properties:
- SH1: `external_wall_insulation` is vetoed (conservation area) and `internal_wall_insulation` is feasible.
- SH2: `cavity_wall_insulation` and `air_source_heat_pump` are feasible.
- SH3: loft and PV are vetoed, and `air_source_heat_pump` is `needs_survey`.

These match the Block 4 "done when" checks.

## R-02 Cost distribution from low / central / high
BEIS 2017 Table 3 gives the lowest, mean and highest cost per home across literature sources. These are skewed (cavity £670 / £1,040 / £1,940), so a triangular with mean = central has no valid mode. We read low/high as P5/P95 and central as the median of a lognormal with a different log-sd on each side. It's always positive, matches all three published points and keeps the skew.

**Caveat:** BEIS's central is a mean, and we treat it as a median. For right-skewed items, the true median is a little lower.

**Proof:** `tests/test_retrofit_catalogue.py::test_ranged_cost_matches_source_percentiles`. On loft insulation, P5/P50/P95 are within 3% of £260 / £620 / £930.

## R-03 Central-only costs
Smart controls (£620), the heat pump (£12,500) and PV (£1,876/kWp) have one figure each. Rather than present a point cost as certain, we sample a lognormal with log-sd 0.25 (P5–P95 ≈ ×0.66–×1.51) and add a warning naming the assumption.

**Worked example:**
- Heat pump gross median £12,500 minus BUS £7,500 gives a net median of about £4,850 (90% range about £640–£11,400).
- Most of that width comes from the assumed spread, not from data.
- Replace the assumption once the DESNZ BUS statistics give a range.

## R-04 Grants
`GRANTS` in `catalogue.py` records an amount and a `confirmed` flag. Only confirmed schemes are subtracted:
- BUS rates are alternatives, never stacked.
- The £9,000 off-gas rate is considered only for off-gas-grid homes, and is listed as not applied until it's confirmed on gov.uk.
- ECO4/GBIS depend on eligibility and are excluded (AGENTS.md "Still to verify").

Net cost is floored at 0.

## R-05 Package cost
Installation costs are additive, unlike savings, which always come from one twin run on the whole package. If any item in a package has no sourced cost, the package cost is `None`, because a partial sum would make the package look cheaper than it is. Such packages are excluded from cost ranking (compliance fix 2).

## R-06 Build order
1. `catalogue.py` (done).
2. `rating.py`: unblocks the projected score and the value "after".
3. `ladder.py`.
4. `potential.py`.
5. `optimise/pareto.py`.

## Open
- The gain thresholds for `p_gain` (LLD: 3% of baseline purchased energy, or Δscore ≥ 2) still need agreeing.
- R-03's spread is an assumption, to replace once ranges are sourced.
- Triple glazing stays feasible on SH3 (`R_GLAZING_DONE` blocks double and secondary only). It's unsourced, so it is excluded from cost ranking anyway.
