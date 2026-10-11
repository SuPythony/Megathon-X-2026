# Learning initial distributions from application submissions

The physics workflow can now use previous application answers to inform both missing-input distributions and initial physical-constant priors. It still performs **no Bayesian meter calibration**.

## What gets used

The application-facing function is `prepare_application_inputs(request, n_samples, history=...)` in `greentwin/twin/application_history.py`. It reads a history snapshot, constructs initial distributions, generates missing inputs, then records the current baseline's supplied facts for future households. It calls no JAX simulator; the existing deterministic ensemble stage consumes its output unchanged.

Recorded occupants, presence, thermostat, heating periods, cooking fuel, flat position, PV/battery ownership and capacities inform missing-input priors. Explicit recorded U-values, ACH, heating/DHW efficiency, retained-heat fraction and device specifications can inform constant priors. Explicit baseload or useful-hot-water demand can inform those input distributions. Whole-house electricity/gas bills are not misinterpreted as baseload, U-values or efficiency: identifying those constants from bills requires the deferred calibration phase.

Matching uses property type and district, then region, then wider application history if local answers are unavailable. Household/demand evidence prefers homes within 30% of the new floor area; daily energy evidence additionally prefers the supplied occupant count where matching answers exist. Fabric evidence can additionally match the age band. Equipment efficiency evidence must match the heating technology; hot-water efficiency also matches its source. Raw customer identifiers are hashed; the population store retains only allowed household facts and district-level location, excluding photos, addresses and EPC certificate identifiers. This is a lightweight population input store, not a replacement for the app's customer database.

**Unmeasured constants cannot be discovered from nothing.** If nobody supplied or measured a particular constant, its provisional physical prior remains. Other supplied household descriptors still set the initial physical model. This phase does not infer hidden fabric performance from meter readings or claim that repeated simulator defaults are observed evidence.

## How distributions update

Each distinct household contributes its latest baseline revision at the requested cutoff. Its observations are weighted by `exp(-age_in_years/10)`. Effective evidence weight N is their total recency weight; numeric pooling weight is `N/(N+20)`.

Numeric means blend the existing fallback mean with the matching recorded-answer mean. Variance retains fallback uncertainty, population spread, a heterogeneity floor and disagreement between means. The resulting numeric marginal is a bounded normal with the original physical domain. Categorical probabilities blend recency-weighted observed counts with twenty pseudo-observations from the existing fallback probabilities. Small histories therefore influence the result without becoming unjustifiably certain.

The current household and, when supplied, its customer group are excluded from population learning. Its latest answers are already exact simulator inputs. Future records and validation/test households are excluded. Preview/retrofit requests do not update the population store. Generated missing-input samples, simulation results and prior means are never persisted as observations. Explicit showcase/synthetic properties are rejected by the history writer.

Every informed prior includes the history path, SHA-256 snapshot identity, cutoff, eligible/excluded household counts and per-variable cohort/sample/pooling details in `PriorBundle.application_history`. Prepared artifacts retain that frozen prior and its samples even if later submissions change the population store. Standalone `build_priors(request)` remains independent of mutable application state; pass `history=ApplicationHistory(...)` to opt into a specific snapshot.

## CLI: automatic use of new entries

The existing `run_uncertain_twin.py prepare/run --request ...` now uses the worktree-local history by default:

```sh
.venv/bin/python scripts/run_uncertain_twin.py prepare --request household.json --samples 128 --output artifacts/new-household-inputs
.venv/bin/python scripts/run_uncertain_twin.py simulate --prepared artifacts/new-household-inputs --output artifacts/new-household-result
```

The default history is `data/processed/application_history.sqlite`. After successful input generation, a baseline request's supplied facts are recorded for the next household. The `simulate --prepared` command does not read or modify history: it replays the frozen prepared inputs.

```sh
.venv/bin/python scripts/run_uncertain_twin.py prepare --request household.json --changes retrofit.json --output artifacts/retrofit-inputs
.venv/bin/python scripts/run_uncertain_twin.py prepare --request hypothetical-household.json --what-if --output artifacts/preview-inputs
```

Both use existing population evidence while keeping proposed changes out of history. Use `--no-history` for an isolated experiment. `--demo SH1/SH2/SH3` also stays isolated. `--history` selects a history file; all writes must remain inside this worktree. Reads do not create a missing database.

## Importing earlier application entries

Previous application exports must contain the submitted facts, not generated simulator outputs. The supported interchange is one `HouseholdRequest`, a list of requests, or a wrapper preserving timestamps and request kind:

```json
{
  "submissions": [
    {
      "submitted_at": "2026-10-01T12:00:00+00:00",
      "kind": "baseline",
      "request": {
        "property_id": "HOME-001",
        "user_inputs": {
          "postcode_district": "BS16",
          "floor_area_m2": 95,
          "property_type": "house",
          "heating_system": "gas_boiler_combi",
          "occupants": 3,
          "thermostat_c": 20
        }
      }
    }
  ]
}
```

```sh
.venv/bin/python scripts/application_history.py import --file previous-submissions.json
.venv/bin/python scripts/application_history.py inspect
```

Keep the original timestamp to preserve temporal eligibility. Without one, import time is used; such a record cannot be used retrospectively before that date. `physics_package` may carry a Satyarth handoff, but only numeric parameters explicitly sourced as user/measured/surveyed/recorded EPC values are eligible for historical learning. Generated package defaults and suggested priors are excluded. A full `Property` supplies explicit recorded building facts; properties marked showcase/synthetic are not admitted as observed application entries.

The accessible worktree contains no connected application database or browser submission export. It contains showcase records and a cached Satyarth example, which are not imported as real observations. The actual application's database/export location is required to connect older submissions. Its app handler can call the function above after converting the existing intake and property into `HouseholdRequest`; no changes were made to Satyarth's branch or the main checkout.

## Validation

Ten new tests cover numeric and categorical prior updates, exact-current-input preservation, use of a newly entered home by the next home, one vote per household revision, temporal and held-out exclusions, incompatible heating technology, known equipment absence, what-if exclusion, missing-file reads, anonymous identifiers and snapshot changes. The history-informed synthetic samples also run through the JAX simulator and conservation validation. These are controlled fixtures demonstrating method behaviour, not measurements of actual customer accuracy.
