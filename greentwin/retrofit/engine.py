"""``RetrofitEngine``: Part 2's implementation of ``greentwin.interfaces.EnergyRetrofitEngine``.

    engine = RetrofitEngine(usage={"SH2": usage_rows}, scorer=GreenHomeScorer(n_samples=1000))
    base   = engine.baseline(prop, samples)                       # metered intensity (brief section 2)
    steps  = engine.ladder(prop, samples, feasibility, "bills")   # RetrofitEffect per step, marginal, per fuel
    after  = engine.package(prop, samples, actions)               # ONE twin run on the whole package
    pot    = engine.improvement_potential(prop, samples, feasibility)
    rating = engine.estimate_rating(prop, samples, actions)       # "estimated rating, not an EPC"
    front  = engine.pareto(prop, samples, feasibility, budget_gbp=10_000)
    engine.warnings[prop.property_id]                             # everything that was assumed, missing or dropped

Twin runs go through ``runner.run_packages`` (batched); meter readings, when given, anchor each fuel's level (R-07).
The stand-in physics reports engine ``mock`` in ``provenance()``, so its numbers can't reach the demo.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from contracts.models import (
    AreaEPCStats, EnergyBreakdown, EnergyPrior, EnergyUsageMonth, FeasibilityReport, ImprovementPotential,
    ParetoPoint, Property, Provenance, RatingEstimate, RetrofitAction, RetrofitEffect,
)
from greentwin.interfaces import PosteriorSamples

from . import catalogue as cat
from .ladder import Goal, build_ladder
from .potential import improvement_potential
from .rating import estimate_rating
from .runner import Meter, anchored, breakdown, engine_name, meter_from_usage, run_packages

MODEL_VERSION = "retrofit/0.1"


class RetrofitEngine:
    """Implements ``greentwin.interfaces.EnergyRetrofitEngine``."""

    def __init__(self, usage: Optional[Mapping[str, Sequence[EnergyUsageMonth]]] = None, scorer=None,
                 peers: Optional[Mapping[str, AreaEPCStats]] = None, seed: int = 42):
        self.usage = dict(usage or {})
        self.scorer = scorer
        self.peers = dict(peers or {})
        self.seed = seed
        self.warnings: dict[str, list[str]] = {}
        self._meters: dict[tuple[str, int], Optional[Meter]] = {}

    # ---------------------------------------------------------------- helpers
    def _warn(self, prop: Property, items: Sequence[str]) -> None:
        w = self.warnings.setdefault(prop.property_id, [])
        w += [x for x in items if x not in w]

    def meter(self, prop: Property, n: int) -> Optional[Meter]:
        key = (prop.property_id, n)
        if key not in self._meters:
            rows = self.usage.get(prop.property_id)
            m = meter_from_usage(prop, rows, n, self.seed) if rows else None
            self._meters[key] = m
            if m is None:
                self._warn(prop, ["No usable meter readings: energy is the twin's prior, not anchored to bills"])
            else:
                self._warn(prop, m.warnings)
        return self._meters[key]

    def _peer(self, prop: Property) -> Optional[AreaEPCStats]:
        return self.peers.get(prop.location.local_authority_code or "")

    @staticmethod
    def _typical(weather: str) -> None:
        if weather != "typical":
            raise NotImplementedError(f"weather scenario {weather!r} needs Part 0's load_weather; only 'typical' runs")

    def provenance(self, n_samples: int, data_version: str = "synthetic-weather") -> Provenance:
        name = engine_name()
        return Provenance(engine="physics_l1" if name == "physics_l1" or name == "physics" else "mock",
                          model_version=f"{name}/{MODEL_VERSION}; catalogue {cat.version()}",
                          data_version=data_version, seed=self.seed, n_samples=n_samples)

    # ---------------------------------------------------------------- Protocol
    def energy_prior(self, prop: Property) -> EnergyPrior:
        from greentwin.models.energy_prior import energy_prior

        return energy_prior(prop)

    def baseline(self, prop: Property, samples: PosteriorSamples, weather: str = "typical") -> EnergyBreakdown:
        self._typical(weather)
        m = self.meter(prop, samples.n_samples)
        run = run_packages(samples, [[]])[0]
        return breakdown(anchored(run, run, m), prop.floor_area_m2, m is not None)

    def package(self, prop: Property, samples: PosteriorSamples, actions: Sequence[RetrofitAction]) -> EnergyBreakdown:
        m = self.meter(prop, samples.n_samples)
        base, after = run_packages(samples, [[], list(actions)])
        return breakdown(anchored(after, base, m), prop.floor_area_m2, m is not None)

    def ladder(self, prop: Property, samples: PosteriorSamples, feasibility: FeasibilityReport,
               goal: Goal = "bills") -> list[RetrofitEffect]:
        res = build_ladder(prop, samples, feasibility, goal, self.meter(prop, samples.n_samples), self.scorer,
                           self._peer(prop), self.seed)
        self._warn(prop, res.warnings)
        return res.effects

    def improvement_potential(self, prop: Property, samples: PosteriorSamples,
                              feasibility: FeasibilityReport) -> ImprovementPotential:
        return improvement_potential(prop, samples, feasibility, self.meter(prop, samples.n_samples), seed=self.seed)

    def estimate_rating(self, prop: Property, samples: PosteriorSamples,
                        actions: Sequence[RetrofitAction]) -> Optional[RatingEstimate]:
        res = estimate_rating(prop, samples, actions)
        self._warn(prop, res.warnings + ([res.unavailable_reason] if res.unavailable_reason else []))
        return res.estimate

    def pareto(self, prop: Property, samples: PosteriorSamples, feasibility: FeasibilityReport,
               budget_gbp: Optional[float] = None, goal: Goal = "bills") -> list[ParetoPoint]:
        from greentwin.optimise.pareto import pareto

        res = pareto(prop, samples, feasibility, self.meter(prop, samples.n_samples), self.scorer, self._peer(prop),
                     budget_gbp, goal, self.seed)
        self._warn(prop, res.warnings)
        return res.points
