"""Retrofit catalogue: load + validate `contracts/retrofit_catalogue.json`, ordering, cost and grant samples.

Savings never live here: they come from rerunning the twin. This module answers "what is this retrofit, where
does it go in the sequence, what does it cost (as samples), which grants count, and why is a cost missing".

Cost rules (AGENTS.md compliance fix 2, decision log `docs/retrofit/DECISIONS.md`):
- `cost_status: unsourced` -> no cost at all (`None` + reason); the item is excluded from cost ranking.
- low / central / high -> two-piece lognormal: median = central, P5 = low, P95 = high (R-02).
- central only -> lognormal around central with an assumed spread, flagged in `warnings` (R-03).
- `per_kwp` costs scale with the action's size, else the catalogue's own size edit (e.g. PV 3.5 kWp).
- Only confirmed grants are subtracted (R-04); net cost never goes below 0.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from contracts.models import Distribution, RetrofitAction, RetrofitItem
from contracts.summarise import summarise

ROOT = Path(__file__).resolve().parents[2]
CATALOGUE_PATH = ROOT / "contracts" / "retrofit_catalogue.json"
RULES_PATH = ROOT / "contracts" / "feasibility_rules.json"

Z95 = 1.6448536269514722  # standard normal P95: low/high are read as P5/P95
UNRANGED_LOG_SD = 0.25    # R-03 assumption: central-only costs get P5-P95 of about x0.66 to x1.51

# R-04: grants subtracted from net cost. Only confirmed schemes carry an amount; the rest are listed with a reason.
GRANTS: dict[str, dict] = {
    "BUS_7500": {"amount_gbp": 7500.0, "confirmed": True,
                 "note": "Boiler Upgrade Scheme, England & Wales, MCS installer (Energy Saving Trust)."},
    "BUS_9000_off_gas": {"amount_gbp": 9000.0, "confirmed": False,
                         "note": "Off-gas-grid rate; dates to confirm on gov.uk, not used in net cost yet."},
    "ECO4": {"amount_gbp": None, "confirmed": False, "note": "Eligibility-based; status to confirm, not used in net cost."},
    "GBIS": {"amount_gbp": None, "confirmed": False, "note": "Eligibility-based; status to confirm, not used in net cost."},
}


@lru_cache(maxsize=1)
def _raw() -> dict:
    return json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def items() -> dict[str, RetrofitItem]:
    """Every catalogue item, validated against the contract, keyed by id. Cross-references are checked."""
    out = {d["id"]: RetrofitItem.model_validate(d) for d in _raw()["items"]}
    for it in out.values():
        for ref in it.depends_on + it.excludes:
            if ref not in out:
                raise ValueError(f"{it.id}: refers to unknown retrofit {ref!r}")
        for g in it.grants:
            if g not in GRANTS:
                raise ValueError(f"{it.id}: unknown grant {g!r}")
    return out


def get(retrofit_id: str) -> RetrofitItem:
    try:
        return items()[retrofit_id]
    except KeyError:
        raise KeyError(f"unknown retrofit {retrofit_id!r}") from None


def version() -> str:
    return str(_raw()["version"])


# --------------------------------------------------------------------------- ordering
@lru_cache(maxsize=1)
def ordering_groups() -> list[list[str]]:
    """Fabric -> glazing -> controls -> heating -> PV -> battery -> tariff (feasibility_rules.json)."""
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))["ordering"]


def sequence_group(retrofit_id: str) -> int:
    """Index of the ordering group; items missing from the ordering go last."""
    for i, group in enumerate(ordering_groups()):
        if retrofit_id in group:
            return i
    return len(ordering_groups())


def sequence_position(retrofit_id: str) -> tuple[int, int]:
    """(group, position within the group) for a stable fabric-first sort."""
    g = sequence_group(retrofit_id)
    groups = ordering_groups()
    return g, (groups[g].index(retrofit_id) if g < len(groups) else 0)


def conflicts(ids: Sequence[str]) -> list[tuple[str, str]]:
    """Pairs in `ids` that the catalogue marks as mutually exclusive."""
    s = set(ids)
    return sorted({tuple(sorted((a, b))) for a in s for b in get(a).excludes if b in s})


def valid_package(ids: Sequence[str]) -> bool:
    """No mutually exclusive pair. `depends_on` is an ordering hint, not a requirement, except battery needs PV."""
    return not conflicts(ids)


# --------------------------------------------------------------------------- cost
@dataclass
class CostSamples:
    """Installed cost of one retrofit as samples (GBP), or None with a reason when it has no source."""

    retrofit_id: str
    cost_status: str
    gross_gbp: Optional[np.ndarray]
    grant_gbp: float = 0.0
    grants_applied: list[str] = field(default_factory=list)
    grants_not_applied: list[str] = field(default_factory=list)
    unavailable_reason: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    source: str = ""

    @property
    def available(self) -> bool:
        return self.gross_gbp is not None

    @property
    def net_gbp(self) -> Optional[np.ndarray]:
        return None if self.gross_gbp is None else np.maximum(self.gross_gbp - self.grant_gbp, 0.0)

    def distribution(self, net: bool = True) -> Optional[Distribution]:
        x = self.net_gbp if net else self.gross_gbp
        return None if x is None else summarise(x, "GBP", worst_tail="high")


def _size(item: RetrofitItem, action: Optional[RetrofitAction]) -> Optional[float]:
    """Size that a per-unit cost scales with: the action's size, else the catalogue's numeric size edit."""
    if action is not None and action.size is not None:
        return float(action.size)
    for key in ("pv_kwp", "battery_kwh"):
        v = item.twin_edits.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _two_piece_lognormal(low: float, central: float, high: float, z: np.ndarray) -> np.ndarray:
    """Median `central`, P5 `low`, P95 `high`; asymmetric, always positive."""
    mu = np.log(central)
    sd_lo = (mu - np.log(low)) / Z95
    sd_hi = (np.log(high) - mu) / Z95
    return np.exp(mu + np.where(z < 0, z * sd_lo, z * sd_hi))


def cost_samples(retrofit_id: str, n: int, seed: int = 42, action: Optional[RetrofitAction] = None,
                 off_gas_grid: bool = False) -> CostSamples:
    """Sampled installed cost (and grants) for one retrofit. Seeded per item so packages are reproducible."""
    item = get(retrofit_id)
    base = dict(retrofit_id=retrofit_id, cost_status=item.cost_status, source=item.source)
    if item.cost_status == "unsourced" or item.cost_gbp_central is None:
        return CostSamples(gross_gbp=None, **base,
                           unavailable_reason=f"No sourced cost for {item.name.lower()} yet ({item.source}); "
                                              "excluded from cost ranking")
    warnings: list[str] = []
    if item.cost_status == "secondary_source":
        warnings.append(f"{item.name}: cost from a secondary source, to confirm against the primary publication")

    lo, c, hi = item.cost_gbp_low, item.cost_gbp_central, item.cost_gbp_high
    if c == 0 and not lo and not hi:
        gross = np.zeros(n)
    else:
        z = np.random.default_rng([seed, zlib.crc32(retrofit_id.encode())]).standard_normal(n)
        if lo is not None and hi is not None and lo < c < hi:
            gross = _two_piece_lognormal(lo, c, hi, z)
        else:
            gross = c * np.exp(UNRANGED_LOG_SD * z)
            warnings.append(f"{item.name}: source gives one figure (GBP {c:,.0f}); spread is an assumption "
                            f"(log-sd {UNRANGED_LOG_SD})")

    if item.cost_basis == "per_kwp":
        size = _size(item, action)
        if size is None:
            return CostSamples(gross_gbp=None, **base, unavailable_reason=f"{item.name}: per-kWp cost but no size")
        gross = gross * size
    elif item.cost_basis != "per_home":
        return CostSamples(gross_gbp=None, **base,
                           unavailable_reason=f"{item.name}: cost basis {item.cost_basis!r} not supported yet")

    applied, not_applied, grant = [], [], 0.0
    for g in item.grants:
        info = GRANTS[g]
        if g == "BUS_9000_off_gas" and not off_gas_grid:
            continue  # not relevant to this home at all
        if info["confirmed"] and info["amount_gbp"]:
            if info["amount_gbp"] > grant:  # BUS rates are alternatives, never stacked
                grant = info["amount_gbp"]
                applied = [g]
        else:
            not_applied.append(f"{g}: {info['note']}")
    return CostSamples(gross_gbp=gross, grant_gbp=grant, grants_applied=applied, grants_not_applied=not_applied,
                       warnings=warnings, **base)


def package_cost(actions: Sequence[RetrofitAction], n: int, seed: int = 42,
                 off_gas_grid: bool = False) -> tuple[Optional[np.ndarray], list[CostSamples]]:
    """Net package cost per sample = sum of item net costs (costs add; savings never do).
    None when any item has no sourced cost: a partial total would understate the package."""
    parts = [cost_samples(a.retrofit_id, n, seed, a, off_gas_grid) for a in actions]
    if any(not p.available for p in parts):
        return None, parts
    return np.sum([p.net_gbp for p in parts] + [np.zeros(n)], axis=0), parts
