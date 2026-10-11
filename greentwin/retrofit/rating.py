"""Post-retrofit rating estimator: ``contracts/sap_estimator.json`` -> ``RatingEstimate``.

1. The twin runs in STANDARD mode (SAP-style heating pattern and occupancy, never this household's behaviour)
   before and after the package, on every posterior sample.
2. Energy is priced with ``tariffs.json`` (unit rates + standing charge of every fuel still supplied). Documented
   deviation: SAP prices with its own fuel table. PV export is not credited (no sourced export tariff), so PV's
   rating gain is understated.
3. SAP_model = f(ECF), ECF = deflator x cost / (TFA + 45).
4. Anchor: SAP_after = SAP_recorded + (SAP_model_after - SAP_model_before); only the model's change is used, so its
   level bias cancels. No recorded SAP -> SAP_model_after, ``anchored_to_recorded_sap = False``.
5. Band probabilities from the thresholds. Always labelled "estimated rating, not an EPC".
Unsupported main fuel (no sourced price) -> ``None`` + reason, never a guess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from contracts.models import Property, RatingEstimate, RetrofitAction
from contracts.summarise import category_probabilities, summarise
from greentwin.interfaces import PosteriorSamples, TwinRun

from .runner import fuel_supported, run_packages, standing_charges, unit_rates

SPEC_PATH = Path(__file__).resolve().parents[2] / "contracts" / "sap_estimator.json"
BANDS = ["A", "B", "C", "D", "E", "F", "G"]
GAS_SUPPLY_KWH = 1.0  # below this the home has no gas supply, so no gas standing charge


@lru_cache(maxsize=1)
def spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def sap_from_cost(cost_gbp: np.ndarray, floor_area_m2: float, clamp: bool = True) -> np.ndarray:
    """SAP 2012 section 13. ``clamp=False`` for the anchored delta: clamping both ends first would erase the
    change for homes whose modelled SAP sits below 1 (R-11); the final SAP is clamped instead."""
    f = spec()["formula"]
    ecf = f["deflator"] * np.asarray(cost_gbp, dtype=float) / (floor_area_m2 + 45.0)
    sap = np.where(ecf >= 3.5, 117.0 - 121.0 * np.log10(np.maximum(ecf, 1e-9)), 100.0 - 13.95 * ecf)
    lo, hi = f["clamp"]
    return np.clip(sap, lo, hi) if clamp else sap


def band_of(sap: np.ndarray | float) -> np.ndarray:
    th = spec()["band_thresholds"]
    s = np.atleast_1d(np.asarray(sap, dtype=float))
    out = np.full(s.shape, "G", dtype=object)
    for b in reversed(BANDS):  # G..A: each higher threshold overwrites
        out[s >= th[b]] = b
    return out


def standard_cost(run: TwinRun) -> np.ndarray:
    r, sc = unit_rates(), standing_charges()
    e = np.asarray(run.electricity_import_kwh, dtype=float)
    g = np.asarray(run.gas_kwh, dtype=float)
    return e * r["electricity"] + g * r["gas"] + sc["electricity"] + np.where(g > GAS_SUPPLY_KWH, sc["gas"], 0.0)


@dataclass
class RatingResult:
    estimate: Optional[RatingEstimate]
    unavailable_reason: Optional[str] = None
    sap_samples: Optional[np.ndarray] = None
    warnings: list[str] = field(default_factory=list)


def rating_from_runs(prop: Property, std_before: TwinRun, std_after: TwinRun) -> RatingResult:
    if not fuel_supported(prop):
        return RatingResult(None, f"No sourced price for {prop.heating.main_fuel}: rating after unavailable")
    tfa = prop.floor_area_m2
    model_before = sap_from_cost(standard_cost(std_before), tfa, clamp=False)
    model_after = sap_from_cost(standard_cost(std_after), tfa, clamp=False)
    warnings = ["Standard-mode cost uses Ofgem Q4 2026 prices, not the SAP fuel price table",
                "PV export is not credited (no sourced export tariff)"]
    if prop.epc.sap_score is not None:
        sap = prop.epc.sap_score + (model_after - model_before)
        anchored = True
    else:
        sap = model_after
        anchored = False
        warnings.append("No recorded EPC: rating is the model's own level, not anchored")
    lo, hi = spec()["formula"]["clamp"]
    sap = np.clip(sap, lo, hi)
    band, probs = category_probabilities(list(band_of(sap)), BANDS)
    est = RatingEstimate(
        sap=summarise(sap, "SAP points", tails=[("band D or worse (SAP < 69)", 68.5, "below")]),
        band=band, band_probabilities=probs, anchored_to_recorded_sap=anchored)
    return RatingResult(est, sap_samples=sap, warnings=warnings)


def estimate_rating(prop: Property, samples: PosteriorSamples, actions: Sequence[RetrofitAction]) -> RatingResult:
    before, after = run_packages(samples, [[], list(actions)], mode="standard")
    return rating_from_runs(prop, before, after)
