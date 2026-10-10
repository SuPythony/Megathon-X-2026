"""Draw samples for uncertain score inputs.

* A ``Distribution`` (posterior summary from another part) is resampled through its quantile function
  (piecewise-linear through P1..P99), so the score sees the same spread the producer reported.
* A missing rating is sampled from the local-authority band mix (decision D-02).
* An unknown loft depth is sampled from an age-band prior (decision D-04); never 0 by default.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from contracts.models import Distribution

BANDS = "ABCDEFG"
SAP_RANGES = {"A": (92, 100), "B": (81, 91), "C": (69, 80), "D": (55, 68), "E": (39, 54), "F": (21, 38), "G": (1, 20)}
_Q = np.array([0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
PRIORS_PATH = Path(__file__).with_name("priors.json")


def from_distribution(d: Distribution, n: int, rng: np.random.Generator) -> np.ndarray:
    """Inverse-CDF resampling from the reported percentiles; tails extrapolated linearly (P1->P0, P99->P100)."""
    if d.std == 0:
        return np.full(n, d.mean)
    p = d.percentiles
    mid = [p.p1, p.p5, p.p10, p.p25, p.p50, p.p75, p.p90, p.p95, p.p99]
    lo = p.p1 - (p.p5 - p.p1) * 0.25  # 1% of mass below P1, slope of the P1..P5 segment
    hi = p.p99 + (p.p99 - p.p95) * 0.25
    return np.interp(rng.uniform(size=n), _Q, [lo, *mid, hi])


@lru_cache(maxsize=1)
def priors() -> dict:
    return json.loads(PRIORS_PATH.read_text(encoding="utf-8"))


def peer_band_probs(band_counts: Optional[Mapping[str, int]]) -> tuple[dict[str, float], str]:
    """Band probabilities from an EPC live-table row (A..G counts); falls back to the national mix."""
    if band_counts:
        tot = sum(band_counts.get(b, 0) for b in BANDS)
        if tot > 0:
            return {b: band_counts.get(b, 0) / tot for b in BANDS}, "local_authority"
    nat = priors()["national_band_mix"]
    return {b: nat["probs"][b] for b in BANDS}, nat["source"]


def sap_from_band_mix(probs: Mapping[str, float], n: int, rng: np.random.Generator) -> np.ndarray:
    bands = rng.choice(list(BANDS), size=n, p=[probs[b] for b in BANDS])
    lo = np.array([SAP_RANGES[b][0] for b in bands], dtype=float)
    hi = np.array([SAP_RANGES[b][1] for b in bands], dtype=float)
    return rng.uniform(lo, hi + 1.0)  # uniform over the integer range of the band


def loft_mm_prior(age_band: Optional[str], n: int, rng: np.random.Generator) -> np.ndarray:
    """Unknown loft depth: sample from the empirical depth distribution for the age band."""
    from .mapping import age_band_start
    pri = priors()["loft_mm_by_age"]
    start = age_band_start(age_band)
    key = "all"
    for k in pri:
        if k == "all":
            continue
        a, b = (int(x) for x in k.split("-"))
        if start is not None and a <= start <= b:
            key = k
            break
    depths = np.array([float(x) for x in pri[key]["probs"]])
    probs = np.array(list(pri[key]["probs"].values()), dtype=float)
    return rng.choice(depths, size=n, p=probs / probs.sum())
