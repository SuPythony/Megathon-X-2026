"""Turn posterior samples into the shared ``Distribution`` summary.

Every producer (twin, Bayesian calibration, retrofit engine, value model, dashboard)
must build uncertain outputs with ``summarise`` so the UI and the LLM always see the
same fields computed the same way. Categorical outputs (score band, EPC band) use
``category_probabilities``.

Example::

    gas_saving = summarise(samples_kwh, "kWh/yr",
                           tails=[("saving is negative (gas use rises)", 0.0, "below")])
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Literal, Sequence

import numpy as np
from scipy.stats import gaussian_kde, skew

from .models import CredibleInterval, Distribution, Histogram, Percentiles, TailProbability

PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
SKEW_HEADLINE_MEDIAN = 0.5  # |skewness| above this -> headline is the median


def _mode(x: np.ndarray) -> float:
    """Peak of a Gaussian KDE on a fine grid; falls back to the median when the KDE is degenerate."""
    if x.size < 3 or np.ptp(x) == 0:
        return float(np.median(x))
    try:
        kde = gaussian_kde(x)
    except np.linalg.LinAlgError:
        return float(np.median(x))
    grid = np.linspace(x.min(), x.max(), 512)
    return float(grid[np.argmax(kde(grid))])


def summarise(
    samples: Iterable[float],
    unit: str,
    *,
    tails: Sequence[tuple[str, float, Literal["below", "above"]]] = (),
    worst_tail: Literal["low", "high"] = "low",
    bins: int = 30,
) -> Distribution:
    """Summarise posterior samples into a ``Distribution``.

    ``tails`` are (label, threshold, direction) triples, e.g.
    ("bill rises by more than £100/yr", -100.0, "below"); each becomes a tail probability.
    ``worst_tail`` says which side is the bad outcome: "low" for savings, "high" for costs.
    """
    x = np.asarray(list(samples), dtype=float)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("samples must be non-empty and finite; missing values are not zeros")

    q = np.percentile(x, PERCENTILES)
    pct = Percentiles(**{f"p{p}": float(v) for p, v in zip(PERCENTILES, q)})
    degenerate = np.ptp(x) == 0
    sk = 0.0 if degenerate or x.size < 3 else float(skew(x))

    k = max(1, int(np.ceil(0.05 * x.size)))
    xs = np.sort(x)
    worst = xs[:k] if worst_tail == "low" else xs[-k:]

    if degenerate:
        hist = Histogram(bin_edges=[float(x[0]), float(x[0])], counts=[int(x.size)])
    else:
        counts, edges = np.histogram(x, bins=bins)
        hist = Histogram(bin_edges=[float(e) for e in edges], counts=[int(c) for c in counts])

    return Distribution(
        unit=unit,
        n_samples=int(x.size),
        mean=float(x.mean()),
        median=pct.p50,
        mode=_mode(x),
        std=float(x.std(ddof=1)) if x.size > 1 else 0.0,
        skewness=sk,
        percentiles=pct,
        credible_intervals=[
            CredibleInterval(level=0.5, lo=pct.p25, hi=pct.p75),
            CredibleInterval(level=0.9, lo=pct.p5, hi=pct.p95),
        ],
        tail_probabilities=[
            TailProbability(
                label=label,
                threshold=float(t),
                direction=d,
                probability=float(np.mean(x < t) if d == "below" else np.mean(x > t)),
            )
            for label, t, d in tails
        ],
        worst_tail=worst_tail,
        worst_5pct_mean=float(worst.mean()),
        histogram=hist,
        headline="median" if abs(sk) > SKEW_HEADLINE_MEDIAN else "mean",
    )


def exact(value: float, unit: str, n_samples: int = 1) -> Distribution:
    """A value known exactly (std = 0), e.g. gas use after the boiler is removed."""
    return summarise([value] * max(1, n_samples), unit)


def category_probabilities(labels: Iterable[str], categories: Sequence[str]) -> tuple[str, dict[str, float]]:
    """(mode, P(each category)) for categorical samples such as score bands; probabilities sum to 1."""
    labs = list(labels)
    if not labs:
        raise ValueError("no samples")
    c = Counter(labs)
    probs = {cat: c.get(cat, 0) / len(labs) for cat in categories}
    mode = max(categories, key=lambda cat: probs[cat])
    return mode, probs
