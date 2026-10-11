"""ValueModel (Part 4 interface): indicative value range before vs after a retrofit package.

uplift_s = exp(beta_s[region x type] * dSAP_s / 10) - 1, pairing posterior samples of the rating effect with
samples of the SAP change from Part 2's RatingEstimate (anchored to the recorded SAP). Applied to the CURRENT value
estimate, never to a historical sale price. Always labelled 'indicative' with ValueImpact.wording.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from contracts.models import FeatureContribution, GreenTwin, Property, RatingEstimate, ValueImpact
from contracts.summarise import summarise
from greentwin.score.sampling import from_distribution

from .hedonic import SAP_SCALE

ROOT = Path(__file__).resolve().parents[2]
REGION_CODE = {  # contracts Location.region (ONS name) -> NSPL rgn code used in training
    "North East": "E12000001", "North West": "E12000002", "Yorkshire and The Humber": "E12000003",
    "East Midlands": "E12000004", "West Midlands": "E12000005", "East of England": "E12000006",
    "London": "E12000007", "South East": "E12000008", "South West": "E12000009", "Wales": "W99999999",
}


def ppd_type(prop: Property) -> str:
    """contracts property type + built form -> Land Registry type used in training."""
    if prop.property_type.value in ("flat", "maisonette"):
        return "flat"
    f = prop.built_form.value
    if f == "detached":
        return "detached"
    if f == "semi_detached":
        return "semi"
    return "terraced"


class HedonicValueModel:
    """Implements ``greentwin.interfaces.ValueModel``."""

    def __init__(self, version: str = "v1", n_samples: int = 2000, seed: int = 42, artifact_dir: Optional[str] = None):
        d = Path(artifact_dir) if artifact_dir else ROOT / "artifacts" / "value_model" / version
        self.version = version
        self.post = dict(np.load(d / "posterior.npz"))
        self.levels = json.loads((d / "encoding.json").read_text())
        eff = d / "rating_effects.json"
        self.effects = json.loads(eff.read_text()) if eff.exists() else None
        self.n, self.seed = n_samples, seed

    def _beta_bayes(self, prop: Property, rng, n: int) -> np.ndarray:
        """Bayesian hierarchical draws, per SAP point (log points)."""
        key = f"{REGION_CODE.get(prop.location.region, '')}|{ppd_type(prop)}"
        lv = self.levels["rt"]
        b = self.post["beta_rt"]
        col = lv.index(key) if key in lv else None
        draws = b[:, col] if col is not None else self.post["beta0"]  # unseen region x type -> pooled mean
        return rng.choice(draws, size=n, replace=True) / SAP_SCALE

    def _beta(self, prop: Property, rng) -> np.ndarray:
        """V-13: triangulated rating effect. Each sample comes from one design (equal weights): OLS with LA effects,
        within-neighbourhood cells, Bayesian hierarchical. The spread across designs is model uncertainty."""
        if not self.effects:
            return self._beta_bayes(prop, rng, self.n)
        designs = self.effects["designs"]
        names = list(designs)
        w = np.array([self.effects["weights"][k] for k in names])
        pick = rng.choice(len(names), size=self.n, p=w / w.sum())
        out = np.empty(self.n)
        for i, k in enumerate(names):
            idx = np.flatnonzero(pick == i)
            spec = designs[k]
            out[idx] = (self._beta_bayes(prop, rng, len(idx)) if spec["type"] == "posterior"
                        else rng.normal(spec["mean"], spec["sd"], len(idx)))
        return out

    def value_impact(self, prop: Property, rating_after: Optional[RatingEstimate]) -> ValueImpact:
        rng = np.random.default_rng(self.seed)
        cur = prop.current_value_estimate_gbp
        before = prop.epc.current_band
        if rating_after is None:
            return ValueImpact(current_value_gbp=cur, band_before=before, method="unavailable",
                               unavailable_reason="No post-retrofit rating estimate (sap_estimator) for this package, "
                                                  "so the value change cannot be estimated.")
        sap_before = prop.epc.sap_score
        if sap_before is None:
            return ValueImpact(current_value_gbp=cur, band_before=before, band_after=rating_after.band,
                               method="unavailable",
                               unavailable_reason="No recorded SAP rating, so the rating change is not anchored.")
        d_sap = from_distribution(rating_after.sap, self.n, rng) - sap_before
        beta = self._beta(prop, rng)
        uplift = np.exp(beta * d_sap) - 1          # beta is per SAP point (log points)
        per_pt = (np.exp(beta) - 1) * 100
        tails = [("no clear increase (uplift <= 0)", 0.0, "below")]
        out = dict(
            current_value_gbp=cur, band_before=before, band_after=rating_after.band,
            uplift_pct=summarise(uplift * 100, "%", tails=tails),
            rating_effect_pct_per_sap_point=summarise(per_pt, "% per SAP point"),
            method="bayesian_hedonic",
            price_drivers=[FeatureContribution(feature="SAP change (estimated, not an EPC)",
                                               value=round(float(np.median(d_sap)), 1),
                                               contribution=round(float(np.median(uplift) * 100), 2), unit="%")],
        )
        if cur is not None:
            out["value_after_gbp"] = summarise(cur * (1 + uplift), "GBP")
        else:
            out["unavailable_reason"] = "No current value estimate: uplift shown in % only."
        return ValueImpact(**out)

    def green_twins(self, prop: Property, k: int = 3) -> list[GreenTwin]:
        """V-10: greener comparable sales as ranges. Scored sales if available (git-ignored data), else the
        precomputed showcase file, else empty ('not enough comparable greener sales')."""
        scored = ROOT / "data" / "processed" / f"sales_scored_{self.version}.parquet"
        if scored.exists():
            import pandas as pd
            from .green_twins import find
            return find(prop, pd.read_parquet(scored).dropna(subset=["green_score"]), k=k)
        show = ROOT / "artifacts" / "value_model" / self.version / "green_twins_showcase.json"
        if show.exists():
            hit = json.loads(show.read_text()).get(prop.property_id)
            if hit:
                return [GreenTwin.model_validate(t) for t in hit["twins"]][:k]
        return []
