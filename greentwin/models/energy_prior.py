"""M04 energy-intensity prior: EBM on NEED 2025 -> per-fuel kWh/m2 Distribution + glass-box contributions.

The prior for homes WITHOUT meter data (and the prior mean for calibration). It never produces customer
savings: the physics twin does. See docs/M04_ENERGY_PRIOR.md for data, method and metrics.

- One EBM per fuel, trained on raw kWh/m2 so contributions are additive in kWh/m2 (they sum to the prediction).
- Monotone where physics requires: better EPC band / loft / cavity insulation never raise gas; PV never raises
  electricity import. Monotone features are kept out of interaction terms so the guarantee holds overall.
- Uncertainty: split-conformal relative residuals (y - yhat) / yhat from the 'val' split, per stratum
  (fuel x property type), pushed through the prediction to give a full predictive Distribution.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from contracts.models import EnergyPrior, FeatureContribution, Property, Provenance
from contracts.summarise import exact, summarise
from greentwin.data.need import DATA_VERSION, FLOOR_BAND_AREA_M2, FLOOR_BAND_EDGES_M2, REGION_CODES

MODEL_ID = "m04_energy_prior"
MODEL_VERSION = "v2"
DEFAULT_ARTIFACT = Path("artifacts") / MODEL_ID / MODEL_VERSION / "model.pkl"
UNIT = "kWh/m2/yr"
SEED = 42
N_SAMPLES = 1000

FUELS = ("electricity", "gas")
NOMINAL = ["prop_type", "region", "main_heat_fuel"]
FEATURES = {
    "electricity": ["prop_type", "age_band", "floor_band", "region", "epc_band", "imd_quintile",
                    "conservatory", "li_flag", "cwi_flag", "pv_flag", "main_heat_fuel"],
    # Gas model is fitted on gas-heated homes only, so main_heat_fuel is constant and dropped.
    "gas": ["prop_type", "age_band", "floor_band", "region", "epc_band", "imd_quintile",
            "conservatory", "li_flag", "cwi_flag", "pv_flag"],
}
# +1 = prediction may only rise with the feature, -1 = may only fall. EPC ordinal: 1 = A/B ... 5 = F/G.
MONOTONE = {
    "electricity": {"pv_flag": -1},
    "gas": {"epc_band": +1, "li_flag": -1, "cwi_flag": -1},
}
# Conformal strata: residual spread differs a lot between flats/detached and gas/electric heating.
STRATA = {"electricity": ["main_heat_fuel", "prop_type"], "gas": ["prop_type"]}
FEATURE_LABELS = {
    "prop_type": "property type", "age_band": "age band", "floor_band": "floor-area band", "region": "region",
    "epc_band": "EPC band", "imd_quintile": "deprivation quintile (IMD)", "conservatory": "conservatory",
    "li_flag": "loft insulation installed", "cwi_flag": "cavity wall insulation installed",
    "pv_flag": "solar PV", "main_heat_fuel": "main heating fuel",
}
AGE_BAND_LABELS = {1: "before 1930", 2: "1930-1972", 3: "1973-1999", 4: "2000 or later"}
FLOOR_BAND_LABELS = {1: "<=50 m2", 2: "51-100 m2", 3: "101-150 m2", 4: "151-200 m2", 5: ">200 m2"}
EPC_LABELS = {1: "A/B", 2: "C", 3: "D", 4: "E", 5: "F/G"}


# --------------------------------------------------------------------------- features
def _v(x) -> str:
    """Enum or plain string (contracts may store enum values as str)."""
    return getattr(x, "value", x)


def interaction_pairs(fuel: str) -> list[tuple[str, str]]:
    """Candidate pairwise terms: every pair of non-monotone features (keeps monotonicity global)."""
    free = [f for f in FEATURES[fuel] if f not in MONOTONE[fuel]]
    return list(combinations(free, 2))


def prepare_frame(df: pd.DataFrame, fuel: str) -> pd.DataFrame:
    """Select + type the model columns. Nominals as strings, ordinals as float (NaN = unknown, never 0)."""
    X = df[FEATURES[fuel]].copy()
    for c in X.columns:
        X[c] = X[c].astype("string").fillna("unknown").astype(str) if c in NOMINAL else X[c].astype(float)
    return X


def _age_band(prop: Property) -> float:
    year = prop.year_built
    if year is None:
        nums = [int(n) for n in re.findall(r"\d{4}", prop.construction_age_band)]
        if not nums:
            return np.nan
        band = prop.construction_age_band.lower()
        year = nums[0] - 10 if "before" in band else nums[0] + 3 if "onward" in band else int(np.mean(nums))
    return float(1 if year < 1930 else 2 if year <= 1972 else 3 if year <= 1999 else 4)


def _prop_type(prop: Property) -> str:
    if _v(prop.property_type) in ("flat", "maisonette"):
        return "Flat"
    if _v(prop.property_type) == "bungalow":
        return "Bungalow"
    return {"detached": "Detached", "semi_detached": "Semi detached", "mid_terrace": "Mid terrace",
            "enclosed_mid_terrace": "Mid terrace", "end_terrace": "End terrace"}.get(_v(prop.built_form), "unknown")


def _region(prop: Property) -> Optional[str]:
    name = prop.location.region.strip().lower()
    aliases = {"east": "East of England", "yorkshire and the humber": "Yorkshire and The Humber",
               "yorkshire & the humber": "Yorkshire and The Humber"}
    for canonical in REGION_CODES.values():
        if canonical.lower() == name:
            return canonical
    return aliases.get(name)


def property_to_features(prop: Property, imd_quintile: Optional[int] = None) -> pd.DataFrame:
    """Map a contract Property onto NEED feature space (one row). Unknown stays NaN / 'unknown'.

    NEED's LI_FLAG / CWI_FLAG mean "installed under a government scheme since 2000"; we proxy them from the
    current fabric (>= 200 mm loft insulation; filled cavity). IMD needs an LSOA lookup (data layer); pass it in.
    """
    area = prop.floor_area_m2
    floor_band = 1 + sum(area > e for e in FLOOR_BAND_EDGES_M2)
    epc = _v(prop.epc.current_band) if prop.epc.current_band is not None else None
    epc_ord = {"A": 1, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 5}.get(epc, np.nan) if epc else np.nan
    row = {
        "prop_type": _prop_type(prop),
        "age_band": _age_band(prop),
        "floor_band": float(floor_band),
        "region": _region(prop) or "unknown",
        "epc_band": epc_ord,
        "imd_quintile": float(imd_quintile) if imd_quintile is not None else np.nan,
        "conservatory": np.nan,  # not in the Property contract
        "li_flag": float(_v(prop.roof.type) == "pitched_loft" and (prop.roof.loft_insulation_mm or 0) >= 200),
        "cwi_flag": float(_v(prop.walls.type) == "cavity" and _v(prop.walls.insulation) == "cavity_filled"),
        "pv_flag": float(prop.renewables.pv_kwp > 0),
        "main_heat_fuel": "gas" if prop.heating.main_fuel == "mains_gas" else "not_gas",
    }
    return pd.DataFrame([row])


# --------------------------------------------------------------------------- conformal
QUANTILE_GRID = np.linspace(0.0005, 0.9995, N_SAMPLES)


def _stratum_key(row: pd.Series | dict, fuel: str) -> str:
    return "|".join(str(row[c]) for c in STRATA[fuel])


def fit_residual_quantiles(df: pd.DataFrame, y: np.ndarray, yhat: np.ndarray, fuel: str,
                           min_n: int = 200) -> dict[str, np.ndarray]:
    """Relative residual quantiles per stratum (+ '__all__' fallback) from the calibration split."""
    r = (y - yhat) / yhat
    keys = df[STRATA[fuel]].astype(str).agg("|".join, axis=1).to_numpy()
    out = {"__all__": np.quantile(r, QUANTILE_GRID)}
    for k in np.unique(keys):
        m = keys == k
        if m.sum() >= min_n:
            out[k] = np.quantile(r[m], QUANTILE_GRID)
    return out


def interval_from_quantiles(yhat: np.ndarray, keys: np.ndarray, rq: dict[str, np.ndarray],
                            level: float = 0.90) -> tuple[np.ndarray, np.ndarray]:
    lo_i = int(np.searchsorted(QUANTILE_GRID, (1 - level) / 2))
    hi_i = int(np.searchsorted(QUANTILE_GRID, 1 - (1 - level) / 2)) - 1
    q = np.stack([rq.get(k, rq["__all__"]) for k in keys])
    return yhat * (1 + q[:, lo_i]), yhat * (1 + q[:, hi_i])


# --------------------------------------------------------------------------- model
@dataclass
class EnergyPriorModel:
    """Loaded M04 artefact. ``energy_prior(prop)`` is what EnergyRetrofitEngine.energy_prior delegates to."""

    models: dict  # fuel -> fitted ExplainableBoostingRegressor
    residual_quantiles: dict  # fuel -> {stratum: quantiles}
    metadata: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ARTIFACT) -> "EnergyPriorModel":
        # joblib unpickles: only load artefacts we trained ourselves (scripts/train_energy_prior.py), never downloads.
        blob = joblib.load(path)
        return cls(models=blob["models"], residual_quantiles=blob["residual_quantiles"],
                   metadata=blob.get("metadata", {}))

    def save(self, path: str | Path = DEFAULT_ARTIFACT) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models, "residual_quantiles": self.residual_quantiles,
                     "metadata": self.metadata}, path, compress=3)

    def predict_point(self, df: pd.DataFrame, fuel: str) -> np.ndarray:
        """Point prediction (kWh/m2/yr) for a NEED-feature frame; floored at 1 so relative residuals apply."""
        return np.maximum(self.models[fuel].predict(prepare_frame(df, fuel)), 1.0)

    def samples(self, features: pd.DataFrame, fuel: str) -> np.ndarray:
        """Predictive samples for ONE home: yhat * (1 + conformal relative residual quantiles)."""
        yhat = float(self.predict_point(features, fuel)[0])
        rq = self.residual_quantiles[fuel]
        q = rq.get(_stratum_key(features.iloc[0], fuel), rq["__all__"])
        return np.maximum(yhat * (1 + q), 0.0)

    def contributions(self, features: pd.DataFrame, fuel: str) -> list[FeatureContribution]:
        """EBM additive terms for one home in kWh/m2/yr, incl. the intercept; they sum to the point prediction."""
        ebm = self.models[fuel]
        X = prepare_frame(features, fuel)
        scores = ebm.eval_terms(X)[0]
        row = X.iloc[0]
        out = [FeatureContribution(feature="average home (intercept)", value=None,
                                   contribution=round(float(ebm.intercept_), 2), unit=UNIT)]
        for name, s in zip(ebm.term_names_, scores):
            parts = name.split(" & ")
            label = " x ".join(FEATURE_LABELS.get(p, p) for p in parts)
            value = ", ".join(_display_value(p, row[p]) for p in parts)
            out.append(FeatureContribution(feature=label, value=value, contribution=round(float(s), 2), unit=UNIT))
        return out[:1] + sorted(out[1:], key=lambda c: -abs(c.contribution))  # intercept first, then by size

    def area_factor(self, prop: Property, fuel: str) -> float:
        """Within-band floor-area adjustment of intensity: (A / A_band)^(beta - 1).

        NEED only has floor-area bands, so the EBM predicts at the band's representative area. beta is the
        elasticity of annual kWh to floor area, estimated from NEED (scripts/bayes_energy_prior.py).
        Without it (beta unknown) the factor is 1 and intensity is taken as constant within the band.
        """
        beta = self.metadata.get("area_elasticity", {}).get(fuel)
        if beta is None:
            return 1.0
        band = 1 + sum(prop.floor_area_m2 > e for e in FLOOR_BAND_EDGES_M2)
        return float((prop.floor_area_m2 / FLOOR_BAND_AREA_M2[band]) ** (beta - 1.0))

    def home_samples(self, prop: Property, features: pd.DataFrame, fuel: str) -> np.ndarray:
        return self.samples(features, fuel) * self.area_factor(prop, fuel)

    def _summary(self, prop: Property, features: pd.DataFrame, fuel: str):
        row = features.iloc[0]
        tails = []
        peer = self.metadata.get("peer_median_kwh_m2", {}).get(fuel, {})
        key = f"{row['prop_type']}|{int(row['age_band']) if row['age_band'] == row['age_band'] else 'nan'}"
        if key in peer:
            tails.append((f"above the NEED median for {row['prop_type'].lower()} homes of this age",
                          float(peer[key]), "above"))
        dist = summarise(self.home_samples(prop, features, fuel), UNIT, tails=tails, worst_tail="high")
        contrib = self.contributions(features, fuel)
        f = self.area_factor(prop, fuel)
        if f != 1.0:  # keep the waterfall exact: the area adjustment is its own term
            point = sum(c.contribution for c in contrib)
            contrib.append(FeatureContribution(feature="floor area within band", value=f"{prop.floor_area_m2:g} m2",
                                               contribution=round(point * (f - 1.0), 2), unit=UNIT))
        return dist, contrib

    def energy_prior(self, prop: Property, imd_quintile: Optional[int] = None) -> EnergyPrior:
        features = property_to_features(prop, imd_quintile)
        row = features.iloc[0]
        elec, contrib_e = self._summary(prop, features, "electricity")
        if row["main_heat_fuel"] == "gas":
            gas, contrib_g = self._summary(prop, features, "gas")
        else:
            # No mains gas supply: gas use is structurally 0 (not a missing value).
            gas, contrib_g = exact(0.0, UNIT), []
        ood = (not 25 <= prop.floor_area_m2 <= 400) or row["region"] == "unknown" or row["prop_type"] == "unknown"
        return EnergyPrior(
            property_id=prop.property_id,
            electricity_kwh_m2=elec,
            gas_kwh_m2=gas,
            contributions_electricity=contrib_e,
            contributions_gas=contrib_g,
            out_of_distribution=bool(ood),
            provenance=Provenance(engine="ebm", model_version=f"{MODEL_ID}-{MODEL_VERSION}",
                                  data_version=DATA_VERSION, seed=SEED, n_samples=N_SAMPLES),
        )


def _display_value(feature: str, v) -> str:
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "unknown"
    v = int(v)
    if feature == "age_band":
        return AGE_BAND_LABELS.get(v, str(v))
    if feature == "floor_band":
        return FLOOR_BAND_LABELS.get(v, str(v))
    if feature == "epc_band":
        return EPC_LABELS.get(v, str(v))
    if feature in ("li_flag", "cwi_flag", "pv_flag", "conservatory"):
        return "yes" if v else "no"
    return str(v)
