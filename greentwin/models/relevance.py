"""Relevance model (plan 4c.3): P(an EPC assessor would recommend measure X | the home's features), one EBM per measure.

Labels: the real EPC recommendations table (improvement_summary_text mapped to catalogue ids below), joined to the
same certificate. Features: what the certificate records at assessment time, parsed by greentwin/score/epc_features
(element sub-scores, heating system, glazing, controls, lighting, PV, SAP, form, age, floor area). No `potential_*`
fields (leakage, core rule 9). Split by property (UPRN, else certificate), assigned once by hash.
Baseline: rules only (a measure is "recommended" when its element is in a poor state); same metrics.

The model RANKS feasible options and explains them; it never vetoes (rules win) and never produces savings.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from contracts.models import Property, RoofType
from greentwin.score import mapping as m

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "relevance" / "v1"

# EPC improvement summary text -> catalogue ids (several ids when the EPC names a choice, e.g. internal OR external)
MEASURES: list[tuple[str, tuple[str, ...]]] = [
    (r"loft insulation|insulate.*loft|room-in-roof insulation|flat roof|sloping ceiling insulation", ("loft_insulation",)),
    (r"cavity wall insulation", ("cavity_wall_insulation",)),
    (r"internal or external wall|solid wall insulation|wall insulation", ("internal_wall_insulation", "external_wall_insulation")),
    (r"floor insulation", ("floor_insulation",)),
    (r"draught proof", ("draught_proofing",)),
    (r"secondary glazing", ("secondary_glazing",)),
    (r"double glaz|glazed windows|replace .*windows|replacement glazing|triple glaz", ("double_glazing",)),
    (r"heating controls|room thermostat|thermostatic radiator|programmer|time and temperature zone", ("smart_thermostat",)),
    (r"air source heat pump|air or ground source heat pump", ("air_source_heat_pump",)),
    (r"high heat retention storage", ("hhr_storage_heaters",)),
    (r"photovoltaic|solar pv", ("solar_pv",)),
    (r"low energy lighting|led", ("led_lighting",)),
]
TARGETS = sorted({c for _, ids in MEASURES for c in ids})
NUM = ["tfa", "sap", "age_start", "s_walls", "s_roof", "s_floor", "roof_absent", "floor_absent", "s_windows_glazing",
       "pv_kwp_est", "controls_level", "low_energy_lighting_pct", "mains_gas", "is_heat_pump", "is_storage",
       "is_gas_boiler", "s_heating_system"]
CAT = ["property_type", "built_form"]
FEATURES = NUM + CAT


def catalogue_ids(summary: str) -> tuple[str, ...]:
    s = (summary or "").lower()
    for pat, ids in MEASURES:
        if re.search(pat, s):
            return ids
    return ()


def _norm(s: pd.Series) -> pd.Series:
    return s.fillna("unknown").str.strip().str.lower().str.replace(r"[\s-]+", "_", regex=True)


def features_from_epc(feat: pd.DataFrame) -> pd.DataFrame:
    """greentwin.score.epc_features.build output -> model features."""
    hs = feat["heating_system"].fillna("")
    out = pd.DataFrame(index=feat.index)
    for c in ("tfa", "sap", "s_walls", "s_roof", "s_floor", "roof_absent", "floor_absent", "s_windows_glazing",
              "pv_kwp_est", "controls_level", "low_energy_lighting_pct", "mains_gas", "s_heating_system"):
        out[c] = pd.to_numeric(feat[c], errors="coerce")
    out["age_start"] = feat["age_band"].map(lambda b: m.age_band_start(b) if isinstance(b, str) else None).astype(float)
    out["is_heat_pump"] = hs.str.contains("heat_pump").astype(float)
    out["is_storage"] = hs.str.contains("storage").astype(float)
    out["is_gas_boiler"] = hs.str.startswith("gas_boiler").astype(float)
    out["property_type"] = _norm(feat["property_type"])
    out["built_form"] = _norm(feat["built_form"])
    return out


def features_from_property(prop: Property) -> pd.DataFrame:
    """The same features for one of our homes (recorded state; nothing from after a retrofit)."""
    roof_absent = float(prop.roof.type == RoofType.another_dwelling_above)
    loft = prop.roof.loft_insulation_mm
    if prop.roof.type == RoofType.pitched_loft and loft is None and prop.roof.u_value is None:
        s_roof = np.nan  # unknown depth stays unknown (EBM handles missing), never 0
    else:
        r = m.roof_sub(prop)
        s_roof = np.nan if r is None else float(np.asarray(r))
    fl = m.floor_sub(prop)
    hs = prop.heating.main_system.value
    row = {
        "tfa": prop.floor_area_m2, "sap": prop.epc.sap_score, "age_start": m.age_band_start(prop.construction_age_band),
        "s_walls": m.wall_sub(prop.walls.type, prop.walls.insulation, prop.construction_age_band),
        "s_roof": s_roof, "s_floor": np.nan if fl is None else float(fl),
        "roof_absent": roof_absent, "floor_absent": float(fl is None),
        "s_windows_glazing": m.glazing_sub(prop.glazing), "pv_kwp_est": prop.renewables.pv_kwp,
        "controls_level": m.controls_level(prop.controls), "low_energy_lighting_pct": np.nan,
        "mains_gas": float(str(getattr(prop.heating.main_fuel, "value", prop.heating.main_fuel)) == "mains_gas"),
        "is_heat_pump": float("heat_pump" in hs), "is_storage": float("storage" in hs),
        "is_gas_boiler": float(hs.startswith("gas_boiler")), "s_heating_system": m.heating_sub(prop.heating.main_system),
        "property_type": prop.property_type.value, "built_form": prop.built_form.value,
    }
    return pd.DataFrame([row])[FEATURES]


def rules_baseline(X: pd.DataFrame) -> pd.DataFrame:
    """Rules only: recommend a measure when its element is in a poor state (0/1 per measure)."""
    f = lambda c: X[c].fillna(-1)  # noqa: E731 - unknown never triggers a rule
    no_roof = X["roof_absent"].fillna(0) > 0
    out = pd.DataFrame(index=X.index)
    out["loft_insulation"] = ((f("s_roof") >= 0) & (f("s_roof") < 90) & ~no_roof)
    out["cavity_wall_insulation"] = f("s_walls") == 25
    out["internal_wall_insulation"] = f("s_walls").isin([0, 30])
    out["external_wall_insulation"] = out["internal_wall_insulation"]
    out["floor_insulation"] = f("s_floor") == 30
    out["draught_proofing"] = (f("age_start") >= 0) & (f("age_start") < 1950)
    out["double_glazing"] = (f("s_windows_glazing") >= 0) & (f("s_windows_glazing") < 60)
    out["secondary_glazing"] = out["double_glazing"]
    out["smart_thermostat"] = (f("controls_level") >= 0) & (f("controls_level") < 60)
    out["air_source_heat_pump"] = (X["is_heat_pump"] == 0) & (X["mains_gas"] == 0)
    out["hhr_storage_heaters"] = X["is_storage"] == 1
    out["solar_pv"] = (f("pv_kwp_est") == 0) & ~no_roof
    out["led_lighting"] = (f("low_energy_lighting_pct") >= 0) & (f("low_energy_lighting_pct") < 100)
    return out[TARGETS].astype(float)


def split_of(key: str, seed: int = 42) -> str:
    h = int(hashlib.md5(f"{seed}:{key}".encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else ("val" if h < 80 else "test")


class RelevanceModel:
    """One EBM classifier per measure (trained by scripts/train_relevance.py)."""

    def __init__(self, models: dict, metrics: Optional[dict] = None):
        self.models = models
        self.metrics = metrics or {}

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({t: mdl.predict_proba(X[FEATURES])[:, 1] for t, mdl in self.models.items()}, index=X.index)

    def for_property(self, prop: Property, retrofit_ids=None) -> dict[str, dict]:
        """retrofit_id -> {p_assessor_recommends, top_reasons}; ids without a trained model are left out."""
        X = features_from_property(prop)
        out = {}
        for t, mdl in self.models.items():
            if retrofit_ids is not None and t not in retrofit_ids:
                continue
            p = float(mdl.predict_proba(X)[0, 1])
            reasons = []
            try:
                exp = mdl.explain_local(X)
                d = exp.data(0)
                pairs = sorted(zip(d["names"], d["scores"]), key=lambda x: -abs(x[1]))[:3]
                reasons = [{"feature": str(n), "contribution_logit": round(float(s), 3)} for n, s in pairs]
            except Exception:  # noqa: BLE001 - explanation is optional
                pass
            out[t] = {"p_assessor_recommends": round(p, 3), "top_reasons": reasons,
                      "auc_test": self.metrics.get("per_measure", {}).get(t, {}).get("ebm", {}).get("auc")}
        return out


@lru_cache(maxsize=1)
def load(path: Path = ARTIFACT) -> Optional[RelevanceModel]:
    f = path / "model.pkl"
    if not f.exists():
        return None
    import pickle

    with open(f, "rb") as fh:
        models = pickle.load(fh)
    metrics = json.loads((path / "metrics.json").read_text()) if (path / "metrics.json").exists() else {}
    return RelevanceModel(models, metrics)
