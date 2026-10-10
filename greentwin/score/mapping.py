"""Sub-score mappings from ``contracts/score_mapping.json``.

Every function here is pure and monotonic in the "better state" direction (see tests/test_score.py, which
sweeps each one). Inputs that can be uncertain (intensity, SAP, loft depth) accept numpy arrays so the
same code scores every posterior sample at once.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from contracts.models import Control, Glazing, HeatingSystem, Property, RoofType, WallInsulation, WallType

MAPPING_PATH = Path(__file__).resolve().parents[2] / "contracts" / "score_mapping.json"

COMPONENTS = (
    "energy_efficiency", "energy_intensity", "insulation", "heating_system",
    "windows_glazing", "renewable_energy", "smart_energy_management",
)

BOILERS = {
    HeatingSystem.gas_boiler_condensing, HeatingSystem.gas_boiler_non_condensing, HeatingSystem.oil_boiler,
    HeatingSystem.lpg_boiler, HeatingSystem.communal_boiler, HeatingSystem.solid_fuel,
}


@lru_cache(maxsize=1)
def mapping() -> dict:
    return json.loads(MAPPING_PATH.read_text(encoding="utf-8"))


def weights() -> dict[str, float]:
    return dict(mapping()["weights"])


def _interp(x, points: Sequence[Sequence[float]]):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return np.interp(np.asarray(x, dtype=float), xs, ys)  # clamps at both ends


def age_band_start(band: Optional[str]) -> Optional[int]:
    """'1900-1929' -> 1900; 'before 1900' -> 1800; '2012 onwards' -> 2012; unknown -> None."""
    if not band:
        return None
    b = band.lower()
    if "before" in b:
        return 1800
    digits = "".join(ch if ch.isdigit() else " " for ch in b).split()
    return int(digits[0]) if digits else None


# --------------------------------------------------------------------------- energy efficiency
def ee_from_sap(sap):
    """Sub-score = SAP rating, clamped to 0..100 (SAP can exceed 100 for net exporters)."""
    return np.clip(np.asarray(sap, dtype=float), 0.0, 100.0)


def band_midpoint(band: str) -> float:
    return float(mapping()["components"]["energy_efficiency"]["band_midpoints"][band])


# --------------------------------------------------------------------------- intensity
def intensity_sub(kwh_m2):
    """Annualised metered kWh/m2 (lower is better) -> 0..100."""
    return _interp(kwh_m2, mapping()["components"]["energy_intensity"]["points"])


# --------------------------------------------------------------------------- insulation
def loft_sub(mm):
    return _interp(mm, mapping()["components"]["insulation"]["roof_loft_mm_points"])


def roof_u_sub(u):
    """Roof U-value -> equivalent loft depth (RdSAP 10 Table 16) -> loft curve. Lower U never scores lower."""
    pts = mapping()["components"]["insulation"]["roof_u_to_loft_mm"]["points"]
    return loft_sub(_interp(u, pts))


def wall_u_sub(u):
    return _interp(u, mapping()["components"]["insulation"]["wall_u_points"])


def floor_u_sub(u):
    return _interp(u, mapping()["components"]["insulation"]["floor_u_points"])


ROOF_INSULATED_U = 0.35  # W/m2K; RdSAP improvement threshold for roof rooms (RdSAP10 note 13)
ROOF_AS_BUILT_INSULATED_FROM = 1983  # age bands from 1983 assumed insulated as built (decision D-11)
WALL_AS_BUILT_INSULATED_FROM = 1976  # first wall U-value requirement in Building Regulations (decision D-10)


def roof_other_insulated(prop: Property) -> bool:
    """Flat / room-in-roof: the contract has no 'insulated' flag, so infer it (decision D-11)."""
    if prop.roof.u_value is not None:
        return prop.roof.u_value <= ROOF_INSULATED_U
    if prop.roof.loft_insulation_mm is not None:
        return prop.roof.loft_insulation_mm >= 100
    start = age_band_start(prop.construction_age_band)
    return start is not None and start >= ROOF_AS_BUILT_INSULATED_FROM


def roof_sub(prop: Property, loft_mm=None):
    """Roof sub-score; ``loft_mm`` overrides the property value (used for sampled unknown depths)."""
    c = mapping()["components"]["insulation"]
    rt = prop.roof.type
    if rt == RoofType.another_dwelling_above:
        return None
    if prop.roof.u_value is not None and loft_mm is None:
        return float(roof_u_sub(prop.roof.u_value))
    if rt == RoofType.pitched_loft:
        mm = prop.roof.loft_insulation_mm if loft_mm is None else loft_mm
        if mm is None:
            raise ValueError("unknown loft depth must be sampled, never treated as 0")
        return loft_sub(mm)
    key = "flat" if rt == RoofType.flat else "room_in_roof"
    key += "_insulated" if roof_other_insulated(prop) else "_uninsulated"
    return float(c["roof_other"][key])


def wall_sub(wall_type: WallType, insulation: WallInsulation, age_band: Optional[str] = None) -> float:
    """Wall sub-score. 'as_built' only means insulated when built after wall-insulation regulations (D-10)."""
    w = mapping()["components"]["insulation"]["walls"]
    if insulation == WallInsulation.as_built:
        start = age_band_start(age_band)
        if start is not None and start < WALL_AS_BUILT_INSULATED_FROM:
            insulation = WallInsulation.none
    if insulation == WallInsulation.none:
        if wall_type in (WallType.solid_brick, WallType.solid_stone):
            return float(w["solid_none"])
        if wall_type == WallType.cavity:
            return float(w["cavity_none"])
        return float(w["timber_or_system_none"])
    return float(w[insulation.value])


def floor_sub(prop: Property):
    from contracts.models import FloorType
    if prop.floor.type == FloorType.another_dwelling_below:
        return None
    if prop.floor.u_value is not None:
        return float(floor_u_sub(prop.floor.u_value))
    f = mapping()["components"]["insulation"]["floor"]
    return float(f["insulated" if prop.floor.insulated else "uninsulated"])


def insulation_sub(prop: Property, loft_mm=None):
    """Element-weighted roof/walls/floor; absent elements dropped and weights renormalised."""
    ew = mapping()["components"]["insulation"]["element_weights"]
    parts = {
        "roof": roof_sub(prop, loft_mm),
        "walls": (float(wall_u_sub(prop.walls.u_value)) if prop.walls.u_value is not None
                  else wall_sub(prop.walls.type, prop.walls.insulation, prop.construction_age_band)),
        "floor": floor_sub(prop),
    }
    present = {k: v for k, v in parts.items() if v is not None}
    tot = sum(ew[k] for k in present)
    return sum(ew[k] / tot * np.asarray(v, dtype=float) for k, v in present.items())


# --------------------------------------------------------------------------- heating
def heating_sub(system: HeatingSystem, age_years: Optional[int] = None) -> float:
    h = mapping()["components"]["heating_system"]
    s = float(h["map"][system.value])
    if system in BOILERS and age_years is not None and age_years > 12:
        s = max(0.0, s - (age_years - 12))
    return s


# --------------------------------------------------------------------------- glazing
def glazing_sub(glazing: Sequence[Glazing]) -> float:
    m = mapping()["components"]["windows_glazing"]["map"]
    tot = sum(g.share_pct for g in glazing)
    if tot <= 0:
        raise ValueError("glazing shares sum to 0")
    return sum(m[g.type.value] * g.share_pct / tot for g in glazing)  # shares renormalised to 100%


# --------------------------------------------------------------------------- renewables
def renewables_sub(pv_kwp: float, battery_kwh: float = 0.0, solar_thermal: bool = False) -> float:
    r = mapping()["components"]["renewable_energy"]
    s = float(_interp(pv_kwp, r["pv_kwp_points"]))
    if solar_thermal:
        s += r["bonuses"]["solar_thermal"]
    if battery_kwh > 0:
        s += r["bonuses"]["battery_any"]
    return min(float(r["cap"]), s)


# --------------------------------------------------------------------------- smart
def controls_level(controls: Sequence[Control]) -> float:
    """Highest controls level whose requirements are all present (0-100)."""
    have = {c.value if hasattr(c, "value") else c for c in controls}
    best = 0.0
    for lvl in mapping()["components"]["smart_energy_management"]["levels"]:
        if set(lvl["requires"]) <= have:
            best = max(best, float(lvl["sub_score"]))
    return best


def smart_sub(controls: Sequence[Control], lighting_pct):
    """Controls and efficient electricity use (D-26): (1 - a) x controls level + a x low-energy lighting %.

    ``lighting_pct`` may be an array of samples; it must be known or sampled, never defaulted to 0.
    """
    a = mapping()["components"]["smart_energy_management"]["lighting_share"]
    light = np.clip(np.asarray(lighting_pct, dtype=float), 0.0, 100.0)
    return (1 - a) * controls_level(controls) + a * light


def classify(score_rounded: int) -> str:
    for b in mapping()["classification"]:
        if b["min"] <= score_rounded <= b["max"]:
            return b["band"]
    raise ValueError(score_rounded)


def round_half_up(x: float) -> int:
    """Brief: round to the nearest integer; half up (Python's round() is banker's rounding). Decision D-03."""
    return int(np.floor(x + 0.5))
