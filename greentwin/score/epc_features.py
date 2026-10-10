"""EPC register rows -> Green Home Score sub-scores + targets (training data for the learned weights).

Parsing is done once per distinct description string (a few hundred per field), then mapped onto rows, so
millions of certificates are cheap. Sub-scores reuse ``greentwin.score.mapping`` so the learned weights apply to
exactly the sub-scores the product shows. Unparseable values stay NaN (never 0).
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from contracts.models import Glazing, GlazingType, HeatingSystem, WallInsulation, WallType

from . import mapping as m

STARS = {"very poor": 1, "poor": 2, "average": 3, "good": 4, "very good": 5}
_U = re.compile(r"thermal transmittance\s*([0-9.]+)", re.I)
_MM = re.compile(r"(\d+)\+?\s*mm", re.I)
ABSENT = ("dwelling above", "premises above", "dwelling below", "premises below")
ABSENT_ELEMENT = -1.0  # sentinel: element does not exist (dwelling above/below); pandas would turn None into NaN


def _u(desc: str) -> Optional[float]:
    g = _U.search(desc)
    return float(g.group(1)) if g else None


# --------------------------------------------------------------------------- walls
def wall_score(desc) -> float:
    if not isinstance(desc, str):
        return np.nan
    d = desc.lower()
    u = _u(d)
    if u is not None:
        return float(m.wall_u_sub(u))
    if "cavity" in d:
        wt = WallType.cavity
    elif "solid brick" in d:
        wt = WallType.solid_brick
    elif any(k in d for k in ("stone", "granite", "whinstone", "sandstone", "limestone", "cob")):
        wt = WallType.solid_stone
    elif "timber" in d or "park home" in d:
        wt = WallType.timber_frame
    elif "system" in d:
        wt = WallType.system_built
    else:
        return np.nan
    none = m.wall_sub(wt, WallInsulation.none)
    if "filled cavity" in d:
        return m.wall_sub(wt, WallInsulation.cavity_filled)
    if "external insulation" in d:
        return m.wall_sub(wt, WallInsulation.external)
    if "internal insulation" in d:
        return m.wall_sub(wt, WallInsulation.internal)
    if "partial insulation" in d:  # EPC: partly insulated as built -> halfway between none and as built
        return (none + m.wall_sub(wt, WallInsulation.as_built)) / 2
    if "no insulation" in d:
        return none
    if "insulated" in d or "additional insulation" in d:  # explicit on the EPC, so no age gating (D-10)
        return m.wall_sub(wt, WallInsulation.as_built)
    return np.nan


# --------------------------------------------------------------------------- roof
ROOF_ASSUMED_INSULATED_MM = 100  # "Pitched, insulated (assumed)" with no depth: cautious 100 mm (D-16)
ROOF_LIMITED_MM = 25


def roof_score(desc) -> Optional[float]:
    """Returns NaN for unknown, None for an absent element (dwelling above)."""
    if not isinstance(desc, str):
        return np.nan
    d = desc.lower()
    if any(k in d for k in ABSENT[:2]):
        return ABSENT_ELEMENT
    u = _u(d)
    if u is not None:
        return float(m.roof_u_sub(u))
    ins = mapping_roof_other()
    if "pitched" in d and "rafters" not in d:
        g = _MM.search(d)
        if g:
            return float(m.loft_sub(int(g.group(1))))
        if "no insulation" in d:
            return float(m.loft_sub(0))
        if "limited insulation" in d:
            return float(m.loft_sub(ROOF_LIMITED_MM))
        if "insulated" in d:
            return float(m.loft_sub(ROOF_ASSUMED_INSULATED_MM))
        return np.nan
    kind = "flat" if "flat" in d else "room_in_roof" if ("roof room" in d or "rafters" in d or "thatched" in d) else None
    if kind is None:
        return np.nan
    if "no insulation" in d:
        return float(ins[f"{kind}_uninsulated"])
    if "limited insulation" in d:
        return (ins[f"{kind}_uninsulated"] + ins[f"{kind}_insulated"]) / 2
    if "insulated" in d or _MM.search(d):
        return float(ins[f"{kind}_insulated"])
    return np.nan


def mapping_roof_other() -> dict:
    return m.mapping()["components"]["insulation"]["roof_other"]


# --------------------------------------------------------------------------- floor
def floor_score(desc) -> Optional[float]:
    if not isinstance(desc, str):
        return np.nan
    d = desc.lower()
    if any(k in d for k in ABSENT[2:]):
        return ABSENT_ELEMENT
    u = _u(d)
    if u is not None:
        return float(m.floor_u_sub(u))
    f = m.mapping()["components"]["insulation"]["floor"]
    if "no insulation" in d:
        return float(f["uninsulated"])
    if "limited insulation" in d:
        return (f["uninsulated"] + f["insulated"]) / 2
    if "insulated" in d:
        return float(f["insulated"])
    return np.nan


def insulation_from(walls: float, roof, floor) -> float:
    ew = m.mapping()["components"]["insulation"]["element_weights"]
    parts = {"walls": walls, "roof": roof, "floor": floor}
    if any(np.isnan(v) for v in parts.values()):
        return np.nan  # unknown element -> unknown insulation (never 0)
    present = {k: v for k, v in parts.items() if v != ABSENT_ELEMENT}
    tot = sum(ew[k] for k in present)
    return sum(ew[k] / tot * v for k, v in present.items())


# --------------------------------------------------------------------------- heating
_OIL = re.compile(r"\boil\b")


class _Unknown:
    value = "gas_boiler_unknown_condensing"


GAS_BOILER_UNKNOWN = _Unknown()


def heating_score(h) -> float:
    if not isinstance(h, str):
        return np.nan
    if h == GAS_BOILER_UNKNOWN.value:
        return (m.heating_sub(HeatingSystem.gas_boiler_condensing) + m.heating_sub(HeatingSystem.gas_boiler_non_condensing)) / 2
    return m.heating_sub(HeatingSystem(h), None)
SOLID = ("coal", "wood", "anthracite", "smokeless", "dual fuel", "biomass", "pellet", "peat")


def heating_system(desc, fuel, eff, controls) -> Optional[HeatingSystem]:
    if not isinstance(desc, str):
        return None
    d = desc.lower().split(", boiler")[0]
    f = (fuel or "").lower() if isinstance(fuel, str) else ""
    c = (controls or "").lower() if isinstance(controls, str) else ""
    if "heat pump" in d:
        return HeatingSystem.ground_source_heat_pump if ("ground" in d or "water source" in d) \
            else HeatingSystem.air_source_heat_pump
    if "community" in d:
        return HeatingSystem.district_heating
    if "storage heater" in d or "electricaire" in d:
        return HeatingSystem.electric_storage_hhr if "high heat retention" in (d + c) else HeatingSystem.electric_storage
    if any(k in d for k in ("electric underfloor", "electric ceiling", "electric heaters", "portable electric")) \
            or ("electric" in d and ("room heater" in d or "boiler" in d)):
        return HeatingSystem.electric_direct
    if _OIL.search(d) or _OIL.search(f):
        return HeatingSystem.oil_boiler
    if "lpg" in d or "lpg" in f:
        return HeatingSystem.lpg_boiler
    if any(k in d for k in SOLID) or any(k in f for k in SOLID):
        return HeatingSystem.solid_fuel
    if "gas" in d or "gas" in f:
        # The EPC text never says "condensing". Star rating: Very good -> condensing, Average/Poor -> not;
        # "Good" (99.5% of 2019 gas boilers) is ambiguous -> GAS_BOILER_UNKNOWN, scored as the 50/50 mix (D-16).
        e = eff.lower() if isinstance(eff, str) else ""
        if "boiler" in d and e == "very good":
            return HeatingSystem.gas_boiler_condensing
        if "boiler" in d and e == "good":
            return GAS_BOILER_UNKNOWN
        return HeatingSystem.gas_boiler_non_condensing
    if "electric" in d or "electric" in f:
        return HeatingSystem.electric_direct
    return None


# --------------------------------------------------------------------------- glazing
def glazing_score(desc, gtype, prop_pct) -> float:
    if not isinstance(desc, str):
        return np.nan
    d = desc.lower()
    if "single" in d:
        return m.glazing_sub([Glazing(type=GlazingType.single)])
    t = (gtype or "").lower() if isinstance(gtype, str) else ""
    if "triple" in d:
        kinds = [(GlazingType.triple, 1.0)]
    elif "secondary" in d:
        kinds = [(GlazingType.secondary, 1.0)]
    elif "high performance" in d or "after 2002" in t:
        kinds = [(GlazingType.double_post_2002, 1.0)]
    elif "before 2002" in t:
        kinds = [(GlazingType.double_pre_2002, 1.0)]
    else:  # double / multiple glazing, install date unknown: half pre-2002, half post-2002 (D-16)
        kinds = [(GlazingType.double_pre_2002, 0.5), (GlazingType.double_post_2002, 0.5)]
    if d.startswith("fully") or "throughout" in d or "high performance" in d:
        share = 100.0
    else:
        try:
            share = float(prop_pct)
        except (TypeError, ValueError):
            share = np.nan
        if np.isnan(share):
            share = 80.0 if d.startswith("mostly") else 50.0 if d.startswith("partial") else 20.0
    share = min(max(share, 0.0), 100.0)
    gl = [Glazing(type=k, share_pct=share * w) for k, w in kinds if share * w > 0]
    if share < 100:
        gl.append(Glazing(type=GlazingType.single, share_pct=100 - share))
    return m.glazing_sub(gl)


# --------------------------------------------------------------------------- controls
def controls_list(desc) -> Optional[list[str]]:
    if not isinstance(desc, str):
        return None
    # drop negated phrases first ("Programmer, no room thermostat", "No time or thermostatic control ...")
    d = re.sub(r"\bno (time or )?(room thermostat|thermostatic control[^,]*|time control[^,]*)", "", desc.lower())
    out = set()
    if "programmer" in d or "time and temperature zone" in d or "automatic charge" in d or "high heat retention" in d:
        out.add("programmer")
    if "room thermostat" in d or "room stat" in d or "appliance thermostat" in d or "time and temperature zone" in d \
            or "automatic charge" in d or "high heat retention" in d:
        out.add("room_thermostat")
    if "trv" in d or "time and temperature zone" in d or "high heat retention" in d:
        out.add("trvs")
    if "zone control" in d or "at least two room" in d:
        out.add("zoned_control")
    if "smart" in d:
        out.add("smart_thermostat")
    return sorted(out)


# --------------------------------------------------------------------------- renewables
PV_KWP_PER_M2 = 0.18   # modern panels ~ 400 W / 2.2 m2
PITCH_FACTOR = 1.15    # sloped roof area per m2 of footprint


def pv_kwp(pct, tfa, ptype) -> float:
    try:
        p = float(pct)
    except (TypeError, ValueError):
        return 0.0 if pct is None or (isinstance(pct, float) and np.isnan(pct)) else np.nan
    if np.isnan(p) or p <= 0:
        return 0.0
    storeys = 1 if str(ptype).lower() in ("bungalow", "flat", "maisonette") else 2
    return p / 100 * float(tfa) / storeys * PITCH_FACTOR * PV_KWP_PER_M2


# --------------------------------------------------------------------------- frame builder
def _map_unique(s: pd.Series, f) -> pd.Series:
    u = pd.Series(s.unique())
    lut = dict(zip(u, u.map(f)))
    return s.map(lut)


def build(df: pd.DataFrame) -> pd.DataFrame:
    """One row per certificate -> sub-scores, extras, targets and EPC element stars."""
    df = df.copy()
    num = lambda c: pd.to_numeric(df[c], errors="coerce")  # noqa: E731
    out = pd.DataFrame(index=df.index)
    out["uprn"] = df["UPRN"]
    out["certificate"] = df["CERTIFICATE_NUMBER"]
    out["lodgement_date"] = pd.to_datetime(df["LODGEMENT_DATE"], errors="coerce")
    out["local_authority"] = df["LOCAL_AUTHORITY"]
    out["property_type"] = df["PROPERTY_TYPE"]
    out["built_form"] = df["BUILT_FORM"]
    out["age_band"] = df["CONSTRUCTION_AGE_BAND"].str.replace("England and Wales: ", "", regex=False)
    out["tfa"] = num("TOTAL_FLOOR_AREA")
    out["report_type"] = df["REPORT_TYPE"]
    out["tenure"] = df["TENURE"]

    # --- targets (modelled by SAP/RdSAP)
    out["sap"] = num("CURRENT_ENERGY_EFFICIENCY")
    out["eir"] = num("ENVIRONMENT_IMPACT_CURRENT")
    out["co2_m2"] = num("CO2_EMISS_CURR_PER_FLOOR_AREA")
    out["pe_m2"] = num("ENERGY_CONSUMPTION_CURRENT")
    out["band"] = df["CURRENT_ENERGY_RATING"]

    # --- sub-scores
    out["s_energy_efficiency"] = m.ee_from_sap(out["sap"])
    walls = _map_unique(df["WALLS_DESCRIPTION"], wall_score)
    roof = _map_unique(df["ROOF_DESCRIPTION"], roof_score)
    floor = _map_unique(df["FLOOR_DESCRIPTION"], floor_score)
    out["s_insulation"] = [insulation_from(w, r, f) for w, r, f in zip(walls, roof, floor)]
    # element columns: NaN = unknown; absent elements recorded separately
    out["s_walls"] = walls
    out["s_roof"] = roof.where(roof != ABSENT_ELEMENT)
    out["s_floor"] = floor.where(floor != ABSENT_ELEMENT)
    out["roof_absent"] = (roof == ABSENT_ELEMENT).astype(float)
    out["floor_absent"] = (floor == ABSENT_ELEMENT).astype(float)

    key = df["MAINHEAT_DESCRIPTION"].fillna("") + "|" + df["MAIN_FUEL"].fillna("") + "|" + \
        df["MAINHEAT_ENERGY_EFF"].fillna("") + "|" + df["MAINHEATCONT_DESCRIPTION"].fillna("")
    hs = _map_unique(key, lambda k: getattr(heating_system(*[x or None for x in k.split("|")]), "value", None))
    out["heating_system"] = hs.astype(object)
    out["s_heating_system"] = [heating_score(h) for h in hs]

    gkey = df["WINDOWS_DESCRIPTION"].fillna("§") + "|" + df["GLAZED_TYPE"].fillna("") + "|" + \
        df["MULTI_GLAZE_PROPORTION"].fillna("")
    out["s_windows_glazing"] = _map_unique(
        gkey, lambda k: (lambda a, b, c: glazing_score(None if a == "§" else a, b or None, c or None))(*k.split("|")))

    kwp = [pv_kwp(p, t, ty) for p, t, ty in zip(df["PHOTO_SUPPLY"], out["tfa"], df["PROPERTY_TYPE"])]
    out["pv_kwp_est"] = kwp
    solar_th = df["SOLAR_WATER_HEATING_FLAG"].eq("Y")
    out["s_renewable_energy"] = [m.renewables_sub(k, 0.0, bool(st)) if not np.isnan(k) else np.nan
                                 for k, st in zip(kwp, solar_th)]

    ctl = _map_unique(df["MAINHEATCONT_DESCRIPTION"], controls_list)
    out["s_smart_energy_management"] = ctl.map(lambda c: m.smart_sub(c) if c is not None else np.nan)

    # --- extras (candidate parameters not in the brief's mapping; D-08)
    out["low_energy_lighting_pct"] = num("LOW_ENERGY_LIGHTING")
    out["open_fireplaces"] = num("NUMBER_OPEN_FIREPLACES")
    out["mech_vent"] = df["MECHANICAL_VENTILATION"].str.contains("mechanical", na=False).astype(float)
    out["dual_tariff"] = df["ENERGY_TARIFF"].str.contains("dual|off-peak|24 hour", case=False, na=False).astype(float)
    out["mains_gas"] = df["MAINS_GAS_FLAG"].eq("Y").astype(float)

    # --- EPC element ratings (1-5 stars), for validating our sub-score mappings
    for el, col in [("walls", "WALLS_ENERGY_EFF"), ("roof", "ROOF_ENERGY_EFF"), ("floor", "FLOOR_ENERGY_EFF"),
                    ("windows", "WINDOWS_ENERGY_EFF"), ("mainheat", "MAINHEAT_ENERGY_EFF"),
                    ("controls", "MAINHEATC_ENERGY_EFF"), ("lighting", "LIGHTING_ENERGY_EFF"),
                    ("hot_water", "HOT_WATER_ENERGY_EFF")]:
        out[f"stars_{el}"] = df[col].str.lower().map(STARS)
    return out


def latest_per_property(feat: pd.DataFrame) -> pd.DataFrame:
    """Current state = latest certificate per UPRN (rows without UPRN kept as their own property)."""
    key = feat["uprn"].fillna("cert:" + feat["certificate"].astype(str))
    return feat.assign(_k=key).sort_values("lodgement_date").groupby("_k", as_index=False).tail(1).drop(columns="_k")
