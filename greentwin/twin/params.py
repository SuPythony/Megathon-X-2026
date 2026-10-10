"""params_from_property: one home's EPC record (+ fitted floor plan) -> one ``TwinParams`` parameter set.

Order of precedence for every field: measured value on the EPC -> fitted floor plan geometry -> RdSAP 2012 default
table (``rdsap_defaults.json``) -> SAP 2012 formula -> labelled assumption. ``params_with_sources`` reports which one
was used for each field, so the UI, the LLM and the physics owner can see where every number came from.

Geometry comes from a ``TwinLayout`` (the prefit archetype fitted to this home). Without one, the closest archetype
is fitted to the EPC floor area and built form.
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from contracts.models import BuiltForm, FloorType, Property, PropertyType, RoofType, TwinParams
from greentwin.twin.archetypes import TwinLayout, build_layout
from greentwin.twin.physics_api import FuelConfig

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def rdsap() -> dict:
    return json.loads((HERE / "rdsap_defaults.json").read_text(encoding="utf-8"))


def age_band_index(construction_age_band: str, year_built: Optional[int] = None) -> int:
    """RdSAP England & Wales band A..L as 0..11, from the EPC age band text ('1900-1929', 'before 1900', ...)."""
    year = year_built
    if year is None:
        years = [int(y) for y in re.findall(r"\d{4}", construction_age_band)]
        if not years:
            return 4  # unknown: band E (1967-1975), the stock median; reported as an assumption by the caller
        year = years[0] if "before" not in construction_age_band.lower() else years[0] - 1
    for i, (lo, hi) in enumerate(rdsap()["age_bands"].values()):
        if lo <= year <= hi:
            return i
    return 11


def choose_archetype(prop: Property) -> str:
    """Closest prefit archetype for a home with no floor plan."""
    if prop.property_type in (PropertyType.flat, PropertyType.maisonette):
        return "flat_2bed"
    if prop.property_type == PropertyType.bungalow:
        return "bungalow_2bed"
    band = age_band_index(prop.construction_age_band, prop.year_built)
    if prop.built_form == BuiltForm.detached:
        return "postwar_detached_4bed"
    if prop.built_form == BuiltForm.semi_detached:
        return "interwar_semi_3bed"
    return "victorian_terrace_2bed" if band <= 2 else "modern_terrace_3bed"  # terraces: pre-1950 vs later


def sap_occupancy(tfa: float) -> float:
    """SAP 2012 Table 1b: assumed occupants N from total floor area."""
    if tfa <= 13.9:
        return 1.0
    return 1 + 1.76 * (1 - math.exp(-0.000349 * (tfa - 13.9) ** 2)) + 0.0013 * (tfa - 13.9)


def floor_u_iso13370(area: float, exposed_perimeter: float, insulated: bool) -> float:
    """BS EN ISO 13370 slab-on-ground U-value, as RdSAP 2012 Appendix S5 uses it."""
    f = rdsap()["floor"]
    if area <= 0 or exposed_perimeter <= 0:
        return 0.0
    lam, w = f["lambda_ground"], f["wall_thickness_m"]
    b = 2 * area / exposed_perimeter
    dt = w + lam * (f["rsi"] + (f["insulation_r"] if insulated else 0.0) + f["rse"])
    if dt < b:
        return 2 * lam / (math.pi * b + dt) * math.log(math.pi * b / dt + 1)
    return lam / (0.457 * b + dt)


def _loft_u(mm: int) -> float:
    table = {int(k): v for k, v in rdsap()["loft_u_by_mm"].items() if not k.startswith("_")}
    keys = sorted(table)
    if mm <= keys[0]:
        return table[keys[0]]
    if mm >= keys[-1]:
        return table[keys[-1]]
    lo = max(k for k in keys if k <= mm)
    hi = min(k for k in keys if k >= mm)
    if lo == hi:
        return table[lo]
    return table[lo] + (table[hi] - table[lo]) * (mm - lo) / (hi - lo)


def hot_water_fuel(prop: Property) -> str:
    if prop.heating.hot_water == "electric_immersion":
        return "electricity"
    return prop.heating.main_fuel


def fuel_config(prop: Property) -> FuelConfig:
    return FuelConfig(heating_fuel=prop.heating.main_fuel, hot_water_fuel=hot_water_fuel(prop))


def params_with_sources(prop: Property, layout: Optional[TwinLayout] = None,
                        appliance_kwh_per_day: Optional[float] = None) -> tuple[TwinParams, dict[str, str]]:
    """TwinParams for one home plus, per field, where the value came from."""
    d = rdsap()
    band = age_band_index(prop.construction_age_band, prop.year_built)
    src: dict[str, str] = {}

    if layout is None:
        layout = build_layout(choose_archetype(prop), total_floor_area_m2=prop.floor_area_m2, built_form=prop.built_form)
        geo_src = f"layout:{layout.archetype_id} (fitted to EPC floor area)"
    else:
        geo_src = f"layout:{layout.archetype_id} (floor plan)"
    env = layout.envelope()
    for k in ("wall_area_m2", "roof_area_m2", "floor_area_m2", "window_area_m2", "door_area_m2",
              "party_wall_area_m2", "volume_m3"):
        src[k] = geo_src

    roof_area, floor_area = env.roof_area_m2, env.floor_area_m2
    if prop.roof.type == RoofType.another_dwelling_above:
        roof_area, src["roof_area_m2"] = 0.0, "epc (dwelling above)"
    if prop.floor.type == FloorType.another_dwelling_below:
        floor_area, src["floor_area_m2"] = 0.0, "epc (dwelling below)"

    window_area = env.window_area_m2
    ratio = next((g.window_to_floor_ratio for g in prop.glazing if g.window_to_floor_ratio), None)
    wall_area = env.wall_area_m2
    if ratio:
        new_window = ratio * prop.floor_area_m2
        wall_area = max(0.0, wall_area + window_area - new_window)
        window_area, src["window_area_m2"] = new_window, "epc (window-to-floor ratio)"

    # Walls
    if prop.walls.u_value is not None:
        wall_u, src["wall_u"] = prop.walls.u_value, "epc (measured)"
    else:
        entry = d["wall_u"][prop.walls.type.value].get(prop.walls.insulation.value,
                                                        d["wall_u"][prop.walls.type.value]["none"])
        wall_u = entry[band] if isinstance(entry, list) else entry
        src["wall_u"] = f"rdsap_default ({d['wall_u']['_status']})"

    # Roof
    if prop.roof.u_value is not None:
        roof_u, src["roof_u"] = prop.roof.u_value, "epc (measured)"
    elif prop.roof.type == RoofType.another_dwelling_above:
        roof_u, src["roof_u"] = 0.0, "epc (dwelling above)"
    elif prop.roof.type == RoofType.pitched_loft and prop.roof.loft_insulation_mm is not None:
        roof_u, src["roof_u"] = _loft_u(prop.roof.loft_insulation_mm), f"rdsap_default loft table ({d['loft_u_by_mm']['_status']})"
    else:
        roof_u = d["roof_u_unknown_insulation"][prop.roof.type.value][band]
        src["roof_u"] = f"rdsap_default by age ({d['roof_u_unknown_insulation']['_status']})"

    # Floor
    if prop.floor.u_value is not None:
        floor_u, src["floor_u"] = prop.floor.u_value, "epc (measured)"
    elif floor_area == 0:
        floor_u, src["floor_u"] = 0.0, "epc (dwelling below)"
    else:
        perimeter = sum(w.length_m for w in layout.walls if w.kind == "exterior" and w.storey == 0)
        floor_u = floor_u_iso13370(floor_area, perimeter, prop.floor.insulated)
        src["floor_u"] = f"iso13370 ({d['floor']['_status']})"

    # Windows: share-weighted
    shares = [(d["window_u"][g.type.value], g.share_pct) for g in prop.glazing] or [(d["window_u"]["double_pre_2002"], 100)]
    total_share = sum(s for _, s in shares) or 100
    window_u = sum(u * s for u, s in shares) / total_share
    src["window_u"] = f"rdsap_default glazing ({d['window_u']['_status']})"

    door_u, src["door_u"] = d["door_u"]["value"], f"rdsap_default ({d['door_u']['_status']})"
    src["thermal_bridging_y"] = "twinparams_default"

    ach_cfg = d["air_changes_per_hour"]
    ach = ach_cfg["by_band"][band] - ach_cfg["draught_proofing_max_reduction"] * (prop.draught_proofed_pct or 0) / 100
    src["air_changes_per_hour"] = "assumption by age band, less draught proofing"

    if prop.heating.seasonal_efficiency is not None:
        eff, src["heating_efficiency"] = prop.heating.seasonal_efficiency, "epc"
    else:
        eff = d["heating_efficiency"][prop.heating.main_system.value]
        src["heating_efficiency"] = f"rdsap_default by system ({d['heating_efficiency']['_status']})"

    # Occupancy-driven demands (SAP 2012 Appendix J and L)
    tfa = prop.floor_area_m2
    n = sap_occupancy(tfa)
    hot_water = (25 * n + 36) * 4.19 * 37 / 3600  # litres/day x specific heat x 37 K mean rise -> kWh/day useful heat
    src["hot_water_kwh_per_day"] = "sap_2012 appendix J (useful heat, 37 K mean rise)"
    e_appliances = 207.8 * (tfa * n) ** 0.4714
    low_energy = (prop.low_energy_lighting_pct if prop.low_energy_lighting_pct is not None else 50) / 100
    e_lighting = 59.73 * (tfa * n) ** 0.4714 * (1 - 0.5 * low_energy)
    e_cooking = (35 + 7 * n) * 8.76
    sap_baseload = (e_appliances + e_lighting + e_cooking) / 365
    if appliance_kwh_per_day is not None:
        baseload, src["baseload_kwh_per_day"] = appliance_kwh_per_day + e_lighting / 365, "appliance list + sap lighting"
    else:
        baseload, src["baseload_kwh_per_day"] = sap_baseload, "sap_2012 appendix L (appliances + lighting + cooking)"
    lighting_share = (e_lighting / 365) / baseload if baseload > 0 else 0.15
    src["lighting_share_of_baseload"] = "sap_2012 appendix L"
    internal_gains = 60 * n - 40 * n + (e_appliances + 0.85 * e_lighting) * 1000 / 8760 + 35 + 7 * n
    src["internal_gains_w"] = "sap_2012 table 5 (metabolic, appliances, lighting, cooking, less evaporation)"

    src.update({"heat_loss_multiplier": "default 1.0 (calibrated)", "setpoint_c": "default 19.5 (calibrated)",
                "heating_hours_per_day": "twinparams_default", "solar_gain_factor": "twinparams_default",
                "pv_kwp": "epc", "pv_azimuth_deg": "epc or 180", "pv_tilt_deg": "epc or 35", "battery_kwh": "epc",
                "heating_fuel": "epc", "tariff": "default standard"})

    params = TwinParams(
        property_id=prop.property_id,
        wall_area_m2=round(wall_area, 2), roof_area_m2=round(roof_area, 2), floor_area_m2=round(floor_area, 2),
        window_area_m2=round(window_area, 2), door_area_m2=round(env.door_area_m2, 2),
        party_wall_area_m2=round(env.party_wall_area_m2, 2),
        wall_u=round(wall_u, 3), roof_u=round(roof_u, 3), floor_u=round(floor_u, 3), window_u=round(window_u, 3),
        door_u=door_u, air_changes_per_hour=round(max(ach, 0.2), 3), volume_m3=round(env.volume_m3, 2),
        heating_efficiency=eff, heating_fuel=prop.heating.main_fuel,
        hot_water_kwh_per_day=round(hot_water, 3), baseload_kwh_per_day=round(baseload, 3),
        lighting_share_of_baseload=round(lighting_share, 3), internal_gains_w=round(internal_gains, 1),
        pv_kwp=prop.renewables.pv_kwp, pv_azimuth_deg=prop.renewables.pv_azimuth_deg or 180.0,
        pv_tilt_deg=prop.renewables.pv_tilt_deg or 35.0, battery_kwh=prop.renewables.battery_kwh,
    )
    return params, src


def params_from_property(prop: Property, layout: Optional[TwinLayout] = None) -> TwinParams:
    """``DigitalTwin.params_from_property``: RdSAP defaults (+ fitted floor plan geometry) -> one parameter set."""
    return params_with_sources(prop, layout)[0]
