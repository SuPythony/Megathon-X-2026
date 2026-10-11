"""The home's recorded features after a retrofit package: ``Property`` -> ``Property``.

The twin works on physics parameters (``twin_edits``); the Green Home Score's insulation, heating, glazing,
renewables and smart-management sub-scores read ``Property`` fields. This maps each catalogue retrofit to the
EPC-style fields it changes, so a projected score sees the same package the twin ran (R-10). Pure function.
"""

from __future__ import annotations

from typing import Callable, Sequence

from contracts.models import (
    Control, Glazing, GlazingType, HeatingSystem, Property, RetrofitAction, WallInsulation,
)

from . import catalogue as cat


def _num(rid: str, key: str) -> float:
    return float(cat.get(rid).twin_edits[key])


def _glazing(t: GlazingType) -> Callable[[dict, RetrofitAction], None]:
    def edit(p: dict, a: RetrofitAction) -> None:
        old = p["glazing"]
        ratio = old[0].get("window_to_floor_ratio") if old else None
        p["glazing"] = [Glazing(type=t, share_pct=100, window_to_floor_ratio=ratio).model_dump()]
    return edit


def _wall(ins: WallInsulation, rid: str) -> Callable[[dict, RetrofitAction], None]:
    def edit(p: dict, a: RetrofitAction) -> None:
        p["walls"]["insulation"] = ins.value
        p["walls"]["u_value"] = _num(rid, "wall_u_value")
    return edit


def _add_control(*cs: Control) -> Callable[[dict, RetrofitAction], None]:
    def edit(p: dict, a: RetrofitAction) -> None:
        have = [str(getattr(c, "value", c)) for c in p["controls"] if str(getattr(c, "value", c)) != "none"]
        p["controls"] = have + [c.value for c in cs if c.value not in have]
    return edit


def _heat_pump(p: dict, a: RetrofitAction) -> None:
    p["heating"].update(main_system=HeatingSystem.air_source_heat_pump.value, main_fuel="electricity",
                        seasonal_efficiency=_num("air_source_heat_pump", "scop"), system_age_years=0,
                        hot_water="heat_pump_cylinder")


def _hhr(p: dict, a: RetrofitAction) -> None:
    p["heating"].update(main_system=HeatingSystem.electric_storage_hhr.value, main_fuel="electricity",
                        system_age_years=0)


def _loft(p: dict, a: RetrofitAction) -> None:
    p["roof"]["loft_insulation_mm"] = max(270, p["roof"].get("loft_insulation_mm") or 0)
    p["roof"]["u_value"] = _num("loft_insulation", "roof_u_value")


def _floor(p: dict, a: RetrofitAction) -> None:
    p["floor"].update(insulated=True, u_value=_num("floor_insulation", "floor_u_value"))


def _pv(p: dict, a: RetrofitAction) -> None:
    kwp = float(a.size) if a.size is not None else _num("solar_pv", "pv_kwp")
    p["renewables"]["pv_kwp"] = p["renewables"]["pv_kwp"] + kwp
    if p["renewables"].get("pv_azimuth_deg") is None:
        p["renewables"]["pv_azimuth_deg"] = p["constraints"].get("best_roof_azimuth_deg") or 180.0


def _battery(p: dict, a: RetrofitAction) -> None:
    p["renewables"]["battery_kwh"] = p["renewables"]["battery_kwh"] + (
        float(a.size) if a.size is not None else _num("battery", "battery_kwh"))


EDITS: dict[str, Callable[[dict, RetrofitAction], None]] = {
    "draught_proofing": lambda p, a: p.__setitem__("draught_proofed_pct", 100.0),
    "loft_insulation": _loft,
    "cavity_wall_insulation": _wall(WallInsulation.cavity_filled, "cavity_wall_insulation"),
    "internal_wall_insulation": _wall(WallInsulation.internal, "internal_wall_insulation"),
    "external_wall_insulation": _wall(WallInsulation.external, "external_wall_insulation"),
    "floor_insulation": _floor,
    "double_glazing": _glazing(GlazingType.double_post_2002),
    "triple_glazing": _glazing(GlazingType.triple),
    "secondary_glazing": _glazing(GlazingType.secondary),
    "smart_thermostat": _add_control(Control.smart_thermostat, Control.trvs),
    "air_source_heat_pump": _heat_pump,
    "hhr_storage_heaters": _hhr,
    "solar_pv": _pv,
    "battery": _battery,
    "led_lighting": lambda p, a: p.__setitem__("low_energy_lighting_pct", 100.0),
    "time_of_use_tariff": _add_control(Control.smart_meter),
}


def property_after(prop: Property, actions: Sequence[RetrofitAction]) -> Property:
    """New Property with each retrofit's recorded-feature changes applied in order. Unknown ids raise."""
    p = prop.model_dump(mode="json")
    for a in actions:
        if a.retrofit_id not in EDITS:
            raise KeyError(f"no Property mapping for retrofit {a.retrofit_id!r}")
        EDITS[a.retrofit_id](p, a)
    return Property.model_validate(p)
