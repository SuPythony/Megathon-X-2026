"""apply_retrofits: catalogue retrofits -> edits to the twin's parameters. Pure: returns a new parameter set.

Packages are applied in order and then re-simulated as a whole (never summed). ``contracts/retrofit_catalogue.json``
keys don't all match ``TwinParams`` names yet, so ``KEY_MAP`` translates them. PROVISIONAL until the contract owner
renames the keys; each translation's interpretation is written next to it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

from contracts.models import RetrofitAction, TwinParams

CATALOGUE = Path(__file__).resolve().parents[2] / "contracts" / "retrofit_catalogue.json"

Edit = Callable[[dict, object, RetrofitAction], None]


@lru_cache(maxsize=1)
def catalogue() -> dict[str, dict]:
    return {item["id"]: item for item in json.loads(CATALOGUE.read_text(encoding="utf-8"))["items"]}


def _set(field: str) -> Edit:
    def edit(p: dict, v, a: RetrofitAction) -> None:
        p[field] = float(v)
    return edit


def _heat_pump(p: dict, v, a: RetrofitAction) -> None:
    # main_system: air_source_heat_pump -> electric heating; SCOP comes from the 'scop' edit
    p["heating_fuel"] = "electricity"
    if "hot_water_fuel" in p:
        p["hot_water_fuel"] = "electricity"


def _main_system(p: dict, v, a: RetrofitAction) -> None:
    if v == "air_source_heat_pump":
        _heat_pump(p, v, a)
    elif v == "electric_storage_hhr":
        p["heating_fuel"] = "electricity"
        p["heating_efficiency"] = 1.0
        if "hot_water_fuel" in p:  # HHR storage usually comes with an immersion cylinder
            p["hot_water_fuel"], p["hot_water_efficiency"] = "electricity", 1.0
    else:
        raise ValueError(f"main_system {v!r} has no twin mapping yet")


def _scop(p: dict, v, a: RetrofitAction) -> None:
    p["heating_efficiency"] = float(v)
    if "hot_water_efficiency" in p:
        p["hot_water_efficiency"] = round(float(v) * 0.8, 3)  # DHW at higher flow temperature: assumption


KEY_MAP: dict[str, Edit] = {
    "roof_u_value": _set("roof_u"),
    "wall_u_value": _set("wall_u"),
    "floor_u_value": _set("floor_u"),
    "window_u_value": _set("window_u"),
    # Interpreted as an ABSOLUTE reduction in air changes per hour (floor 0.2). Provisional.
    "ach_reduction": lambda p, v, a: p.__setitem__("air_changes_per_hour", max(0.2, p["air_changes_per_hour"] - float(v))),
    # Internal insulation takes floor space: interpreted as a factor on heated volume. Provisional.
    "floor_area_factor": lambda p, v, a: p.__setitem__("volume_m3", p["volume_m3"] * float(v)),
    "setpoint_reduction_c": lambda p, v, a: p.__setitem__("setpoint_c", p["setpoint_c"] - float(v)),
    "heating_hours_factor": lambda p, v, a: p.__setitem__("heating_hours_per_day", p["heating_hours_per_day"] * float(v)),
    "main_system": _main_system,
    "scop": _scop,
    # Better controls on storage heaters: interpreted as fewer effective heating hours. Provisional.
    "controllability_gain": lambda p, v, a: p.__setitem__("heating_hours_per_day", p["heating_hours_per_day"] * (1 - float(v))),
}


def _generic(p: dict, key: str, v, a: RetrofitAction) -> None:
    """Contract rule: a number sets the field, 'x<k>' multiplies it, another string sets a categorical field.
    A sized action (e.g. PV kWp) overrides the catalogue number."""
    if key not in p:
        raise KeyError(f"twin edit {key!r} is neither a TwinParams field nor in retrofits.KEY_MAP")
    if isinstance(v, str) and v.startswith("x"):
        factor = float(v[1:])
        if key == "lighting_share_of_baseload":  # LED: lighting energy falls, so baseload falls with it
            share = p[key]
            p["baseload_kwh_per_day"] = p["baseload_kwh_per_day"] * (1 - share + share * factor)
            p[key] = share * factor / (1 - share + share * factor)
        else:
            p[key] = p[key] * factor
    elif isinstance(v, str):
        p[key] = v
    else:
        p[key] = float(a.size) if a.size is not None else float(v)


def apply_retrofits(params: TwinParams, actions: Sequence[RetrofitAction]) -> TwinParams:
    """New parameter set with each action's twin edits applied in order. Unknown retrofits or edits raise."""
    p = params.model_dump()
    items = catalogue()
    for a in actions:
        if a.retrofit_id not in items:
            raise KeyError(f"unknown retrofit {a.retrofit_id!r}")
        for key, v in items[a.retrofit_id]["twin_edits"].items():
            (KEY_MAP[key] if key in KEY_MAP else lambda q, val, act, k=key: _generic(q, k, val, act))(p, v, a)
    return type(params).model_validate(p)
