"""params_from_property and weather: every field filled and sourced, sensible values, documented fallbacks."""

from pathlib import Path

import pytest

from contracts.models import Property, TwinParams, WeatherMonth
from greentwin.twin.archetypes import build_layout
from greentwin.twin.params import (
    _loft_u,
    age_band_index,
    choose_archetype,
    floor_u_iso13370,
    fuel_config,
    params_with_sources,
    sap_occupancy,
)
from greentwin.twin.weather import resolve_weather, typical_year, weather_arrays

EX = Path(__file__).resolve().parents[1] / "contracts" / "examples"


def _prop(pid):
    return Property.model_validate_json((EX / f"{pid}_property.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("pid", ["SH1", "SH2", "SH3"])
def test_showcase_params_complete_and_sourced(pid):
    params, src = params_with_sources(_prop(pid))
    TwinParams.model_validate(params.model_dump())
    numeric = [k for k, v in params.model_dump().items() if isinstance(v, float)]
    assert set(numeric) <= set(src), f"unsourced fields: {set(numeric) - set(src)}"
    assert params.volume_m3 > 0 and params.wall_area_m2 > 0 and params.window_area_m2 > 0
    assert 0.2 <= params.air_changes_per_hour <= 1.5
    assert 0 < params.hot_water_kwh_per_day < 10 and 3 < params.baseload_kwh_per_day < 20


def test_showcase_specifics():
    sh1, _ = params_with_sources(_prop("SH1"))
    sh3, src3 = params_with_sources(_prop("SH3"))
    assert sh1.wall_u == pytest.approx(2.1)            # 1900-1929 solid brick, uninsulated
    assert sh1.roof_u == pytest.approx(0.68)           # 50 mm loft
    assert sh1.window_u == pytest.approx(0.6 * 4.8 + 0.4 * 2.8)
    assert sh1.party_wall_area_m2 > 0                  # mid-terrace
    assert sh3.roof_area_m2 == 0 and sh3.floor_area_m2 == 0 and sh3.heating_fuel == "electricity"
    assert src3["roof_u"].startswith("epc")
    assert fuel_config(_prop("SH3")).hot_water_fuel == "electricity"


def test_layout_from_floor_plan_is_used():
    layout = build_layout("interwar_semi_3bed", total_floor_area_m2=95)
    _, src = params_with_sources(_prop("SH2"), layout=layout)
    assert "floor plan" in src["wall_area_m2"]


def test_helpers():
    assert age_band_index("1900-1929") == 1 and age_band_index("before 1900") == 0 and age_band_index("2012 onwards") == 11
    assert age_band_index("unknown") == 4
    assert _loft_u(100) == pytest.approx(0.40) and 0.40 > _loft_u(110) > 0.35
    assert sap_occupancy(10) == 1 and 2 < sap_occupancy(85) < 3
    assert floor_u_iso13370(50, 20, False) > floor_u_iso13370(50, 20, True) > 0
    assert floor_u_iso13370(100, 20, False) < floor_u_iso13370(50, 20, False)  # bigger floor, lower U
    assert choose_archetype(_prop("SH1")) == "victorian_terrace_2bed"
    assert choose_archetype(_prop("SH3")) == "flat_2bed"


def test_weather_typical_year_and_filling():
    year = typical_year("M14")
    assert len(year) == 12 and all(w.source == "synthetic" for w in year)
    arr = weather_arrays(year)
    assert arr["hdd_15_5"][0] > arr["hdd_15_5"][6] >= 0 and arr["days_in_month"].sum() == 365
    gappy = [WeatherMonth(area_code="M14", month="2025-01", hdd_15_5=340.0, source="haduk_grid")]
    filled = weather_arrays(gappy)
    assert filled["mean_temp_c"][0] == pytest.approx(15.5 + 1.0 - 340 / 31) and filled["ghi_kwh_m2"][0] > 0
    _, label = resolve_weather("M14")
    assert label == "synthetic"
