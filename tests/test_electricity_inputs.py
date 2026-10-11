from pathlib import Path

import numpy as np
import pytest

from contracts.electricity import ElectricityIntake, ElectricitySeries
from contracts.models import Property
from greentwin.twin.electricity_inputs import prepare_electricity_inputs, timeline
from greentwin.twin.params import params_from_property
from greentwin.twin.weather import typical_year

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def home():
    prop = Property.model_validate_json((ROOT / "contracts/examples/SH2_property.json").read_text())
    return params_from_property(prop), typical_year("BS16"), prop.location


def test_generated_inputs_reproducible_and_match_monthly_weather(home):
    params, weather, location = home
    a = prepare_electricity_inputs(
        params, weather, development=True, latitude=location.lat, longitude=location.lon
    )
    b = prepare_electricity_inputs(
        params, weather, development=True, latitude=location.lat, longitude=location.lon
    )
    assert a == b
    assert len(a.timestamps_utc) == 365 * 48
    index = np.array(a.month_index)
    for i, record in enumerate(weather):
        assert np.mean(np.array(a.temperature_c)[index == i]) == pytest.approx(record.mean_temp_c)
    night = (np.array(a.local_hour) < 4) | (np.array(a.local_hour) > 23)
    assert np.all(np.array(a.pv_kwh_per_kwp)[night] == 0)
    assert any("synthetic" in value for value in a.sources.values())


def test_dst_leap_year_and_month_coverage():
    assert len(timeline(("2025-03",))[0]) == (31 * 24 - 1) * 2
    assert len(timeline(("2025-10",))[0]) == (31 * 24 + 1) * 2
    assert len(timeline(("2024-02",))[0]) == 29 * 48
    with pytest.raises(ValueError, match="consecutive"):
        timeline(("2025-01", "2025-03"))


def test_sourced_mode_does_not_invent_weather_or_equipment(home):
    params, weather, _ = home
    with pytest.raises(ValueError, match="sourced"):
        prepare_electricity_inputs(params, weather)
    a = prepare_electricity_inputs(params, weather, development=True)
    series = ElectricitySeries(
        timestamps_utc=a.timestamps_utc,
        temperature_c=a.temperature_c,
        pv_kwh_per_kwp=a.pv_kwh_per_kwp,
        temperature_source="synthetic",
        pv_source="synthetic",
    )
    with pytest.raises(ValueError, match="development"):
        prepare_electricity_inputs(params, weather, series=series)
    series.temperature_source, series.pv_source = "supplied ERA5 series", "supplied PVGIS series"
    with pytest.raises(ValueError, match="Battery needs"):
        prepare_electricity_inputs(params.model_copy(update={"battery_kwh": 10}), weather, series=series)
    with pytest.raises(ValueError, match="inverter"):
        prepare_electricity_inputs(params.model_copy(update={"pv_kwp": 4}), weather, series=series)


def test_profile_and_schedule_validation(home):
    params, weather, _ = home
    with pytest.raises(ValueError):
        ElectricityIntake(heating_periods=[(6, 9), (8, 10)])
    with pytest.raises(ValueError):
        prepare_electricity_inputs(params, weather, development=True, appliance_hourly_profile_kw=[0] * 24)
    inputs = prepare_electricity_inputs(
        params.model_copy(update={"pv_kwp": 4, "battery_kwh": 10}),
        weather,
        development=True,
        intake=ElectricityIntake(heating_periods=[(22, 6)]),
    )
    assert inputs.battery.charge_kw == 5
    assert inputs.battery.charge_efficiency * inputs.battery.discharge_efficiency < 1
    assert np.all(np.asarray(inputs.heating_on_fraction)[np.asarray(inputs.local_hour) == 12.25] == 0)
