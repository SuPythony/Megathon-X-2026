"""Independent checks of the uncertainty boundary, replay and coupled fuel balance."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from contracts.models import Property
from contracts.uncertain_twin import ContextRecord, HouseholdRequest
from greentwin.twin.ensemble import paired_differences, simulate_ensemble
from greentwin.twin.ensemble_io import load_inputs, save_inputs
from greentwin.twin.missing_inputs import generate_missing_inputs
from greentwin.twin.priors import build_priors

ROOT = Path(__file__).resolve().parents[1]


def request(**facts):
    prop = Property.model_validate_json((ROOT / "contracts/examples/SH2_property.json").read_text())
    return HouseholdRequest(
        property_id="SH2",
        property=prop,
        months=["2025-01", "2025-02"],
        user_inputs=facts,
        allow_development_weather=True,
    )


def test_prepare_import_does_not_load_jax():
    result = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            "-B",
            "-c",
            "import sys; from greentwin.twin.missing_inputs import generate_missing_inputs; assert 'jax' not in sys.modules",
        ],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_supplied_values_exact_missing_values_uncertain():
    req = request(
        occupants=4,
        thermostat_c=20,
        wall_u=0.3,
        heating_efficiency=0.91,
        baseload_kwh_per_day=5,
        gas_cooking_kwh_per_day=1,
    )
    before = req.model_dump_json()
    samples = generate_missing_inputs(req, 32)
    for key, value in {
        "setpoint_c": 20,
        "wall_u": 0.3,
        "heating_efficiency": 0.91,
        "baseload_kwh_per_day": 5,
    }.items():
        np.testing.assert_array_equal(samples.parameters[key], value)
        assert samples.summary.parameters[key].variance == 0
    assert np.std(samples.parameters["air_changes_per_hour"]) > 0
    assert (samples.parameters["hot_water_efficiency"] <= 1).all()
    assert (samples.parameters["wall_area_m2"] > 0).all()
    np.testing.assert_array_equal(samples.weather["extra_gas_kwh_monthly"], np.tile([31, 28], (32, 1)))
    assert req.model_dump_json() == before
    assert samples.summary.parameters["wall_u"].unit == "W/m2K"


def test_floor_area_total_is_not_exposed_floor():
    s = generate_missing_inputs(request(floor_area_m2=120, storeys=2), 8)
    np.testing.assert_array_equal(s.parameters["floor_area_m2"], 60)
    s2 = generate_missing_inputs(request(floor_area_m2=120, exposed_floor_area_m2=10), 8)
    np.testing.assert_array_equal(s2.parameters["floor_area_m2"], 10)


def test_appliances_route_outdoor_secondary_and_immersion_separately():
    s = generate_missing_inputs(
        request(
            appliances=[
                {"type": "ev_charger"},
                {"type": "electric_heater"},
                {"type": "immersion_heater", "answers": {"jacket": "none"}},
            ]
        ),
        8,
    )
    assert (s.weather["extra_electricity_kwh_monthly"] > 0).all()
    assert (s.weather["secondary_heating_kwh_monthly"] > 0).all()
    assert s.prior.equipment["hot_water_fuel"] == "electricity"
    np.testing.assert_allclose(s.parameters["hot_water_efficiency"], 1)
    np.testing.assert_allclose(
        s.parameters["hot_water_standing_loss_kwh_per_day"], 3 * s.constants["cylinder_loss_multiplier"]
    )


def test_supplied_constants_are_held_exact():
    s = generate_missing_inputs(
        request(
            retained_electric_heat_fraction=0.7, water_temperature_rise_c=40, heating_auxiliary_power_kw=0.05
        ),
        8,
    )
    np.testing.assert_array_equal(s.constants["water_temperature_rise_c"], 40)
    np.testing.assert_array_equal(s.parameters["retained_electric_heat_fraction"], 0.7)


@pytest.mark.parametrize(
    "facts",
    [
        {"wall_u": -1},
        {"occupants": 1.5},
        {"has_battery": "no"},
        {"gas_cooking_kwh_per_day": -1},
        {"invented_field": 1},
        {"thermostat_c": 100},
        {"has_solar_pv": False, "pv_kwp": 2},
    ],
)
def test_invalid_facts_fail_explicitly(facts):
    with pytest.raises((ValueError, TypeError)):
        generate_missing_inputs(request(**facts), 8)


@pytest.mark.parametrize("position", ["ground", "middle", "top"])
def test_basic_flat_input_preserves_exposure(position):
    req = HouseholdRequest(
        property_id="NEW",
        months=["2025-01"],
        allow_development_weather=True,
        user_inputs={
            "postcode_district": "BS16",
            "floor_area_m2": 60,
            "property_type": "flat",
            "heating_system": "gas_boiler_combi",
            "flat_position": position,
            "has_solar_pv": False,
            "has_battery": False,
        },
    )
    s = generate_missing_inputs(req, 8)
    np.testing.assert_array_equal(s.parameters["roof_area_m2"], 60 if position == "top" else 0)
    np.testing.assert_array_equal(s.parameters["floor_area_m2"], 60 if position == "ground" else 0)
    assert (s.parameters["roof_u"] > 0).all()


def test_context_pooling_filters_and_avoids_aggregate_double_count():
    req = request()
    records = [
        ContextRecord(
            variable="air_changes_per_hour",
            values=[0.2, 0.25, 0.3, 0.35],
            area_code="BS16",
            year=2025,
            source="development local test fixture",
            status="development",
        ),
        ContextRecord(
            variable="air_changes_per_hour",
            values=[1, 1.1],
            region="South West",
            year=2025,
            source="development regional fixture",
            status="development",
        ),
        ContextRecord(variable="wall_u", values=[0.3, 0.4], year=2027, source="future fixture"),
        ContextRecord(variable="roof_u", values=[0.3, 0.4], year=2025, source="held out", split="test"),
    ]
    informed = build_priors(req.model_copy(update={"context": records}))
    assert (
        informed.marginals["air_changes_per_hour"].centre
        < build_priors(req).marginals["air_changes_per_hour"].centre
    )
    reasons = {a["source"]: a["reason"] for a in informed.context_audit}
    assert "overlapping" in reasons["development regional fixture"]
    assert "future" in reasons["future fixture"]
    assert "held-out" in reasons["held out"]
    exact = build_priors(
        request(wall_u=0.8).model_copy(
            update={
                "context": [ContextRecord(variable="wall_u", values=[0.2, 0.3], year=2025, source="fixture")]
            }
        )
    )
    assert exact.marginals["wall_u"].centre == 0.8


def test_seed_and_common_random_scenario_streams():
    a = generate_missing_inputs(request(), 8)
    b = generate_missing_inputs(request(wall_u=0.3), 8)
    c = generate_missing_inputs(request().model_copy(update={"seed": 43}), 8)
    for key in ("volume_m3", "window_area_m2", "hot_water_kwh_per_day"):
        np.testing.assert_array_equal(a.parameters[key], b.parameters[key])
    assert not np.array_equal(a.parameters["volume_m3"], c.parameters["volume_m3"])
    np.testing.assert_array_equal(
        a.parameters["wall_u"], generate_missing_inputs(request(), 16).parameters["wall_u"][:8]
    )


@pytest.fixture(scope="module")
def run():
    return simulate_ensemble(
        generate_missing_inputs(request(occupants=4, thermostat_c=20, gas_cooking_kwh_per_day=1), 8),
        batch_size=4,
        include_intervals=True,
    )


def test_output_moments_and_every_realisation_conserves(run):
    a = run.samples["gas_kwh"].astype(float)
    d = run.result.totals["gas_kwh"]
    assert d.mean == pytest.approx(a.mean())
    assert d.variance == pytest.approx(a.var(ddof=1))
    assert d.monte_carlo_standard_error == pytest.approx(a.std(ddof=1) / np.sqrt(len(a)))
    assert run.result.calibration_status == "not_performed"
    assert run.result.annual_totals == {}
    assert max(run.result.max_conservation_error_kwh.values()) < 0.05
    np.testing.assert_allclose(
        run.samples["interval_demand_kwh"].sum(axis=1), run.samples["electricity_demand_kwh"], rtol=2e-6
    )


def test_batch_replay_and_paired_retrofit(run):
    req = request(occupants=4, thermostat_c=20, gas_cooking_kwh_per_day=1)
    other = simulate_ensemble(generate_missing_inputs(req, 8), batch_size=8)
    np.testing.assert_allclose(run.samples["gas_kwh"], other.samples["gas_kwh"], rtol=2e-6)
    upgraded = simulate_ensemble(
        generate_missing_inputs(
            request(occupants=4, thermostat_c=20, gas_cooking_kwh_per_day=1, wall_u=0.1), 8
        ),
        batch_size=8,
    )
    assert (upgraded.samples["space_heating_kwh"] <= run.samples["space_heating_kwh"]).all()
    assert paired_differences(run, upgraded)["gas_kwh"].mean > 0


@pytest.mark.parametrize(
    "system,fuel",
    [
        ("oil_boiler", "oil"),
        ("lpg_boiler", "lpg"),
        ("solid_fuel", "solid"),
        ("district_heating", "district"),
        ("air_source_heat_pump", "electricity"),
    ],
)
def test_fuel_switch_and_retained_cooking(system, fuel):
    s = generate_missing_inputs(
        request(heating_system=system, gas_cooking_kwh_per_day=1, cooking_fuel="mains_gas"), 4
    )
    r = simulate_ensemble(s, batch_size=4)
    np.testing.assert_allclose(r.samples["gas_kwh"], 59, atol=1e-5)
    expected = r.samples["space_heating_kwh"] / s.parameters["heating_efficiency"]
    np.testing.assert_allclose(r.samples["monthly_heating_fuel_kwh"].sum(axis=1), expected, rtol=2e-6)
    if fuel != "electricity":
        assert (r.samples[f"{fuel}_kwh"] > 0).all()
    np.testing.assert_allclose(
        r.samples["all_fuel_purchased_kwh"],
        sum(
            r.samples[k]
            for k in ["electricity_import_kwh", "gas_kwh", "oil_kwh", "lpg_kwh", "solid_kwh", "district_kwh"]
        ),
        rtol=2e-6,
    )


def test_artifact_replay_and_tamper_detection():
    import tempfile

    with tempfile.TemporaryDirectory(dir=ROOT / ".pytest_tmp") as directory:
        out = Path(directory) / "prepared"
        s = generate_missing_inputs(request(), 8)
        save_inputs(s, out)
        restored = load_inputs(out)
        for key, a in s.parameters.items():
            np.testing.assert_array_equal(a, restored.parameters[key])
        with pytest.raises(ValueError):
            save_inputs(s, out)
        (out / "request.json").write_text(json.dumps({"corrupted": True}))
        with pytest.raises(ValueError, match="hash mismatch"):
            load_inputs(out)
