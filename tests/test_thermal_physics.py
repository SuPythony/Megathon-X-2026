"""Analytic balances, physical response and the differentiable integration seam."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from contracts.models import Property, TwinParams, WeatherMonth
from greentwin.twin.generate import GeneratorConfig, Simulator, generate_dataset
from greentwin.twin.params import fuel_config, params_from_property
from greentwin.twin.physics import VERSION, ThermalAssumptions, run_physics, simulate_thermal
from greentwin.twin.physics_api import FuelConfig
from greentwin.twin.weather import typical_year

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def params():
    return {
        "wall_area_m2": 100.0,
        "roof_area_m2": 50.0,
        "floor_area_m2": 50.0,
        "window_area_m2": 20.0,
        "door_area_m2": 4.0,
        "party_wall_area_m2": 60.0,
        "wall_u": 1.0,
        "roof_u": 0.2,
        "floor_u": 0.3,
        "window_u": 2.0,
        "door_u": 2.0,
        "thermal_bridging_y": 0.1,
        "air_changes_per_hour": 0.5,
        "volume_m3": 250.0,
        "heat_loss_multiplier": 1.0,
        "setpoint_c": 20.0,
        "heating_hours_per_day": 24.0,
        "heating_efficiency": 0.8,
        "hot_water_kwh_per_day": 2.0,
        "baseload_kwh_per_day": 4.0,
        "lighting_share_of_baseload": 0.15,
        "internal_gains_w": 200.0,
        "solar_gain_factor": 1.0,
        "pv_kwp": 0.0,
        "pv_azimuth_deg": 180.0,
        "pv_tilt_deg": 35.0,
        "battery_kwh": 0.0,
    }


@pytest.fixture
def weather():
    return {"days_in_month": np.array([30.0]), "mean_temp_c": np.array([5.0]), "ghi_kwh_m2": np.array([0.0])}


GAS = FuelConfig("mains_gas", "mains_gas")


def test_analytic_ua_ventilation_bridges_and_kwh(params, weather):
    result = run_physics(params, weather, "uncalibrated", GAS)
    # Independent hand calculation: UA=173, yA=22.4, ventilation=41.25 W/K.
    expected_h = 236.65
    expected_loss = expected_h * 15 * 720 / 1000
    expected_heat = expected_loss - 200 * 720 / 1000
    assert float(result["heat_loss_coefficient_w_per_k"]) == pytest.approx(expected_h)
    assert float(result["thermal_bridging_w_per_k"]) == pytest.approx(22.4)
    assert float(result["space_heating_kwh"]) == pytest.approx(expected_heat)
    assert float(result["gas_kwh"]) == pytest.approx((expected_heat + 60) / 0.8)
    assert float(result["electricity_import_kwh"]) == pytest.approx(120)
    assert float(result["element_heat_loss_w_per_k"]["party_walls"]) == 0
    assert sum(float(x[0]) for x in result["element_monthly_heat_loss_kwh"].values()) == pytest.approx(
        expected_loss
    )


def test_setback_two_balances_not_simple_duty_scaling(params, weather):
    full = run_physics(params, weather, "uncalibrated", GAS)
    setback = run_physics(params | {"heating_hours_per_day": 0.0}, weather, "uncalibrated", GAS)
    half = run_physics(params | {"heating_hours_per_day": 12.0}, weather, "uncalibrated", GAS)
    assert float(setback["space_heating_kwh"]) > 0  # maintained setback, not switched off
    assert float(half["space_heating_kwh"]) == pytest.approx(
        (float(full["space_heating_kwh"]) + float(setback["space_heating_kwh"])) / 2
    )
    assert float(half["space_heating_kwh"]) > 0.5 * float(full["space_heating_kwh"])


def test_heat_balance_across_season_gains_and_schedule(params):
    weather = {
        "days_in_month": np.full(12, 30.0),
        "mean_temp_c": np.linspace(-5, 30, 12),
        "ghi_kwh_m2": np.linspace(0, 150, 12),
        "baseload_multiplier": np.linspace(0.5, 1.5, 12),
    }
    r = run_physics(params | {"heating_hours_per_day": 9.0}, weather, "uncalibrated", GAS)
    np.testing.assert_allclose(
        r["monthly_space_heating_kwh"] + r["monthly_heat_gains_kwh"],
        r["monthly_heat_loss_kwh"] + r["monthly_unused_heat_gains_kwh"],
        rtol=2e-6,
    )
    assert r["monthly_space_heating_kwh"][-1] == 0
    assert r["monthly_unused_heat_gains_kwh"][-1] > 0
    assert r["monthly_heat_loss_kwh"][-1] == 0
    assert np.all(np.asarray(r["monthly_space_heating_kwh"]) >= 0)


@pytest.mark.parametrize("fuel_name", ["mains_gas", "electricity", "oil", "lpg", "solid", "district"])
def test_all_fuels_and_conversion_balance(params, weather, fuel_name):
    efficiency = 3.0 if fuel_name == "electricity" else 0.8
    r = run_physics(
        params | {"heating_efficiency": efficiency}, weather, "uncalibrated", FuelConfig(fuel_name, fuel_name)
    )
    heat = float(r["space_heating_kwh"] + r["hot_water_kwh"])
    if fuel_name == "electricity":
        purchased = float(r["electricity_import_kwh"]) - 120
    elif fuel_name == "mains_gas":
        purchased = float(r["gas_kwh"])
    else:
        purchased = float(r[f"{fuel_name}_kwh"])
    assert purchased == pytest.approx(heat / efficiency)
    np.testing.assert_allclose(
        r["monthly_heating_fuel_kwh"] + r["monthly_heating_ambient_heat_kwh"],
        r["monthly_space_heating_kwh"] + r["monthly_heating_conversion_loss_kwh"],
        rtol=2e-6,
    )


def test_separate_immersion_and_profile_multipliers(params, weather):
    weather |= {"hot_water_multiplier": np.array([1.5]), "baseload_multiplier": np.array([2.0])}
    r = run_physics(params, weather, "uncalibrated", FuelConfig("mains_gas", "electricity", 1.0))
    assert float(r["hot_water_kwh"]) == pytest.approx(90)
    assert float(r["electricity_import_kwh"]) == pytest.approx(240 + 90)
    assert float(r["gas_kwh"]) == pytest.approx(float(r["space_heating_kwh"]) / 0.8)
    # Appliance retention rises by 0.85*4 kWh/day; the occupant part stays fixed.
    assert float(r["monthly_internal_gains_kwh"][0]) == pytest.approx(144 + 0.85 * 4 * 30)


def test_monotonic_fabric_weather_gains_and_efficiency(params, weather):
    baseline = float(run_physics(params, weather, "uncalibrated", GAS)["space_heating_kwh"])
    for key in (
        "wall_u",
        "roof_u",
        "floor_u",
        "window_u",
        "door_u",
        "air_changes_per_hour",
        "heat_loss_multiplier",
        "thermal_bridging_y",
    ):
        lower = float(
            run_physics(params | {key: params[key] / 2}, weather, "uncalibrated", GAS)["space_heating_kwh"]
        )
        assert lower < baseline
    colder = float(
        run_physics(params, weather | {"mean_temp_c": np.array([0.0])}, "uncalibrated", GAS)[
            "space_heating_kwh"
        ]
    )
    assert colder > baseline
    gains = float(
        run_physics(params | {"internal_gains_w": 400.0}, weather, "uncalibrated", GAS)["space_heating_kwh"]
    )
    assert gains < baseline
    efficient = run_physics(params | {"heating_efficiency": 0.95}, weather, "uncalibrated", GAS)
    assert float(efficient["space_heating_kwh"]) == pytest.approx(baseline)
    assert float(efficient["gas_kwh"]) < float(run_physics(params, weather, "uncalibrated", GAS)["gas_kwh"])


def test_jit_vmap_and_analytic_gradients(params, weather):
    compiled = jax.jit(run_physics, static_argnames=("mode", "fuel", "assumptions"))
    eager = run_physics(params, weather, "uncalibrated", GAS)
    result = compiled(params, weather, "uncalibrated", GAS)
    np.testing.assert_allclose(result["gas_kwh"], eager["gas_kwh"], rtol=2e-6)
    sweep = jnp.array([0.2, 0.5, 1.0, 2.0])
    batch = jax.jit(
        jax.vmap(
            lambda u: run_physics(params | {"wall_u": u}, weather, "uncalibrated", GAS)["space_heating_kwh"]
        )
    )(sweep)
    assert np.all(np.diff(np.asarray(batch)) > 0)
    gradient = jax.grad(
        lambda u: run_physics(params | {"wall_u": u}, weather, "uncalibrated", GAS)["space_heating_kwh"]
    )(1.0)
    assert float(gradient) == pytest.approx(100 * 15 * 720 / 1000)
    all_gradients = jax.grad(lambda p: run_physics(p, weather, "uncalibrated", GAS)["gas_kwh"])(
        {key: jnp.asarray(value) for key, value in params.items()}
    )
    assert all(np.isfinite(value) for value in all_gradients.values())
    assert all_gradients["heating_efficiency"] < 0
    assert all_gradients["setpoint_c"] > 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("pv_kwp", 2.0),
        ("battery_kwh", 4.0),
        ("wall_u", -1.0),
        ("heating_efficiency", 0.0),
        ("heating_efficiency", 1.2),
        ("heating_hours_per_day", 25.0),
        ("heat_loss_multiplier", 0.0),
        ("volume_m3", 0.0),
        ("setpoint_c", np.nan),
        ("solar_gain_factor", 3.0),
    ],
)
def test_invalid_domains_raise_eager_and_fail_closed_jitted(params, weather, key, value):
    with pytest.raises(ValueError):
        run_physics(params | {key: value}, weather, "uncalibrated", GAS)
    compiled = jax.jit(run_physics, static_argnames=("mode", "fuel"))
    result = compiled(params | {key: value}, weather, "uncalibrated", GAS)
    assert np.isnan(float(result["gas_kwh"]))


def test_shapes_weather_and_mode_rejected(params, weather):
    for bad in (
        weather | {"mean_temp_c": np.array([1.0, 2.0])},
        weather | {"days_in_month": np.array([0.0])},
        weather | {"ghi_kwh_m2": np.array([np.nan])},
        weather | {"baseload_multiplier": np.array([-1.0])},
    ):
        with pytest.raises(ValueError):
            run_physics(params, bad, "uncalibrated", GAS)
    with pytest.raises(ValueError, match="standard"):
        run_physics(params, weather, "standard", GAS)
    with pytest.raises(ValueError, match="explicit"):
        run_physics(params, weather, "uncalibrated", FuelConfig("mains_gas", "electricity"))


def test_contract_facade_and_synthetic_pipeline():
    properties = [
        Property.model_validate_json(path.read_text())
        for path in sorted((ROOT / "contracts/examples").glob("SH*_property.json"))
    ]
    for prop in properties:
        result = simulate_thermal(
            params_from_property(prop),
            typical_year(prop.location.postcode_district),
            fuel_config(prop),
            data_version="showcase-test",
        )
        assert result.calibration_status == "uncalibrated"
        assert result.provenance.engine == "physics_l1"
        assert result.weather_sources == ["synthetic"]
        assert len(result.months) == 12
        assert sum(element.share_pct for element in result.elements) == pytest.approx(100)
        assert max(result.conservation_max_error_kwh.values()) < 0.01
        assert all(element.estimated for element in result.elements)
        assert len(result.model_dump_json()) > 0
    config = GeneratorConfig(development=True, n_homes=30, n_months=24)
    dataset = generate_dataset(properties, config, simulator=Simulator(run_physics, VERSION))
    assert all(home.baseline is not None and len(home.observations) == 24 for home in dataset.homes)
    assert len({home.baseline.totals["space_heating_kwh"] for home in dataset.homes}) == 30
    assert all(max(home.baseline.conservation_max_error_kwh.values()) < 0.02 for home in dataset.homes)


def test_facade_does_not_impute_or_mix_weather(params):
    p = TwinParams(property_id="test", heating_fuel="mains_gas", **params)
    w = WeatherMonth(
        area_code="M14", month="2025-01", hdd_15_5=100, mean_temp_c=5, ghi_kwh_m2=10, source="synthetic"
    )
    for records in (
        [w, w],
        [w.model_copy(update={"mean_temp_c": None})],
        [w, w.model_copy(update={"month": "2025-02", "area_code": "M15"})],
    ):
        with pytest.raises(ValueError):
            simulate_thermal(p, records, GAS)
    with pytest.raises(ValueError):
        simulate_thermal(p, [w], replace(GAS, heating_fuel="electricity"))
    with pytest.raises(ValueError):
        ThermalAssumptions(solar_aperture_factor=1.1)


def test_independent_heat_pump_dhw_requires_own_cop():
    prop = Property.model_validate_json((ROOT / "contracts/examples/SH1_property.json").read_text())
    prop.heating.hot_water = "heat_pump_cylinder"
    with pytest.raises(ValueError, match="DHW COP"):
        fuel_config(prop)
    config = fuel_config(prop, hot_water_efficiency=2.5)
    assert config.hot_water_fuel == "electricity"
    assert config.hot_water_efficiency == 2.5


def test_generator_rejects_unmodelled_solar_thermal():
    prop = Property.model_validate_json((ROOT / "contracts/examples/SH1_property.json").read_text())
    prop.renewables.solar_thermal = True
    with pytest.raises(ValueError, match="solar_thermal"):
        generate_dataset(
            [prop], GeneratorConfig(development=True, n_homes=1), simulator=Simulator(run_physics, VERSION)
        )


def test_cli_runs_and_cannot_write_outside_or_overwrite(tmp_path, capsys):
    from scripts.run_thermal_twin import main

    path = tmp_path / "thermal.json"
    main(["--demo", "SH3", "--output", str(path)])
    original = path.read_bytes()
    with pytest.raises(SystemExit) as exc:
        main(["--demo", "SH3", "--output", str(path)])
    assert exc.value.code == 2
    assert path.read_bytes() == original
    outside = ROOT.parent / "thermal-test-must-not-create.json"
    with pytest.raises(SystemExit) as exc:
        main(["--demo", "SH3", "--output", str(outside)])
    assert exc.value.code == 2
    assert "inside" in capsys.readouterr().err
    assert not outside.exists()
