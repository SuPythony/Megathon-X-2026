"""Independent electricity/storage balances and actual thermal integration."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from contracts.electricity import ElectricityIntake
from contracts.models import Property
from greentwin.twin.electricity import _dispatch
from greentwin.twin.electricity_inputs import development_electricity_factory, prepare_electricity_inputs
from greentwin.twin.electricity_simulation import simulate_electricity, validate_interval_output
from greentwin.twin.generate import GeneratorConfig, RetrofitScenario, Simulator, generate_dataset
from greentwin.twin.params import params_from_property
from greentwin.twin.physics import VERSION, run_physics
from greentwin.twin.physics_api import PHYSICS_PARAM_KEYS, FuelConfig
from greentwin.twin.satyarth_inputs import adapt_satyarth_package
from greentwin.twin.weather import typical_year, weather_arrays

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def home():
    prop = Property.model_validate_json((ROOT / "contracts/examples/SH2_property.json").read_text())
    p = params_from_property(prop)
    w = typical_year("BS16")[:2]
    return prop, p, w


def prepared(p, w, **kwargs):
    return prepare_electricity_inputs(p, w, development=True, latitude=51.5, longitude=-2.5, **kwargs)


def raw_run(p, w, inputs, fuel=None):
    numeric = {key: getattr(p, key) for key in PHYSICS_PARAM_KEYS}
    arrays = weather_arrays(w) | {"electricity_inputs": inputs.numeric_arrays()}
    return run_physics(numeric, arrays, "uncalibrated", fuel or FuelConfig("mains_gas", "mains_gas"))


def test_battery_hand_calculation_and_power_limits():
    r = _dispatch(
        jnp.array([0.0, 1.0]),
        jnp.array([2.0, 0.0]),
        jnp.array([0.5, 0.5]),
        2.0,
        10.0,
        10.0,
        0.8,
        0.8,
        0.0,
        100.0,
    )
    np.testing.assert_allclose(r["charge"], [2, 0])
    np.testing.assert_allclose(r["discharge"], [0, 1])
    np.testing.assert_allclose(r["stored"], [1.6, 0.35], atol=1e-6)
    np.testing.assert_allclose(r["battery_loss"], [0.4, 0.25], atol=1e-6)
    limited = _dispatch(
        jnp.array([0.0, 5.0]),
        jnp.array([5.0, 0.0]),
        jnp.array([0.5, 0.5]),
        10.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
    )
    np.testing.assert_allclose(limited["charge"], [0.5, 0])
    np.testing.assert_allclose(limited["discharge"], [0, 0.5])
    np.testing.assert_allclose(limited["curtailment"], [4.5, 0])
    np.testing.assert_allclose(limited["import"], [0, 4.5])


def test_no_pv_battery_matches_monthly_thermal(home):
    _, p, w = home
    inputs = prepared(p, w)
    r = simulate_electricity(p, w, FuelConfig("mains_gas", "mains_gas"), inputs)
    baseline = run_physics(
        {key: getattr(p, key) for key in PHYSICS_PARAM_KEYS},
        weather_arrays(w),
        "uncalibrated",
        FuelConfig("mains_gas", "mains_gas"),
    )
    assert r.totals["gas_kwh"] == pytest.approx(float(baseline["gas_kwh"]))
    assert r.totals["electricity_import_kwh"] == pytest.approx(
        float(baseline["electricity_import_kwh"]), rel=2e-5
    )
    assert r.totals["pv_generation_kwh"] == 0
    assert r.self_consumption_pct is None
    assert r.self_sufficiency_pct == 0
    assert r.element_heat_loss_estimated is True
    assert r.import_bill_gbp is None  # 2025 is outside the 2026Q4 tariff
    assert r.time_of_use_bill_gbp is None
    assert r.calibration_status == "uncalibrated"


def test_pv_battery_curtailment_conservation_and_continuity(home):
    _, p, w = home
    p = p.model_copy(update={"pv_kwp": 8.0, "battery_kwh": 10.0})
    inputs = prepared(
        p, w, intake=ElectricityIntake(export_limit_kw=0.0, battery_charge_kw=1.0, battery_discharge_kw=1.0)
    )
    r = simulate_electricity(p, w, FuelConfig("mains_gas", "mains_gas"), inputs, include_intervals=True)
    assert r.totals["export_kwh"] == 0
    assert sum(r.monthly["monthly_curtailment_kwh"]) > 0
    assert sum(r.monthly["monthly_battery_charge_kwh"]) > 0
    assert sum(r.monthly["monthly_battery_loss_kwh"]) > 0
    assert max(r.conservation_max_error_kwh.values()) < 0.02
    assert all(-1e-6 <= value <= 9.0002 for value in r.intervals["interval_stored_kwh"])  # float32 rounding
    # Continuous telescoping storage through the month boundary, not reset to zero.
    stored = np.asarray(r.intervals["interval_stored_kwh"])
    delta = np.asarray(r.intervals["interval_battery_delta_kwh"])
    np.testing.assert_allclose(stored, np.cumsum(delta), atol=2e-4)


def test_initial_energy_not_created_from_nothing(home):
    _, p, w = home
    p = p.model_copy(update={"battery_kwh": 10.0})
    inputs = prepared(p, w, intake=ElectricityIntake(initial_battery_fraction=0.5))
    r = simulate_electricity(p, w, FuelConfig("mains_gas", "mains_gas"), inputs)
    assert r.battery_initial_kwh == pytest.approx(4.5)
    assert r.battery_final_kwh == pytest.approx(0, abs=1e-5)
    assert sum(r.monthly["monthly_battery_discharge_kwh"]) == pytest.approx(4.5 * 0.95, rel=1e-5)


def test_jit_vmap_and_gradients_of_coupled_engine(home):
    _, p, w = home
    p = p.model_copy(update={"pv_kwp": 4.0, "battery_kwh": 10.0})
    numeric = {key: float(getattr(p, key)) for key in PHYSICS_PARAM_KEYS}
    arrays = weather_arrays(w) | {"electricity_inputs": prepared(p, w).numeric_arrays()}
    f = FuelConfig("mains_gas", "mains_gas")
    compiled = jax.jit(run_physics, static_argnames=("mode", "fuel"))
    r = compiled(numeric, arrays, "uncalibrated", f)
    assert np.isfinite(float(r["electricity_import_kwh"]))
    batch = jax.jit(
        jax.vmap(
            lambda size: run_physics(numeric | {"pv_kwp": size}, arrays, "uncalibrated", f)[
                "electricity_import_kwh"
            ]
        )
    )(jnp.array([0.0, 2.0, 4.0]))
    assert np.all(np.diff(np.asarray(batch)) <= 0.01)
    gradient = jax.grad(
        lambda base: run_physics(numeric | {"baseload_kwh_per_day": base}, arrays, "uncalibrated", f)[
            "electricity_import_kwh"
        ]
    )(p.baseload_kwh_per_day)
    assert np.isfinite(float(gradient)) and gradient > 0
    bad = arrays | {
        "electricity_inputs": arrays["electricity_inputs"] | {"battery_charge_efficiency": jnp.array(1.2)}
    }
    assert np.isnan(float(compiled(numeric, bad, "uncalibrated", f)["electricity_import_kwh"]))


def test_interval_validation_catches_corruption(home):
    _, p, w = home
    p = p.model_copy(update={"pv_kwp": 4.0, "battery_kwh": 10.0})
    inputs = prepared(p, w)
    raw = raw_run(p, w, inputs)
    corrupt = raw | {"interval_stored_kwh": raw["interval_stored_kwh"] + 1}
    with pytest.raises(ValueError, match="storage_state"):
        validate_interval_output(corrupt, p, inputs)


def test_new_satyarth_end_uses_not_doubled(home):
    _, p, w = home
    inputs = prepared(p, w)
    profiles = {
        "secondary_heating_kwh_monthly": np.array([20.0, 20.0]),
        "extra_electricity_kwh_monthly": np.array([100.0, 100.0]),
        "extra_gas_kwh_monthly": np.array([5.0, 5.0]),
        "gains_multiplier": np.ones(2),
    }
    extras = {"hot_water_standing_loss_kwh_per_day": 1.0, "shower_electric_kwh_per_day": 0.5}
    r = simulate_electricity(
        p, w, FuelConfig("mains_gas", "mains_gas", 0.7), inputs, profiles=profiles, extra_params=extras
    )
    base = simulate_electricity(p, w, FuelConfig("mains_gas", "mains_gas", 0.7), inputs)
    expected_electric = base.totals["electricity_import_kwh"] + 200 + 40 + 0.5 * (31 + 28)
    assert r.totals["electricity_import_kwh"] == pytest.approx(expected_electric, rel=2e-5)
    expected_gas = base.totals["gas_kwh"] - 40 / p.heating_efficiency + (31 + 28) / 0.7 + 10
    assert r.totals["gas_kwh"] == pytest.approx(expected_gas, rel=2e-5)
    assert r.totals["space_heating_kwh"] == pytest.approx(base.totals["space_heating_kwh"])


def test_synthetic_generator_uses_same_prepared_household_for_packages(home):
    prop, _, _ = home

    def scenarios(prop, p, fuel):
        yield RetrofitScenario(
            name="pv-battery",
            retrofit_ids=["solar_pv", "battery"],
            params=p.model_copy(update={"pv_kwp": 4.0, "battery_kwh": 10.0}),
            fuel=fuel,
        )

    dataset = generate_dataset(
        [prop],
        GeneratorConfig(development=True, n_homes=3, n_months=2),
        simulator=Simulator(run_physics, VERSION),
        scenario_factory=scenarios,
        electricity_factory=development_electricity_factory,
    )
    for row in dataset.homes:
        assert row.electricity_input_summary["interval_count"] == 59 * 48
        assert (
            row.scenarios["pv-battery"].totals["electricity_import_kwh"]
            <= row.baseline.totals["electricity_import_kwh"]
        )
        assert row.scenarios["pv-battery"].totals["gas_kwh"] == row.baseline.totals["gas_kwh"]
        assert max(row.scenarios["pv-battery"].conservation_max_error_kwh.values()) < 0.02


def test_satyarth_adapter_preserves_handoff(home):
    _, p, w = home
    package = {
        "inputs_version": "physics_inputs/0.1",
        "property_id": p.property_id,
        "params": p.model_dump(),
        "home": {"postcode_district": "BS16"},
        "hot_water": {
            "fuel": "mains_gas",
            "efficiency": 0.75,
            "standing_loss_kwh_per_day": 1.0,
            "shower_electric_kwh_per_day": 0.5,
        },
        "occupancy": {"presence": "home_all_day"},
        "heating_behaviour": {"periods": [[6, 9], [17, 23]]},
        "appliances": {"hourly_profile_kw": [1] * 24},
        "weather": {
            "months": [x.month for x in w],
            "source": "synthetic",
            "arrays": {k: np.asarray(v).tolist() for k, v in weather_arrays(w).items()},
        },
        "schedules": {"extra_electricity_kwh_monthly": [100.0, 100.0], "appliance_month_factor": [1.0, 1.0]},
    }
    a = adapt_satyarth_package(package)
    assert a.fuel.hot_water_efficiency == 0.75
    assert a.extra_params["shower_electric_kwh_per_day"] == 0.5
    assert a.profiles["extra_electricity_kwh_monthly"].sum() == 200
    assert a.appliance_hourly_profile_kw == [1] * 24


def test_cli_prepared_only_and_output_guard(home, tmp_path, capsys):
    from scripts.run_electricity_twin import main

    _, p, w = home
    package = tmp_path / "inputs.json"
    package.write_text(
        json.dumps(
            {
                "parameters": p.model_dump(),
                "weather": [x.model_dump() for x in w],
                "equipment": {"heating_fuel": "mains_gas", "hot_water_fuel": "mains_gas"},
                "electricity_inputs": prepared(p, w).model_dump(),
            }
        )
    )
    output = tmp_path / "result.json"
    main(["--prepared", str(package), "--output", str(output)])
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        main(["--prepared", str(package), "--output", str(output)])
    assert output.read_bytes() == original
    with pytest.raises(SystemExit):
        main(["--prepared", str(package), "--output", str(ROOT.parent / "must-not-create-electricity.json")])
    assert "inside" in capsys.readouterr().err
