"""Generator integration checks. The conserving fixture is not a production simulator."""

import json
from dataclasses import replace

import numpy as np
import pytest

from contracts.models import AreaEPCStats, EnergyUsageMonth, HeatingSystem, Property, ReferenceLabel
from greentwin.twin.generate import (
    ROOT,
    GeneratorConfig,
    RetrofitScenario,
    Simulator,
    generate_dataset,
    month_sequence,
    population_weights,
    write_dataset,
)
from greentwin.twin.params import params_with_sources
from greentwin.twin.physics_api import FuelConfig
from greentwin.twin.weather import typical_year


@pytest.fixture
def properties():
    return [
        Property.model_validate_json((ROOT / "contracts/examples" / f"SH{i}_property.json").read_text())
        for i in (1, 2, 3)
    ]


def conserving_fixture(p, w, mode, fuel):
    """Minimal test-only heat/fuel accounting for testing the generator, not accuracy."""
    days = w["days_in_month"]
    n = len(days)
    z = np.zeros(n)
    elements = {
        key: p[f"{key[:-1] if key in ('walls', 'windows', 'doors') else key}_u"]
        * p[f"{key[:-1] if key in ('walls', 'windows', 'doors') else key}_area_m2"]
        for key in ("walls", "roof", "floor", "windows", "doors")
    }
    elements.update(ventilation=0.33 * p["air_changes_per_hour"] * p["volume_m3"], party_walls=0)
    elements = {k: v * p["heat_loss_multiplier"] for k, v in elements.items()}
    loss = sum(elements.values()) * np.maximum(p["setpoint_c"] - w["mean_temp_c"], 0) * 24 * days / 1000
    gains = p["internal_gains_w"] * 24 * days / 1000
    space = np.maximum(loss - gains, 0)
    hot = p["hot_water_kwh_per_day"] * days * w["hot_water_multiplier"]
    by_fuel = {}
    for key, value in (
        (fuel.heating_fuel, space / p["heating_efficiency"]),
        (fuel.hot_water_fuel, hot / (fuel.hot_water_efficiency or p["heating_efficiency"])),
    ):
        by_fuel[key] = by_fuel.get(key, z) + value
    demand = p["baseload_kwh_per_day"] * days * w["baseload_multiplier"] + by_fuel.get("electricity", z)
    pv = p["pv_kwp"] * w["ghi_kwh_m2"] * 0.85
    used = np.minimum(pv * 0.30, demand)
    monthly = {
        "monthly_electricity_kwh": demand - used,
        "monthly_gas_kwh": by_fuel.get("mains_gas", z),
        "monthly_space_heating_kwh": space,
        "monthly_hot_water_kwh": hot,
        "monthly_pv_generation_kwh": pv,
        "monthly_export_kwh": pv - used,
        "monthly_electricity_demand_kwh": demand,
        "monthly_battery_loss_kwh": z,
        "monthly_battery_delta_kwh": z,
        "monthly_heat_loss_kwh": loss,
        "monthly_heat_gains_kwh": gains,
        "monthly_unused_heat_gains_kwh": np.maximum(gains - loss, 0),
    }
    for f, v in by_fuel.items():
        if f not in ("mains_gas", "electricity"):
            monthly[f"monthly_{f}_kwh"] = v
    totals = {
        k: monthly[v].sum()
        for k, v in {
            "electricity_import_kwh": "monthly_electricity_kwh",
            "gas_kwh": "monthly_gas_kwh",
            "space_heating_kwh": "monthly_space_heating_kwh",
            "hot_water_kwh": "monthly_hot_water_kwh",
            "pv_generation_kwh": "monthly_pv_generation_kwh",
            "export_kwh": "monthly_export_kwh",
        }.items()
    }
    return monthly | totals | {"element_heat_loss_w_per_k": elements}


SIMULATOR = Simulator(conserving_fixture, "test-only-conservation-fixture")


def test_inputs_only_random_reproducible_and_does_not_import_jax(properties):
    config = GeneratorConfig(development=True, n_homes=60)
    first = generate_dataset(properties, config)
    again = generate_dataset(properties, config)
    assert first.model_dump() == again.model_dump()
    assert all(h.baseline is None and not h.observations and not h.reference_labels for h in first.homes)
    assert first.metadata["mode"] == "inputs_only"
    assert len({h.true_params.setpoint_c for h in first.homes}) == 60
    different = generate_dataset(properties, config.model_copy(update={"seed": 43}))
    assert first.homes[0].true_params != different.homes[0].true_params
    for h in first.homes:
        assert (
            0
            < h.true_params.heating_efficiency
            <= (6 if "heat_pump" in h.true_property.heating.main_system else 1)
        )
        assert np.isclose(np.mean(h.latent.baseload_multiplier), 1)
        assert h.true_property.epc.potential_band is None
        assert h.true_property.current_value_estimate_gbp is None


def test_observation_noise_does_not_change_truth_and_missing_is_not_zero(properties):
    config = GeneratorConfig(
        development=True,
        n_homes=10,
        missing_rate=0,
        missing_block_rate=0,
        missing_input_rate=0,
        conflict_rate=0,
        meter_noise_relative=0,
    )
    clean = generate_dataset(properties, config, simulator=SIMULATOR)
    masked = generate_dataset(properties, config.model_copy(update={"missing_rate": 1}), simulator=SIMULATOR)
    noisy = generate_dataset(
        properties, config.model_copy(update={"meter_noise_relative": 0.2}), simulator=SIMULATOR
    )
    for a, b, c in zip(clean.homes, masked.homes, noisy.homes):
        assert a.baseline == b.baseline == c.baseline
        assert a.true_params == b.true_params == c.true_params
        assert all(r.electricity_import_kwh is None for r in b.observations)
        for i, row in enumerate(a.observations):
            assert row.electricity_import_kwh == a.baseline.monthly["monthly_electricity_kwh"][i]
        if a.baseline.totals["gas_kwh"] == 0:
            assert all(r.gas_kwh == 0 for r in b.observations)
        else:
            assert all(r.gas_kwh is None for r in b.observations)
    assert any(a.observations != b.observations for a, b in zip(clean.homes, noisy.homes))


def test_source_groups_never_span_splits(properties):
    properties[0].customer_id = properties[1].customer_id = "shared-source-customer"
    properties[0].split = "test"
    dataset = generate_dataset(properties, GeneratorConfig(development=True, n_homes=40))
    assignments = {}
    for h in dataset.homes:
        assignments.setdefault(h.split_group, set()).add(h.true_property.split)
        if h.source_property_id in ("SH1", "SH2"):
            assert h.true_property.split == "test"
    assert all(len(values) == 1 for values in assignments.values())
    properties[1].split = "train"
    with pytest.raises(ValueError, match="multiple splits"):
        generate_dataset(properties, GeneratorConfig(development=True))


@pytest.mark.parametrize(
    "failure", ["electricity", "heat", "conversion", "monthly_sum", "nonfinite", "battery"]
)
def test_invalid_physics_is_rejected(properties, failure):
    def corrupt(p, w, mode, fuel):
        raw = conserving_fixture(p, w, mode, fuel)
        if failure == "electricity":
            raw["monthly_export_kwh"] = raw["monthly_export_kwh"] + 20
            raw["export_kwh"] = raw["monthly_export_kwh"].sum()
        elif failure == "heat":
            raw["monthly_heat_loss_kwh"] += 20
        elif failure == "conversion":
            raw["monthly_gas_kwh"] += 20
            raw["gas_kwh"] = raw["monthly_gas_kwh"].sum()
        elif failure == "monthly_sum":
            raw["electricity_import_kwh"] += 20
        elif failure == "nonfinite":
            raw["monthly_heat_gains_kwh"][0] = np.nan
        else:
            raw["monthly_battery_delta_kwh"] = np.ones(len(w["days_in_month"])) * 10
            raw["monthly_electricity_kwh"] += 10
            raw["electricity_import_kwh"] = raw["monthly_electricity_kwh"].sum()
        return raw

    with pytest.raises(ValueError):
        generate_dataset(
            [properties[1]],
            GeneratorConfig(development=True, n_homes=1),
            simulator=Simulator(corrupt, "invalid-test-fixture"),
        )


def test_package_labels_are_paired_reruns_and_heat_pump_electricity_can_rise(properties):
    def scenarios(prop, p, fuel):
        loft = p.model_copy(update={"roof_u": p.roof_u * 0.5})
        hp = p.model_copy(update={"heating_fuel": "electricity", "heating_efficiency": 2.9})
        combined = hp.model_copy(update={"roof_u": loft.roof_u})
        return [
            RetrofitScenario("loft", ("loft_insulation",), loft, fuel),
            RetrofitScenario("hp", ("air_source_heat_pump",), hp, FuelConfig("electricity", "electricity")),
            RetrofitScenario(
                "package",
                ("loft_insulation", "air_source_heat_pump"),
                combined,
                FuelConfig("electricity", "electricity"),
                all_feasible=True,
            ),
        ]

    dataset = generate_dataset(
        [properties[1]],
        GeneratorConfig(development=True, n_homes=4),
        simulator=SIMULATOR,
        scenario_factory=scenarios,
    )
    for h in dataset.homes:
        for label in h.reference_labels:
            ReferenceLabel.model_validate(label.model_dump())
        hp_elec = next(
            r
            for r in h.reference_labels
            if r.retrofit_ids == ["air_source_heat_pump"] and r.fuel == "electricity"
        )
        assert hp_elec.value < 0
        assert h.scenarios["hp"].totals["gas_kwh"] == 0
        package_saving = h.baseline.totals["gas_kwh"] - h.scenarios["package"].totals["gas_kwh"]
        assert (
            next(r.value for r in h.reference_labels if r.label_type == "package_saving" and r.fuel == "gas")
            == package_saving
        )
        assert any(
            r.label_type == "improvement_potential" and r.value is not None for r in h.reference_labels
        )


def test_other_fuel_is_preserved(properties):
    prop = properties[1].model_copy(deep=True)
    prop.heating.main_fuel = "oil"
    prop.heating.main_system = HeatingSystem.oil_boiler
    home = generate_dataset([prop], GeneratorConfig(development=True, n_homes=1), simulator=SIMULATOR).homes[
        0
    ]
    assert home.baseline.totals["oil_kwh"] > 0
    assert home.baseline.totals["gas_kwh"] == 0


def test_production_requires_real_sources_and_complete_weather(properties):
    with pytest.raises(ValueError, match="real EPC"):
        generate_dataset(properties, GeneratorConfig(n_homes=1))
    prop = properties[1].model_copy(update={"source": "epc_real"})
    with pytest.raises(ValueError, match="Missing real weather"):
        generate_dataset([prop], GeneratorConfig(n_homes=1))
    weather = [w.model_copy(update={"source": "open_meteo_era5"}) for w in typical_year("BS16")]
    dataset = generate_dataset([prop], GeneratorConfig(n_homes=1, n_months=12), weather_records=weather)
    assert dataset.metadata["weather_sources"] == ["open_meteo_era5"]
    assert dataset.metadata["empirical_validation"] == "not_run"


def test_zero_pv_angles_are_preserved(properties):
    prop = properties[1].model_copy(deep=True)
    prop.renewables.pv_azimuth_deg = 0
    prop.renewables.pv_tilt_deg = 0
    p, _ = params_with_sources(prop)
    assert p.pv_azimuth_deg == p.pv_tilt_deg == 0


def test_written_contracts_and_manifest_and_no_overwrite(properties, tmp_path, monkeypatch):
    import greentwin.twin.generate as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    data = generate_dataset(properties, GeneratorConfig(development=True, n_homes=3), simulator=SIMULATOR)
    output = tmp_path / "dataset"
    write_dataset(data, output)
    for name, model in (
        ("properties", Property),
        ("usage", EnergyUsageMonth),
        ("reference_labels", ReferenceLabel),
    ):
        for line in (output / f"{name}.jsonl").read_text().splitlines():
            model.model_validate_json(line)
    meta = json.loads((output / "meta.json").read_text())
    assert meta["config"]["seed"] == 42 and meta["row_counts"]["truth"] == 3
    with pytest.raises(FileExistsError):
        write_dataset(data, output)
    with pytest.raises(ValueError, match="worktree"):
        write_dataset(data, tmp_path.parent / "outside")


def test_month_sequence_and_leap_year():
    assert month_sequence("2024-12", 3) == ["2024-12", "2025-01", "2025-02"]
    with pytest.raises(ValueError):
        GeneratorConfig(start_month="2025-13")


def test_stub_not_allowed():
    def physics_stub(p, w, m, f):
        return {}

    with pytest.raises(ValueError, match="stub"):
        Simulator(physics_stub, "v1")
    with pytest.raises(ValueError, match="physics_l1"):
        replace(SIMULATOR, engine="mock")


def test_population_raking_targets_and_missing_categories(properties):
    first, second = properties[0].model_copy(deep=True), properties[1].model_copy(deep=True)
    first.location.local_authority_code = second.location.local_authority_code = "TEST-LA"
    first.epc.current_band, second.epc.current_band = "F", "D"
    stats = AreaEPCStats(
        area_code="TEST-LA",
        area_type="local_authority",
        period="2025/1",
        lodgements=100,
        band_counts={"F": 80, "D": 20},
        property_type_counts={"house": 100},
        release="test",
    )
    np.testing.assert_allclose(population_weights([first, second], [stats]), [0.8, 0.2])
    with pytest.raises(ValueError, match="category absent"):
        population_weights([first, second], [stats.model_copy(update={"band_counts": {"A": 100}})])


def test_invalid_weather_rejected(properties):
    weather = typical_year("BS16")
    weather[0].mean_temp_c = np.nan
    with pytest.raises(ValueError, match="finite"):
        generate_dataset(
            [properties[1]],
            GeneratorConfig(development=True, n_homes=1, n_months=12),
            weather_records=weather,
        )
