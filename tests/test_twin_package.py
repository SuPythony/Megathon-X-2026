"""Physics inputs package, appliances, household, retrofits, room allocation and the DigitalTwin class."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from contracts.models import EnergyUsageMonth, Property, RetrofitAction
from greentwin.interfaces import DigitalTwin
from greentwin.twin.appliances import catalogue, summarise_appliances
from greentwin.twin.archetypes import build_layout
from greentwin.twin.inputs import build_physics_inputs
from greentwin.twin.intake import MeterReading, TwinIntake
from greentwin.twin.params import params_with_sources
from greentwin.twin.physics_stub import run_physics
from greentwin.twin.retrofits import apply_retrofits
from greentwin.twin.retrofits import catalogue as retrofit_catalogue
from greentwin.twin.scene import AppliancePlacement as P
from greentwin.twin.scene import appliance_visuals
from greentwin.twin.simulate import prior_samples
from greentwin.twin.twin import GreenDigitalTwin

EX = Path(__file__).resolve().parents[1] / "contracts" / "examples"


def _prop(pid):
    return Property.model_validate_json((EX / f"{pid}_property.json").read_text(encoding="utf-8"))


def _usage(pid):
    return [EnergyUsageMonth.model_validate(r) for r in json.loads((EX / f"{pid}_usage_2025.json").read_text())]


APPS = [
    P(appliance_id="a", type="fridge_freezer", room_id="kitchen", answers={"size": "large", "age": "old"}),
    P(appliance_id="b", type="washing_machine", room_id="kitchen"),
    P(appliance_id="c", type="hob", room_id="kitchen", answers={"type": "gas"}),
    P(appliance_id="d", type="tv", room_id="living"),
    P(appliance_id="e", type="electric_shower", room_id="bath"),
    P(appliance_id="f", type="electric_heater", room_id="bed1"),
    P(appliance_id="g", type="ev_charger", room_id="outside"),
]


# ---------------------------------------------------------------- appliances
def test_every_visual_type_has_energy_data():
    assert set(appliance_visuals()) == set(catalogue()["types"])


def test_appliance_categories_and_schedules():
    s = summarise_appliances(APPS, occupants=2.4)
    by_type = {a.type: a for a in s.appliances}
    assert by_type["hob"].fuel == "mains_gas" and s.gas_cooking_kwh_per_day > 0
    assert by_type["electric_shower"].category == "hot_water" and s.shower_electric_kwh_per_day > 0
    assert sum(s.secondary_heating_kwh_monthly[5:8]) == 0 and s.secondary_heating_kwh_monthly[0] > 0  # winter only
    assert by_type["ev_charger"].heat_to_room == 0 and sum(s.outdoor_kwh_monthly) > 0
    assert by_type["washing_machine"].defaulted == ["drum_kg"]
    for a in s.appliances:
        assert sum(a.monthly_kwh) == pytest.approx(a.kwh_per_year, rel=0.01)
    assert np.mean(s.appliance_month_factor) == pytest.approx(1.0, abs=0.02)
    old = summarise_appliances([P(appliance_id="x", type="fridge_freezer", answers={"age": "old"})], 2.4)
    new = summarise_appliances([P(appliance_id="x", type="fridge_freezer", answers={"age": "new"})], 2.4)
    assert old.baseload_kwh_per_day > new.baseload_kwh_per_day


def test_more_occupants_more_laundry():
    two = summarise_appliances([P(appliance_id="w", type="washing_machine")], 2)
    five = summarise_appliances([P(appliance_id="w", type="washing_machine")], 5)
    assert five.baseload_kwh_per_day > two.baseload_kwh_per_day


# ---------------------------------------------------------------- household + hot water
def test_household_answers_change_params():
    default, _ = params_with_sources(_prop("SH2"))
    told, src = params_with_sources(_prop("SH2"), occupants=5, thermostat_c=22, heating_periods=[(6, 9), (16, 23)])
    assert told.occupants == 5 and told.hot_water_kwh_per_day > default.hot_water_kwh_per_day
    assert told.setpoint_c == 21.0 and told.heating_hours_per_day == 10
    assert "user" in src["setpoint_c"]


def test_hot_water_is_separate_from_heating():
    sh3, _ = params_with_sources(_prop("SH3"))
    assert sh3.hot_water_fuel == "electricity" and sh3.hot_water_efficiency == 1.0 and sh3.hot_water_standing_loss_kwh_per_day > 0
    s = summarise_appliances(APPS, 2.4)
    sh2, _ = params_with_sources(_prop("SH2"), appliances=s)
    assert sh2.hot_water_fuel == "mains_gas" and sh2.hot_water_efficiency < sh2.heating_efficiency
    assert sh2.shower_electric_kwh_per_day > 0
    no_shower, _ = params_with_sources(_prop("SH2"))
    assert sh2.hot_water_kwh_per_day < no_shower.hot_water_kwh_per_day  # showers take part of the demand


# ---------------------------------------------------------------- intake
def test_intake_household_and_meters():
    i = TwinIntake(intake_id="i", postcode_district="M14", occupants=3, thermostat_c=20,
                   meter_readings=[MeterReading(month="2025-01", gas_kwh=1800, electricity_kwh=300)])
    assert any("Only 1 month of" in w for w in i.warnings)
    with pytest.raises(ValueError):
        TwinIntake(intake_id="i", postcode_district="M14",
                   meter_readings=[MeterReading(month="2025-01"), MeterReading(month="2025-01")])
    with pytest.raises(ValueError):
        MeterReading(month="2025-13")


# ---------------------------------------------------------------- physics inputs package
def test_package_complete_and_runs_on_stub():
    pkg = build_physics_inputs(_prop("SH2"), appliances=APPS, usage=_usage("SH2"))
    assert pkg.calibration_readiness.method == "ready"
    assert len(pkg.weather.months) == len(pkg.meter.months) == 12
    assert {e.element for e in pkg.fabric} == {"walls", "roof", "floor", "windows", "doors", "party_walls"}
    assert pkg.ventilation.heat_loss_w_per_k > 0 and pkg.hot_water.fuel == "mains_gas"
    assert {p.param for p in pkg.priors} >= {"heat_loss_multiplier", "setpoint_c", "baseload_kwh_per_day", "heating_efficiency"}
    for key in ("appliance_month_factor", "secondary_heating_kwh_monthly", "extra_gas_kwh_monthly"):
        assert len(pkg.schedules[key]) == 12
    out = run_physics({k: jnp.asarray(v) for k, v in pkg.physics_params().items()},
                      {k: jnp.asarray(v) for k, v in pkg.physics_timeseries().items()}, "calibrated", pkg.fuel_config())
    assert float(out["gas_kwh"]) > 0 and float(out["electricity_import_kwh"]) > 0
    json.loads(pkg.model_dump_json())  # serialisable


def test_package_readiness_tiers_and_missing_months():
    assert build_physics_inputs(_prop("SH1")).calibration_readiness.method == "prior_only"
    few = _usage("SH1")[:4]
    assert build_physics_inputs(_prop("SH1"), usage=few).calibration_readiness.method == "weak"
    gappy = _usage("SH1")
    gappy[3] = gappy[3].model_copy(update={"gas_kwh": None})
    pkg = build_physics_inputs(_prop("SH1"), usage=gappy)
    assert pkg.meter.gas_kwh[3] is None  # missing stays missing, never 0


def test_schedules_reach_the_physics():
    pkg = build_physics_inputs(_prop("SH2"), appliances=APPS)
    p = {k: jnp.asarray(v) for k, v in pkg.physics_params().items()}
    ts = {k: jnp.asarray(v) for k, v in pkg.physics_timeseries().items()}
    full = run_physics(p, ts, "calibrated", pkg.fuel_config())
    no_ev = dict(ts, extra_electricity_kwh_monthly=jnp.zeros(12))
    assert float(full["electricity_import_kwh"]) > float(run_physics(p, no_ev, "calibrated", pkg.fuel_config())["electricity_import_kwh"])
    no_heater = dict(ts, secondary_heating_kwh_monthly=jnp.zeros(12))
    assert float(full["gas_kwh"]) < float(run_physics(p, no_heater, "calibrated", pkg.fuel_config())["gas_kwh"])


# ---------------------------------------------------------------- retrofits
@pytest.mark.parametrize("rid", sorted(retrofit_catalogue()))
def test_every_catalogue_retrofit_applies(rid):
    params, _ = params_with_sources(_prop("SH2"))
    after = apply_retrofits(params, [RetrofitAction(retrofit_id=rid)])
    assert type(after) is type(params)


def test_retrofit_effects():
    params, _ = params_with_sources(_prop("SH1"))
    loft = apply_retrofits(params, [RetrofitAction(retrofit_id="loft_insulation")])
    assert loft.roof_u < params.roof_u
    pv = apply_retrofits(params, [RetrofitAction(retrofit_id="solar_pv", size=4.0)])
    assert pv.pv_kwp == 4.0
    led = apply_retrofits(params, [RetrofitAction(retrofit_id="led_lighting")])
    assert led.baseload_kwh_per_day < params.baseload_kwh_per_day
    hp = apply_retrofits(params, [RetrofitAction(retrofit_id="air_source_heat_pump")])
    assert hp.heating_fuel == "electricity" and hp.hot_water_fuel == "electricity" and hp.heating_efficiency > 2
    assert params.roof_u == apply_retrofits(params, []).roof_u  # pure
    with pytest.raises(KeyError):
        apply_retrofits(params, [RetrofitAction(retrofit_id="magic_paint")])


# ---------------------------------------------------------------- DigitalTwin + rooms
@pytest.fixture(scope="module")
def twin_and_samples():
    t = GreenDigitalTwin(n_samples=200)
    params = t.params_from_property(_prop("SH1"))
    return t, prior_samples(params, 200)


def test_protocol_and_package_rerun(twin_and_samples):
    t, s = twin_and_samples
    assert isinstance(t, DigitalTwin)
    base = t.simulate(s, [])
    package = t.simulate(s, [RetrofitAction(retrofit_id="loft_insulation"), RetrofitAction(retrofit_id="draught_proofing")])
    assert package.gas_kwh.mean() < base.gas_kwh.mean()
    assert t.last_engine in ("physics_stub", "physics_l1")


def test_element_heat_loss_and_scene(twin_and_samples):
    t, s = twin_and_samples
    elements = t.element_heat_loss(s)
    assert sum(e.share_pct for e in elements) == pytest.approx(100, abs=0.5)
    assert all(e.estimated for e in elements)
    scene = t.scene(_prop("SH1"), s)
    assert sum(x.heat_loss_share_pct for x in scene.surfaces) <= 100.5


def test_calibrate_prior_only_without_bills(twin_and_samples):
    # With bills, calibrate() runs P2's Bayesian calibration: tests/test_twin_p2_integration.py
    t, _ = twin_and_samples
    result, samples = t.calibrate(_prop("SH1"), [], [])
    assert result.method == "prior_only" and result.provenance.engine in ("mock", "physics_l1")


def test_rooms_sum_to_whole_house(twin_and_samples):
    t, s = twin_and_samples
    layout = build_layout("victorian_terrace_2bed", total_floor_area_m2=85)
    alloc = t.room_energy(_prop("SH1"), layout, s, [P(appliance_id="a", type="fridge_freezer", room_id="kitchen"),
                                                     P(appliance_id="b", type="tv", room_id="living")])
    assert sum(r.heat_loss_share_pct for r in alloc.rooms) == pytest.approx(100, abs=0.5)
    run = t.simulate(s, [])
    whole = sum(np.asarray(v) for v in run.element_heat_loss_w_per_k.values()).mean()
    assert sum(r.heat_loss_w_per_k.mean for r in alloc.rooms) == pytest.approx(whole, rel=1e-3)
    assert sum(r.space_heating_kwh.mean for r in alloc.rooms) == pytest.approx(run.space_heating_kwh.mean(), rel=1e-3)
    kitchen = next(r for r in alloc.rooms if r.room_id == "kitchen")
    assert kitchen.appliance_kwh.mean > 0 and alloc.unaccounted_electricity_kwh is not None
    assert all(r.estimated for r in alloc.rooms)
