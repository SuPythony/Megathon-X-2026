"""Customer answers: optional follow-ups asked only when relevant, explicit answers fixed, "not sure" uncertain,
technical quantities generated (never asked), each with provenance."""

import numpy as np
import pytest

from contracts.models import HeatingSystem
from greentwin.twin.household import follow_up_questions
from greentwin.twin.inputs import build_physics_inputs
from greentwin.twin.intake import TwinIntake
from greentwin.twin.scene import AppliancePlacement
from greentwin.twin.showcase import load_showcase
from greentwin.twin.simulate import prior_samples, simulate

HEATER = AppliancePlacement(appliance_id="h1", type="electric_heater", room_id=None, answers={"rating_kw": "2"})


def _pkg(pid="SH2", **answers):
    sc = load_showcase(pid)
    return sc, build_physics_inputs(sc.prop, layout=sc.layout, intake=TwinIntake(intake_id="t", postcode_district="M14", **answers))


def _prior(pkg, param):
    return next((p for p in pkg.priors if p.param == param), None)


def _prov(pkg, field):
    return next(p for p in pkg.input_provenance if p.field == field)


def test_follow_ups_only_when_relevant():
    sc = load_showcase("SH2")
    gas_full_led = sc.prop.model_copy(update={"low_energy_lighting_pct": 100.0})
    assert follow_up_questions(gas_full_led, "gas_boiler_combi", []) == []
    keys = lambda qs: {q.key for q in qs}  # noqa: E731
    assert keys(follow_up_questions(gas_full_led, None, ["electric_heater"])) == {"plug_in_heater_use"}
    assert keys(follow_up_questions(gas_full_led, "electric_storage", [])) == {"plug_in_heater_use"}
    storage = sc.prop.model_copy(update={"heating": sc.prop.heating.model_copy(
        update={"main_system": HeatingSystem.electric_storage})})
    assert "plug_in_heater_use" in keys(follow_up_questions(storage, None, []))
    assert keys(follow_up_questions(None, None, [])) == {"led_lighting"}  # no EPC: lighting unknown
    partial = sc.prop.model_copy(update={"low_energy_lighting_pct": 40.0})
    assert "40%" in follow_up_questions(partial, None, [])[0].reason


def test_explicit_answers_are_fixed_and_not_sure_is_uncertain():
    _, given = _pkg(occupants=3, thermostat_c=21, heating_periods=[(6, 9), (17, 22)])
    _, unsure = _pkg()
    for param in ("setpoint_c", "heating_hours_per_day"):
        assert _prior(given, param).kind == "fixed" and param not in given.prior_table()
        assert _prior(unsure, param).kind == "normal_add" and _prior(unsure, param).sigma > 0
    assert given.params.setpoint_c == 20.0 and given.params.heating_hours_per_day == 8.0
    assert _prior(unsure, "hot_water_kwh_per_day").sigma > _prior(given, "hot_water_kwh_per_day").sigma
    # Fixed answers stay at their value in every sample
    s = prior_samples(given.params, 200, priors=given.prior_table())
    assert np.all(s.params["setpoint_c"] == np.float32(20.0))
    # Provenance says which were answered
    assert _prov(given, "setpoint_c").customer_answer == "given" and _prov(given, "setpoint_c").status == "stated"
    assert _prov(unsure, "setpoint_c").customer_answer == "not_sure"
    assert _prov(unsure, "occupants").customer_answer == "not_sure"
    assert _prov(given, "wall_u").customer_answer is None  # technical: never asked


def test_lighting_answer_changes_baseload_and_is_recorded():
    _, led = _pkg(led_lighting="all")
    _, few = _pkg(led_lighting="few")
    _, unsure = _pkg()
    assert led.params.low_energy_lighting_pct == 100 and few.params.low_energy_lighting_pct == 15
    assert led.params.baseload_kwh_per_day < few.params.baseload_kwh_per_day
    assert _prov(led, "low_energy_lighting_pct").customer_answer == "given"
    assert _prov(unsure, "low_energy_lighting_pct").customer_answer == "not_sure"
    assert _prior(unsure, "baseload_kwh_per_day").sigma > _prior(led, "baseload_kwh_per_day").sigma


def test_heater_answer_sets_secondary_heating():
    _, never = _pkg(appliances=[HEATER], plug_in_heater_use="never")
    _, most = _pkg(appliances=[HEATER], plug_in_heater_use="most_days")
    _, unsure = _pkg(appliances=[HEATER])
    sec = lambda p: sum(p.schedules["secondary_heating_kwh_monthly"])  # noqa: E731
    assert sec(never) == 0 < sec(unsure) < sec(most)
    assert _prior(most, "secondary_heating_scale").kind == "fixed"
    assert _prior(unsure, "secondary_heating_scale").kind == "lognormal"
    # Heaters in use but none placed: one typical heater is added
    _, stated = _pkg("SH2", heating_system="electric_storage", plug_in_heater_use="some_evenings")
    assert sec(stated) > 0 and any(a.appliance_id == "plug_in_heater_stated" for a in stated.appliances.appliances)
    # An answer to a question that wasn't relevant is ignored
    _, irrelevant = _pkg(plug_in_heater_use="most_days")
    assert irrelevant.appliances is None and not any(f.question.key == "plug_in_heater_use" for f in irrelevant.follow_ups)


def test_technical_quantities_are_generated_with_sources():
    _, pkg = _pkg()
    names = {t.quantity for t in pkg.technical_inputs}
    for needed in ("Wall U-value", "Ventilation rate", "Heating seasonal efficiency", "Outdoor temperature and degree days",
                   "Solar irradiation (horizontal)", "Appliance load profile", "Battery round-trip efficiency"):
        assert needed in names
    assert all(t.source and t.status and t.replace_with for t in pkg.technical_inputs)
    _, hp = _pkg(heating_system="air_source_heat_pump")
    assert any(t.quantity == "Heating seasonal COP" and t.value > 1.5 for t in hp.technical_inputs)


def test_battery_losses_reduce_self_use():
    sc, pkg = _pkg()
    base = pkg.params.model_copy(update={"pv_kwp": 4.0, "battery_kwh": 10.0})
    # battery_round_trip_efficiency is the stand-in's battery model; P2's physics takes its battery efficiencies from
    # the prepared electricity inputs (ElectricityIntake), so this checks the stub explicitly.
    from greentwin.twin.physics_stub import run_physics as stub_run
    run = lambda rte: simulate(prior_samples(base.model_copy(update={"battery_round_trip_efficiency": rte}), 1, priors={}),  # noqa: E731
                               pkg.physics_timeseries(), pkg.fuel_config(), run_physics=stub_run)[0]
    lossless, lossy = run(1.0), run(0.8)
    assert lossy.electricity_import_kwh.sum() > lossless.electricity_import_kwh.sum()
    assert lossy.pv_generation_kwh.sum() == pytest.approx(lossless.pv_generation_kwh.sum())


def test_solar_and_battery_answers_override_the_epc():
    _, sized = _pkg(solar_pv="yes", pv_kwp=4.0, home_battery="yes", battery_kwh=9.5)
    assert sized.params.pv_kwp == 4.0 and sized.params.battery_kwh == 9.5
    assert _prior(sized, "pv_kwp").kind == "fixed" and "pv_kwp" not in sized.prior_table()
    assert _prov(sized, "pv_kwp").customer_answer == "given" and _prov(sized, "pv_kwp").status == "stated"
    _, unsized = _pkg(solar_pv="yes", home_battery="no")
    assert unsized.params.pv_kwp == 3.5 and _prior(unsized, "pv_kwp").kind == "lognormal"
    assert unsized.params.battery_kwh == 0 and _prior(unsized, "battery_kwh").kind == "fixed"
    sc, none = _pkg(solar_pv="no")
    assert none.params.pv_kwp == 0
    _, unsure = _pkg()
    assert _prov(unsure, "pv_kwp").customer_answer == "not_sure"
    assert any("Not sure whether the home has solar" in f for f in unsure.data_quality["flags"])
    # The physics sees the panels: imports fall with solar
    run = lambda pkg: simulate(prior_samples(pkg.params, 1, priors={}), pkg.physics_timeseries(), pkg.fuel_config())[0]  # noqa: E731
    assert run(sized).electricity_import_kwh.sum() < run(none).electricity_import_kwh.sum()
    # A "no" against an EPC that records panels is flagged
    epc_pv = sc.prop.model_copy(update={"renewables": sc.prop.renewables.model_copy(update={"pv_kwp": 2.0})})
    pkg = build_physics_inputs(epc_pv, layout=sc.layout, intake=TwinIntake(intake_id="t", postcode_district="M14", solar_pv="no"))
    assert pkg.params.pv_kwp == 0 and any("you said none" in f for f in pkg.data_quality["flags"])


def test_appliance_completeness_tops_up_unlisted_use():
    fridge = AppliancePlacement(appliance_id="f", type="fridge_freezer", room_id=None)
    _, everything = _pkg(appliances=[fridge], appliance_completeness="all")
    _, few = _pkg(appliances=[fridge], appliance_completeness="few")
    _, unsure = _pkg(appliances=[fridge])
    b = lambda p: p.params.baseload_kwh_per_day  # noqa: E731
    assert b(everything) < b(unsure) < b(few)
    assert everything.params.internal_gains_w < few.params.internal_gains_w
    assert _prior(everything, "baseload_kwh_per_day").sigma < _prior(few, "baseload_kwh_per_day").sigma
    assert _prov(everything, "appliance_completeness").customer_answer == "given"
    assert _prov(unsure, "appliance_completeness").customer_answer == "not_sure"
    _, no_list = _pkg(appliance_completeness="all")  # not asked without a list: ignored
    assert not any(p.field == "appliance_completeness" for p in no_list.input_provenance)
