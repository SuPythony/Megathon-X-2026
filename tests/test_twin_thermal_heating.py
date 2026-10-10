"""Thermal mass (construction -> heat capacity -> time constant in the physics) and user-given heating answers."""

import jax
import jax.numpy as jnp
import pytest

from contracts.models import WallType
from greentwin.twin.household import apply_user_heating, meter_fuel_conflicts
from greentwin.twin.inputs import build_physics_inputs
from greentwin.twin.intake import TwinIntake
from greentwin.twin.params import params_with_sources
from greentwin.twin.physics_api import EXTRA_PARAM_KEYS, PHYSICS_PARAM_KEYS
from greentwin.twin.physics_stub import run_physics
from greentwin.twin.showcase import load_showcase
from greentwin.twin.thermal import thermal_mass
from greentwin.twin.weather import typical_year, weather_arrays


def _inputs(pid="SH2", **overrides):
    sc = load_showcase(pid)
    p, _ = params_with_sources(sc.prop, layout=sc.layout)
    params = {k: jnp.asarray(float(getattr(p, k))) for k in (*PHYSICS_PARAM_KEYS, *EXTRA_PARAM_KEYS)}
    params.update({k: jnp.asarray(float(v)) for k, v in overrides.items()})
    w = {k: jnp.asarray(v) for k, v in weather_arrays(typical_year("BS16")).items()}
    from greentwin.twin.params import fuel_config
    return params, w, fuel_config(p)


def test_thermal_mass_follows_construction():
    sc = load_showcase("SH2")
    solid = thermal_mass(sc.prop, sc.layout, "solid")
    hollow = thermal_mass(sc.prop, sc.layout, "hollow")
    assert solid.heat_capacity_kj_per_k > hollow.heat_capacity_kj_per_k
    timber = sc.prop.model_copy(update={"walls": sc.prop.walls.model_copy(update={"type": WallType.timber_frame})})
    assert thermal_mass(timber, sc.layout).heat_capacity_kj_per_k < thermal_mass(sc.prop, sc.layout).heat_capacity_kj_per_k
    flat = load_showcase("SH3")
    assert thermal_mass(flat.prop, flat.layout).thermal_class == "high"  # concrete floors and ceilings
    assert sum(solid.breakdown_kj_per_k.values()) == pytest.approx(solid.heat_capacity_kj_per_k, abs=2)


def test_intermittent_heating_saves_more_in_light_homes():
    def heat(capacity, hours):
        p, w, f = _inputs(heat_capacity_kj_per_k=capacity, heating_hours_per_day=hours)
        return float(run_physics(p, w, "calibrated", f)["space_heating_kwh"])
    light_saving = 1 - heat(8000, 9) / heat(8000, 24)
    heavy_saving = 1 - heat(50000, 9) / heat(50000, 24)
    assert light_saving > heavy_saving > 0


def test_thermal_mass_gradient_is_finite():
    p, w, f = _inputs()
    g = jax.grad(lambda c: run_physics(dict(p, heat_capacity_kj_per_k=c), w, "calibrated", f)["gas_kwh"])(p["heat_capacity_kj_per_k"])
    assert jnp.isfinite(g)


def test_user_heating_overrides_epc_and_flags():
    sc = load_showcase("SH2")
    prop, conflicts, src = apply_user_heating(sc.prop, "air_source_heat_pump", "heat_pump_cylinder")
    assert prop.heating.main_fuel == "electricity" and prop.heating.hot_water == "heat_pump_cylinder"
    assert conflicts and "using your answer" in conflicts[0] and src["heating"].startswith("user")
    assert sc.prop.heating.main_fuel == "mains_gas"  # original untouched
    same, none, _ = apply_user_heating(sc.prop, "gas_boiler_combi", None)
    assert none == [] and same.heating.main_system == sc.prop.heating.main_system  # keeps condensing type
    assert meter_fuel_conflicts(prop, [1500.0] * 12, has_gas_cooking=False)


def test_package_uses_user_heating_and_thermal_mass():
    sc = load_showcase("SH2")
    intake = TwinIntake(intake_id="t", postcode_district="BS16", heating_system="air_source_heat_pump",
                        hot_water_source="heat_pump_cylinder", internal_walls="hollow")
    pkg = build_physics_inputs(sc.prop, layout=sc.layout, appliances=sc.appliances, intake=intake, usage=sc.usage)
    assert pkg.heating_system.fuel == "electricity" and pkg.fuel["hot_water_fuel"] == "electricity"
    assert pkg.thermal_mass.partitions == "user answer"
    assert pkg.physics_params()["heat_capacity_kj_per_k"] == pkg.params.heat_capacity_kj_per_k
    assert any("you said" in f for f in pkg.data_quality["flags"])
    assert any("Gas bills" in f for f in pkg.data_quality["flags"])
    assert any(p.param == "heat_capacity_kj_per_k" for p in pkg.priors)
