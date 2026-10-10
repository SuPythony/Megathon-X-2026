"""Retrofit scenarios for the viewer: packages re-simulated (never summed), savings per sample, options, dev server."""

import pytest

from greentwin.twin.inputs import build_physics_inputs
from greentwin.twin.scenarios import run_scenario
from greentwin.twin.scene import AppliancePlacement, build_scene, retrofit_options, scenario_key
from greentwin.twin.showcase import load_showcase
from greentwin.twin.simulate import prior_samples
from greentwin.twin.twin import GreenDigitalTwin


@pytest.fixture(scope="module")
def sh1():
    sc = load_showcase("SH1")
    pkg = build_physics_inputs(sc.prop, layout=sc.layout, appliances=sc.appliances, intake=sc.intake, usage=sc.usage)
    return sc, prior_samples(pkg.params, 300, priors=pkg.prior_table()), GreenDigitalTwin(n_samples=300)


def test_baseline_has_zero_savings(sh1):
    sc, samples, twin = sh1
    v = run_scenario(twin, sc.layout, samples, [], sc.appliances)
    assert v.key == "" and v.savings.gas_kwh.headline == 0 and v.savings.electricity_kwh.headline == 0


def test_package_is_rerun_not_summed(sh1):
    sc, samples, twin = sh1
    loft = run_scenario(twin, sc.layout, samples, ["loft_insulation"])
    glazing = run_scenario(twin, sc.layout, samples, ["double_glazing"])
    both = run_scenario(twin, sc.layout, samples, ["double_glazing", "loft_insulation"])
    assert both.key == "double_glazing+loft_insulation"
    assert both.savings.gas_kwh.headline > max(loft.savings.gas_kwh.headline, glazing.savings.gas_kwh.headline)
    summed = loft.savings.gas_kwh.headline + glazing.savings.gas_kwh.headline
    assert both.savings.gas_kwh.headline != pytest.approx(summed, rel=1e-6)  # interactions come from the physics


def test_heat_pump_saves_gas_but_raises_electricity(sh1):
    sc, samples, twin = sh1
    hp = run_scenario(twin, sc.layout, samples, ["air_source_heat_pump"], sc.appliances)
    assert hp.after.gas_kwh.headline == pytest.approx(0, abs=1)
    assert hp.savings.electricity_kwh.headline < 0  # negative saving = use goes up
    assert hp.savings.carbon_kg.headline > 0
    assert sum(r.heat_loss_share_pct for r in hp.rooms_after.rooms) == pytest.approx(100, abs=0.5)


def test_tariff_warns_and_pv_cuts_import(sh1):
    sc, samples, twin = sh1
    assert run_scenario(twin, sc.layout, samples, ["time_of_use_tariff"]).warnings
    assert run_scenario(twin, sc.layout, samples, ["solar_pv"]).savings.electricity_kwh.headline > 0


def test_retrofit_options_geometry():
    sh1, sh3 = load_showcase("SH1"), load_showcase("SH3")
    o1 = {o.retrofit_id: o for o in retrofit_options(sh1.layout, sh1.prop)}
    o3 = {o.retrofit_id: o for o in retrofit_options(sh3.layout, sh3.prop)}
    assert o1["loft_insulation"].applicable and not o1["cavity_wall_insulation"].applicable  # solid brick
    assert not o3["loft_insulation"].applicable and not o3["solar_pv"].applicable and not o3["floor_insulation"].applicable
    assert "triple_glazing" in o1["double_glazing"].excludes
    assert o1["triple_glazing"].cost_gbp_central is None and o1["triple_glazing"].cost_status == "unsourced"
    assert scenario_key(["b", "a"]) == "a+b"


def test_scene_carries_answers():
    sc = load_showcase("SH2")
    scene = build_scene(sc.layout, "SH2", [AppliancePlacement(appliance_id="x", type="tv", room_id="living", answers={"inches": "65"})])
    assert scene.appliances[0].answers == {"inches": "65"} and scene.retrofits


def test_dev_server_runs_a_package():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import importlib.util
    import sys
    from pathlib import Path

    from fastapi.testclient import TestClient

    path = Path(__file__).resolve().parents[1] / "scripts" / "twin_server.py"
    spec = importlib.util.spec_from_file_location("twin_server", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["twin_server"] = module  # pydantic resolves the request/response models through sys.modules
    spec.loader.exec_module(module)
    app = module.app

    client = TestClient(app)
    assert client.get("/api/twin/health").json()["homes"] == ["SH1", "SH2", "SH3"]
    r = client.post("/api/twin/run", json={"property_id": "SH2", "retrofit_ids": ["loft_insulation", "solar_pv"]})
    assert r.status_code == 200 and r.json()["scenario"]["key"] == "loft_insulation+solar_pv"
    assert client.post("/api/twin/run", json={"property_id": "SH9"}).status_code == 404
    assert client.post("/api/twin/run", json={"property_id": "SH1", "retrofit_ids": ["magic"]}).status_code == 422
