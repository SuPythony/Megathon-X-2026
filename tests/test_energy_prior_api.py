"""M04 HTTP router: contract in, contract out, Bayesian update, health."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "artifacts/m04_energy_prior/v2/model.pkl").exists(), reason="train first")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from greentwin.api.energy_prior import app
    return TestClient(app)


def _load(name):
    return json.loads((ROOT / "contracts/examples" / name).read_text())


def test_prior_endpoint_returns_contract(client):
    from contracts.models import EnergyPrior
    r = client.post("/m04/energy-prior", json=_load("SH2_property.json"))
    assert r.status_code == 200
    prior = EnergyPrior.model_validate(r.json())
    assert prior.provenance.engine == "ebm" and prior.gas_kwh_m2.std > 0


def test_update_endpoint_narrows_and_handles_missing(client):
    r = client.post("/m04/energy-prior/update", json={"property": _load("SH1_property.json"),
                                                      "usage": _load("SH1_usage_2025.json")})
    assert r.status_code == 200
    gas = r.json()["update"]["fuels"]["gas"]
    assert gas["observed"]["imputed_months"] == ["2025-08"]
    assert gas["posterior"]["std"] < gas["prior"]["std"]


def test_bad_input_rejected(client):
    bad = _load("SH2_property.json") | {"floor_area_m2": "big"}
    assert client.post("/m04/energy-prior", json=bad).status_code == 422


def test_health(client):
    h = client.get("/m04/health").json()
    assert h["bayes_layer"] and h["fuels"] == ["electricity", "gas"]
