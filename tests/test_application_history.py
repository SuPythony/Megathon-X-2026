"""Recorded population answers inform priors without circular generated evidence."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from contracts.models import Property
from contracts.uncertain_twin import HouseholdRequest
from greentwin.twin.application_history import ApplicationHistory, prepare_application_inputs
from greentwin.twin.ensemble import simulate_ensemble
from greentwin.twin.priors import build_priors

ROOT = Path(__file__).resolve().parents[1]
AS_OF = datetime(2026, 10, 11, tzinfo=UTC)


def request(pid="new", **facts):
    return HouseholdRequest(
        property_id=pid,
        months=["2025-01"],
        allow_development_weather=True,
        user_inputs={
            "postcode_district": "BS16",
            "property_type": "house",
            "floor_area_m2": 95,
            "heating_system": "gas_boiler_combi",
            "has_solar_pv": False,
            "has_battery": False,
        }
        | facts,
    )


@pytest.fixture
def history(tmp_path):
    return ApplicationHistory(tmp_path / "application.sqlite")


def test_history_changes_initial_constant_and_missing_input_distributions(history):
    for i in range(8):
        history.record(
            request(str(i), wall_u=0.3 + i * 0.01, occupants=4, thermostat_c=22, baseload_kwh_per_day=9),
            submitted_at=AS_OF,
        )
    prior = build_priors(request(), history=history, history_as_of=AS_OF)
    default = build_priors(request())
    assert prior.marginals["wall_u"].centre < default.marginals["wall_u"].centre
    assert prior.marginals["setpoint_c"].centre > default.marginals["setpoint_c"].centre
    assert prior.marginals["occupants"].categories["4"] > default.marginals["occupants"].categories["4"]
    assert prior.marginals["wall_u"].scale > 0
    assert prior.application_history["eligible_households"] == 8
    assert len(prior.application_history["snapshot_sha256"]) == 64


def test_latest_revision_counts_once_and_current_home_is_excluded(history):
    history.record(request("old", wall_u=0.3), submitted_at=datetime(2026, 1, 1, tzinfo=UTC))
    history.record(request("old", wall_u=0.4), submitted_at=AS_OF)
    history.record(request("new", wall_u=2), submitted_at=AS_OF)
    p = build_priors(request(), history=history, history_as_of=AS_OF)
    assert p.application_history["eligible_households"] == 1
    assert p.application_history["excluded_households"] == 1
    wall = next(r for r in p.application_history["variables"] if r["variable"] == "wall_u")
    assert wall["n_households"] == 1
    assert history.snapshot(AS_OF)[0].facts["wall_u"] in (0.4, 2)


def test_new_answer_is_exact_then_available_for_next_home(history):
    current = prepare_application_inputs(
        request("one", occupants=6, wall_u=0.35, thermostat_c=21), 8, history=history, submitted_at=AS_OF
    )
    np.testing.assert_array_equal(current.parameters["wall_u"], 0.35)
    assert current.prior.application_history["eligible_households"] == 0
    next_home = prepare_application_inputs(request("two"), 8, history=history, submitted_at=AS_OF)
    assert next_home.prior.application_history["eligible_households"] == 1
    assert next_home.prior.marginals["occupants"].categories["6"] > 0
    assert "application history" in next_home.prior.marginals["wall_u"].source
    assert "internal_gains_w" not in history.snapshot(AS_OF)[0].facts
    simulated = simulate_ensemble(next_home, batch_size=8)
    assert max(simulated.result.max_conservation_error_kwh.values()) < 0.01


def test_what_ifs_do_not_pollute_history(history):
    assert history.record(request("x", wall_u=0.1), kind="what_if") is False
    assert not history.path.exists()
    prepare_application_inputs(
        request("x", wall_u=0.1), 4, history=history, kind="what_if", submitted_at=AS_OF
    )
    assert not history.path.exists()


def test_future_rows_and_held_out_rows_are_not_used(history):
    history.record(request("future", wall_u=0.1), submitted_at=datetime(2027, 1, 1, tzinfo=UTC))
    prop = Property.model_validate_json((ROOT / "contracts/examples/SH2_property.json").read_text())
    prop = prop.model_copy(update={"property_id": "held", "source": "self_serve", "split": "test"})
    history.record(
        HouseholdRequest(property_id="held", property=prop, user_inputs={"wall_u": 0.1}), submitted_at=AS_OF
    )
    p = build_priors(request(), history=history, history_as_of=AS_OF)
    assert p.application_history["eligible_households"] == 0


def test_known_facts_remain_exact_and_wrong_equipment_does_not_mix(history):
    history.record(
        request("one", wall_u=0.2, heating_efficiency=0.5, heating_system="solid_fuel"), submitted_at=AS_OF
    )
    p = build_priors(request(wall_u=0.8), history=history, history_as_of=AS_OF)
    assert p.marginals["wall_u"].family == "exact"
    assert p.marginals["wall_u"].centre == 0.8
    assert "application history" not in p.marginals["heating_efficiency"].source


def test_known_equipment_absence_not_changed_by_history(history):
    history.record(request("one", has_solar_pv=True, pv_kwp=4), submitted_at=AS_OF)
    p = build_priors(request(), history=history, history_as_of=AS_OF)
    assert p.marginals["has_solar_pv"].categories == {"no": 1}


def test_read_missing_history_does_not_create_files(history):
    assert history.snapshot(AS_OF) == []
    assert not history.path.exists()


def test_import_rejects_showcase_and_full_postcode(history):
    prop = Property.model_validate_json((ROOT / "contracts/examples/SH2_property.json").read_text())
    with pytest.raises(ValueError, match="Showcase"):
        history.record(HouseholdRequest(property_id="SH2", property=prop))
    with pytest.raises(ValueError, match="postcode districts"):
        history.record(request(postcode_district="BS16 1AB"))


def test_snapshot_frozen_priors_and_anonymous_fields(history):
    history.record(request("alice-home", wall_u=0.3), submitted_at=AS_OF)
    first = build_priors(request(), history=history, history_as_of=AS_OF)
    history.record(request("bob-home", wall_u=0.4), submitted_at=AS_OF)
    second = build_priors(request(), history=history, history_as_of=AS_OF)
    assert first.application_history["snapshot_sha256"] != second.application_history["snapshot_sha256"]
    assert first.application_history["eligible_households"] == 1
    raw = history.path.read_bytes()
    assert b"alice-home" not in raw and b"bob-home" not in raw
