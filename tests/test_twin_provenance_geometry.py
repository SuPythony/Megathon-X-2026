"""Input provenance and confirmed geometry."""

import pytest
from pydantic import ValidationError

from greentwin.twin.archetypes import build_layout
from greentwin.twin.geometry import GeometryConfirmation, apply_geometry, confirm, from_layout
from greentwin.twin.inputs import build_physics_inputs
from greentwin.twin.provenance import classify
from greentwin.twin.showcase import load_showcase


def _prior_sigma(pkg):
    return next(p.sigma for p in pkg.priors if p.param == "heat_loss_multiplier")


def test_footprint_scales_each_axis():
    layout = build_layout("interwar_semi_3bed", footprint_m=(8.0, 6.0))
    assert layout.footprint_m == (8.0, 6.0)
    assert layout.envelope().total_floor_area_m2 == pytest.approx(8.0 * 6.0 * 2, abs=0.1)
    assert build_layout("interwar_semi_3bed").geometry_source == "archetype_default"
    assert build_layout("interwar_semi_3bed", total_floor_area_m2=90).geometry_source == "epc_fitted"


def test_confirmed_geometry_is_used_and_flagged():
    sc = load_showcase("SH2")
    g = confirm(from_layout(sc.layout).model_copy(update={"footprint_width_m": 7.2, "footprint_depth_m": 8.0}))
    assert g.confirmed_by_user and g.confirmed_at
    layout, flags = apply_geometry(g, sc.prop)
    assert layout.geometry_source == "user_confirmed" and layout.footprint_m == (7.2, 8.0)
    assert any("differs from the EPC" in f for f in flags)  # 115 m2 vs 95 m2
    draft, draft_flags = apply_geometry(from_layout(sc.layout), sc.prop)
    assert draft.geometry_source == "vision_fitted" and "not confirmed" in draft_flags[-1]
    with pytest.raises(ValidationError):
        GeometryConfirmation(archetype_id="flat_2bed", footprint_width_m=1, footprint_depth_m=5)


def test_provenance_and_prior_follow_confirmation():
    sc = load_showcase("SH2")
    fitted = build_physics_inputs(sc.prop, layout=sc.layout, appliances=sc.appliances, intake=sc.intake, usage=sc.usage)
    g = confirm(from_layout(sc.layout))
    confirmed = build_physics_inputs(sc.prop, layout=sc.layout, appliances=sc.appliances,
                                     intake=sc.intake.model_copy(update={"geometry": g}), usage=sc.usage)
    prov = {p.field: p for p in confirmed.input_provenance}
    assert prov["wall_area_m2"].confirmed and prov["wall_area_m2"].status == "confirmed"
    assert prov["wall_u"].status == "to_verify" and not prov["wall_u"].confirmed
    assert prov["occupants"].source == "user_stated" and prov["occupants"].ref.startswith("intake:")
    assert prov["weather"].status == "synthetic"
    assert {p.field for p in fitted.input_provenance} >= {"wall_u", "volume_m3", "heat_capacity_kj_per_k", "meter_readings"}
    assert _prior_sigma(confirmed) < _prior_sigma(fitted)
    s = fitted.provenance_summary
    assert sum(s.heat_loss_share_by_status.values()) == pytest.approx(100, abs=0.5)
    assert confirmed.geometry.confirmed_by_user and confirmed.provenance_summary.geometry_source == "user_confirmed"


def test_classify_rules():
    assert classify("wall_u", "epc (measured)", "epc_fitted")[:2] == ("epc_measured", "measured")
    assert classify("wall_area_m2", "layout:x (floor plan)", "vision_fitted")[0] == "vision_draft"
    assert classify("roof_u", "rdsap_default loft table (to_verify)", "epc_fitted")[1] == "to_verify"
    assert classify("air_changes_per_hour", "assumption by age band", "epc_fitted")[1] == "assumption"
