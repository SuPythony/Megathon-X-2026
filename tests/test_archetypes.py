"""Archetype library and photo-to-twin drafts: geometry is consistent, fitting is deterministic, no live API calls."""

from types import SimpleNamespace

import pytest

from contracts.models import BuiltForm
from greentwin.twin.archetypes import ARCHETYPES, build_layout
from greentwin.twin.vision import (
    ExtractedRoom,
    FloorPlanExtraction,
    FuseBoardExtraction,
    VisionError,
    extract_floor_plan,
    extract_fuse_board,
    fit_layout,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


@pytest.mark.parametrize("aid", sorted(ARCHETYPES))
def test_rooms_tile_each_storey_exactly(aid):
    a = ARCHETYPES[aid]
    for storey in range(a.storeys):
        rooms = [r for r in a.rooms if r.storey == storey]
        assert rooms, f"{aid} storey {storey} has no rooms"
        for r in rooms:
            assert 0 <= r.x0 < r.x1 <= a.width_m and 0 <= r.y0 < r.y1 <= a.depth_m, r.key
        area = sum((r.x1 - r.x0) * (r.y1 - r.y0) for r in rooms)
        assert area == pytest.approx(a.width_m * a.depth_m), f"{aid} storey {storey} gaps or overlaps"
        for i, p in enumerate(rooms):
            for q in rooms[i + 1:]:
                overlap_x = min(p.x1, q.x1) - max(p.x0, q.x0)
                overlap_y = min(p.y1, q.y1) - max(p.y0, q.y0)
                assert not (overlap_x > 1e-9 and overlap_y > 1e-9), f"{aid}: {p.key} overlaps {q.key}"


@pytest.mark.parametrize("aid", sorted(ARCHETYPES))
def test_room_keys_unique_and_bedrooms_match(aid):
    a = ARCHETYPES[aid]
    keys = [r.key for r in a.rooms]
    assert len(keys) == len(set(keys))
    assert sum(r.room_type == "bedroom" for r in a.rooms) == a.bedrooms


@pytest.mark.parametrize("aid", sorted(ARCHETYPES))
def test_scaling_hits_target_area(aid):
    layout = build_layout(aid, total_floor_area_m2=100.0)
    assert layout.envelope().total_floor_area_m2 == pytest.approx(100.0, abs=0.1)


@pytest.mark.parametrize("aid", sorted(ARCHETYPES))
def test_no_openings_on_party_or_internal_walls(aid):
    layout = build_layout(aid)
    kinds = {w.wall_id: w.kind for w in layout.walls}
    assert all(kinds[o.wall_id] == "exterior" for o in layout.openings)


def test_built_form_sets_party_walls():
    detached = build_layout("interwar_semi_3bed", built_form=BuiltForm.detached).envelope()
    semi = build_layout("interwar_semi_3bed", built_form=BuiltForm.semi_detached).envelope()
    terrace = build_layout("interwar_semi_3bed", built_form=BuiltForm.mid_terrace).envelope()
    assert detached.party_wall_area_m2 == 0
    assert 0 < semi.party_wall_area_m2 < terrace.party_wall_area_m2
    # Turning an exterior wall into a party wall removes exposed wall area.
    assert detached.wall_area_m2 > semi.wall_area_m2 > terrace.wall_area_m2


def test_flat_has_no_roof_or_ground_floor_loss():
    env = build_layout("flat_2bed").envelope()
    assert env.roof_area_m2 == 0 and env.floor_area_m2 == 0 and env.party_wall_area_m2 > 0


def test_envelope_accounts_for_all_exterior_area():
    layout = build_layout("postwar_detached_4bed")
    env = layout.envelope()
    gross = sum(w.length_m * w.height_m for w in layout.walls if w.kind == "exterior")
    assert env.wall_area_m2 + env.window_area_m2 + env.door_area_m2 == pytest.approx(gross, abs=0.05)
    w, d = layout.footprint_m
    assert gross == pytest.approx(2 * (w + d) * layout.ceiling_height_m * layout.storeys, abs=0.05)


def test_orientation_follows_front_azimuth():
    layout = build_layout("bungalow_2bed", front_azimuth_deg=90.0)  # front faces east
    by_side = {w.side: w.orientation_deg for w in layout.walls if w.side}
    assert by_side == {"front": 90.0, "back": 270.0, "left": 180.0, "right": 0.0}


def test_floor_plan_contract_validates():
    plan = build_layout("modern_terrace_3bed", total_floor_area_m2=85).to_floor_plan("SH1", epc_floor_area_m2=80)
    assert plan.source == "parametric" and len(plan.rooms) == 8
    assert plan.area_vs_epc_pct == pytest.approx(6.2, abs=0.1)


def _extraction(**overrides):
    base = dict(
        archetype_id="interwar_semi_3bed", archetype_confidence=0.9, archetype_reason="bay window, side hall",
        built_form_guess=BuiltForm.semi_detached, storeys=2, bedrooms=3,
        rooms=[
            ExtractedRoom(label="Lounge", room_type="living", storey=0, confidence=0.95),
            ExtractedRoom(label="Kitchen", room_type="kitchen", storey=0, confidence=0.9),
            ExtractedRoom(label="Conservatory", room_type="conservatory", storey=0, confidence=0.8),
            ExtractedRoom(label="Master Bedroom", room_type="bedroom", storey=1, confidence=0.4),
        ],
    )
    base.update(overrides)
    return FloorPlanExtraction(**base)


def test_fit_layout_maps_labels_and_flags_gaps():
    fitted = fit_layout(_extraction(total_floor_area_m2=92.0))
    labels = {r.room_id: r.label for r in fitted.layout.rooms}
    assert labels["living"] == "Lounge" and labels["bed1"] == "Master Bedroom"  # largest bedroom
    assert fitted.unmatched_rooms == ["Conservatory"]
    assert fitted.needs_confirmation == ["Master Bedroom"]
    assert fitted.area_source == "printed_total"
    assert fitted.layout.envelope().total_floor_area_m2 == pytest.approx(92.0, abs=0.1)


def test_fit_layout_low_archetype_confidence_needs_confirmation():
    fitted = fit_layout(_extraction(archetype_confidence=0.3))
    assert fitted.needs_confirmation[0] == "archetype_id"
    assert fitted.area_source == "archetype_default"


class FakeMessages:
    def __init__(self, parsed, stop_reason="end_turn"):
        self.parsed, self.stop_reason, self.calls = parsed, stop_reason, []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(stop_reason=self.stop_reason, stop_details=None, parsed_output=self.parsed)


def test_extract_floor_plan_uses_sonnet_structured_output_and_caches(tmp_path):
    fake = SimpleNamespace(messages=FakeMessages(_extraction()))
    first = extract_floor_plan(PNG, client=fake, cache_dir=tmp_path)
    second = extract_floor_plan(PNG, client=fake, cache_dir=tmp_path)
    assert first == second
    assert len(fake.messages.calls) == 1, "second call should come from cache"
    call = fake.messages.calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    assert call["output_format"] is FloorPlanExtraction
    assert call["messages"][0]["content"][0]["source"]["media_type"] == "image/png"


def test_extract_fuse_board_refusal_raises(tmp_path):
    fake = SimpleNamespace(messages=FakeMessages(None, stop_reason="refusal"))
    with pytest.raises(VisionError):
        extract_fuse_board(PNG, client=fake, cache_dir=tmp_path)
    assert not list(tmp_path.iterdir()), "failed calls must not be cached"


def test_fuse_board_schema_example():
    FuseBoardExtraction.model_validate({
        "board_type": "split_load_rcd", "main_switch_rating_a": 100,
        "circuits": [{"position": 1, "device": "mcb", "rating_a": 40, "curve": "B", "label_raw": "Shower",
                      "circuit_kind": "shower", "rooms_guess": ["bathroom"], "confidence": 0.9}],
    })
