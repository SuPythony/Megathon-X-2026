"""Room scene JSON: every house type looks distinct, appliances land inside their rooms, custom placement wins."""

import json

import pytest

from contracts.models import BuiltForm
from greentwin.twin.archetypes import ARCHETYPES, build_layout
from greentwin.twin.scene import (
    AppliancePlacement,
    RoomScene,
    _bbox,
    _half_extents,
    appliance_visuals,
    build_scene,
    house_styles,
    write_examples,
)


def test_every_archetype_has_a_distinct_style():
    styles = house_styles()
    assert set(styles) == set(ARCHETYPES)
    looks = {(s.roof, s.wall_material, s.wall_colour, s.roof_colour) for s in styles.values()}
    assert len(looks) == len(styles), "two house types share the same look"


def _inside(app, scene):
    room = next(r for r in scene.rooms if r.room_id == app.room_id)
    x0, y0, x1, y1 = _bbox(room)
    hx, hy = _half_extents(app.size_m[0], app.size_m[1], app.rotation_deg)
    x, y, _ = app.position_m
    return x0 + hx - 1e-6 <= x <= x1 - hx + 1e-6 and y0 + hy - 1e-6 <= y <= y1 - hy + 1e-6


@pytest.mark.parametrize("aid", sorted(ARCHETYPES))
def test_auto_placed_appliances_sit_inside_rooms_without_overlap(aid):
    room = next(r for r in ARCHETYPES[aid].rooms if r.room_type == "kitchen")
    apps = [AppliancePlacement(appliance_id=f"a{i}", type=t, room_id=room.key)
            for i, t in enumerate(["fridge_freezer", "washing_machine", "dishwasher", "hob", "kettle"])]
    scene = build_scene(build_layout(aid), "T", apps)
    floor = [a for a in scene.appliances if a.mount == "floor"]
    for a in scene.appliances:
        assert a.placement == "auto"
        assert _inside(a, scene), f"{a.type} outside {a.room_id}"
    for i, p in enumerate(floor):
        for q in floor[i + 1:]:
            (px, py, _), (qx, qy, _) = p.position_m, q.position_m
            phx, phy = _half_extents(p.size_m[0], p.size_m[1], p.rotation_deg)
            qhx, qhy = _half_extents(q.size_m[0], q.size_m[1], q.rotation_deg)
            assert abs(px - qx) >= phx + qhx - 1e-6 or abs(py - qy) >= phy + qhy - 1e-6, f"{p.type} overlaps {q.type}"


def test_custom_placement_is_respected_and_clamped():
    layout = build_layout("victorian_terrace_2bed")
    apps = [
        AppliancePlacement(appliance_id="tv", type="tv", room_id="living", u=0.5, v=0.5, rotation_deg=90),
        AppliancePlacement(appliance_id="fridge", type="fridge_freezer", room_id="kitchen", u=0.0, v=1.0),
    ]
    scene = build_scene(layout, "T", apps)
    tv, fridge = scene.appliances
    living = next(r for r in scene.rooms if r.room_id == "living")
    x0, y0, x1, y1 = _bbox(living)
    assert tv.placement == "custom" and tv.rotation_deg == 90
    assert tv.position_m[:2] == pytest.approx(((x0 + x1) / 2, (y0 + y1) / 2), abs=1e-3)
    assert tv.position_m[2] == pytest.approx(0.9)  # wall-mounted height
    assert _inside(fridge, scene) and "moved inside the room" in fridge.warnings
    assert fridge.rotation_deg in (90.0, 0.0)  # faces away from the nearest wall


def _overlap(p, q):
    (px, py, _), (qx, qy, _) = p.position_m, q.position_m
    phx, phy = _half_extents(p.size_m[0], p.size_m[1], p.rotation_deg)
    qhx, qhy = _half_extents(q.size_m[0], q.size_m[1], q.rotation_deg)
    return abs(px - qx) < phx + qhx - 1e-6 and abs(py - qy) < phy + qhy - 1e-6


def test_auto_placement_steps_around_custom_items_and_keeps_order():
    apps = [
        AppliancePlacement(appliance_id="wm", type="washing_machine", room_id="kitchen"),
        AppliancePlacement(appliance_id="ff", type="fridge_freezer", room_id="kitchen", u=0.0, v=1.0),
        AppliancePlacement(appliance_id="dw", type="dishwasher", room_id="kitchen"),
    ]
    scene = build_scene(build_layout("victorian_terrace_2bed"), "T", apps)
    assert [a.appliance_id for a in scene.appliances] == ["wm", "ff", "dw"]
    wm, ff, dw = scene.appliances
    assert ff.placement == "custom" and not _overlap(ff, wm) and not _overlap(ff, dw) and not _overlap(wm, dw)


def test_example_scenes_have_no_3d_overlaps(tmp_path):
    for path in write_examples(tmp_path):
        if not path.name.endswith(".scene.json"):
            continue
        apps = RoomScene.model_validate_json(path.read_text(encoding="utf-8")).appliances
        for i, p in enumerate(apps):
            for q in apps[i + 1:]:
                if p.room_id is None or p.room_id != q.room_id:
                    continue
                pz, qz = p.position_m[2], q.position_m[2]
                if pz + p.size_m[2] <= qz or qz + q.size_m[2] <= pz:
                    continue  # stacked, e.g. hob above oven
                assert not _overlap(p, q), f"{path.name}: {p.type} overlaps {q.type}"


def test_appliance_on_upper_floor_gets_storey_height():
    layout = build_layout("interwar_semi_3bed")
    scene = build_scene(layout, "T", [AppliancePlacement(appliance_id="c", type="immersion_heater", room_id="landing")])
    assert scene.appliances[0].position_m[2] == pytest.approx(layout.ceiling_height_m)


def test_unusual_room_warns_and_unknown_inputs_raise():
    layout = build_layout("bungalow_2bed")
    scene = build_scene(layout, "T", [AppliancePlacement(appliance_id="x", type="hob", room_id="bed1")])
    assert scene.appliances[0].warnings
    with pytest.raises(ValueError):
        build_scene(layout, "T", [AppliancePlacement(appliance_id="x", type="toaster_9000", room_id="kitchen")])
    with pytest.raises(ValueError):
        build_scene(layout, "T", [AppliancePlacement(appliance_id="x", type="tv", room_id="ballroom")])


def test_outdoor_appliances_sit_outside_the_footprint():
    layout = build_layout("postwar_detached_4bed")
    scene = build_scene(layout, "T", [AppliancePlacement(appliance_id="ev", type="ev_charger", room_id="outside"),
                                      AppliancePlacement(appliance_id="tub", type="hot_tub")])
    w, d = layout.footprint_m
    ev, tub = scene.appliances
    assert ev.room_id is None and ev.position_m[1] < 0
    assert tub.position_m[1] > d


def test_neighbours_follow_built_form():
    detached = build_scene(build_layout("interwar_semi_3bed", built_form=BuiltForm.detached), "T")
    terrace = build_scene(build_layout("interwar_semi_3bed", built_form=BuiltForm.mid_terrace), "T")
    flat = build_scene(build_layout("flat_2bed"), "T")
    assert detached.context == []
    assert {c.relation for c in terrace.context} == {"left", "right"}
    assert {c.relation for c in flat.context} == {"left", "right", "above", "below"}


def test_garage_moves_off_a_party_wall_or_is_dropped():
    semi = build_scene(build_layout("postwar_detached_4bed", built_form=BuiltForm.semi_detached), "T")
    terrace = build_scene(build_layout("postwar_detached_4bed", built_form=BuiltForm.mid_terrace), "T")
    garage = [f for f in semi.features if f.kind == "garage"]
    assert garage and garage[0].side == "left"
    assert not [f for f in terrace.features if f.kind == "garage"]


def test_roof_and_windows_heights():
    scene = build_scene(build_layout("victorian_terrace_2bed"), "T")
    assert scene.roof.ridge_z_m > scene.roof.eave_z_m == pytest.approx(2 * scene.ceiling_height_m)
    for o in scene.openings:
        assert o.sill_m + o.height_m <= scene.ceiling_height_m


def test_examples_and_schemas_round_trip(tmp_path):
    paths = write_examples(tmp_path)
    scenes = [p for p in paths if p.name.endswith(".scene.json")]
    assert len(scenes) == len(ARCHETYPES)
    for p in scenes:
        RoomScene.model_validate_json(p.read_text(encoding="utf-8"))
    schema = json.loads((tmp_path / "room_scene.schema.json").read_text(encoding="utf-8"))
    assert "appliances" in schema["properties"]


def test_visual_catalogue_types_cover_question_set():
    needed = {"fridge_freezer", "washing_machine", "tumble_dryer", "dishwasher", "electric_oven", "hob",
              "electric_shower", "immersion_heater", "tv", "desktop_pc", "electric_heater", "dehumidifier",
              "ev_charger", "hot_tub"}
    assert needed <= set(appliance_visuals())
