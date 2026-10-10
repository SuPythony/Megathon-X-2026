"""Room-level 3D scene JSON for the twin viewer.

``build_scene`` turns a fitted ``TwinLayout`` plus the user's appliances into one JSON document the 3D viewer
renders directly: rooms, walls, openings, roof, exterior features, faded neighbour blocks and appliances.

- Each archetype has its own exterior look (``house_styles.json``): roof form, materials, colours, bay windows,
  porches, chimneys, garages, balconies. Homes of one type look alike; different types look different.
- Appliances are placed by the user (``u``/``v`` = position as a fraction of the room's width/depth, optional
  rotation) or, when no position is given, auto-placed along the room's walls. Visuals come from
  ``appliance_visuals.json``.
- The scene never calculates energy. Numbers are attached later from the twin's results.

Conventions: metres; plan x across the front, y from the front (0) to the back; z up from the ground floor.
Every box position is its bottom-centre. Rotation is about z in degrees: 0 = the object's front faces -y
(towards the street), 90 = faces +x, 180 = faces +y, 270 = faces -x.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from greentwin.twin.archetypes import (
    ARCHETYPES,
    LayoutOpening,
    LayoutWall,
    RoomType,
    Side,
    TwinLayout,
    build_layout,
)

SCENE_VERSION = "room_scene/0.1"
HERE = Path(__file__).resolve().parent
BASIS_LABEL = ("Layout from the closest standard UK house type, fitted to this home; room heat loss is estimated, "
               "only whole-house totals are calibrated to bills")

Vec3 = tuple[float, float, float]
_FACING: dict[Side, float] = {"back": 0.0, "right": 270.0, "front": 180.0, "left": 90.0}  # rotation facing into room


# --------------------------------------------------------------------------- catalogues
class ApplianceVisual(BaseModel):
    label: str
    shape: str
    w: float
    d: float
    h: float
    mount: Literal["floor", "counter", "wall", "ceiling", "outdoor"]
    mount_height_m: float
    colour: str
    accent: str
    icon: str
    rooms: list[str]


class Feature(BaseModel):
    kind: Literal["bay_window", "porch", "chimney", "garage", "balcony", "garden_wall"]
    side: Optional[Side] = None
    room: Optional[str] = None
    position: Optional[Literal["left_end", "right_end", "centre", "party_side"]] = None
    width_m: Optional[float] = None
    depth_m: Optional[float] = None
    height_m: Optional[float] = None
    projection_m: Optional[float] = None
    storeys: Optional[int] = None
    style: Optional[str] = None
    roof: Optional[str] = None
    door_colour: Optional[str] = None
    balustrade: Optional[str] = None


class HouseStyle(BaseModel):
    wall_material: str
    wall_colour: str
    wall_upper_colour: Optional[str] = None
    roof: Literal["gable", "hip", "flat"]
    roof_pitch_deg: float
    ridge_axis: Literal["x", "y"]
    roof_colour: str
    roof_material: str
    frame_colour: str
    window_style: str
    door_colour: str
    features: list[Feature] = Field(default_factory=list)


@lru_cache(maxsize=1)
def appliance_visuals() -> dict[str, ApplianceVisual]:
    raw = json.loads((HERE / "appliance_visuals.json").read_text(encoding="utf-8"))
    return {k: ApplianceVisual.model_validate(v) for k, v in raw["types"].items()}


@lru_cache(maxsize=1)
def house_styles() -> dict[str, HouseStyle]:
    raw = json.loads((HERE / "house_styles.json").read_text(encoding="utf-8"))
    return {k: HouseStyle.model_validate(v) for k, v in raw["styles"].items()}


# --------------------------------------------------------------------------- input: user appliances
class AppliancePlacement(BaseModel):
    """One appliance as the user entered it. Leave u/v empty to auto-place along the room's walls."""

    appliance_id: str
    type: str = Field(description="Key in appliance_visuals.json")
    room_id: Optional[str] = Field(None, description="Archetype room key; None or 'outside' = outdoors")
    label: Optional[str] = None
    u: Optional[float] = Field(None, description="Indoor: 0..1 across the room's width (x)")
    v: Optional[float] = Field(None, description="Indoor: 0..1 across the room's depth (y)")
    x_m: Optional[float] = Field(None, description="Outdoor only: plan x in metres")
    y_m: Optional[float] = Field(None, description="Outdoor only: plan y in metres (negative = in front)")
    rotation_deg: Optional[float] = None


# --------------------------------------------------------------------------- output scene
class SceneRoom(BaseModel):
    room_id: str
    label: str
    room_type: RoomType
    storey: int
    polygon_m: list[tuple[float, float]]
    area_m2: float
    floor_z_m: float
    height_m: float


class SceneWall(LayoutWall):
    z0_m: float


class SceneOpening(LayoutOpening):
    sill_m: float
    z0_m: float


class SceneRoof(BaseModel):
    type: Literal["gable", "hip", "flat"]
    pitch_deg: float
    ridge_axis: Literal["x", "y"]
    eave_z_m: float
    ridge_z_m: float
    colour: str
    material: str


class SceneFeature(BaseModel):
    kind: str
    position_m: Vec3
    size_m: Vec3
    side: Optional[Side] = None
    room_id: Optional[str] = None
    style: Optional[str] = None
    colour: Optional[str] = None
    extra: dict[str, str | float | int] = Field(default_factory=dict)


class ContextBlock(BaseModel):
    """A neighbouring dwelling, drawn faded so party walls read clearly."""

    relation: Literal["left", "right", "back", "above", "below"]
    position_m: Vec3
    size_m: Vec3


class SceneAppliance(BaseModel):
    appliance_id: str
    type: str
    label: str
    room_id: Optional[str]
    placement: Literal["custom", "auto"]
    position_m: Vec3
    rotation_deg: float
    size_m: Vec3
    shape: str
    mount: str
    colour: str
    accent: str
    icon: str
    warnings: list[str] = Field(default_factory=list)


class SceneStyle(BaseModel):
    wall_material: str
    wall_colour: str
    wall_upper_colour: Optional[str]
    frame_colour: str
    window_style: str
    door_colour: str


class RoomScene(BaseModel):
    scene_version: str = SCENE_VERSION
    property_id: str
    archetype_id: str
    archetype_name: str
    built_form: str
    storeys: int
    footprint_m: tuple[float, float]
    ceiling_height_m: float
    front_azimuth_deg: float
    style: SceneStyle
    rooms: list[SceneRoom]
    walls: list[SceneWall]
    openings: list[SceneOpening]
    roof: SceneRoof
    features: list[SceneFeature]
    context: list[ContextBlock]
    appliances: list[SceneAppliance]
    basis: str = BASIS_LABEL


# --------------------------------------------------------------------------- helpers
def _bbox(room: SceneRoom) -> tuple[float, float, float, float]:
    xs = [p[0] for p in room.polygon_m]
    ys = [p[1] for p in room.polygon_m]
    return min(xs), min(ys), max(xs), max(ys)


def _half_extents(w: float, d: float, rot: float) -> tuple[float, float]:
    r = math.radians(rot)
    return abs(w / 2 * math.cos(r)) + abs(d / 2 * math.sin(r)), abs(w / 2 * math.sin(r)) + abs(d / 2 * math.cos(r))


def _r(v: float) -> float:
    return round(v, 3)


def _sill(kind: str, h: float, ceiling: float) -> float:
    if kind != "window":
        return 0.0
    return max(0.1, min(0.9, ceiling - h - 0.15))


# --------------------------------------------------------------------------- appliance placement
COUNTER_HEIGHT_M = 0.9


def _layers(vis: ApplianceVisual) -> list[str]:
    """Vertical bands an appliance occupies: below worktop height ('floor'), above it ('raised'), or both."""
    bottom, top = vis.mount_height_m, vis.mount_height_m + vis.h
    return [name for name, hit in (("floor", bottom < COUNTER_HEIGHT_M), ("raised", top > COUNTER_HEIGHT_M)) if hit]


class _WallCursor:
    """Finds free wall space per room and layer (floor-standing vs counter/wall-mounted), stepping around
    anything already placed, including the user's custom placements."""

    ORDER: tuple[Side, ...] = ("back", "left", "right", "front")
    STEP = 0.05
    GAP = 0.05

    def __init__(self) -> None:
        self.occupied: dict[tuple[str, str], list[tuple[float, float, float, float]]] = {}

    def reserve(self, room_id: str, layers: list[str], x: float, y: float, hx: float, hy: float) -> None:
        for layer in layers:
            self.occupied.setdefault((room_id, layer), []).append((x - hx, y - hy, x + hx, y + hy))

    def free(self, room_id: str, layers: list[str], x: float, y: float, hx: float, hy: float) -> bool:
        g = self.GAP
        return all(x + hx + g <= a0 or x - hx - g >= a1 or y + hy + g <= b0 or y - hy - g >= b1
                   for layer in layers for a0, b0, a1, b1 in self.occupied.get((room_id, layer), []))

    def place(self, room: SceneRoom, layers: list[str], w: float, d: float) -> Optional[tuple[float, float, float]]:
        x0, y0, x1, y1 = _bbox(room)
        for side in self.ORDER:
            along = (x1 - x0) if side in ("back", "front") else (y1 - y0)
            across = (y1 - y0) if side in ("back", "front") else (x1 - x0)
            if w + 2 * self.GAP > along or d > across:
                continue
            s = self.GAP
            while s + w + self.GAP <= along + 1e-9:
                if side == "back":
                    x, y = x0 + s + w / 2, y1 - d / 2
                elif side == "front":
                    x, y = x1 - s - w / 2, y0 + d / 2
                elif side == "left":
                    x, y = x0 + d / 2, y1 - s - w / 2
                else:
                    x, y = x1 - d / 2, y0 + s + w / 2
                rot = _FACING[side]
                hx, hy = _half_extents(w, d, rot)
                if self.free(room.room_id, layers, x, y, hx, hy):
                    self.reserve(room.room_id, layers, x, y, hx, hy)
                    return x, y, rot
                s += self.STEP
        return None


def _place_appliances(
    placements: list[AppliancePlacement], rooms: dict[str, SceneRoom], room_types: dict[str, str],
    footprint: tuple[float, float],
) -> list[SceneAppliance]:
    visuals = appliance_visuals()
    width, depth = footprint
    cursor = _WallCursor()
    results: dict[int, SceneAppliance] = {}
    custom_first = sorted(range(len(placements)),
                          key=lambda i: placements[i].u is None or placements[i].v is None)

    for i in custom_first:
        p = placements[i]
        if p.type not in visuals:
            raise ValueError(f"unknown appliance type {p.type!r}; known: {sorted(visuals)}")
        vis = visuals[p.type]
        warnings: list[str] = []
        outdoor = p.room_id in (None, "outside")

        if outdoor:
            if vis.mount != "outdoor":
                warnings.append(f"{vis.label} is normally indoors")
            if p.x_m is not None and p.y_m is not None:
                x, y, rot, mode = p.x_m, p.y_m, p.rotation_deg or 0.0, "custom"
            elif p.type == "hot_tub":
                x, y, rot, mode = width / 2, depth + 1.5 + vis.d / 2, 180.0, "auto"
            else:  # wall-mounted outdoor units go on the front wall, outside
                x, y, rot, mode = width * 0.85, -vis.d / 2, 0.0, "auto"
            z = vis.mount_height_m
            room_id = None
        else:
            if p.room_id not in rooms:
                raise ValueError(f"appliance {p.appliance_id}: unknown room {p.room_id!r}; known: {sorted(rooms)}")
            room = rooms[p.room_id]
            room_id = room.room_id
            if room_types[room_id] not in vis.rooms:
                warnings.append(f"{vis.label} is unusual in a {room_types[room_id]}")
            x0, y0, x1, y1 = _bbox(room)

            if p.u is not None and p.v is not None:
                mode = "custom"
                x = x0 + max(0.0, min(1.0, p.u)) * (x1 - x0)
                y = y0 + max(0.0, min(1.0, p.v)) * (y1 - y0)
                if p.rotation_deg is not None:
                    rot = p.rotation_deg
                else:  # face away from the nearest wall
                    dist = {"left": x - x0, "right": x1 - x, "front": y - y0, "back": y1 - y}
                    rot = _FACING[min(dist, key=dist.get)]
                hx, hy = _half_extents(vis.w, vis.d, rot)
                cx = min(max(x, x0 + hx), x1 - hx) if x1 - x0 >= 2 * hx else (x0 + x1) / 2
                cy = min(max(y, y0 + hy), y1 - hy) if y1 - y0 >= 2 * hy else (y0 + y1) / 2
                if abs(cx - x) > 1e-6 or abs(cy - y) > 1e-6:
                    warnings.append("moved inside the room")
                x, y = cx, cy
                layers = _layers(vis)
                if not cursor.free(room_id, layers, x, y, hx, hy):
                    warnings.append("overlaps another appliance")
                cursor.reserve(room_id, layers, x, y, hx, hy)
            else:
                mode = "auto"
                spot = cursor.place(room, _layers(vis), vis.w, vis.d)
                if spot is None:
                    x, y, rot = (x0 + x1) / 2, (y0 + y1) / 2, 0.0
                    warnings.append("no free wall space; placed in the room centre")
                else:
                    x, y, rot = spot
                if p.rotation_deg is not None:
                    rot = p.rotation_deg
            z = room.floor_z_m + vis.mount_height_m

        results[i] = SceneAppliance(
            appliance_id=p.appliance_id, type=p.type, label=p.label or vis.label, room_id=room_id,
            placement=mode, position_m=(_r(x), _r(y), _r(z)), rotation_deg=rot % 360.0,
            size_m=(vis.w, vis.d, vis.h), shape=vis.shape, mount=vis.mount,
            colour=vis.colour, accent=vis.accent, icon=vis.icon, warnings=warnings,
        )
    return [results[i] for i in range(len(placements))]


# --------------------------------------------------------------------------- exterior
def _roof(layout: TwinLayout, style: HouseStyle) -> SceneRoof:
    width, depth = layout.footprint_m
    eave = layout.storeys * layout.ceiling_height_m
    span = depth if style.ridge_axis == "x" else width
    rise = 0.0 if style.roof == "flat" else span / 2 * math.tan(math.radians(style.roof_pitch_deg))
    return SceneRoof(type=style.roof, pitch_deg=style.roof_pitch_deg, ridge_axis=style.ridge_axis,
                     eave_z_m=_r(eave), ridge_z_m=_r(eave + rise), colour=style.roof_colour,
                     material=style.roof_material)


def _features(layout: TwinLayout, style: HouseStyle, rooms: dict[str, SceneRoom], roof: SceneRoof) -> list[SceneFeature]:
    width, depth = layout.footprint_m
    ch = layout.ceiling_height_m
    party = set(layout.party_sides)
    out: list[SceneFeature] = []

    for f in style.features:
        room = rooms.get(f.room) if f.room else None
        cx_room = (sum(_bbox(room)[0::2]) / 2) if room else width / 2
        if f.kind == "bay_window":
            x0, _, x1, _ = _bbox(room)
            w = min(f.width_m, 0.8 * (x1 - x0))
            out.append(SceneFeature(kind=f.kind, side="front", room_id=f.room, colour=style.frame_colour,
                                    position_m=(_r(cx_room), _r(-f.projection_m / 2), 0.0),
                                    size_m=(_r(w), f.projection_m, _r((f.storeys or 1) * ch))))
        elif f.kind == "porch":
            enclosed = f.style == "enclosed"
            out.append(SceneFeature(kind=f.kind, side="front", room_id=f.room, style=f.style,
                                    colour=style.wall_colour if enclosed else style.roof_colour,
                                    position_m=(_r(cx_room), _r(-f.depth_m / 2), 0.0 if enclosed else 2.2),
                                    size_m=(f.width_m, f.depth_m, 2.4 if enclosed else 0.15)))
        elif f.kind == "chimney":
            if roof.type == "flat":
                continue
            pos = f.position
            if pos == "party_side":
                pos = "right_end" if "right" in party else "left_end" if "left" in party else "centre"
            x = {"left_end": f.width_m / 2, "right_end": width - f.width_m / 2, "centre": width / 2}[pos]
            out.append(SceneFeature(kind=f.kind, colour=style.wall_colour,
                                    position_m=(_r(x), _r(depth / 2), _r(roof.ridge_z_m - 0.5)),
                                    size_m=(f.width_m, f.depth_m, _r(f.height_m + 0.5))))
        elif f.kind == "garage":
            side = f.side if f.side not in party else next((s for s in ("right", "left") if s not in party), None)
            if side is None:
                continue  # no free side (terraced): no garage
            x = width + f.width_m / 2 if side == "right" else -f.width_m / 2
            out.append(SceneFeature(kind=f.kind, side=side, style=f.roof, colour=style.wall_colour,
                                    position_m=(_r(x), _r(depth / 2), 0.0), size_m=(f.width_m, _r(depth), f.height_m),
                                    extra={"door_colour": f.door_colour or style.door_colour, "heated": 0}))
        elif f.kind == "balcony":
            out.append(SceneFeature(kind=f.kind, side="front", room_id=f.room, style=f.balustrade,
                                    colour=style.frame_colour,
                                    position_m=(_r(cx_room), _r(-f.depth_m / 2), room.floor_z_m if room else 0.0),
                                    size_m=(f.width_m, f.depth_m, 1.0)))
        elif f.kind == "garden_wall":
            out.append(SceneFeature(kind=f.kind, side="front", colour=style.wall_colour,
                                    position_m=(_r(width / 2), _r(-f.depth_m), 0.0),
                                    size_m=(_r(width), 0.25, f.height_m)))
    return out


def _context(layout: TwinLayout) -> list[ContextBlock]:
    width, depth = layout.footprint_m
    h = layout.storeys * layout.ceiling_height_m
    out: list[ContextBlock] = []
    offsets = {"left": (-width / 2, depth / 2, 0.0), "right": (width * 1.5, depth / 2, 0.0),
               "back": (width / 2, depth * 1.5, 0.0)}
    for side in layout.party_sides:
        if side in offsets:
            out.append(ContextBlock(relation=side, position_m=tuple(_r(v) for v in offsets[side]),
                                    size_m=(_r(width), _r(depth), _r(h))))
    if not layout.roof_exposed:
        out.append(ContextBlock(relation="above", position_m=(_r(width / 2), _r(depth / 2), _r(h)),
                                size_m=(_r(width), _r(depth), _r(h))))
    if not layout.ground_floor_exposed:
        out.append(ContextBlock(relation="below", position_m=(_r(width / 2), _r(depth / 2), _r(-h)),
                                size_m=(_r(width), _r(depth), _r(h))))
    return out


# --------------------------------------------------------------------------- entry point
def build_scene(layout: TwinLayout, property_id: str,
                appliances: Optional[list[AppliancePlacement]] = None) -> RoomScene:
    style = house_styles()[layout.archetype_id]
    ch = layout.ceiling_height_m
    rooms = {
        r.room_id: SceneRoom(room_id=r.room_id, label=r.label, room_type=layout.room_types[r.room_id],
                             storey=r.storey, polygon_m=r.polygon_m, area_m2=r.area_m2,
                             floor_z_m=_r(r.storey * ch), height_m=ch)
        for r in layout.rooms
    }
    walls = [SceneWall(**w.model_dump(), z0_m=_r(w.storey * ch)) for w in layout.walls]
    storey_of = {w.wall_id: w.storey for w in layout.walls}
    openings = []
    for o in layout.openings:
        sill = _sill(o.kind, o.height_m, ch)
        openings.append(SceneOpening(**o.model_dump(), sill_m=_r(sill), z0_m=_r(storey_of[o.wall_id] * ch + sill)))
    roof = _roof(layout, style)

    return RoomScene(
        property_id=property_id, archetype_id=layout.archetype_id,
        archetype_name=ARCHETYPES[layout.archetype_id].name, built_form=layout.built_form.value,
        storeys=layout.storeys, footprint_m=layout.footprint_m, ceiling_height_m=ch,
        front_azimuth_deg=layout.front_azimuth_deg,
        style=SceneStyle(**style.model_dump(include=set(SceneStyle.model_fields))),
        rooms=list(rooms.values()), walls=walls, openings=openings, roof=roof,
        features=_features(layout, style, rooms, roof), context=_context(layout),
        appliances=_place_appliances(appliances or [], rooms, layout.room_types, layout.footprint_m),
    )


# --------------------------------------------------------------------------- examples + schema
_EXAMPLE_APPLIANCES: dict[str, list[dict]] = {
    "victorian_terrace_2bed": [
        {"appliance_id": "a1", "type": "fridge_freezer", "room_id": "kitchen", "u": 0.08, "v": 0.9},
        {"appliance_id": "a2", "type": "washing_machine", "room_id": "kitchen"},
        {"appliance_id": "a3", "type": "electric_oven", "room_id": "kitchen"},
        {"appliance_id": "a4", "type": "hob", "room_id": "kitchen"},
        {"appliance_id": "a5", "type": "tv", "room_id": "living"},
        {"appliance_id": "a6", "type": "electric_shower", "room_id": "bath"},
    ],
    "interwar_semi_3bed": [
        {"appliance_id": "a1", "type": "fridge_freezer", "room_id": "kitchen"},
        {"appliance_id": "a2", "type": "tumble_dryer", "room_id": "kitchen"},
        {"appliance_id": "a3", "type": "dishwasher", "room_id": "kitchen"},
        {"appliance_id": "a4", "type": "tv", "room_id": "living", "u": 0.5, "v": 0.95, "rotation_deg": 0},
        {"appliance_id": "a5", "type": "games_console", "room_id": "living"},
        {"appliance_id": "a6", "type": "immersion_heater", "room_id": "landing"},
    ],
    "postwar_detached_4bed": [
        {"appliance_id": "a1", "type": "fridge_freezer", "room_id": "kitchen"},
        {"appliance_id": "a2", "type": "electric_oven", "room_id": "kitchen"},
        {"appliance_id": "a3", "type": "hob", "room_id": "kitchen"},
        {"appliance_id": "a4", "type": "desktop_pc", "room_id": "study"},
        {"appliance_id": "a5", "type": "tv", "room_id": "living"},
        {"appliance_id": "a6", "type": "ev_charger", "room_id": "outside"},
        {"appliance_id": "a7", "type": "hot_tub", "room_id": "outside"},
    ],
    "modern_terrace_3bed": [
        {"appliance_id": "a1", "type": "fridge_freezer", "room_id": "kitchen"},
        {"appliance_id": "a2", "type": "washing_machine", "room_id": "kitchen"},
        {"appliance_id": "a3", "type": "microwave", "room_id": "kitchen"},
        {"appliance_id": "a4", "type": "tv", "room_id": "living"},
        {"appliance_id": "a5", "type": "tv", "room_id": "bed1", "label": "Bedroom TV"},
    ],
    "bungalow_2bed": [
        {"appliance_id": "a1", "type": "fridge_freezer", "room_id": "kitchen"},
        {"appliance_id": "a2", "type": "freezer", "room_id": "kitchen"},
        {"appliance_id": "a3", "type": "electric_heater", "room_id": "living", "u": 0.15, "v": 0.2},
        {"appliance_id": "a4", "type": "dehumidifier", "room_id": "bed2"},
        {"appliance_id": "a5", "type": "tv", "room_id": "living"},
    ],
    "flat_2bed": [
        {"appliance_id": "a1", "type": "fridge_freezer", "room_id": "kitchen"},
        {"appliance_id": "a2", "type": "washing_machine", "room_id": "kitchen"},
        {"appliance_id": "a3", "type": "electric_heater", "room_id": "living"},
        {"appliance_id": "a4", "type": "electric_heater", "room_id": "bed1"},
        {"appliance_id": "a5", "type": "immersion_heater", "room_id": "hall"},
        {"appliance_id": "a6", "type": "electric_shower", "room_id": "bath"},
    ],
}


def write_examples(out_dir: Path = HERE / "examples") -> list[Path]:
    """Write one example scene per archetype plus the JSON Schema of the scene and of appliance input."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for aid, apps in _EXAMPLE_APPLIANCES.items():
        scene = build_scene(build_layout(aid), f"EXAMPLE_{aid}", [AppliancePlacement(**a) for a in apps])
        path = out_dir / f"{aid}.scene.json"
        path.write_text(scene.model_dump_json(indent=2), encoding="utf-8")
        paths.append(path)
    for name, model in (("room_scene", RoomScene), ("appliance_placement", AppliancePlacement)):
        path = out_dir / f"{name}.schema.json"
        path.write_text(json.dumps(model.model_json_schema(), indent=2), encoding="utf-8")
        paths.append(path)
    return paths


if __name__ == "__main__":
    for p in write_examples():
        print(p.relative_to(HERE.parents[1]))
