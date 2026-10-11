"""Prefit digital twins of standard UK house types (archetypes).

Each archetype is a rectangular footprint per storey, tiled by rectangular rooms, with doors and windows
placed on room walls. A home's twin is built by picking an archetype and fitting it: scale to the home's
floor area, set ceiling height, built form (which side walls are party walls) and orientation. Homes of the
same type look alike in 3D, but their parameters (areas, U-values, appliances, behaviour) differ.

Plan coordinates, in metres: x runs across the front facade, increasing to the right of someone standing in
the street facing the house; y runs from the front facade (y = 0) to the back (y = depth).

Geometry only. No energy numbers are produced here; areas feed ``TwinParams`` and the room allocation.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field

from contracts.models import BuiltForm, FloorPlan, Room

Side = Literal["front", "back", "left", "right"]
WallKind = Literal["exterior", "party", "internal"]
OpeningKind = Literal["door", "window", "patio_door"]
RoomType = Literal[
    "living", "kitchen", "dining", "bedroom", "bathroom", "wc", "hall", "landing",
    "utility", "study", "garage", "conservatory", "other",
]
ArchetypeId = Literal[
    "victorian_terrace_2bed",
    "interwar_semi_3bed",
    "postwar_detached_4bed",
    "modern_terrace_3bed",
    "bungalow_2bed",
    "flat_2bed",
]

SIDES: tuple[Side, ...] = ("front", "back", "left", "right")


# --------------------------------------------------------------------------- template models
class TemplateOpening(BaseModel):
    side: Side
    kind: OpeningKind
    width_m: float
    height_m: float
    position: float = Field(0.5, description="Centre along the wall, 0..1 from the wall's start")


class TemplateRoom(BaseModel):
    key: str
    label: str
    room_type: RoomType
    storey: int
    x0: float
    y0: float
    x1: float
    y1: float
    openings: list[TemplateOpening] = Field(default_factory=list)


class Archetype(BaseModel):
    archetype_id: ArchetypeId
    name: str
    description: str = Field(description="Shown to the vision model when it picks an archetype")
    property_type: Literal["house", "bungalow", "flat", "maisonette"]
    default_built_form: BuiltForm
    typical_age_band: str
    bedrooms: int
    storeys: int
    width_m: float
    depth_m: float
    ceiling_height_m: float
    roof_exposed: bool = True
    ground_floor_exposed: bool = True
    default_party_sides: tuple[Side, ...] = ()
    rooms: list[TemplateRoom]

    @property
    def reference_floor_area_m2(self) -> float:
        return self.width_m * self.depth_m * self.storeys


# --------------------------------------------------------------------------- fitted layout models
class LayoutWall(BaseModel):
    wall_id: str
    room_id: str
    storey: int
    side: Optional[Side] = Field(None, description="Facade side for exterior/party walls; None if internal")
    kind: WallKind
    start_m: tuple[float, float]
    end_m: tuple[float, float]
    length_m: float
    height_m: float
    orientation_deg: Optional[float] = Field(None, description="Outward normal azimuth, 180 = south")


class LayoutOpening(BaseModel):
    opening_id: str
    wall_id: str
    room_id: str
    kind: OpeningKind
    width_m: float
    height_m: float
    centre_m: tuple[float, float]


class Envelope(BaseModel):
    """Whole-house areas in ``TwinParams`` field names."""

    wall_area_m2: float = Field(description="Exterior wall area net of windows and doors")
    roof_area_m2: float
    floor_area_m2: float = Field(description="Ground-floor heat-loss area")
    window_area_m2: float
    door_area_m2: float
    party_wall_area_m2: float
    volume_m3: float
    total_floor_area_m2: float


class TwinLayout(BaseModel):
    archetype_id: ArchetypeId
    built_form: BuiltForm
    storeys: int
    footprint_m: tuple[float, float] = Field(description="(width, depth)")
    ceiling_height_m: float
    front_azimuth_deg: float
    party_sides: list[Side]
    roof_exposed: bool
    ground_floor_exposed: bool
    rooms: list[Room]
    room_types: dict[str, RoomType]
    walls: list[LayoutWall]
    openings: list[LayoutOpening]

    def envelope(self) -> Envelope:
        opening_area: dict[str, float] = {}
        window_area = door_area = 0.0
        for o in self.openings:
            a = o.width_m * o.height_m
            opening_area[o.wall_id] = opening_area.get(o.wall_id, 0.0) + a
            if o.kind == "door":
                door_area += a
            else:
                window_area += a  # patio doors are glazed: counted as window area
        wall_net = party = 0.0
        for w in self.walls:
            gross = w.length_m * w.height_m
            if w.kind == "exterior":
                wall_net += gross - opening_area.get(w.wall_id, 0.0)
            elif w.kind == "party":
                party += gross
        footprint = self.footprint_m[0] * self.footprint_m[1]
        total = sum(r.area_m2 for r in self.rooms)
        return Envelope(
            wall_area_m2=round(wall_net, 2),
            roof_area_m2=round(footprint if self.roof_exposed else 0.0, 2),
            floor_area_m2=round(footprint if self.ground_floor_exposed else 0.0, 2),
            window_area_m2=round(window_area, 2),
            door_area_m2=round(door_area, 2),
            party_wall_area_m2=round(party, 2),
            volume_m3=round(total * self.ceiling_height_m, 2),
            total_floor_area_m2=round(total, 2),
        )

    def to_floor_plan(self, property_id: str, epc_floor_area_m2: Optional[float] = None) -> FloorPlan:
        env = self.envelope()
        ext_len = sum(w.length_m for w in self.walls if w.kind == "exterior" and w.storey == 0)
        diff = None
        if epc_floor_area_m2:
            diff = round((env.total_floor_area_m2 - epc_floor_area_m2) / epc_floor_area_m2 * 100, 1)
        return FloorPlan(
            property_id=property_id,
            rooms=self.rooms,
            total_area_m2=env.total_floor_area_m2,
            area_vs_epc_pct=diff,
            exterior_wall_length_m=round(ext_len, 2),
            window_area_m2=env.window_area_m2,
            source="parametric",
        )


# --------------------------------------------------------------------------- library
def _o(side: Side, kind: OpeningKind, w: float, h: float, pos: float = 0.5) -> TemplateOpening:
    return TemplateOpening(side=side, kind=kind, width_m=w, height_m=h, position=pos)


def _r(key, label, rtype, storey, x0, y0, x1, y1, *openings) -> TemplateRoom:
    return TemplateRoom(key=key, label=label, room_type=rtype, storey=storey,
                        x0=x0, y0=y0, x1=x1, y1=y1, openings=list(openings))


ARCHETYPES: dict[str, Archetype] = {a.archetype_id: a for a in [
    Archetype(
        archetype_id="victorian_terrace_2bed",
        name="Victorian two-up two-down terrace",
        description="Pre-1919 solid-brick terraced house, 2 storeys, 2 bedrooms, narrow frontage (~5 m), "
                    "front living room and rear kitchen downstairs, bathroom upstairs at the back.",
        property_type="house", default_built_form=BuiltForm.mid_terrace, typical_age_band="1900-1929",
        bedrooms=2, storeys=2, width_m=4.8, depth_m=8.0, ceiling_height_m=2.7,
        rooms=[
            _r("living", "Living room", "living", 0, 0, 0, 4.8, 3.8,
               _o("front", "window", 1.6, 1.6, 0.6), _o("front", "door", 0.9, 2.0, 0.12)),
            _r("kitchen", "Kitchen", "kitchen", 0, 0, 3.8, 4.8, 8.0,
               _o("back", "window", 1.2, 1.1, 0.35), _o("back", "door", 0.9, 2.0, 0.8)),
            _r("bed1", "Bedroom 1", "bedroom", 1, 0, 0, 4.8, 3.8, _o("front", "window", 1.4, 1.4)),
            _r("bed2", "Bedroom 2", "bedroom", 1, 0, 3.8, 2.8, 8.0, _o("back", "window", 1.0, 1.2)),
            _r("bath", "Bathroom", "bathroom", 1, 2.8, 3.8, 4.8, 8.0, _o("back", "window", 0.6, 0.9)),
        ],
    ),
    Archetype(
        archetype_id="interwar_semi_3bed",
        name="1930s semi-detached",
        description="1919-1944 semi-detached house, 2 storeys, 3 bedrooms including a small box room, "
                    "side hall with stairs, front lounge with bay window, rear kitchen and dining room.",
        property_type="house", default_built_form=BuiltForm.semi_detached, typical_age_band="1930-1949",
        bedrooms=3, storeys=2, width_m=6.5, depth_m=7.0, ceiling_height_m=2.5,
        default_party_sides=("right",),
        rooms=[
            _r("hall", "Hall", "hall", 0, 0, 0, 2.0, 4.0,
               _o("front", "door", 0.9, 2.0), _o("left", "window", 0.6, 1.0, 0.6)),
            _r("living", "Lounge", "living", 0, 2.0, 0, 6.5, 4.0, _o("front", "window", 2.0, 1.5)),
            _r("kitchen", "Kitchen", "kitchen", 0, 0, 4.0, 3.0, 7.0,
               _o("back", "window", 1.2, 1.1), _o("left", "door", 0.9, 2.0, 0.5)),
            _r("dining", "Dining room", "dining", 0, 3.0, 4.0, 6.5, 7.0, _o("back", "patio_door", 1.8, 2.0)),
            _r("bed3", "Bedroom 3", "bedroom", 1, 0, 0, 2.4, 3.0, _o("front", "window", 1.0, 1.2)),
            _r("bed1", "Bedroom 1", "bedroom", 1, 2.4, 0, 6.5, 4.0, _o("front", "window", 1.8, 1.3)),
            _r("landing", "Landing", "landing", 1, 0, 3.0, 2.4, 4.5, _o("left", "window", 0.6, 1.0)),
            _r("bath", "Bathroom", "bathroom", 1, 0, 4.5, 2.4, 7.0, _o("back", "window", 0.7, 0.9)),
            _r("bed2", "Bedroom 2", "bedroom", 1, 2.4, 4.0, 6.5, 7.0, _o("back", "window", 1.4, 1.2)),
        ],
    ),
    Archetype(
        archetype_id="postwar_detached_4bed",
        name="1960s-80s detached",
        description="1950-1990 detached house, 2 storeys, 4 bedrooms, wide frontage (~9 m), through lounge "
                    "on one side, hall and study at the front, kitchen-diner at the back.",
        property_type="house", default_built_form=BuiltForm.detached, typical_age_band="1967-1975",
        bedrooms=4, storeys=2, width_m=9.0, depth_m=7.5, ceiling_height_m=2.4,
        rooms=[
            _r("living", "Lounge", "living", 0, 0, 0, 4.5, 7.5,
               _o("front", "window", 2.4, 1.4), _o("back", "patio_door", 2.4, 2.0), _o("left", "window", 1.0, 1.0)),
            _r("hall", "Hall", "hall", 0, 4.5, 0, 6.5, 3.5, _o("front", "door", 1.0, 2.0)),
            _r("study", "Study", "study", 0, 6.5, 0, 9.0, 3.5,
               _o("front", "window", 1.4, 1.2), _o("right", "window", 0.8, 1.0)),
            _r("kitchen", "Kitchen-diner", "kitchen", 0, 4.5, 3.5, 9.0, 7.5,
               _o("back", "window", 1.6, 1.1), _o("right", "door", 0.9, 2.0, 0.7)),
            _r("bed1", "Bedroom 1", "bedroom", 1, 0, 0, 4.5, 4.0,
               _o("front", "window", 1.8, 1.3), _o("left", "window", 0.8, 1.0)),
            _r("bed2", "Bedroom 2", "bedroom", 1, 0, 4.0, 4.5, 7.5, _o("back", "window", 1.6, 1.2)),
            _r("landing", "Landing", "landing", 1, 4.5, 0, 6.0, 4.0, _o("front", "window", 0.6, 1.0)),
            _r("bath", "Bathroom", "bathroom", 1, 4.5, 4.0, 6.0, 7.5, _o("back", "window", 0.6, 0.9)),
            _r("bed3", "Bedroom 3", "bedroom", 1, 6.0, 0, 9.0, 3.5, _o("front", "window", 1.4, 1.2)),
            _r("bed4", "Bedroom 4", "bedroom", 1, 6.0, 3.5, 9.0, 7.5,
               _o("back", "window", 1.2, 1.2), _o("right", "window", 0.6, 0.9)),
        ],
    ),
    Archetype(
        archetype_id="modern_terrace_3bed",
        name="Post-2000 terrace",
        description="Built after 2000, cavity-wall terraced or town house, 2 storeys, 3 bedrooms, front "
                    "kitchen and hall, rear living room opening to the garden, bathroom in the middle upstairs.",
        property_type="house", default_built_form=BuiltForm.mid_terrace, typical_age_band="2003-2006",
        bedrooms=3, storeys=2, width_m=5.0, depth_m=8.5, ceiling_height_m=2.4,
        rooms=[
            _r("kitchen", "Kitchen", "kitchen", 0, 0, 0, 3.0, 3.5, _o("front", "window", 1.2, 1.1)),
            _r("hall", "Hall", "hall", 0, 3.0, 0, 5.0, 3.5, _o("front", "door", 1.0, 2.0)),
            _r("living", "Living room", "living", 0, 0, 3.5, 5.0, 8.5,
               _o("back", "patio_door", 1.8, 2.1, 0.4), _o("back", "window", 1.0, 1.2, 0.85)),
            _r("bed2", "Bedroom 2", "bedroom", 1, 0, 0, 2.8, 3.0, _o("front", "window", 1.2, 1.2)),
            _r("bed3", "Bedroom 3", "bedroom", 1, 2.8, 0, 5.0, 3.0, _o("front", "window", 0.9, 1.2)),
            _r("bath", "Bathroom", "bathroom", 1, 0, 3.0, 2.5, 5.0),
            _r("landing", "Landing", "landing", 1, 2.5, 3.0, 5.0, 5.0),
            _r("bed1", "Bedroom 1", "bedroom", 1, 0, 5.0, 5.0, 8.5, _o("back", "window", 1.6, 1.3)),
        ],
    ),
    Archetype(
        archetype_id="bungalow_2bed",
        name="Detached bungalow",
        description="Single-storey detached bungalow, 2 bedrooms, central hall, living room and kitchen on "
                    "one side, bedrooms and bathroom on the other, loft above.",
        property_type="bungalow", default_built_form=BuiltForm.detached, typical_age_band="1950-1966",
        bedrooms=2, storeys=1, width_m=10.0, depth_m=7.0, ceiling_height_m=2.4,
        rooms=[
            _r("living", "Living room", "living", 0, 0, 0, 4.5, 4.0,
               _o("front", "window", 2.0, 1.4), _o("left", "window", 1.0, 1.0)),
            _r("kitchen", "Kitchen", "kitchen", 0, 0, 4.0, 4.5, 7.0,
               _o("back", "window", 1.2, 1.1), _o("left", "door", 0.9, 2.0)),
            _r("hall", "Hall", "hall", 0, 4.5, 0, 5.5, 7.0, _o("front", "door", 0.9, 2.0)),
            _r("bed1", "Bedroom 1", "bedroom", 0, 5.5, 0, 10.0, 4.0,
               _o("front", "window", 1.8, 1.3), _o("right", "window", 0.8, 1.0)),
            _r("bed2", "Bedroom 2", "bedroom", 0, 5.5, 4.0, 8.0, 7.0, _o("back", "window", 1.2, 1.2)),
            _r("bath", "Bathroom", "bathroom", 0, 8.0, 4.0, 10.0, 7.0, _o("back", "window", 0.6, 0.9)),
        ],
    ),
    Archetype(
        archetype_id="flat_2bed",
        name="Mid-floor purpose-built flat",
        description="Flat in a purpose-built block, one level, 2 bedrooms, flats above and below and on "
                    "both sides, windows front and back only, living room and kitchen at the front.",
        property_type="flat", default_built_form=BuiltForm.not_applicable, typical_age_band="1967-1975",
        bedrooms=2, storeys=1, width_m=8.0, depth_m=7.5, ceiling_height_m=2.4,
        roof_exposed=False, ground_floor_exposed=False, default_party_sides=("left", "right"),
        rooms=[
            _r("living", "Living room", "living", 0, 0, 0, 4.5, 4.0, _o("front", "window", 2.0, 1.4)),
            _r("kitchen", "Kitchen", "kitchen", 0, 4.5, 0, 8.0, 4.0, _o("front", "window", 1.2, 1.1)),
            _r("bed1", "Bedroom 1", "bedroom", 0, 0, 4.0, 3.5, 7.5, _o("back", "window", 1.4, 1.2)),
            _r("hall", "Hall", "hall", 0, 3.5, 4.0, 4.5, 7.5, _o("back", "door", 0.9, 2.0)),
            _r("bath", "Bathroom", "bathroom", 0, 4.5, 4.0, 6.0, 7.5, _o("back", "window", 0.6, 0.9)),
            _r("bed2", "Bedroom 2", "bedroom", 0, 6.0, 4.0, 8.0, 7.5, _o("back", "window", 1.0, 1.2)),
        ],
    ),
]}

_PARTY_SIDES: dict[BuiltForm, tuple[Side, ...]] = {
    BuiltForm.detached: (),
    BuiltForm.semi_detached: ("right",),
    BuiltForm.end_terrace: ("right",),
    BuiltForm.mid_terrace: ("left", "right"),
    BuiltForm.enclosed_mid_terrace: ("left", "right", "back"),
}


def get_archetype(archetype_id: str) -> Archetype:
    try:
        return ARCHETYPES[archetype_id]
    except KeyError:
        raise ValueError(f"unknown archetype {archetype_id!r}; known: {sorted(ARCHETYPES)}") from None


def party_sides_for(arch: Archetype, built_form: BuiltForm) -> tuple[Side, ...]:
    if built_form == BuiltForm.not_applicable or arch.property_type == "flat":
        return arch.default_party_sides
    return _PARTY_SIDES[built_form]


def side_azimuth(side: Side, front_azimuth_deg: float) -> float:
    offset = {"front": 0.0, "back": 180.0, "left": 90.0, "right": -90.0}[side]
    return (front_azimuth_deg + offset) % 360.0


# --------------------------------------------------------------------------- fitting
def build_layout(
    archetype_id: str,
    *,
    total_floor_area_m2: Optional[float] = None,
    ceiling_height_m: Optional[float] = None,
    built_form: Optional[BuiltForm] = None,
    front_azimuth_deg: float = 180.0,
    room_labels: Optional[dict[str, str]] = None,
) -> TwinLayout:
    """Fit an archetype to one home. Scales both axes equally so the layout keeps its proportions."""
    arch = get_archetype(archetype_id)
    s = math.sqrt(total_floor_area_m2 / arch.reference_floor_area_m2) if total_floor_area_m2 else 1.0
    width, depth = arch.width_m * s, arch.depth_m * s
    height = ceiling_height_m or arch.ceiling_height_m
    form = built_form or arch.default_built_form
    party = party_sides_for(arch, form)
    labels = room_labels or {}

    rooms: list[Room] = []
    room_types: dict[str, RoomType] = {}
    walls: list[LayoutWall] = []
    openings: list[LayoutOpening] = []
    eps = 1e-6

    for tr in arch.rooms:
        x0, y0, x1, y1 = tr.x0 * s, tr.y0 * s, tr.x1 * s, tr.y1 * s
        rid = tr.key
        rooms.append(Room(
            room_id=rid, label=labels.get(rid, tr.label), storey=tr.storey,
            polygon_m=[(round(x0, 3), round(y0, 3)), (round(x1, 3), round(y0, 3)),
                       (round(x1, 3), round(y1, 3)), (round(x0, 3), round(y1, 3))],
            area_m2=round((x1 - x0) * (y1 - y0), 2), ceiling_height_m=height, confidence=1.0,
        ))
        room_types[rid] = tr.room_type

        # Each edge, oriented so position 0..1 runs left-to-right as seen from outside that edge.
        edges: dict[Side, tuple[tuple[float, float], tuple[float, float], bool]] = {
            "front": ((x0, y0), (x1, y0), abs(y0) < eps),
            "back": ((x1, y1), (x0, y1), abs(y1 - depth) < eps * max(1.0, s)),
            "left": ((x0, y1), (x0, y0), abs(x0) < eps),
            "right": ((x1, y0), (x1, y1), abs(x1 - width) < eps * max(1.0, s)),
        }
        for side, (a, b, on_boundary) in edges.items():
            kind: WallKind = ("party" if side in party else "exterior") if on_boundary else "internal"
            wid = f"{rid}:{side}"
            length = math.dist(a, b)
            walls.append(LayoutWall(
                wall_id=wid, room_id=rid, storey=tr.storey, side=side if on_boundary else None, kind=kind,
                start_m=(round(a[0], 3), round(a[1], 3)), end_m=(round(b[0], 3), round(b[1], 3)),
                length_m=round(length, 3), height_m=height,
                orientation_deg=side_azimuth(side, front_azimuth_deg) if on_boundary else None,
            ))
            for i, op in enumerate(o for o in tr.openings if o.side == side):
                if kind != "exterior":
                    continue  # no openings on party or internal walls
                w = min(op.width_m, 0.8 * length)
                h = min(op.height_m, height - 0.2)
                cx = a[0] + (b[0] - a[0]) * op.position
                cy = a[1] + (b[1] - a[1]) * op.position
                openings.append(LayoutOpening(
                    opening_id=f"{wid}:{op.kind}{i}", wall_id=wid, room_id=rid, kind=op.kind,
                    width_m=round(w, 3), height_m=round(h, 3), centre_m=(round(cx, 3), round(cy, 3)),
                ))

    return TwinLayout(
        archetype_id=arch.archetype_id, built_form=form, storeys=arch.storeys,
        footprint_m=(round(width, 3), round(depth, 3)), ceiling_height_m=height,
        front_azimuth_deg=front_azimuth_deg % 360.0, party_sides=list(party),
        roof_exposed=arch.roof_exposed, ground_floor_exposed=arch.ground_floor_exposed,
        rooms=rooms, room_types=room_types, walls=walls, openings=openings,
    )

