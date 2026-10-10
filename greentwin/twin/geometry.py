"""Confirmed geometry: the user checks the fitted layout and confirms or corrects it.

What can be confirmed (v1): house type, footprint width and depth (each axis scales on its own), ceiling height,
which way the front faces, built form (which walls are shared), and room names. Individual room sizes are not
editable yet: rooms keep the house type's proportions inside the confirmed footprint.

Once confirmed, the layout's ``geometry_source`` is ``user_confirmed``: areas, volume and thermal mass then count as
confirmed inputs in the provenance, and the heat-loss prior narrows.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from contracts.models import BuiltForm, Property
from greentwin.twin.archetypes import ArchetypeId, TwinLayout, build_layout

AREA_MISMATCH_PCT = 15.0


class GeometryConfirmation(BaseModel):
    archetype_id: ArchetypeId
    footprint_width_m: float = Field(gt=2, lt=40, description="Across the front")
    footprint_depth_m: float = Field(gt=2, lt=40, description="Front to back")
    ceiling_height_m: float = Field(2.4, ge=2.0, le=4.0)
    front_azimuth_deg: float = Field(180.0, ge=0, lt=360, description="Way the front faces; 180 = south")
    built_form: Optional[BuiltForm] = None
    room_labels: dict[str, str] = Field(default_factory=dict)
    confirmed_by_user: bool = Field(False, description="True only after the user pressed confirm")
    confirmed_at: Optional[str] = None

    @field_validator("confirmed_at")
    @classmethod
    def _iso(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            datetime.fromisoformat(v)
        return v


def confirm(geometry: GeometryConfirmation) -> GeometryConfirmation:
    """Mark a geometry as confirmed by the user now."""
    return geometry.model_copy(update={"confirmed_by_user": True,
                                       "confirmed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})


def from_layout(layout: TwinLayout) -> GeometryConfirmation:
    """The geometry the user is asked to confirm: the current fitted layout."""
    return GeometryConfirmation(
        archetype_id=layout.archetype_id, footprint_width_m=layout.footprint_m[0],
        footprint_depth_m=layout.footprint_m[1], ceiling_height_m=layout.ceiling_height_m,
        front_azimuth_deg=layout.front_azimuth_deg, built_form=layout.built_form,
        room_labels={r.room_id: r.label for r in layout.rooms},
    )


def apply_geometry(geometry: GeometryConfirmation, prop: Optional[Property] = None) -> tuple[TwinLayout, list[str]]:
    """Layout from a (confirmed) geometry, plus flags, e.g. floor area far from the EPC's."""
    layout = build_layout(
        geometry.archetype_id,
        footprint_m=(geometry.footprint_width_m, geometry.footprint_depth_m),
        ceiling_height_m=geometry.ceiling_height_m,
        built_form=geometry.built_form or (prop.built_form if prop else None),
        front_azimuth_deg=geometry.front_azimuth_deg,
        room_labels=geometry.room_labels,
        geometry_source="user_confirmed" if geometry.confirmed_by_user else "vision_fitted",
    )
    flags: list[str] = []
    if prop is not None and prop.floor_area_m2:
        area = layout.envelope().total_floor_area_m2
        diff = (area - prop.floor_area_m2) / prop.floor_area_m2 * 100
        if abs(diff) > AREA_MISMATCH_PCT:
            flags.append(f"Confirmed floor area {area:.0f} m2 differs from the EPC's {prop.floor_area_m2:.0f} m2 by "
                         f"{diff:+.0f}%: check the dimensions (the twin uses the confirmed geometry)")
    if not geometry.confirmed_by_user:
        flags.append("Geometry not confirmed by the user yet")
    return layout, flags
