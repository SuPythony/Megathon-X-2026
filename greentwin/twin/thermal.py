"""Thermal mass: how much heat the home's fabric stores, from its construction and the fitted layout.

Heat capacity C (kJ/K) = sum over inner surfaces of areal heat capacity kappa (kJ/m2K) x area: external walls (heavy
masonry vs timber frame; internal insulation hides the masonry), party walls, internal partitions (solid vs hollow),
ground floor, intermediate floors and ceilings. Thermal mass parameter TMP = C / total floor area, as in SAP.

The physics uses C with the heat-loss coefficient to get the time constant tau = C / (3.6 x HLC) hours, which sets
(a) how much of the solar and internal gains are useful and (b) how far the home cools when the heating is off.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from contracts.models import FloorType, Property, RoofType, WallInsulation
from greentwin.twin.archetypes import TwinLayout

PartitionAnswer = Literal["solid", "hollow"]


class ThermalMass(BaseModel):
    heat_capacity_kj_per_k: float
    tmp_kj_per_m2k: float = Field(description="Thermal mass parameter: heat capacity per m2 of floor (SAP)")
    thermal_class: Literal["low", "medium", "high"]
    breakdown_kj_per_k: dict[str, float]
    partitions: str = Field(description="How partitions were classed: user answer or age-band assumption")
    source: str


def thermal_mass(prop: Property, layout: TwinLayout, internal_walls: Optional[PartitionAnswer] = None,
                 age_band_index: int = 4) -> ThermalMass:
    from greentwin.twin.params import rdsap  # local: params imports this module

    t = rdsap()["thermal_mass"]
    env = layout.envelope()
    footprint = layout.footprint_m[0] * layout.footprint_m[1]

    wall_kappa = (t["external_wall_insulated_inside_kappa"] if prop.walls.insulation == WallInsulation.internal
                  else t["external_wall_kappa"][prop.walls.type.value])
    if internal_walls:
        partitions, part_src = internal_walls, "user answer"
    else:
        partitions = "unknown_pre_1950" if age_band_index <= 2 else "unknown_post_1950"
        part_src = "assumption by age (pre-1950 masonry, later plasterboard)"
    partition_area = sum(w.length_m * w.height_m for w in layout.walls if w.kind == "internal")  # both faces

    if not layout.ground_floor_exposed or prop.floor.type == FloorType.another_dwelling_below:
        ground = t["ground_floor_kappa"]["another_dwelling_below"]
    else:
        ground = t["ground_floor_kappa"][prop.floor.type.value]
    top = (t["dwelling_above_ceiling_kappa"]
           if not layout.roof_exposed or prop.roof.type == RoofType.another_dwelling_above else t["ceiling_kappa"])

    breakdown = {
        "external_walls": wall_kappa * env.wall_area_m2,
        "party_walls": t["party_wall_kappa"] * env.party_wall_area_m2,
        "partitions": t["internal_partition_kappa"][partitions] * partition_area,
        "ground_floor": ground * footprint,
        "intermediate_floors": (t["intermediate_floor_kappa"] + t["ceiling_kappa"]) * footprint * (layout.storeys - 1),
        "top_ceiling": top * footprint,
    }
    c = sum(breakdown.values())
    tmp = c / prop.floor_area_m2 if prop.floor_area_m2 > 0 else 0.0
    cls = next(name for name, (lo, hi) in t["classes"].items() if lo <= tmp < hi)
    return ThermalMass(heat_capacity_kj_per_k=round(c, 0), tmp_kj_per_m2k=round(tmp, 1), thermal_class=cls,
                       breakdown_kj_per_k={k: round(v, 0) for k, v in breakdown.items()}, partitions=part_src,
                       source=f"construction x layout areas, kappa from rdsap_defaults.json ({t['_status']})")
