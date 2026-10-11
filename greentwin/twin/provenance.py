"""Input provenance: for every physics input, where it came from, how sure we are, and whether the user confirmed it.

Built from the per-field source notes that ``params_with_sources`` records, plus the geometry status, weather and
meter sources. ``provenance_summary`` reports how much of the home's heat-loss coefficient rests on confirmed or
recorded inputs vs defaults: that share widens or narrows the calibration prior on ``heat_loss_multiplier``.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Source = Literal[
    "user_confirmed", "user_stated", "epc_measured", "epc_recorded", "vision_draft", "layout_confirmed",
    "layout_house_type", "appliance_list", "rdsap_default", "sap_formula", "derived", "assumption", "default",
    "synthetic", "meter",
]
Status = Literal["confirmed", "measured", "recorded", "stated", "estimated", "to_verify", "assumption", "default",
                 "synthetic"]
# Best -> worst; an element's heat loss takes the worse of its area and U-value statuses.
STATUS_RANK: dict[str, int] = {"confirmed": 0, "measured": 1, "recorded": 2, "stated": 3, "estimated": 4,
                               "to_verify": 5, "assumption": 6, "default": 7, "synthetic": 8}
GEOMETRY_FIELDS = ("wall_area_m2", "roof_area_m2", "floor_area_m2", "window_area_m2", "door_area_m2",
                   "party_wall_area_m2", "volume_m3")


class InputProvenance(BaseModel):
    field: str
    value: float | str | None = None
    source: Source
    status: Status
    confirmed: bool = Field(description="The user confirmed this exact value")
    detail: str = Field(description="Human-readable note of how the value was made")
    ref: Optional[str] = Field(None, description="e.g. 'epc:0390-...', 'intake:abc', 'house_type:interwar_semi_3bed'")
    customer_answer: Optional[Literal["given", "not_sure"]] = Field(
        None, description="For inputs asked of the customer: answered (held fixed) or not sure (uncertain prior)")


class ProvenanceSummary(BaseModel):
    geometry_source: str
    counts_by_status: dict[str, int]
    heat_loss_share_by_status: dict[str, float] = Field(description="% of the heat-loss coefficient (UA + ventilation)")
    confirmed_or_recorded_heat_loss_pct: float
    default_or_unverified_heat_loss_pct: float


def classify(field: str, detail: str, geometry_source: str) -> tuple[Source, Status, bool]:
    d = detail.lower()
    if field in GEOMETRY_FIELDS and d.startswith("layout"):
        if geometry_source == "user_confirmed":
            return "layout_confirmed", "confirmed", True
        if geometry_source == "vision_fitted":
            return "vision_draft", "estimated", False
        return "layout_house_type", "estimated", False
    if "epc (measured)" in d or d == "epc (measured)":
        return "epc_measured", "measured", False
    if d.startswith("epc"):
        return "epc_recorded", "recorded", False
    if "user" in d and "appliance" not in d:
        return "user_stated", "stated", False
    if "appliance list" in d:
        return "appliance_list", "stated", False
    if "to_verify" in d and "rdsap" in d:
        return "rdsap_default", "to_verify", False
    if "iso13370" in d or "construction x layout" in d:
        return "derived", "to_verify" if "to_verify" in d else "estimated", False
    if "sap_2012" in d:
        return "sap_formula", "estimated", False
    if "assumption" in d:
        return "assumption", "assumption", False
    if "synthetic" in d:
        return "synthetic", "synthetic", False
    return "default", "default", False


def build_provenance(params: dict, sources: dict[str, str], geometry_source: str, refs: dict[str, str],
                     weather_source: str, meter_source: Optional[str],
                     customer: Optional[dict[str, str]] = None) -> list[InputProvenance]:
    """``customer`` maps the inputs asked of the customer to 'given' or 'not_sure'."""
    customer = customer or {}
    out = []
    for field, detail in sorted(sources.items()):
        src, status, confirmed = classify(field, detail, geometry_source)
        # A user-confirmed geometry also confirms the thermal mass areas, not the kappa values (still to_verify).
        ref = refs.get("epc") if src.startswith("epc") else refs.get("intake") if src in ("user_stated", "appliance_list") \
            else refs.get("house_type") if src.startswith("layout") or src == "vision_draft" else None
        value = params.get(field)
        out.append(InputProvenance(field=field, value=value if isinstance(value, (int, float, str)) else None,
                                   source=src, status=status, confirmed=confirmed, detail=detail, ref=ref,
                                   customer_answer=customer.get(field)))
    out.append(InputProvenance(field="weather", source="synthetic" if weather_source == "synthetic" else "derived",
                               status="synthetic" if weather_source == "synthetic" else "recorded", confirmed=False,
                               detail=f"weather source: {weather_source}"))
    if meter_source:
        out.append(InputProvenance(field="meter_readings", source="meter", status="recorded",
                                   confirmed="user" in meter_source, detail=meter_source, ref=refs.get("intake")))
    return out


def _worse(a: str, b: str) -> str:
    return a if STATUS_RANK[a] >= STATUS_RANK[b] else b


def provenance_summary(prov: list[InputProvenance], params: dict, geometry_source: str) -> ProvenanceSummary:
    by_field = {p.field: p.status for p in prov}
    elements = [("wall_area_m2", "wall_u"), ("roof_area_m2", "roof_u"), ("floor_area_m2", "floor_u"),
                ("window_area_m2", "window_u"), ("door_area_m2", "door_u")]
    share: dict[str, float] = {}
    total = 0.0
    for area, u in elements:
        ua = float(params.get(area, 0)) * float(params.get(u, 0))
        status = _worse(by_field.get(area, "default"), by_field.get(u, "default"))
        share[status] = share.get(status, 0.0) + ua
        total += ua
    vent = 0.33 * float(params.get("air_changes_per_hour", 0)) * float(params.get("volume_m3", 0))
    vstatus = _worse(by_field.get("air_changes_per_hour", "default"), by_field.get("volume_m3", "default"))
    share[vstatus] = share.get(vstatus, 0.0) + vent
    total += vent
    pct = {k: round(v / total * 100, 1) for k, v in sorted(share.items(), key=lambda kv: STATUS_RANK[kv[0]])} if total else {}
    good = sum(v for k, v in pct.items() if STATUS_RANK[k] <= STATUS_RANK["stated"])
    bad = sum(v for k, v in pct.items() if STATUS_RANK[k] >= STATUS_RANK["to_verify"])
    counts: dict[str, int] = {}
    for p in prov:
        counts[p.status] = counts.get(p.status, 0) + 1
    return ProvenanceSummary(geometry_source=geometry_source, counts_by_status=counts, heat_loss_share_by_status=pct,
                             confirmed_or_recorded_heat_loss_pct=round(good, 1),
                             default_or_unverified_heat_loss_pct=round(bad, 1))
