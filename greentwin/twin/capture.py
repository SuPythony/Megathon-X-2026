"""Photo-to-twin (minimal): room photos -> drafted element tags (gpt-6-luna vision) -> agent confirms -> Property.

Drafts only: nothing reaches the home record until a person confirms each tag. Vision never outputs energy
numbers; it only fills descriptive Property fields that the physics and rules already use. Images are processed
in memory and not stored; the browser downscales them (which also drops EXIF/location data) before upload.
"""
from __future__ import annotations

import base64
import json
from typing import Literal

from pydantic import BaseModel, Field

from contracts.models import CaptureResult, Control, ElementTag, GlazingType, HeatingSystem, Property, WallInsulation, WallType
from greentwin.kb.store import execute, new_id, now, q
from greentwin.llm import client as llm

CONFIDENCE_OK = 0.7
ENUMS = {
    "glazing.type": [e.value for e in GlazingType],
    "heating.main_system": [e.value for e in HeatingSystem],
    "walls.type": [e.value for e in WallType],
    "walls.insulation": [e.value for e in WallInsulation],
    "controls": [e.value for e in Control if e.value != "none"],
}
NUMERIC = {"roof.loft_insulation_mm": (0, 400), "low_energy_lighting_pct": (0, 100)}
FIELDS = list(ENUMS) + list(NUMERIC)

INSTRUCTIONS = f"""You look at photos of a UK home taken by a surveyor and draft building-element tags.
Only report what is clearly visible. For each observation give: the photo index (0-based), the field, the value,
your confidence (0-1) and a short visual evidence note. Allowed fields and values:
{json.dumps({**ENUMS, **{k: f"number {lo}-{hi}" for k, (lo, hi) in NUMERIC.items()}}, indent=0)}
Use 'controls' once per visible control (e.g. a smart thermostat, TRVs). Never estimate energy use, costs or
savings. If nothing relevant is visible, return no tags. Ignore people and personal items."""


class _Tag(BaseModel):
    photo_index: int
    field: Literal[tuple(FIELDS)]  # type: ignore[valid-type]
    value: str
    confidence: float = Field(ge=0, le=1)
    evidence: str


class _Draft(BaseModel):
    tags: list[_Tag]
    notes: str


def _valid(field: str, value: str):
    if field in ENUMS:
        v = value.strip().lower().replace(" ", "_").replace("-", "_")
        return v if v in ENUMS[field] else None
    try:
        x = float(value)
    except ValueError:
        return None
    lo, hi = NUMERIC[field]
    return x if lo <= x <= hi else None


def capture(property_id: str, images_b64: list[str], consent: bool) -> dict:
    if not consent:
        return {"ok": False, "error": "Consent from the occupier is required before analysing photos."}
    if not images_b64:
        return {"ok": False, "error": "Add at least one photo."}
    if not llm.has_key():
        return {"ok": False, "error": "No LLM key: add tags manually instead."}
    images_b64 = images_b64[:4]
    for b in images_b64:
        if len(b) > 6_000_000:
            return {"ok": False, "error": "Photo too large (max ~4 MB each after resizing)."}
        base64.b64decode(b, validate=True)  # reject anything that is not base64
    content = [{"type": "input_text", "text": f"{len(images_b64)} photo(s) of the home follow."}]
    content += [{"type": "input_image", "image_url": f"data:image/jpeg;base64,{b}"} for b in images_b64]
    r = llm.client().responses.parse(model=llm.MODEL, instructions=INSTRUCTIONS, reasoning={"effort": "low"},
                                     input=[{"role": "user", "content": content}], text_format=_Draft)
    draft = r.output_parsed
    if draft is None:  # refusal or unparseable -> manual tagging
        return {"ok": False, "error": "The vision model returned no usable draft; tag manually."}
    tags, needs, best = [], [], {}
    for t in draft.tags:
        v = _valid(t.field, t.value)
        if v is None or t.photo_index >= len(images_b64):
            continue
        tags.append(ElementTag(photo_id=f"photo_{t.photo_index + 1}", field=t.field, value=v,
                               confidence=round(t.confidence, 2), source="llm_vision"))
        if t.confidence < CONFIDENCE_OK:
            needs.append(t.field)
        if t.field != "controls" and t.confidence > best.get(t.field, (None, -1))[1]:
            best[t.field] = (v, t.confidence)
    result = CaptureResult(property_id=property_id, tags=tags, draft_property_fields={k: v for k, (v, _) in best.items()},
                           needs_confirmation=sorted(set(needs)), vision_model=llm.MODEL, consent_recorded=True,
                           faces_blurred=False)
    cid = new_id("cap")
    execute("INSERT INTO captures VALUES (?, ?, ?, ?, 'draft', ?)",
            [cid, property_id, result.model_dump_json(), json.dumps([t.evidence for t in draft.tags]), now()])
    return {"ok": True, "capture_id": cid, "result": result.model_dump(mode="json"),
            "evidence": [t.evidence for t in draft.tags if _valid(t.field, t.value) is not None], "notes": draft.notes}


def apply(property_id: str, capture_id: str, accepted: list[int]) -> dict:
    """Write the tags the person confirmed into the home record (audit kept in `captures`)."""
    rows = q("SELECT result FROM captures WHERE capture_id = ? AND property_id = ?", [capture_id, property_id])
    if not rows:
        return {"ok": False, "error": "unknown capture for this home"}
    result = CaptureResult.model_validate_json(rows[0]["result"])
    prop = Property.model_validate(q("SELECT data FROM properties WHERE property_id = ?", [property_id])[0]["data"])
    d = prop.model_dump(mode="json")
    applied = []
    for i in accepted:
        if not 0 <= i < len(result.tags):
            continue
        t = result.tags[i]
        if t.field == "glazing.type":
            d["glazing"] = [{"type": t.value, "share_pct": 100, "window_to_floor_ratio": None}]
        elif t.field == "controls":
            if t.value not in d["controls"]:
                d["controls"].append(t.value)
        elif t.field == "roof.loft_insulation_mm":
            d["roof"]["loft_insulation_mm"] = int(t.value)
        else:
            sect, key = t.field.split(".") if "." in t.field else (None, t.field)
            if sect:
                d[sect][key] = t.value
            else:
                d[key] = t.value
        result.tags[i].confirmed_by_agent = True
        applied.append(f"{t.field} = {t.value}")
    new = Property.model_validate(d)  # validate before saving
    new.data_quality.imputed_fields = sorted(set(new.data_quality.imputed_fields) - {a.split(" = ")[0] for a in applied})
    execute("UPDATE properties SET data = ? WHERE property_id = ?", [new.model_dump_json(), property_id])
    execute("UPDATE captures SET status = 'applied', result = ? WHERE capture_id = ?", [result.model_dump_json(), capture_id])
    from greentwin.llm.tools import _prop
    _prop.cache_clear()
    return {"ok": True, "applied": applied}
