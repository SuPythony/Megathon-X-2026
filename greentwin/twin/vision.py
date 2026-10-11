"""Photo-to-twin: vision reads a floor plan or fuse box photo into a structured draft.

Uses the team's one LLM provider and client (``greentwin.llm.client``: OpenAI ``gpt-6-luna``, approved by the team
lead for every call), like P4's room-photo capture (``capture.py``). The two read different photos: this module the
floor plan (rooms, dimensions -> layout) and fuse box (circuits); ``capture.py`` room photos (element tags).
The model returns a schema-validated Pydantic object (structured outputs via ``client.responses.parse``).
Drafts never reach the twin directly: the user confirms them first, and vision never produces energy numbers.
For floor plans the model picks the closest prefit archetype (``archetypes.py``) and reads room labels and any
printed dimensions; ``fit_layout`` turns that into a ``TwinLayout``. Responses are cached on disk by image hash
and prompt version, so the same photo always gives the same draft and the demo can replay offline.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Optional, TypeVar

from pydantic import BaseModel, Field

from contracts.models import BuiltForm, Property
from greentwin.twin.archetypes import (
    ARCHETYPES,
    ArchetypeId,
    RoomType,
    TwinLayout,
    build_layout,
    get_archetype,
)

from greentwin.llm import client as llm  # noqa: E402

MODEL = llm.MODEL
PROMPT_VERSION = "v1"
MAX_TOKENS = 16000
CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "llm" / "vision"

T = TypeVar("T", bound=BaseModel)


class VisionError(RuntimeError):
    """The model did not return a usable draft (refusal or truncated output)."""


# --------------------------------------------------------------------------- floor plan schema
class ExtractedRoom(BaseModel):
    label: str = Field(description="Room name as printed on the plan, e.g. 'Kitchen/Diner'")
    room_type: RoomType
    storey: int = Field(description="0 = ground floor")
    width_m: Optional[float] = Field(None, description="Only if a dimension is printed on the plan")
    depth_m: Optional[float] = Field(None, description="Only if a dimension is printed on the plan")
    window_count: Optional[int] = None
    confidence: float = Field(ge=0, le=1)


class FloorPlanExtraction(BaseModel):
    archetype_id: ArchetypeId = Field(description="Closest prefit archetype")
    archetype_confidence: float = Field(ge=0, le=1)
    archetype_reason: str
    built_form_guess: Optional[BuiltForm] = None
    storeys: int
    bedrooms: int
    rooms: list[ExtractedRoom]
    total_floor_area_m2: Optional[float] = Field(None, description="Only if printed on the plan")
    front_azimuth_deg: Optional[float] = Field(None, description="Only if a north arrow is shown; 180 = south")
    notes: list[str] = Field(default_factory=list, description="Anything unclear or unreadable")


# --------------------------------------------------------------------------- fuse box schema
class ExtractedCircuit(BaseModel):
    position: int = Field(description="Slot number counted left to right, starting at 1")
    device: Literal["mcb", "rcbo", "rcd", "main_switch", "fuse", "spd", "unknown"]
    rating_a: Optional[int] = None
    curve: Optional[Literal["B", "C", "D"]] = None
    label_raw: Optional[str] = Field(None, description="Label exactly as written; null if none")
    circuit_kind: Literal[
        "lighting", "sockets", "cooker", "shower", "immersion", "heating", "ev_charger", "heat_pump",
        "storage_heaters", "smoke_alarm", "spare", "other", "unknown",
    ]
    rooms_guess: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class FuseBoardExtraction(BaseModel):
    board_type: Literal["split_load_rcd", "all_rcbo", "single_rcd", "rewireable_fuses", "unknown"]
    main_switch_rating_a: Optional[int] = None
    circuits: list[ExtractedCircuit]
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- prompts
def _archetype_menu() -> str:
    return "\n".join(f"- {a.archetype_id}: {a.description}" for a in ARCHETYPES.values())


FLOOR_PLAN_PROMPT = f"""This is a floor plan of a UK home. Read it into the schema.

Pick the closest archetype from this list. Match on property type, storeys, bedroom count and layout:
{_archetype_menu()}

Rules:
- List every room you can see, with its printed name and storey (0 = ground floor).
- Give width_m and depth_m only when a dimension is printed for that room. Convert feet to metres. Never guess.
- Give total_floor_area_m2 only if a total area is printed.
- Give front_azimuth_deg only if a north arrow is shown.
- Put anything unreadable or ambiguous in notes. Use low confidence rather than guessing."""

FUSE_BOARD_PROMPT = """This is a photo of a UK consumer unit (fuse box). Read every device into the schema.

Rules:
- One entry per device, left to right, including the main switch and RCDs.
- rating_a is the number printed on the device (e.g. B32 means curve B, 32 A).
- label_raw is the circuit label exactly as written. Null if there is no label. Never invent one.
- circuit_kind from the label, or from the rating if unlabelled (e.g. 40-45 A often means shower or cooker);
  use low confidence when inferring from rating alone.
- Put anything unreadable in notes."""


# --------------------------------------------------------------------------- client calls
def _media_type(image: bytes) -> str:
    if image.startswith(b"\x89PNG"):
        return "image/png"
    if image.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if image[:4] == b"RIFF" and image[8:12] == b"WEBP":
        return "image/webp"
    if image[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    raise ValueError("unsupported image type: use PNG, JPEG, WebP or GIF")


def _cache_key(schema: type[BaseModel], prompt: str, image: bytes) -> str:
    h = hashlib.sha256()
    for part in (MODEL, PROMPT_VERSION, schema.__name__, prompt):
        h.update(part.encode())
    h.update(image)
    return h.hexdigest()[:32]


def _parse(schema: type[T], prompt: str, image: bytes, effort: str, client: Any, cache_dir: Optional[Path]) -> T:
    path = cache_dir / f"{schema.__name__}_{_cache_key(schema, prompt, image)}.json" if cache_dir else None
    if path and path.exists():
        return schema.model_validate_json(path.read_text(encoding="utf-8"))

    if client is None:
        if not llm.has_key():
            raise VisionError("no OPENAI_API_KEY: confirm the house type and rooms by hand instead")
        client = llm.client()

    data_url = f"data:{_media_type(image)};base64,{base64.standard_b64encode(image).decode()}"
    response = client.responses.parse(
        model=MODEL,
        reasoning={"effort": effort},
        max_output_tokens=MAX_TOKENS,
        input=[{"role": "user", "content": [
            {"type": "input_image", "image_url": data_url},
            {"type": "input_text", "text": prompt},
        ]}],
        text_format=schema,
    )
    if getattr(response, "status", None) == "incomplete":
        raise VisionError(f"output incomplete: {getattr(response, 'incomplete_details', None)}")
    result = response.output_parsed
    if result is None:  # a refusal or no parseable output
        raise VisionError("model returned no structured draft (refused or unparseable); confirm by hand")

    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8")
    return result


def extract_floor_plan(image: bytes, *, client: Any = None, cache_dir: Optional[Path] = CACHE_DIR) -> FloorPlanExtraction:
    return _parse(FloorPlanExtraction, FLOOR_PLAN_PROMPT, image, "high", client, cache_dir)


def extract_fuse_board(image: bytes, *, client: Any = None, cache_dir: Optional[Path] = CACHE_DIR) -> FuseBoardExtraction:
    return _parse(FuseBoardExtraction, FUSE_BOARD_PROMPT, image, "medium", client, cache_dir)


# --------------------------------------------------------------------------- draft -> layout
class FittedLayout(BaseModel):
    layout: TwinLayout
    area_source: Literal["printed_total", "printed_rooms", "epc", "archetype_default"]
    unmatched_rooms: list[str] = Field(description="Extracted rooms with no slot in the archetype; user reviews")
    needs_confirmation: list[str]


def fit_layout(extraction: FloorPlanExtraction, prop: Optional[Property] = None,
               confidence_threshold: float = 0.6) -> FittedLayout:
    """Turn a confirmed-or-draft extraction into a fitted archetype layout. Deterministic; no model calls."""
    arch = get_archetype(extraction.archetype_id)

    area, source = None, "archetype_default"
    sized = [r for r in extraction.rooms if r.width_m and r.depth_m]
    if extraction.total_floor_area_m2:
        area, source = extraction.total_floor_area_m2, "printed_total"
    elif extraction.rooms and len(sized) >= 0.6 * len(extraction.rooms):
        # Unsized rooms (halls, landings) are filled at the sized rooms' mean.
        sized_area = sum(r.width_m * r.depth_m for r in sized)
        area = sized_area * len(extraction.rooms) / len(sized)
        source = "printed_rooms"
    elif prop is not None:
        area, source = prop.floor_area_m2, "epc"

    built_form = prop.built_form if prop is not None else extraction.built_form_guess

    # Match extracted rooms to template rooms of the same storey and type: closest area when the plan prints
    # dimensions, otherwise largest first (so "Bedroom 1" / "Master" lands on the biggest bedroom).
    def t_area(t):
        return (t.x1 - t.x0) * (t.y1 - t.y0)

    labels: dict[str, str] = {}
    unmatched: list[str] = []
    free = sorted(arch.rooms, key=t_area, reverse=True)
    scale2 = (area / arch.reference_floor_area_m2) if area else 1.0
    for er in extraction.rooms:
        candidates = [t for t in free if t.storey == er.storey and t.room_type == er.room_type]
        if not candidates:
            unmatched.append(er.label)
            continue
        if er.width_m and er.depth_m:
            slot = min(candidates, key=lambda t: abs(t_area(t) * scale2 - er.width_m * er.depth_m))
        else:
            slot = candidates[0]
        free.remove(slot)
        labels[slot.key] = er.label

    needs = [r.label for r in extraction.rooms if r.confidence < confidence_threshold]
    if extraction.archetype_confidence < confidence_threshold:
        needs.insert(0, "archetype_id")

    layout = build_layout(
        arch.archetype_id,
        total_floor_area_m2=area,
        built_form=built_form,
        front_azimuth_deg=extraction.front_azimuth_deg if extraction.front_azimuth_deg is not None else 180.0,
        room_labels=labels,
        geometry_source="vision_fitted",
    )
    return FittedLayout(layout=layout, area_source=source, unmatched_rooms=unmatched, needs_confirmation=needs)
