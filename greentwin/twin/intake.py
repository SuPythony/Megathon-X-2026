"""What the user gives us to build their home's twin: location, EPC certificate number, photos and appliances.

Privacy (AGENTS.md rule 8): location is kept only as the postcode district (outward code, e.g. 'M14'). A full
postcode the user types is reduced to its district before it is stored or sent; the inward code is never kept.

The EPC certificate number (report reference number, RRN: 20 digits in 5 groups of 4) lets the data layer fetch
the home's EPC, which gives the twin its floor area, built form, age band, walls, roof, glazing and heating.
Fetching is Part 0's job; ``EPCLookup`` is the seam until the data layer exposes a lookup by RRN.
"""

from __future__ import annotations

import re
from typing import Optional, Protocol

from pydantic import BaseModel, Field, field_validator

from contracts.models import Property
from greentwin.twin.scene import AppliancePlacement

_FULL_POSTCODE = re.compile(r"^([A-Z]{1,2}\d[A-Z\d]?)\s*(\d[A-Z]{2})$")
_DISTRICT = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?$")
_RRN = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{4}-\d{4}$")

# Postcode areas outside England & Wales. Our EPC data covers England & Wales only, so these get a warning.
_OUTSIDE_EW = {
    "BT": "Northern Ireland",
    **{a: "Scotland" for a in ("AB", "DD", "DG", "EH", "FK", "G", "HS", "IV", "KA", "KW", "KY", "ML", "PA", "PH", "ZE")},
}


def postcode_district(text: str) -> str:
    """Full postcode or district -> district (outward code). Raises ValueError if it isn't a UK postcode."""
    s = re.sub(r"\s+", " ", text.strip().upper())
    m = _FULL_POSTCODE.match(s)
    if m:
        return m.group(1)
    if _DISTRICT.match(s.replace(" ", "")):
        return s.replace(" ", "")
    raise ValueError(f"{text!r} is not a UK postcode (e.g. 'M14 5AB' or 'M14')")


def postcode_area(district: str) -> str:
    return re.match(r"^[A-Z]+", district).group(0)


def normalise_rrn(text: str) -> str:
    """EPC certificate number with or without dashes/spaces -> '0000-0000-0000-0000-0000'."""
    digits = re.sub(r"[\s-]", "", text.strip())
    if not re.fullmatch(r"\d{20}", digits):
        raise ValueError(f"{text!r} is not an EPC certificate number (20 digits, e.g. 0390-2954-3640-2196-6705)")
    return "-".join(digits[i:i + 4] for i in range(0, 20, 4))


class TwinIntake(BaseModel):
    """One home's intake. Images travel separately (multipart upload); only their file names are recorded here."""

    intake_id: str
    postcode_district: str = Field(description="Outward code only, e.g. 'M14'. Full postcodes are reduced on entry.")
    epc_rrn: Optional[str] = Field(None, description="EPC certificate number, '0000-0000-0000-0000-0000'; None if unknown")
    floor_plan_image: Optional[str] = None
    fuse_box_image: Optional[str] = None
    appliances: list[AppliancePlacement] = Field(default_factory=list)
    photo_consent: bool = Field(False, description="User agreed to interior photos being processed")
    warnings: list[str] = Field(default_factory=list)

    @field_validator("postcode_district", mode="before")
    @classmethod
    def _district_only(cls, v: str) -> str:
        return postcode_district(v)

    @field_validator("epc_rrn", mode="before")
    @classmethod
    def _rrn(cls, v: Optional[str]) -> Optional[str]:
        return normalise_rrn(v) if v not in (None, "") else None

    def model_post_init(self, _ctx) -> None:
        nation = _OUTSIDE_EW.get(postcode_area(self.postcode_district))
        if nation:
            self.warnings.append(f"{self.postcode_district} is in {nation}; our EPC and energy data cover England & Wales only")
        if self.epc_rrn is None:
            self.warnings.append("No EPC number: the twin starts from the closest house type and the floor plan only")
        if (self.floor_plan_image or self.fuse_box_image) and not self.photo_consent:
            self.warnings.append("Photos supplied without consent; they will not be processed")


class EPCLookup(Protocol):
    """Seam to the data layer (Part 0): fetch a home's EPC by certificate number. Returns None if not found."""

    def by_rrn(self, rrn: str, postcode_district: str) -> Optional[Property]: ...


class NoEPCLookup:
    """Used until the data layer offers a lookup by RRN, and when there's no EPC API key: always None."""

    def by_rrn(self, rrn: str, postcode_district: str) -> Optional[Property]:
        return None
