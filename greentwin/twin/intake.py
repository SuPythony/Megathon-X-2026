"""What the user gives us to build their home's twin: location, EPC certificate number, photos and appliances.

Privacy (AGENTS.md rule 8): location is kept only as the postcode district (outward code, e.g. 'M14'). A full
postcode the user types is reduced to its district before it is stored or sent; the inward code is never kept.

The EPC certificate number (report reference number, RRN: 20 digits in 5 groups of 4) lets the data layer fetch
the home's EPC, which gives the twin its floor area, built form, age band, walls, roof, glazing and heating.
Fetching is Part 0's job; ``EPCLookup`` is the seam until the data layer exposes a lookup by RRN.
"""

from __future__ import annotations

import re
from typing import Literal, Optional, Protocol

from pydantic import BaseModel, Field, field_validator, model_validator

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


class MeterReading(BaseModel):
    """One month of bills or smart-meter data as the user entered it. Missing values stay None, never 0."""

    month: str = Field(description="YYYY-MM")
    electricity_kwh: Optional[float] = Field(None, ge=0)
    gas_kwh: Optional[float] = Field(None, ge=0, description="0 only if the home has no gas supply")
    estimated: bool = Field(False, description="Estimated read on the bill")

    @field_validator("month")
    @classmethod
    def _month(cls, v: str) -> str:
        if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", v):
            raise ValueError(f"{v!r} is not a month (YYYY-MM)")
        return v


class TwinIntake(BaseModel):
    """One home's intake. Images travel separately (multipart upload); only their file names are recorded here."""

    intake_id: str
    postcode_district: str = Field(description="Outward code only, e.g. 'M14'. Full postcodes are reduced on entry.")
    epc_rrn: Optional[str] = Field(None, description="EPC certificate number, '0000-0000-0000-0000-0000'; None if unknown")
    floor_plan_image: Optional[str] = None
    fuse_box_image: Optional[str] = None
    appliances: list[AppliancePlacement] = Field(default_factory=list)
    photo_consent: bool = Field(False, description="User agreed to interior photos being processed")
    # Household: all optional; SAP defaults are used (with wider priors) when missing.
    occupants: Optional[int] = Field(None, ge=1, le=12)
    daytime_occupancy: Optional[Literal["out_all_day", "part_day", "home_all_day"]] = None
    thermostat_c: Optional[float] = Field(None, ge=10, le=28, description="Thermostat setting")
    heating_periods: list[tuple[int, int]] = Field(default_factory=list,
                                                   description="Heating on periods, e.g. [(6, 9), (17, 22)]")
    meter_readings: list[MeterReading] = Field(default_factory=list, description="Monthly bills, ideally 12+ months")
    # Heating, as the user knows it today: overrides the EPC, which can be years old.
    heating_system: Optional[Literal[
        "gas_boiler_combi", "gas_boiler_regular", "oil_boiler", "lpg_boiler", "air_source_heat_pump",
        "ground_source_heat_pump", "electric_storage", "electric_storage_hhr", "electric_radiators",
        "district_or_communal", "solid_fuel"]] = None
    hot_water_source: Optional[Literal["from_boiler", "cylinder_from_boiler", "electric_immersion",
                                       "heat_pump_cylinder", "instantaneous_electric"]] = None
    internal_walls: Optional[Literal["solid", "hollow"]] = Field(
        None, description="Do internal walls sound solid or hollow when knocked? (thermal mass)")
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_months(self) -> "TwinIntake":
        months = [r.month for r in self.meter_readings]
        if len(months) != len(set(months)):
            raise ValueError("meter readings repeat a month")
        for start, end in self.heating_periods:
            if not (0 <= start <= 23 and 0 <= end <= 24):
                raise ValueError(f"heating period {(start, end)} is not within 0-24 h")
        return self

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
        if not self.meter_readings:
            self.warnings.append("No meter readings: the twin can't be calibrated to this home yet (prior only)")
        elif len(self.meter_readings) < 12:
            self.warnings.append(f"Only {len(self.meter_readings)} months of meter readings; 12+ covering winter and "
                                 "summer are needed for a reliable calibration")


class EPCLookup(Protocol):
    """Seam to the data layer (Part 0): fetch a home's EPC by certificate number. Returns None if not found."""

    def by_rrn(self, rrn: str, postcode_district: str) -> Optional[Property]: ...


class NoEPCLookup:
    """Used until the data layer offers a lookup by RRN, and when there's no EPC API key: always None."""

    def by_rrn(self, rrn: str, postcode_district: str) -> Optional[Property]:
        return None
