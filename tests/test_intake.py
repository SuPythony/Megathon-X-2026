"""Intake: location is reduced to the postcode district, EPC numbers are normalised, bad input is rejected."""

import pytest
from pydantic import ValidationError

from greentwin.twin.intake import NoEPCLookup, TwinIntake, normalise_rrn, postcode_district


@pytest.mark.parametrize("raw,district", [
    ("M14 5AB", "M14"), ("m145ab", "M14"), ("SW1A 1AA", "SW1A"), ("  cf10  3nb ", "CF10"), ("M14", "M14"),
    ("EC1A", "EC1A"),
])
def test_postcode_reduced_to_district(raw, district):
    assert postcode_district(raw) == district


@pytest.mark.parametrize("raw", ["", "hello", "12345", "M14 5A", "ABCD 1AA"])
def test_bad_postcodes_rejected(raw):
    with pytest.raises(ValueError):
        postcode_district(raw)


def test_rrn_normalised_and_validated():
    assert normalise_rrn("0390 2954 3640 2196 6705") == "0390-2954-3640-2196-6705"
    assert normalise_rrn("03902954364021966705") == "0390-2954-3640-2196-6705"
    for bad in ["0390-2954-3640-2196", "ABCD-2954-3640-2196-6705", "0390-2954-3640-2196-67051"]:
        with pytest.raises(ValueError):
            normalise_rrn(bad)


def test_intake_never_keeps_full_postcode():
    intake = TwinIntake(intake_id="i1", postcode_district="M14 5AB", epc_rrn="03902954364021966705")
    dumped = intake.model_dump_json()
    assert intake.postcode_district == "M14" and "5AB" not in dumped
    assert intake.epc_rrn == "0390-2954-3640-2196-6705"
    assert intake.warnings == []


def test_intake_warnings():
    scot = TwinIntake(intake_id="i2", postcode_district="EH1 1YZ", floor_plan_image="plan.jpg")
    text = " ".join(scot.warnings)
    assert "Scotland" in text and "No EPC number" in text and "consent" in text


def test_intake_rejects_bad_input():
    with pytest.raises(ValidationError):
        TwinIntake(intake_id="i3", postcode_district="not a postcode")
    with pytest.raises(ValidationError):
        TwinIntake(intake_id="i4", postcode_district="M14", epc_rrn="1234")


def test_no_lookup_returns_none():
    assert NoEPCLookup().by_rrn("0390-2954-3640-2196-6705", "M14") is None
