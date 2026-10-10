"""EPC text -> sub-score parsing (training data for the learned weights)."""

import numpy as np

from contracts.models import HeatingSystem
from greentwin.score import epc_features as E
from greentwin.score import mapping as m


def test_walls():
    assert E.wall_score("Solid brick, as built, no insulation (assumed)") == 0
    assert E.wall_score("Cavity wall, as built, no insulation (assumed)") == 25
    assert E.wall_score("Cavity wall, filled cavity") == 75
    assert E.wall_score("Cavity wall, as built, insulated (assumed)") == 100
    assert E.wall_score("Solid brick, with internal insulation") == 85
    assert E.wall_score("Average thermal transmittance 0.18 W/m²K") == 100
    assert np.isnan(E.wall_score(None))


def test_roof_and_floor():
    assert E.roof_score("(another dwelling above)") == E.ABSENT_ELEMENT
    assert E.roof_score("Pitched, 270 mm loft insulation") == 100
    assert E.roof_score("Pitched, 400+ mm loft insulation") == 100
    assert E.roof_score("Pitched, no insulation (assumed)") == 0
    assert E.roof_score("Average thermal transmittance 0.40 W/m²K") == float(m.loft_sub(100))
    assert E.floor_score("(other premises below)") == E.ABSENT_ELEMENT
    assert E.floor_score("Solid, no insulation (assumed)") == 30
    # flat with dwellings above and below: insulation = walls only
    assert E.insulation_from(75.0, E.ABSENT_ELEMENT, E.ABSENT_ELEMENT) == 75.0
    assert np.isnan(E.insulation_from(75.0, np.nan, 30.0))  # unknown roof -> unknown, never 0


def test_heating_parsing():
    hs = E.heating_system
    assert hs("Boiler and radiators, mains gas", "mains gas (not community)", "Very Good", None) == HeatingSystem.gas_boiler_condensing
    assert hs("Boiler and radiators, mains gas", "mains gas (not community)", "Good", None) is E.GAS_BOILER_UNKNOWN
    assert hs("Boiler and radiators, oil", "oil (not community)", "Average", None) == HeatingSystem.oil_boiler
    assert hs("Air source heat pump, radiators, electric", "electricity", "Good", None) == HeatingSystem.air_source_heat_pump
    assert hs("Electric storage heaters", "electricity", "Poor", "Controls for high heat retention storage heaters") \
        == HeatingSystem.electric_storage_hhr
    assert hs("Room heaters, electric", "electricity", "Very Poor", None) == HeatingSystem.electric_direct
    assert hs("Community scheme", "mains gas (community)", "Good", None) == HeatingSystem.district_heating


def test_glazing_and_controls():
    assert E.glazing_score("Single glazed", None, None) == 0
    assert E.glazing_score("Fully double glazed", "double glazing installed during or after 2002", "100") == 80
    assert E.glazing_score("Fully double glazed", "double glazing, unknown install date", "100") == 70
    assert E.glazing_score("Partial double glazing", "double glazing installed before 2002", "50") == 30
    assert m.smart_sub(E.controls_list("Programmer, room thermostat and TRVs")) == 60
    assert m.smart_sub(E.controls_list("Programmer, no room thermostat")) == 20
