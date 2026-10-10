"""Generate the 3 showcase homes + 12 months of synthetic usage + one MOCK scenario result.

These are SYNTHETIC archetypes (no real addresses) used for the demo, UI development
against mocks, and contract tests. Numbers in the mock result are placeholders,
NOT model output: provenance.engine == "mock".

Run:  python scripts/make_showcase_examples.py
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "contracts" / "examples"
OUT.mkdir(parents=True, exist_ok=True)

# Typical UK monthly heating-degree-day shares (Jan..Dec), normalised below.
HDD_SHARE = [0.16, 0.14, 0.12, 0.09, 0.06, 0.03, 0.02, 0.02, 0.04, 0.08, 0.11, 0.14]
HDD_SHARE = [s / sum(HDD_SHARE) for s in HDD_SHARE]
MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]

HOMES = [
    {
        "schema_version": "0.2.0",
        "property_id": "SH1",
        "customer_id": "C-SH1",
        "source": "showcase",
        "split": None,
        "location": {"postcode_district": "M14", "lsoa_code": None, "local_authority": "Manchester",
                     "region": "North West", "lat": 53.447, "lon": -2.222, "altitude_m": 40},
        "property_type": "house", "built_form": "mid_terrace",
        "construction_age_band": "1900-1929", "year_built": 1905, "storeys": 2,
        "floor_area_m2": 85, "habitable_rooms": 4, "bedrooms": 3, "tenure": "owner_occupied",
        "epc": {"current_band": "E", "sap_score": 48, "lodgement_date": "2023-05-10",
                "potential_band": None, "potential_sap_score": None, "estimated_annual_energy_cost_gbp": None},
        "walls": {"type": "solid_brick", "insulation": "none", "u_value": None},
        "roof": {"type": "pitched_loft", "loft_insulation_mm": 50, "u_value": None},
        "floor": {"type": "suspended_timber", "insulated": False, "u_value": None},
        "glazing": [{"type": "single", "share_pct": 60}, {"type": "double_pre_2002", "share_pct": 40}],
        "draught_proofed_pct": 20,
        "heating": {"main_system": "gas_boiler_non_condensing", "main_fuel": "mains_gas", "system_age_years": 22,
                    "seasonal_efficiency": None, "hot_water": "from_main", "cylinder_insulated": True},
        "renewables": {"pv_kwp": 0, "pv_azimuth_deg": None, "pv_tilt_deg": None, "battery_kwh": 0, "solar_thermal": False},
        "controls": ["programmer"],
        "constraints": {"conservation_area": True, "listed_building": False, "leasehold": False,
                        "has_outdoor_space": True, "roof_access": True, "best_roof_azimuth_deg": 200,
                        "off_gas_grid": False, "floor_level": None,
                        "free_text_notes": ["Front facade in conservation area; small rear yard", "Customer keen on heat pump but worried about cost"]},
        "current_value_estimate_gbp": 215000, "value_estimate_date": "2026-09-30",
        "data_quality": {"missing_fields": [], "imputed_fields": [], "conflicts": [], "months_of_metered_data": 11},
        "_annual": {"elec": 3100, "gas": 15500, "gas_base_share": 0.18, "missing_month": "2025-08"},
        "_story": "Victorian solid-wall mid-terrace in a conservation area: EWI blocked -> internal wall insulation; classic 'fabric first, then heat pump' path; one missing month.",
    },
    {
        "schema_version": "0.2.0",
        "property_id": "SH2",
        "customer_id": "C-SH2",
        "source": "showcase",
        "split": None,
        "location": {"postcode_district": "BS16", "lsoa_code": None, "local_authority": "South Gloucestershire",
                     "region": "South West", "lat": 51.487, "lon": -2.512, "altitude_m": 60},
        "property_type": "house", "built_form": "semi_detached",
        "construction_age_band": "1967-1975", "year_built": 1972, "storeys": 2,
        "floor_area_m2": 95, "habitable_rooms": 5, "bedrooms": 3, "tenure": "owner_occupied",
        "epc": {"current_band": "D", "sap_score": 60, "lodgement_date": "2021-11-02",
                "potential_band": None, "potential_sap_score": None, "estimated_annual_energy_cost_gbp": None},
        "walls": {"type": "cavity", "insulation": "none", "u_value": None},
        "roof": {"type": "pitched_loft", "loft_insulation_mm": 100, "u_value": None},
        "floor": {"type": "solid", "insulated": False, "u_value": None},
        "glazing": [{"type": "double_pre_2002", "share_pct": 100}],
        "draught_proofed_pct": 70,
        "heating": {"main_system": "gas_boiler_condensing", "main_fuel": "mains_gas", "system_age_years": 13,
                    "seasonal_efficiency": None, "hot_water": "from_main", "cylinder_insulated": None},
        "renewables": {"pv_kwp": 0, "pv_azimuth_deg": None, "pv_tilt_deg": None, "battery_kwh": 0, "solar_thermal": False},
        "controls": ["programmer", "room_thermostat"],
        "constraints": {"conservation_area": False, "listed_building": False, "leasehold": False,
                        "has_outdoor_space": True, "roof_access": True, "best_roof_azimuth_deg": 165,
                        "off_gas_grid": False, "floor_level": None, "free_text_notes": []},
        "current_value_estimate_gbp": 325000, "value_estimate_date": "2026-09-30",
        "data_quality": {"missing_fields": [], "imputed_fields": [], "conflicts": [], "months_of_metered_data": 12},
        "_annual": {"elec": 3400, "gas": 12800, "gas_base_share": 0.2, "missing_month": None},
        "_story": "1970s cavity semi: every common retrofit feasible; the best Pareto and green-loan demo; shows electricity rising after a heat pump.",
    },
    {
        "schema_version": "0.2.0",
        "property_id": "SH3",
        "customer_id": "C-SH3",
        "source": "showcase",
        "split": None,
        "location": {"postcode_district": "CR0", "lsoa_code": None, "local_authority": "Croydon",
                     "region": "London", "lat": 51.376, "lon": -0.098, "altitude_m": 50},
        "property_type": "flat", "built_form": "not_applicable",
        "construction_age_band": "2003-2006", "year_built": 2004, "storeys": 1,
        "floor_area_m2": 62, "habitable_rooms": 3, "bedrooms": 2, "tenure": "owner_occupied",
        "epc": {"current_band": "C", "sap_score": 70, "lodgement_date": "2024-02-19",
                "potential_band": None, "potential_sap_score": None, "estimated_annual_energy_cost_gbp": None},
        "walls": {"type": "cavity", "insulation": "as_built", "u_value": None},
        "roof": {"type": "another_dwelling_above", "loft_insulation_mm": None, "u_value": None},
        "floor": {"type": "another_dwelling_below", "insulated": False, "u_value": None},
        "glazing": [{"type": "double_post_2002", "share_pct": 100}],
        "draught_proofed_pct": 100,
        "heating": {"main_system": "electric_storage", "main_fuel": "electricity", "system_age_years": 21,
                    "seasonal_efficiency": None, "hot_water": "electric_immersion", "cylinder_insulated": True},
        "renewables": {"pv_kwp": 0, "pv_azimuth_deg": None, "pv_tilt_deg": None, "battery_kwh": 0, "solar_thermal": False},
        "controls": ["programmer"],
        "constraints": {"conservation_area": False, "listed_building": False, "leasehold": True,
                        "has_outdoor_space": None, "roof_access": False, "best_roof_azimuth_deg": None,
                        "off_gas_grid": True, "floor_level": 3,
                        "free_text_notes": ["3rd-floor flat, small balcony facing the street", "Leaseholder: freeholder consent needed for external units"]},
        "current_value_estimate_gbp": 330000, "value_estimate_date": "2026-09-30",
        "data_quality": {"missing_fields": [], "imputed_fields": [], "conflicts": [], "months_of_metered_data": 12},
        "_annual": {"elec": 4600, "gas": 0, "gas_base_share": 0, "missing_month": None},
        "_story": "All-electric flat: no loft/PV (rules), heat pump 'needs survey' (Jev + agent), gas baseline 0 -> per-fuel handling and electricity-only improvement potential.",
    },
]


def usage(home: dict) -> list[dict]:
    a = home["_annual"]
    rows = []
    for m, share in zip(MONTHS, HDD_SHARE):
        gas = a["gas"] * (a["gas_base_share"] / 12 + (1 - a["gas_base_share"]) * share)
        # Electric-heated home: winter-weighted electricity; others: mild seasonality.
        heat_share = 0.55 if home["heating"]["main_fuel"] == "electricity" else 0.15
        elec = a["elec"] * ((1 - heat_share) / 12 + heat_share * share)
        missing = m == a["missing_month"]
        rows.append({
            "property_id": home["property_id"], "month": m,
            "electricity_import_kwh": None if missing else round(elec, 1),
            "gas_kwh": None if missing else round(gas, 1),
            "pv_generation_kwh": None, "electricity_export_kwh": None, "estimated_read": False,
        })
    return rows


def iv(mean, lo, hi, unit):
    return {"mean": mean, "lo": lo, "hi": hi, "level": 0.9, "unit": unit}


BASIS = "Estimated from home type and EPC descriptions; only the total heat loss is calibrated to bills"


def mock_result() -> dict:
    """Shape-only example for UI/LLM development. All numbers are placeholders."""
    res = _mock_result()
    for e in res["element_heat_loss_before"]:
        e.update({"estimated": True, "basis": BASIS})
    return res


def _mock_result() -> dict:
    return {
        "schema_version": "0.2.0", "scenario_id": "SH2-pkg-mock", "property_id": "SH2",
        "provenance": {"engine": "mock", "model_version": "mock-0", "data_version": "mock-0", "seed": 42, "n_samples": 0},
        "baseline": {"electricity_import_kwh": iv(3400, 3250, 3550, "kWh/yr"), "gas_kwh": iv(12800, 12300, 13300, "kWh/yr"),
                     "purchased_total_kwh": iv(16200, 15600, 16800, "kWh/yr"), "intensity_kwh_m2": iv(171, 164, 177, "kWh/m2/yr")},
        "after": {"electricity_import_kwh": iv(5100, 4600, 5700, "kWh/yr"), "gas_kwh": iv(0, 0, 0, "kWh/yr"),
                  "purchased_total_kwh": iv(5100, 4600, 5700, "kWh/yr"), "intensity_kwh_m2": iv(54, 48, 60, "kWh/m2/yr")},
        "electricity_saving_kwh": iv(-1700, -2300, -1200, "kWh/yr"),
        "gas_saving_kwh": iv(12800, 12300, 13300, "kWh/yr"),
        "purchased_energy_saving_kwh": iv(11100, 10400, 11800, "kWh/yr"),
        "improvement_potential_pct": iv(68.5, 64.0, 72.0, "%"),
        "carbon_saving_kgco2e": iv(2010, 1880, 2140, "kgCO2e/yr"),
        "emission_factors": {"electricity": 0.19553, "gas": 0.18296},
        "emission_factor_source": "contracts/emission_factors.json (DESNZ 2025: elec generation + T&D; gas gross CV)",
        "bill_saving_gbp": iv(560, 420, 690, "GBP/yr"),
        "score_before": {"score": 46, "band": "Poor", "band_probabilities": {"Poor": 0.8, "Moderate": 0.2},
                         "components": [], "key_issues": ["Unfilled cavity walls", "Gas boiler", "No renewables"]},
        "score_after": {"score": 78, "band": "Good", "band_probabilities": {"Good": 0.7, "Highly efficient": 0.3},
                        "components": [], "key_issues": []},
        "retrofits": [],
        "rating_after": {"sap": iv(77, 73, 81, "SAP points"), "band": "C", "band_probabilities": {"C": 0.8, "B": 0.15, "D": 0.05},
                         "anchored_to_recorded_sap": True, "method": "sap2012_ecf_delta", "label": "estimated rating, not an EPC"},
        "value": {"current_value_gbp": 325000, "band_before": "D", "band_after": "C", "uplift_pct": iv(3.0, 1.5, 4.5, "%"),
                  "value_after_gbp": iv(334750, 329875, 339625, "GBP"), "method": "hedonic_quantile", "label": "indicative",
                  "wording": "Indicative range based on historical associations in sales data; not a promised or proven increase."},
        "element_heat_loss_before": [
            {"element": "walls", "heat_loss_w_per_k": 110, "share_pct": 37},
            {"element": "roof", "heat_loss_w_per_k": 45, "share_pct": 15},
            {"element": "windows", "heat_loss_w_per_k": 60, "share_pct": 20},
            {"element": "floor", "heat_loss_w_per_k": 35, "share_pct": 12},
            {"element": "ventilation", "heat_loss_w_per_k": 48, "share_pct": 16}],
        "element_heat_loss_after": [],
        "pareto": [], "data_confidence": "high",
        "warnings": ["MOCK RESULT - shape only, numbers are placeholders, not model output"],
    }


def main() -> None:
    for h in HOMES:
        usage_rows = usage(h)
        story = h.pop("_story")
        h.pop("_annual")
        (OUT / f"{h['property_id']}_property.json").write_text(json.dumps(h, indent=2), encoding="utf-8")
        (OUT / f"{h['property_id']}_usage_2025.json").write_text(json.dumps(usage_rows, indent=2), encoding="utf-8")
        print(h["property_id"], "-", story)
    (OUT / "SH2_scenario_spec.json").write_text(json.dumps({
        "schema_version": "0.2.0", "scenario_id": "SH2-pkg-1", "property_id": "SH2",
        "retrofits": [{"retrofit_id": "cavity_wall_insulation", "size": None, "stage_year": 0},
                      {"retrofit_id": "loft_insulation", "size": None, "stage_year": 0},
                      {"retrofit_id": "air_source_heat_pump", "size": 7, "stage_year": 1},
                      {"retrofit_id": "solar_pv", "size": 3.5, "stage_year": 2}],
        "weather": "typical", "n_samples": 500, "include_value": True, "include_pareto": False,
        "include_element_heat_loss": True, "seed": 42, "requested_by": "ui"}, indent=2), encoding="utf-8")
    (OUT / "SH2_scenario_result_MOCK.json").write_text(json.dumps(mock_result(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
