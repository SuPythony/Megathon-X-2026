"""Generate the 3 showcase homes + 12 months of synthetic usage + one MOCK scenario result.

These are SYNTHETIC archetypes (no real addresses) used for the demo, UI development
against mocks, and contract tests. Numbers in the mock result are placeholders,
NOT model output: provenance.engine == "mock".

Run:  python scripts/make_showcase_examples.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contracts.summarise import exact, summarise  # noqa: E402

OUT = ROOT / "contracts" / "examples"
OUT.mkdir(parents=True, exist_ok=True)

# Typical UK monthly heating-degree-day shares (Jan..Dec), normalised below.
HDD_SHARE = [0.16, 0.14, 0.12, 0.09, 0.06, 0.03, 0.02, 0.02, 0.04, 0.08, 0.11, 0.14]
HDD_SHARE = [s / sum(HDD_SHARE) for s in HDD_SHARE]
MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]

HOMES = [
    {
        "schema_version": "0.4.0",
        "property_id": "SH1",
        "customer_id": "C-SH1",
        "source": "showcase",
        "split": None,
        "location": {"postcode_district": "M14", "lsoa_code": None, "local_authority": "Manchester", "local_authority_code": "E08000003",
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
        "schema_version": "0.4.0",
        "property_id": "SH2",
        "customer_id": "C-SH2",
        "source": "showcase",
        "split": None,
        "location": {"postcode_district": "BS16", "lsoa_code": None, "local_authority": "South Gloucestershire", "local_authority_code": "E06000025",
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
        "schema_version": "0.4.0",
        "property_id": "SH3",
        "customer_id": "C-SH3",
        "source": "showcase",
        "split": None,
        "location": {"postcode_district": "CR0", "lsoa_code": None, "local_authority": "Croydon", "local_authority_code": "E09000008",
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


_RNG = np.random.default_rng(42)
N_MOCK = 500


def dist(mean, lo, hi, unit, tails=(), worst_tail="low"):
    """Placeholder Distribution: normal samples with the given mean and ~90% range, summarised
    exactly as real model outputs are (contracts.summarise). lo == hi -> exact value."""
    if lo == hi:
        return exact(mean, unit, N_MOCK).model_dump()
    samples = _RNG.normal(mean, (hi - lo) / 3.29, N_MOCK)
    return summarise(samples, unit, tails=tails, worst_tail=worst_tail).model_dump()


BASIS = "Estimated from home type and EPC descriptions; only the total heat loss is calibrated to bills"


def mock_result() -> dict:
    """Shape-only example for UI/LLM development. All numbers are placeholders."""
    res = _mock_result()
    for e in res["element_heat_loss_before"]:
        e.update({"estimated": True, "basis": BASIS})
    return res


def _mock_result() -> dict:
    return {
        "schema_version": "0.4.0", "scenario_id": "SH2-pkg-mock", "property_id": "SH2",
        "provenance": {"engine": "mock", "model_version": "mock-0", "data_version": "mock-0", "seed": 42, "n_samples": 0},
        "baseline": {"electricity_import_kwh": dist(3400, 3250, 3550, "kWh/yr"), "gas_kwh": dist(12800, 12300, 13300, "kWh/yr"),
                     "purchased_total_kwh": dist(16200, 15600, 16800, "kWh/yr"), "intensity_kwh_m2": dist(171, 164, 177, "kWh/m2/yr")},
        "after": {"electricity_import_kwh": dist(5100, 4600, 5700, "kWh/yr"), "gas_kwh": dist(0, 0, 0, "kWh/yr"),
                  "purchased_total_kwh": dist(5100, 4600, 5700, "kWh/yr"), "intensity_kwh_m2": dist(54, 48, 60, "kWh/m2/yr")},
        "electricity_saving_kwh": dist(-1700, -2300, -1200, "kWh/yr",
                                       tails=[("electricity use rises", 0.0, "below")]),
        "gas_saving_kwh": dist(12800, 12300, 13300, "kWh/yr"),
        "purchased_energy_saving_kwh": dist(11100, 10400, 11800, "kWh/yr"),
        "goal": "bills",
        "improvement_potential": {
            "pct": dist(68.5, 64.0, 72.0, "%"),
            "electricity_change_kwh": dist(-1700, -2300, -1200, "kWh/yr", tails=[("electricity use rises", 0.0, "below")]),
            "gas_change_kwh": dist(12800, 12300, 13300, "kWh/yr"),
            "included_retrofits": ["cavity_wall_insulation", "loft_insulation", "draught_proofing", "smart_thermostat",
                                   "double_glazing", "air_source_heat_pump", "solar_pv"],
            "excluded_needs_survey": [], "p_top_decile": 0.62},
        "carbon_saving_kgco2e": dist(2010, 1880, 2140, "kgCO2e/yr"),
        "emission_factors": {"electricity": 0.19553, "gas": 0.18296},
        "emission_factor_source": "contracts/emission_factors.json (DESNZ 2025: elec generation + T&D; gas gross CV)",
        "bill_saving_gbp": dist(560, 420, 690, "GBP/yr",
                                tails=[("bill rises", 0.0, "below"), ("bill falls by more than £500/yr", 500.0, "above")]),
        "score_before": {"score": 46, "score_distribution": dist(46.2, 41, 51, "points"), "band": "Poor",
                         "band_probabilities": {"Very poor": 0.0, "Poor": 0.8, "Moderate": 0.2, "Good": 0.0,
                                                "Highly efficient": 0.0}, "p_poor_or_very_poor": 0.8,
                         "components": [], "rating_source": "recorded_epc",
                         "peer_context": {"la_share_epc_a_to_c": 0.41, "source": "EPC live table D1, South Gloucestershire"},
                         "key_issues": [
                             {"code": "poor_insulation", "label": "Unfilled cavity walls", "severity": "high",
                              "evidence": {"walls": "cavity, no insulation"}},
                             {"code": "old_heating", "label": "Gas boiler is the main heat source", "severity": "medium",
                              "evidence": {"heating": "gas_boiler_condensing"}},
                             {"code": "no_renewables", "label": "No renewable generation", "severity": "medium",
                              "evidence": {"pv_kwp": 0}}]},
        "score_after": {"score": 78, "score_distribution": dist(77.8, 74, 82, "points"), "band": "Good",
                        "band_probabilities": {"Very poor": 0.0, "Poor": 0.0, "Moderate": 0.0, "Good": 0.7,
                                               "Highly efficient": 0.3}, "p_poor_or_very_poor": 0.0,
                        "components": [], "key_issues": [], "projected": True, "rating_source": "estimated"},
        "retrofits": [{
            "retrofit_id": "air_source_heat_pump", "order": 3,
            "feasibility": {"retrofit_id": "air_source_heat_pump", "status": "feasible", "decided_by": "rules",
                            "reason_codes": [], "p_feasible": 1.0},
            "electricity_saving_kwh": dist(-2300, -2900, -1800, "kWh/yr", tails=[("electricity use rises", 0.0, "below")]),
            "gas_saving_kwh": dist(7150, 6500, 7800, "kWh/yr"),
            "purchased_energy_saving_kwh": dist(4850, 4100, 5600, "kWh/yr"),
            "carbon_saving_kgco2e": dist(860, 700, 1010, "kgCO2e/yr"),
            "bill_saving_gbp": dist(-35, -180, 110, "GBP/yr",
                                    tails=[("bill saving is positive", 0.0, "above"),
                                           ("bill rises by more than £100/yr", -100.0, "below")]),
            "score_after": dist(80, 77, 83, "points"),
            "improvement_potential_cumulative_pct": dist(64, 58, 69, "%"),
            "p_net_benefit": 0.38, "p_suitable": 0.38, "suitability": "Low", "p_gain_above_threshold": 0.99,
            "verdict": "optional",
            "reason": "Large carbon cut; bill change centred near zero at current tariffs (goal = bills)"}],
        "rating_after": {"sap": dist(77, 73, 81, "SAP points"), "band": "C", "band_probabilities": {"C": 0.8, "B": 0.15, "D": 0.05},
                         "anchored_to_recorded_sap": True, "method": "sap2012_ecf_delta", "label": "estimated rating, not an EPC"},
        "value": {"current_value_gbp": 325000, "band_before": "D", "band_after": "C", "uplift_pct": dist(3.0, 1.5, 4.5, "%", tails=[("no value increase", 0.0, "below")]),
                  "value_after_gbp": dist(334750, 329875, 339625, "GBP"), "method": "bayesian_hedonic", "label": "indicative",
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


def mock_scene() -> dict:
    """Shape-only TwinScene for 3D development (SH2, parametric geometry)."""
    surf = [("roof_s", "roof", 180, 48, 2.3, 15), ("roof_n", "roof", 0, 48, 2.3, 0),
            ("wall_s", "walls", 180, 38, 1.5, 9), ("wall_n", "walls", 0, 38, 1.5, 9), ("wall_e", "walls", 90, 40, 1.5, 10),
            ("win_s", "windows", 180, 9, 2.8, 12), ("win_n", "windows", 0, 7, 2.8, 8), ("floor", "floor", None, 48, 0.7, 12),
            ("door", "doors", 0, 3.7, 3.0, 4), ("party", "party_walls", 270, 40, 0.0, 0)]
    return {
        "property_id": "SH2", "scenario_id": None, "geometry_source": "parametric",
        "property_type": "house", "built_form": "semi_detached", "storeys": 2, "footprint_m": [8.0, 6.0],
        "roof_type": "pitched_loft", "roof_pitch_deg": 35,
        "surfaces": [{"surface_id": i, "element": e, "orientation_deg": o, "area_m2": a, "u_value": u,
                      "heat_loss_share_pct": sh, "estimated": True} for i, e, o, a, u, sh in surf],
        "rooms": [],
        "retrofit_slots": [
            {"retrofit_id": "solar_pv", "anchor": "roof_face", "surface_id": "roof_s", "status": "feasible"},
            {"retrofit_id": "air_source_heat_pump", "anchor": "outside_wall", "surface_id": "wall_e",
             "position_m": [8.4, 3.0, 0.0], "status": "feasible"},
            {"retrofit_id": "loft_insulation", "anchor": "loft", "status": "feasible"},
            {"retrofit_id": "cavity_wall_insulation", "anchor": "outside_wall", "status": "feasible"},
            {"retrofit_id": "external_wall_insulation", "anchor": "outside_wall", "status": "infeasible",
             "reason": "Not a solid wall"}],
        "heat_loss_label": BASIS,
    }


def mock_chat() -> dict:
    """Shape-only ChatResponse for UI/LLM development."""
    return {
        "answer": "MOCK: Adding a heat pump removes your gas use (about 7,150 kWh/yr) but raises electricity by about "
                  "2,300 kWh/yr, so your bill change is centred near zero (-£35/yr, 90% range -£180 to +£110). "
                  "These are estimates from your home's model, not guarantees.",
        "route": {"intent": "numeric", "confidence": 0.93, "needs_clarification": False, "router": "llm_fallback",
                  "scenario": {"scenario_id": "SH2-hp", "property_id": "SH2",
                               "retrofits": [{"retrofit_id": "air_source_heat_pump"}]}},
        "tool_calls": [{"name": "run_scenario", "input": {"scenario_id": "SH2-hp"}, "output_ref": "SH2-hp"}],
        "citations": [{"source": "tool_result", "ref": "SH2-hp"}],
        "check": {"numbers_found": ["7,150", "2,300", "35", "180", "110", "90"], "unsupported_numbers": [],
                  "passed": True},
        "scenario_result_id": "SH2-hp", "fallback": "none", "model": None,
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
        "schema_version": "0.4.0", "scenario_id": "SH2-pkg-1", "property_id": "SH2",
        "retrofits": [{"retrofit_id": "cavity_wall_insulation", "size": None, "stage_year": 0},
                      {"retrofit_id": "loft_insulation", "size": None, "stage_year": 0},
                      {"retrofit_id": "air_source_heat_pump", "size": 7, "stage_year": 1},
                      {"retrofit_id": "solar_pv", "size": 3.5, "stage_year": 2}],
        "weather": "typical", "n_samples": 500, "include_value": True, "include_pareto": False,
        "include_element_heat_loss": True, "seed": 42, "goal": "bills", "requested_by": "ui"}, indent=2), encoding="utf-8")
    (OUT / "SH2_scenario_result_MOCK.json").write_text(json.dumps(mock_result(), indent=2), encoding="utf-8")
    (OUT / "SH2_twin_scene_MOCK.json").write_text(json.dumps(mock_scene(), indent=2), encoding="utf-8")
    (OUT / "SH2_chat_response_MOCK.json").write_text(json.dumps(mock_chat(), indent=2, ensure_ascii=False),
                                                     encoding="utf-8")


if __name__ == "__main__":
    main()
