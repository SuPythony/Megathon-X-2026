"""Layer-1 feasibility rules (contracts/feasibility_rules.json) + confirmed customer-memory constraints.

The config's `when` fields are pseudo-code; each is implemented below under its rule id. Rule vetoes always win:
memory constraints can only add vetoes or conditions, never lift one. Rules-only FeasibilityEngine (Jev layer 2
is not wired; `router = "rules_only"`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from contracts.models import Feasibility, FeasibilityReport, Property

ROOT = Path(__file__).resolve().parents[2]
RULES = json.loads((ROOT / "contracts/feasibility_rules.json").read_text())
CATALOGUE = json.loads((ROOT / "contracts/retrofit_catalogue.json").read_text())
ITEMS = {i["id"]: i for i in CATALOGUE["items"]}
ORDER = [rid for group in RULES["ordering"] for rid in group]


def _v(x):
    return getattr(x, "value", x)


WHEN: dict[str, Callable[[Property], bool]] = {
    "R_NO_LOFT": lambda p: _v(p.roof.type) in ("another_dwelling_above", "flat"),
    "R_LOFT_ALREADY_DEEP": lambda p: (p.roof.loft_insulation_mm or 0) >= 270,
    "R_NOT_CAVITY": lambda p: _v(p.walls.type) != "cavity" or _v(p.walls.insulation) != "none",
    "R_SOLID_WALL_ONLY": lambda p: _v(p.walls.type) == "cavity" or _v(p.walls.insulation) != "none",
    "R_CONSERVATION_EWI": lambda p: p.constraints.conservation_area or p.constraints.listed_building,
    "R_LISTED_GLAZING": lambda p: p.constraints.listed_building,
    "R_GLAZING_DONE": lambda p: all(_v(g.type) in ("double_post_2002", "triple") for g in p.glazing),
    "R_NO_ROOF_PV": lambda p: p.constraints.roof_access is False,
    "R_NORTH_ROOF": lambda p: p.constraints.best_roof_azimuth_deg is not None
    and abs(p.constraints.best_roof_azimuth_deg - 180) > 100,
    "R_HP_NO_OUTDOOR": lambda p: p.constraints.has_outdoor_space is not True,
    "R_HP_ALREADY": lambda p: _v(p.heating.main_system) in ("air_source_heat_pump", "ground_source_heat_pump"),
    "R_NO_FLOOR": lambda p: _v(p.floor.type) == "another_dwelling_below" or p.floor.insulated,
    "R_HHR_ONLY_STORAGE": lambda p: _v(p.heating.main_system) not in ("electric_storage", "electric_direct"),
    "R_SMART_DONE": lambda p: "smart_thermostat" in [_v(c) for c in p.controls],
    "R_BATTERY_NEEDS_PV": lambda p: p.renewables.pv_kwp == 0,  # scenario-level PV is judged by run_scenario
    "R_LED_DONE": lambda p: (p.low_energy_lighting_pct or 0) >= 90,
    "R_TOU_NEEDS_SMART_METER": lambda p: "smart_meter" not in [_v(c) for c in p.controls],
}
RANK = {"feasible": 0, "feasible_with_conditions": 1, "needs_survey": 2, "infeasible": 3}

# Confirmed memory constraints -> extra vetoes / conditions (never lift a rule veto).
LANDLORD_ITEMS = ["internal_wall_insulation", "external_wall_insulation", "floor_insulation", "double_glazing",
                  "triple_glazing", "air_source_heat_pump", "solar_pv", "battery", "hhr_storage_heaters"]


def memory_effects(constraints: dict) -> list[tuple[str, str, str, str]]:
    """(retrofit_id, status, code, reason) from confirmed constraints {key: value}."""
    out = []
    if constraints.get("tenure") == "renting":
        out += [(r, "feasible_with_conditions", "M_RENTING", "You rent: needs your landlord's permission")
                for r in LANDLORD_ITEMS]
    if constraints.get("outdoor_space") == "none":
        out.append(("air_source_heat_pump", "infeasible", "M_NO_OUTDOOR", "You said there is no outdoor space for a unit"))
    if constraints.get("roof") == "not_available":
        out += [(r, "infeasible", "M_NO_ROOF", "You said the roof is not available") for r in ("solar_pv", "battery")]
    cap = constraints.get("budget_gbp")
    if cap:
        for rid, it in ITEMS.items():
            c = it.get("cost_gbp_central")
            if it.get("cost_status") != "unsourced" and it.get("cost_basis") == "per_home" and c and c > float(cap):
                out.append((rid, "feasible_with_conditions", "M_OVER_BUDGET",
                            f"Typical cost is above your stated budget of £{float(cap):,.0f}"))
    return out


def assess(prop: Property, constraints: Optional[dict] = None) -> FeasibilityReport:
    status = {rid: ("feasible", [], []) for rid in ITEMS}
    for rule in RULES["rules"]:
        fn = WHEN.get(rule["id"])
        if fn is None or not fn(prop):
            continue
        for rid in rule.get("blocks", []):
            st, codes, reasons = status[rid]
            status[rid] = ("infeasible", codes + [rule["id"]], reasons + [rule["reason"]])
        for rid, new in rule.get("downgrade", {}).items():
            st, codes, reasons = status[rid]
            status[rid] = (max(st, new, key=RANK.get), codes + [rule["id"]], reasons + [rule["reason"]])
    for rid, new, code, reason in memory_effects(constraints or {}):
        st, codes, reasons = status[rid]
        status[rid] = (max(st, new, key=RANK.get), codes + [code], reasons + [reason])
    items = [Feasibility(retrofit_id=rid, status=st, decided_by="rules", reason_codes=codes, reasons=reasons,
                         p_feasible=0.0 if st == "infeasible" else 1.0)
             for rid, (st, codes, reasons) in status.items()]
    items.sort(key=lambda f: ORDER.index(f.retrofit_id) if f.retrofit_id in ORDER else 99)
    return FeasibilityReport(property_id=prop.property_id, items=items, rules_version=str(RULES["version"]),
                             router="rules_only")
