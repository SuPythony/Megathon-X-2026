"""Key issues (brief section 3): why a home performs poorly, with the evidence behind each issue.

Rules follow LLD section 4.3. Intensity is judged per m2 against peers of the same type, so a large home is
not flagged just for using more energy in total.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from contracts.models import GlazingType, KeyIssue, Property

from . import mapping as m
from .mapping import BOILERS

HIGH_INTENSITY_P = 0.7      # P(intensity > peer median) above which intensity is an issue
POOR_SUB_SCORE = 40.0        # sub-score below which a component is an issue
OLD_BOILER_YEARS = 15


def _sev(sub: float) -> str:
    return "high" if sub < 20 else "medium" if sub < POOR_SUB_SCORE else "low"


def key_issues(prop: Property, subs: Mapping[str, float], intensity_samples: np.ndarray,
               peer_p50_kwh_m2: Optional[float], missing: list[str]) -> list[KeyIssue]:
    out: list[KeyIssue] = []

    mean_int = float(np.mean(intensity_samples))
    if peer_p50_kwh_m2 is not None:
        p_above = float(np.mean(intensity_samples > peer_p50_kwh_m2))
        if p_above > HIGH_INTENSITY_P:
            out.append(KeyIssue(
                code="high_intensity", label="High energy use per m2 for a home like this",
                severity="high" if mean_int > 1.5 * peer_p50_kwh_m2 else "medium",
                evidence={"kwh_m2": round(mean_int, 1), "peer_p50_kwh_m2": round(peer_p50_kwh_m2, 1)},
                probability=round(p_above, 3)))
    elif subs["energy_intensity"] < POOR_SUB_SCORE:
        out.append(KeyIssue(
            code="high_intensity", label="High energy use per m2", severity=_sev(subs["energy_intensity"]),
            evidence={"kwh_m2": round(mean_int, 1), "basis": "absolute scale (no peer data)"}))

    # judged per element: a good loft must not hide unfilled cavity walls (D-24)
    elements = {"walls": m.wall_u_sub(prop.walls.u_value) if prop.walls.u_value is not None
                else m.wall_sub(prop.walls.type, prop.walls.insulation, prop.construction_age_band)}
    if prop.roof.type.value != "pitched_loft" or prop.roof.loft_insulation_mm is not None:
        elements["roof"] = m.roof_sub(prop)
    elements["floor"] = m.floor_sub(prop)
    weak = {k: float(v) for k, v in elements.items() if v is not None and float(v) < POOR_SUB_SCORE}
    if subs["insulation"] < POOR_SUB_SCORE or weak:
        ev = {"walls": f"{prop.walls.type.value}, {prop.walls.insulation.value}"}
        if prop.roof.loft_insulation_mm is not None:
            ev["loft_mm"] = prop.roof.loft_insulation_mm
        ev.update({f"{k}_sub_score": round(v, 1) for k, v in weak.items()})
        worst = min([subs["insulation"], *weak.values()])
        out.append(KeyIssue(code="poor_insulation",
                            label="Poorly insulated " + (", ".join(weak) if weak else "walls, roof or floor"),
                            severity=_sev(worst), evidence=ev))

    age = prop.heating.system_age_years
    old_boiler = prop.heating.main_system in BOILERS and age is not None and age > OLD_BOILER_YEARS
    if subs["heating_system"] < POOR_SUB_SCORE or old_boiler:
        ev = {"heating": prop.heating.main_system.value}
        if age is not None:
            ev["age_years"] = age
        out.append(KeyIssue(code="old_heating", label="Inefficient or high-carbon heating system",
                            severity=_sev(subs["heating_system"]), evidence=ev))

    single = sum(g.share_pct for g in prop.glazing if g.type == GlazingType.single)
    if single > 0:
        tot = sum(g.share_pct for g in prop.glazing)
        out.append(KeyIssue(code="single_glazing", label="Single-glazed windows",
                            severity="high" if single / tot >= 0.5 else "medium",
                            evidence={"single_share_pct": round(100 * single / tot, 1)}))

    if subs["renewable_energy"] == 0:
        out.append(KeyIssue(code="no_renewables", label="No renewable generation", severity="low",
                            evidence={"pv_kwp": prop.renewables.pv_kwp}))

    ctl = m.controls_level(prop.controls)  # judged on controls alone: lighting must not hide missing controls
    if ctl < POOR_SUB_SCORE:
        out.append(KeyIssue(code="no_controls", label="Basic or no heating controls",
                            severity=_sev(ctl),
                            evidence={"controls": ", ".join(c.value for c in prop.controls) or "none"}))

    for c in prop.data_quality.conflicts:
        out.append(KeyIssue(code="data_conflict", label="Conflicting data", severity="medium", evidence={"detail": c}))
    if missing:
        out.append(KeyIssue(code="missing_data", label="Some inputs were missing and were estimated",
                            severity="low", evidence={"fields": ", ".join(missing)}))
    return out
