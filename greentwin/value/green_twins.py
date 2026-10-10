"""Green twins (V-09): what greener comparable homes nearby sold for, shown as ranges only (never an address).

A sale's greenness is the Green Home Score at sale WITHOUT the intensity component (no meter data for sales): the
brief's weights renormalised over the other six components, sub-scores computed from the sale's EPC text with the
same parsers as the score (greentwin.score.epc_features). Twins = sales of the same type, built form and age band,
floor area within +-10%, sold 2019-2024, searched in the home's LSOA, then MSOA, then local authority, grouped into
greenness tiers by score gap. Each tier reports the HPI-adjusted price per m2 as a Distribution over the matched
sales. Descriptive, not causal: it shows what greener neighbours sold for, nothing more.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from contracts.models import GreenTwin, Property
from contracts.summarise import summarise
from greentwin.score import epc_features as E
from greentwin.score import mapping as m

TIERS = [(15, 25), (25, 35), (35, 101)]
MIN_SALES = 5          # a tier is shown only if at least this many sales match (privacy + stability)
YEARS = (2019, 2024)
NO_INTENSITY = [c for c in m.COMPONENTS if c != "energy_intensity"]


def _renorm_weights() -> dict[str, float]:
    w = m.weights()
    tot = sum(w[c] for c in NO_INTENSITY)
    return {c: w[c] / tot for c in NO_INTENSITY}


def sale_scores(s: pd.DataFrame) -> pd.Series:
    """Green Home Score at sale without intensity, from the linked EPC's text (NaN when any part is unknown)."""
    num = lambda c: pd.to_numeric(s[c], errors="coerce")  # noqa: E731
    walls = E._map_unique(s["WALLS_DESCRIPTION"], E.wall_score)
    roof = E._map_unique(s["ROOF_DESCRIPTION"], E.roof_score)
    floor = E._map_unique(s["FLOOR_DESCRIPTION"], E.floor_score)
    ins = pd.Series([E.insulation_from(w, r, f) for w, r, f in zip(walls, roof, floor)], index=s.index)
    key = (s["MAINHEAT_DESCRIPTION"].fillna("") + E.SEP + s["MAIN_FUEL"].fillna("") + E.SEP
           + s["MAINHEAT_ENERGY_EFF"].fillna("") + E.SEP + s["MAINHEATCONT_DESCRIPTION"].fillna(""))
    hs = E._map_unique(key, lambda k: getattr(E.heating_system(*[x or None for x in k.split(E.SEP)]), "value", None))
    heat = pd.Series([E.heating_score(h) for h in hs], index=s.index)
    gkey = (s["WINDOWS_DESCRIPTION"].fillna("§") + E.SEP + s["GLAZED_TYPE"].fillna("") + E.SEP
            + s["MULTI_GLAZE_PROPORTION"].fillna(""))
    glz = E._map_unique(gkey, lambda k: (lambda a, b, c: E.glazing_score(None if a == "§" else a, b or None,
                                                                          c or None))(*k.split(E.SEP)))
    kwp = [E.pv_kwp(p, t, ty) for p, t, ty in zip(s["PHOTO_SUPPLY"], s["tfa"], s["ptype"])]
    st = s["SOLAR_WATER_HEATING_FLAG"].eq("Y")
    ren = pd.Series([m.renewables_sub(k, 0.0, bool(x)) if not np.isnan(k) else np.nan for k, x in zip(kwp, st)],
                    index=s.index)
    ctl = E._map_unique(s["MAINHEATCONT_DESCRIPTION"], E.controls_list)
    light = num("LOW_ENERGY_LIGHTING")
    smart = pd.Series([float(m.smart_sub(c, li)) if c is not None and not np.isnan(li) else np.nan
                       for c, li in zip(ctl, light)], index=s.index)
    subs = {"energy_efficiency": m.ee_from_sap(s["sap"]), "insulation": ins, "heating_system": heat,
            "windows_glazing": glz, "renewable_energy": ren, "smart_energy_management": smart}
    w = _renorm_weights()
    return sum(w[c] * pd.to_numeric(subs[c], errors="coerce") for c in NO_INTENSITY)


def home_score(prop: Property, lighting_default: float = 82.0) -> float:
    """The home's own score on the same basis (no intensity). Unknown lighting -> EPC prior mean (82%)."""
    if prop.epc.sap_score is None:
        return np.nan
    loft = prop.roof.loft_insulation_mm
    if prop.roof.type.value == "pitched_loft" and loft is None:
        loft = 150  # prior median depth; only used to place the home, not to score it
    subs = {
        "energy_efficiency": float(m.ee_from_sap(prop.epc.sap_score)),
        "insulation": float(m.insulation_sub(prop, loft)),
        "heating_system": m.heating_sub(prop.heating.main_system, prop.heating.system_age_years),
        "windows_glazing": m.glazing_sub(prop.glazing),
        "renewable_energy": m.renewables_sub(prop.renewables.pv_kwp, prop.renewables.battery_kwh,
                                             prop.renewables.solar_thermal),
        "smart_energy_management": float(m.smart_sub(prop.controls, prop.low_energy_lighting_pct
                                                     if prop.low_energy_lighting_pct is not None else lighting_default)),
    }
    w = _renorm_weights()
    return sum(w[c] * subs[c] for c in NO_INTENSITY)


def find(prop: Property, sales: pd.DataFrame, k: int = 3, lsoa: Optional[str] = None,
         msoa: Optional[str] = None) -> list[GreenTwin]:
    from .model import ppd_type
    own = home_score(prop)
    if np.isnan(own):
        return []
    form = {"semi_detached": "Semi-Detached", "mid_terrace": "Mid-Terrace", "end_terrace": "End-Terrace",
            "detached": "Detached", "enclosed_mid_terrace": "Enclosed Mid-Terrace",
            "enclosed_end_terrace": "Enclosed End-Terrace"}.get(prop.built_form.value)
    c = sales[(sales["ptype"] == ppd_type(prop)) & sales["year"].between(*YEARS)
              & sales["tfa"].between(prop.floor_area_m2 * 0.9, prop.floor_area_m2 * 1.1)]
    if form:
        c = c[c["BUILT_FORM"] == form]
    if prop.construction_age_band:
        c = c[c["age_band"].astype(str).str.startswith(prop.construction_age_band[:4])]
    levels = [("LSOA", "lsoa21", lsoa or prop.location.lsoa_code), ("MSOA", "msoa21", msoa),
              ("local authority", "laua", prop.location.local_authority_code)]
    out: list[GreenTwin] = []
    for lo, hi in TIERS:
        for name, col, code in levels:
            if not code:
                continue
            pool = c[c[col] == code]
            pool = pool[(pool["green_score"] - own).between(lo, hi - 1e-9)]
            if len(pool) >= MIN_SALES:
                ppm2 = pool["price_adj"] / pool["tfa"]
                bands = pool["CURRENT_ENERGY_RATING"].mode()
                out.append(GreenTwin(
                    match_rank=len(out) + 1, score_gap=round(float((pool["green_score"] - own).median()), 1),
                    price_per_m2_gbp=summarise(ppm2, "GBP/m2"),
                    epc_band=str(bands.iloc[0]) if len(bands) else None,
                    same=["property type", "built form", "age band", f"floor area +-10%", name,
                          f"sold {YEARS[0]}-{YEARS[1]}", f"{len(pool)} sales"]))
                break
        if len(out) >= k:
            break
    return out
