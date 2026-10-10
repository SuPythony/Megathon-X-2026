"""Cross-database archetype join: EPC register <-> NEED (decision D-18).

NEED has real meter readings but no postcode or detailed attributes; the EPC register has detailed attributes
but modelled energy only. Neither joins per home, so both are aggregated to archetype cells
(property type x age band x floor-area band, NEED's own banding) and joined on the cell. Cell averages wash out
occupant behaviour, which dominates single-home meter readings. This is an ecological (cell-level) analysis
and is reported as such.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

NEED_TYPES = ["Detached", "Semi detached", "Mid terrace", "End terrace", "Bungalow", "Flat"]


def epc_need_type(ptype, form) -> str | None:
    p = str(ptype).lower()
    f = str(form).lower()
    if p == "bungalow":
        return "Bungalow"
    if p in ("flat", "maisonette"):
        return "Flat"
    if p == "house":
        if "semi" in f:
            return "Semi detached"
        if "mid" in f:
            return "Mid terrace"
        if "end" in f:
            return "End terrace"
        if "detached" in f:
            return "Detached"
    return None


def epc_need_age(band) -> float:
    """NEED PROP_AGE_BAND: 1 <1930, 2 1930-1972, 3 1973-1999, 4 2000+. Straddling EPC bands go to the majority."""
    if not isinstance(band, str):
        return np.nan
    b = band.lower()
    if "before" in b:
        return 1
    yrs = [int(x) for x in re.findall(r"\d{4}", b)]
    if not yrs:
        return np.nan
    start = yrs[0]
    if start < 1930:
        return 1
    if start < 1973 and start != 1967:  # 1930-1949, 1950-1966
        return 2
    if start == 1967:                    # 1967-1975: 6 of 9 years before 1973
        return 2
    if start < 2000 and start != 1996:   # 1976-1982 ... 1991-1995
        return 3
    if start == 1996:                    # 1996-2002: 4 of 7 years before 2000
        return 3
    return 4


def area_band(tfa) -> float:
    """NEED FLOOR_AREA_BAND: 1 <=50, 2 51-100, 3 101-150, 4 151-200, 5 >200 (m2)."""
    return pd.cut(tfa, [0, 50, 100, 150, 200, np.inf], labels=[1, 2, 3, 4, 5]).astype(float)


def epc_cells(feat: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    f = feat.assign(
        cell_type=[epc_need_type(p, b) for p, b in zip(feat["property_type"], feat["built_form"])],
        cell_age=feat["age_band"].map(epc_need_age),
        cell_area=area_band(feat["tfa"]),
    ).dropna(subset=["cell_type", "cell_age", "cell_area"])
    g = f.groupby(["cell_type", "cell_age", "cell_area"])
    out = g[cols].mean()
    out["n_epc"] = g.size()
    return out


def need_cells(nd: pd.DataFrame) -> pd.DataFrame:
    g = nd.dropna(subset=["intensity_kwh_m2"]).groupby(["property_type", "age_band", "area_band"])
    out = pd.DataFrame({
        "metered_kwh_m2_median": g["intensity_kwh_m2"].median(),
        "metered_kwh_m2_mean": g["intensity_kwh_m2"].mean(),
        "n_need": g.size(),
    })
    out.index = out.index.set_names(["cell_type", "cell_age", "cell_area"])
    return out
