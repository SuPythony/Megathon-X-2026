"""NEED anonymised records -> metered intensity + coarse score components (real meter readings).

NEED gives real annual gas and electricity per home, but only banded floor area and coarse attributes, so the
components it can see are: energy efficiency (EPC band), intensity (metered), a heating proxy (gas vs not),
renewables (PV installed), and two insulation proxies (loft / cavity insulation installed under a scheme).
Only readings flagged V (valid) are used; missing is never 0, except gas for homes with no gas meter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from contracts.models import HeatingSystem

from . import mapping as m

# Floor-area band midpoints (m2). Band 5 is open-ended (> 200): 250 assumed (decision D-17).
AREA_MID = {1: 40.0, 2: 75.5, 3: 125.5, 4: 175.5, 5: 250.0}
# EPC band -> SAP midpoint; merged bands use the midpoint of the merged range
BAND_SAP = {"A/B": 91.0, "C": 74.5, "D": 61.5, "E": 46.5, "F/G": 20.0}
PV_ASSUMED_KWP = 3.0  # NEED has a PV flag only; 3 kWp is a typical domestic array (D-17)


NEED_COLS = ["PROP_TYPE", "PROP_AGE_BAND", "FLOOR_AREA_BAND", "REGION", "EPC", "LI_FLAG", "CWI_FLAG", "PV_FLAG",
             "MAIN_HEAT_FUEL"]


def load(path: str, year: int = 2023, chunksize: int = 200_000) -> pd.DataFrame:
    """Chunked read of the NEED CSV (or zip) -> features; the 4M file does not fit in memory as raw text."""
    cols = NEED_COLS + [f"Gcons{year}", f"Econs{year}", f"GasValFlag{year}", f"ElecValFlag{year}"]
    parts, offset = [], 0
    for ch in pd.read_csv(path, usecols=cols, chunksize=chunksize, low_memory=False):
        b = build(ch, year)
        b["row_id"] += offset
        offset += len(ch)
        parts.append(b)
    return pd.concat(parts, ignore_index=True)


def build(df: pd.DataFrame, year: int = 2023) -> pd.DataFrame:
    g, e = df[f"Gcons{year}"], df[f"Econs{year}"]
    gf, ef = df[f"GasValFlag{year}"], df[f"ElecValFlag{year}"]
    gas_heated = df["MAIN_HEAT_FUEL"].eq(1)
    gas = np.where(gf.eq("V"), g, np.where(gf.isna() & ~gas_heated, 0.0, np.nan))  # no gas meter -> true 0
    elec = np.where(ef.eq("V"), e, np.nan)
    area = df["FLOOR_AREA_BAND"].map(AREA_MID)

    out = pd.DataFrame(index=df.index)
    out["row_id"] = np.arange(len(df))
    out["property_type"] = df["PROP_TYPE"]
    out["age_band"] = df["PROP_AGE_BAND"]
    out["area_band"] = df["FLOOR_AREA_BAND"]
    out["region"] = df["REGION"]
    out["epc_band"] = df["EPC"]
    out["gas_kwh"] = gas
    out["elec_kwh"] = elec
    out["area_m2"] = area
    out["intensity_kwh_m2"] = (gas + elec) / area
    out["sap_mid"] = df["EPC"].map(BAND_SAP)

    out["s_energy_efficiency"] = m.ee_from_sap(out["sap_mid"])
    out["s_energy_intensity"] = m.intensity_sub(out["intensity_kwh_m2"].fillna(-1)).astype(float)
    out.loc[out["intensity_kwh_m2"].isna(), "s_energy_intensity"] = np.nan
    gas_sub = m.heating_sub(HeatingSystem.gas_boiler_condensing)
    other_sub = m.heating_sub(HeatingSystem.electric_direct)  # 'not gas' is mostly electric (D-17)
    out["s_heating_proxy"] = np.where(gas_heated, gas_sub, other_sub)
    out["s_renewables_proxy"] = np.where(df["PV_FLAG"].eq(1), m.renewables_sub(PV_ASSUMED_KWP), 0.0)
    out["loft_installed"] = df["LI_FLAG"].astype(float)
    out["cavity_installed"] = df["CWI_FLAG"].astype(float)
    return out
