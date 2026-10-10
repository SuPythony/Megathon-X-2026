"""Sales data preparation for the property-value model (Part 4).

UCL linked sales (V-01) -> clean, market-adjusted, geo-coded, split training table:
1. Market sales only (PPD category A), sane prices, known property type.
2. V-05: keep a sale only if its linked EPC was lodged BEFORE the transfer and within ``max_epc_age_years``
   (the linked EPC is not necessarily the rating at sale).
3. Join NSPL by postcode -> LSOA 2021, MSOA, LA, region, IMD rank, urban/rural; then reduce postcode to its
   outward code (V-06).
4. V-07: adjust price to the reference month with UK HPI by LA x property type (fallback: region, then
   England & Wales), so sales from different years are comparable ("market conditions").
5. Splits by property (address hash) 70/15/15, plus an out-of-time flag for the latest year.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

PPD_TYPE = {"D": "detached", "S": "semi", "T": "terraced", "F": "flat"}
HPI_TYPE_COL = {"detached": "DetachedIndex", "semi": "SemiDetachedIndex", "terraced": "TerracedIndex",
                "flat": "FlatIndex"}
RGN_NAME = {  # ONS region codes (NSPL rgn) -> UK HPI RegionName
    "E12000001": "North East", "E12000002": "North West", "E12000003": "Yorkshire and The Humber",
    "E12000004": "East Midlands", "E12000005": "West Midlands", "E12000006": "East of England",
    "E12000007": "London", "E12000008": "South East", "E12000009": "South West", "W99999999": "Wales",
}


@dataclass
class PrepConfig:
    max_epc_age_years: float = 10.0     # V-05: EPCs are valid for 10 years
    min_price: float = 10_000
    max_price: float = 10_000_000
    ref_month: str = "2026-07"          # latest UK HPI month on hand
    oot_year: int = 2024                # out-of-time test year (latest year in the UCL data)
    seed: int = 42


def split_of(keys: pd.Series, seed: int = 42) -> pd.Series:
    def f(k: str) -> str:
        h = int(hashlib.md5(f"{seed}:{k}".encode()).hexdigest()[:8], 16) % 100
        return "train" if h < 70 else "val" if h < 85 else "test"
    return keys.astype(str).map(f)


def load_hpi(path: str) -> pd.DataFrame:
    h = pd.read_csv(path, usecols=["Date", "RegionName", "AreaCode", "Index", *HPI_TYPE_COL.values()])
    h["month"] = pd.to_datetime(h["Date"], dayfirst=True).dt.strftime("%Y-%m")
    return h


def hpi_factor(sales: pd.DataFrame, hpi: pd.DataFrame, ref_month: str) -> tuple[pd.Series, pd.Series]:
    """Index ratio ref/sale by LA x type; fallback region x type, then England and Wales all-type."""
    long = hpi.melt(id_vars=["AreaCode", "RegionName", "month"], value_vars=list(HPI_TYPE_COL.values()) + ["Index"],
                    var_name="col", value_name="idx").dropna(subset=["idx"])
    by_code = long.set_index(["AreaCode", "col", "month"])["idx"]
    by_name = long.set_index(["RegionName", "col", "month"])["idx"]
    col = sales["ptype"].map(HPI_TYPE_COL)
    m = sales["date"].dt.strftime("%Y-%m")

    def ratio(keys_sale, keys_ref, idx):
        a = idx.reindex(pd.MultiIndex.from_arrays(keys_sale)).to_numpy()
        b = idx.reindex(pd.MultiIndex.from_arrays(keys_ref)).to_numpy()
        return b / a

    ref = pd.Series(ref_month, index=sales.index)
    r_la = ratio([sales["laua"], col, m], [sales["laua"], col, ref], by_code)
    rname = sales["rgn"].map(RGN_NAME)
    r_rg = ratio([rname, col, m], [rname, col, ref], by_name)
    ew = pd.Series("England and Wales", index=sales.index)
    allc = pd.Series("Index", index=sales.index)
    r_ew = ratio([ew, allc, m], [ew, allc, ref], by_name)
    factor = pd.Series(np.where(np.isfinite(r_la), r_la, np.where(np.isfinite(r_rg), r_rg, r_ew)), index=sales.index)
    level = pd.Series(np.where(np.isfinite(r_la), "la_type", np.where(np.isfinite(r_rg), "region_type", "ew_all")),
                      index=sales.index)
    return factor, level


def prepare(sales: pd.DataFrame, nspl: pd.DataFrame, hpi: pd.DataFrame, cfg: PrepConfig = PrepConfig()
            ) -> tuple[pd.DataFrame, dict]:
    log: dict = {"rows_in": len(sales)}
    s = sales.copy()
    s["price"] = pd.to_numeric(s["price"], errors="coerce")
    s["date"] = pd.to_datetime(s["dateoftransfer"], errors="coerce")
    s["epc_date"] = pd.to_datetime(s["LODGEMENT_DATE"], errors="coerce")
    s["ptype"] = s["propertytype"].map(PPD_TYPE)

    def step(name, mask):
        nonlocal s
        log[name] = int((~mask).sum())
        s = s[mask]

    step("drop_not_category_A", s["categorytype"].eq("A"))
    step("drop_price_out_of_range", s["price"].between(cfg.min_price, cfg.max_price))
    step("drop_type_other", s["ptype"].notna())
    gap = (s["date"] - s["epc_date"]).dt.days / 365.25
    log["epc_gap_years_quantiles_before_filter"] = gap.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(2).tolist()
    log["share_epc_after_sale"] = round(float((gap < 0).mean()), 4)
    step("drop_epc_after_sale_or_too_old", gap.between(0, cfg.max_epc_age_years))   # V-05
    s["epc_age_years"] = (s["date"] - s["epc_date"]).dt.days / 365.25

    g = nspl.rename(columns={"pcds": "postcode"})
    s = s.merge(g[["postcode", "laua", "rgn", "ctry", "lsoa21", "msoa21", "lat", "long", "imd", "ru11ind"]],
                on="postcode", how="left")
    step("drop_postcode_not_in_nspl", s["lsoa21"].notna())
    s["postcode_district"] = s["postcode"].str.split(" ").str[0]
    s = s.drop(columns=["postcode"])                                                   # V-06
    s["lat"] = pd.to_numeric(s["lat"], errors="coerce").round(3)
    s["long"] = pd.to_numeric(s["long"], errors="coerce").round(3)

    factor, level = hpi_factor(s, hpi, cfg.ref_month)
    s["hpi_factor"], s["hpi_level"] = factor, level
    step("drop_no_hpi", s["hpi_factor"].notna() & np.isfinite(s["hpi_factor"]))
    s["price_adj"] = s["price"] * s["hpi_factor"]
    s["log_price_adj"] = np.log(s["price_adj"])
    log["hpi_level_counts"] = s["hpi_level"].value_counts().to_dict()

    s["sap"] = pd.to_numeric(s["CURRENT_ENERGY_EFFICIENCY"], errors="coerce")
    s["tfa"] = pd.to_numeric(s["TOTAL_FLOOR_AREA"], errors="coerce")
    s["rooms"] = pd.to_numeric(s["NUMBER_HABITABLE_ROOMS"], errors="coerce")
    step("drop_missing_sap_or_area", s["sap"].between(1, 120) & s["tfa"].between(15, 1000))
    s["log_tfa"] = np.log(s["tfa"])
    s["new_build"] = s["oldnew"].eq("Y")
    s["leasehold"] = s["duration"].eq("L")
    s["age_band"] = s["CONSTRUCTION_AGE_BAND"].str.replace("England and Wales: ", "", regex=False)
    s["year"] = s["date"].dt.year
    s["split"] = split_of(s["prop_key"], cfg.seed)
    s["oot"] = s["year"] >= cfg.oot_year
    log["rows_out"] = len(s)
    log["split_counts"] = s["split"].value_counts().to_dict()
    log["oot_rows"] = int(s["oot"].sum())
    return s.reset_index(drop=True), log
