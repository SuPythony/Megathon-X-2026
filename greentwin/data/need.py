"""NEED 2025 anonymised data -> per-property annual intensity table (M04 training data).

Source: DESNZ, National Energy Efficiency Data-Framework (NEED) anonymised data 2025, OGL v3.
https://www.gov.uk/government/statistics/national-energy-efficiency-data-framework-need-anonymised-data-2025

Facts from the NEED 2025 metadata sheet that shape this table:
- Gas (Gcons) is weather corrected by DESNZ; electricity (Econs) is not. The gas target is a typical-year value.
- Readings are rounded to 100 kWh and suppressed outside 1,000-50,000 kWh (gas) / 500-25,000 kWh (elec).
- Only flag "V" readings are real meter readings; E/L/G/O are estimated, suppressed or missing -> dropped, never 0.
- A blank GasValFlag means no gas meter was matched (usually off the gas grid): gas is structurally absent.
- Floor area is banded only, so intensity uses a representative area per band (ASSUMPTION, see FLOOR_BAND_AREA_M2).
"""
from __future__ import annotations

from pathlib import Path

import duckdb

DATA_VERSION = "need_anon_2025"
TARGET_YEARS = (2021, 2022, 2023)

# Representative floor area per NEED FLOOR_AREA_BAND (m2). Band midpoints; the open-ended bands
# (<=50, >200) are assumptions. Intensity error from this is constant within a band.
FLOOR_BAND_AREA_M2 = {1: 40.0, 2: 75.5, 3: 125.5, 4: 175.5, 5: 250.0}
FLOOR_BAND_EDGES_M2 = (50.0, 100.0, 150.0, 200.0)  # upper edges of bands 1-4

REGION_CODES = {
    "E12000001": "North East", "E12000002": "North West", "E12000003": "Yorkshire and The Humber",
    "E12000004": "East Midlands", "E12000005": "West Midlands", "E12000006": "East of England",
    "E12000007": "London", "E12000008": "South East", "E12000009": "South West", "W92000004": "Wales",
}
EPC_ORDINAL = {"A/B": 1, "C": 2, "D": 3, "E": 4, "F/G": 5}  # "No EPC" -> NULL (missing, not a band)

# Stratified split by property (each NEED row is one property), assigned once. Inside each stratum rows are
# ordered by a seeded hash and cut at these cumulative fractions: train 60 / tune 10 / cal 10 / test 20.
SPLIT_SEED = 42
SPLIT_STRATA = ("PROP_TYPE", "PROP_AGE_BAND", "FLOOR_AREA_BAND", "REGION", "MAIN_HEAT_FUEL")
SPLIT_CUTS = (("train", 0.60), ("tune", 0.70), ("cal", 0.80), ("test", 1.00))


def _valid_mean(prefix: str, flag_prefix: str) -> str:
    """SQL for the mean of valid (flag 'V') readings over TARGET_YEARS; NULL if none are valid."""
    num = " + ".join(f"CASE WHEN {flag_prefix}{y} = 'V' THEN {prefix}{y} ELSE 0 END" for y in TARGET_YEARS)
    den = " + ".join(f"CASE WHEN {flag_prefix}{y} = 'V' THEN 1 ELSE 0 END" for y in TARGET_YEARS)
    return f"CASE WHEN ({den}) > 0 THEN ({num}) / ({den}) END"


def build_intensity_table(csv_path: str | Path, out_path: str | Path) -> int:
    """Read the NEED CSV (only needed columns) and write one row per property to Parquet. Returns row count."""
    csv_path, out_path = Path(csv_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    area = "CASE FLOOR_AREA_BAND " + " ".join(f"WHEN {k} THEN {v}" for k, v in FLOOR_BAND_AREA_M2.items()) + " END"
    region = "CASE REGION " + " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in REGION_CODES.items()) + " END"
    epc = "CASE EPC " + " ".join(f"WHEN '{k}' THEN {v}" for k, v in EPC_ORDINAL.items()) + " END"
    # percent_rank within the stratum (0..1) on a seeded hash -> exact stratified fractions, deterministic.
    rank = (f"percent_rank() OVER (PARTITION BY {', '.join(SPLIT_STRATA)} "
            f"ORDER BY hash(need_row, {SPLIT_SEED}), need_row)")
    split = "CASE " + " ".join(f"WHEN {rank} < {c} THEN '{n}'" for n, c in SPLIT_CUTS[:-1]) + " ELSE 'test' END"
    # Single valid years too: needed to estimate year-to-year (behaviour) noise for the Bayesian update.
    per_year = ",\n        ".join(
        f"(CASE WHEN {flag}{y} = 'V' THEN {col}{y} END)::DOUBLE AS {name}_{y}"
        for col, flag, name in (("Gcons", "GasValFlag", "gas"), ("Econs", "ElecValFlag", "elec")) for y in TARGET_YEARS)
    no_gas_meter = " AND ".join(f"GasValFlag{y} IS NULL" for y in TARGET_YEARS)  # blank flag = no gas meter
    sql = f"""
    COPY (
      WITH raw AS (
        SELECT row_number() OVER () AS need_row, *
        FROM read_csv('{csv_path}', header = true, all_varchar = false, sample_size = -1)
      )
      SELECT
        'need_' || need_row::VARCHAR                      AS property_id,
        PROP_TYPE                                         AS prop_type,
        PROP_AGE_BAND::INT                                AS age_band,
        FLOOR_AREA_BAND::INT                              AS floor_band,
        ({area})::DOUBLE                                  AS floor_area_rep_m2,
        {region}                                          AS region,
        {epc}                                             AS epc_band,
        COALESCE(IMD_BAND_ENG, IMD_BAND_WALES)::DOUBLE    AS imd_quintile,
        CONSERVATORY_FLAG::DOUBLE                         AS conservatory,
        COALESCE(LI_FLAG, 0)::INT                         AS li_flag,
        COALESCE(CWI_FLAG, 0)::INT                        AS cwi_flag,
        COALESCE(PV_FLAG, 0)::INT                         AS pv_flag,
        CASE MAIN_HEAT_FUEL WHEN 1 THEN 'gas' ELSE 'not_gas' END AS main_heat_fuel,
        ({no_gas_meter})                                  AS no_gas_meter,
        ({_valid_mean("Gcons", "GasValFlag")})::DOUBLE    AS gas_kwh,
        ({_valid_mean("Econs", "ElecValFlag")})::DOUBLE   AS elec_kwh,
        {per_year},
        {split}                                           AS split
      FROM raw
    ) TO '{out_path}' (FORMAT parquet)
    """
    con = duckdb.connect()
    con.execute("SET memory_limit = '3GB'; SET preserve_insertion_order = true;")
    final, out_path = out_path, out_path.with_suffix(".tmp.parquet")  # write then rename: readers never see half a file
    con.execute(sql.replace(f"TO '{final}'", f"TO '{out_path}'"))
    out_path.replace(final)
    n = con.execute(f"SELECT count(*) FROM read_parquet('{final}')").fetchone()[0]
    return int(n)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default="data/raw/need/anon2025_4million.csv")
    ap.add_argument("--out", default="data/processed/need_intensity.parquet")
    a = ap.parse_args()
    print(f"wrote {build_intensity_table(a.csv, a.out):,} rows -> {a.out}")
