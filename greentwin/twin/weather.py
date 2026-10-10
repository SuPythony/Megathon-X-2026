"""Weather input for the physics: ``WeatherMonth`` records -> the arrays ``run_physics`` reads (``WEATHER_KEYS``).

Real weather comes from the data layer (``DataLayer.load_weather(area_code, months)``, Part 0) keyed by the home's
postcode district. Until that's wired, ``typical_year`` gives a SYNTHETIC typical UK year, labelled
``source="synthetic"``: fine for running the pipeline, not for real numbers.
"""

from __future__ import annotations

import calendar
from typing import Optional, Sequence

import numpy as np

from contracts.models import WeatherMonth

# Synthetic typical year for lowland England: rounded monthly mean temperature (deg C) and global horizontal
# irradiation (kWh/m2), in the range of Met Office 1991-2020 climate averages and PVGIS. Placeholder only.
_TYPICAL_TEMP_C = (4.5, 4.8, 6.8, 9.0, 12.2, 15.1, 17.2, 17.0, 14.5, 11.1, 7.4, 5.0)
_TYPICAL_GHI = (20.0, 35.0, 70.0, 110.0, 145.0, 150.0, 150.0, 125.0, 85.0, 50.0, 25.0, 15.0)
# Daily temperature swings mean real HDDs exceed (base - mean) x days; this uplift is a rough allowance.
_HDD_SPREAD_UPLIFT_K = 1.0


def _days(month: str) -> int:
    y, m = (int(p) for p in month.split("-"))
    return calendar.monthrange(y, m)[1]


def typical_year(area_code: str, year: int = 2025) -> list[WeatherMonth]:
    """SYNTHETIC typical year (see module docstring). HDD15.5 derived from the mean temperature."""
    out = []
    for m in range(1, 13):
        month = f"{year}-{m:02d}"
        t = _TYPICAL_TEMP_C[m - 1]
        hdd = max(0.0, 15.5 - t + _HDD_SPREAD_UPLIFT_K) * _days(month)
        out.append(WeatherMonth(area_code=area_code, month=month, hdd_15_5=round(hdd, 1), mean_temp_c=t,
                                ghi_kwh_m2=_TYPICAL_GHI[m - 1], source="synthetic"))
    return out


def synthetic_months(area_code: str, months: Sequence[str]) -> list[WeatherMonth]:
    """SYNTHETIC weather for specific months (e.g. the months a home has meter readings for): the typical-year values
    for each calendar month. Stand-in until Part 0's load_weather(area_code, months) returns the real months."""
    by_month = {int(w.month[5:]): w for w in typical_year(area_code)}
    out = []
    for month in months:
        t = by_month[int(month[5:7])]
        hdd = max(0.0, 15.5 - t.mean_temp_c + _HDD_SPREAD_UPLIFT_K) * _days(month)
        out.append(WeatherMonth(area_code=area_code, month=month, hdd_15_5=round(hdd, 1), mean_temp_c=t.mean_temp_c,
                                ghi_kwh_m2=t.ghi_kwh_m2, source="synthetic"))
    return out


def weather_arrays(months: Sequence[WeatherMonth]) -> dict[str, np.ndarray]:
    """``WeatherMonth`` list -> {key: array (n_months,)}. Missing temperature is estimated from HDD; missing
    irradiation falls back to the typical-year value for that calendar month. Both are filled, never zero."""
    if not months:
        raise ValueError("no weather months")
    days = np.array([_days(w.month) for w in months], dtype=np.float32)
    moy = np.array([int(w.month.split("-")[1]) for w in months], dtype=np.float32)
    hdd = np.array([w.hdd_15_5 for w in months], dtype=np.float32)
    temp = np.array([
        w.mean_temp_c if w.mean_temp_c is not None else 15.5 + _HDD_SPREAD_UPLIFT_K - w.hdd_15_5 / _days(w.month)
        for w in months
    ], dtype=np.float32)
    ghi = np.array([
        w.ghi_kwh_m2 if w.ghi_kwh_m2 is not None else _TYPICAL_GHI[int(w.month.split("-")[1]) - 1] for w in months
    ], dtype=np.float32)
    return {"hdd_15_5": hdd, "mean_temp_c": temp, "ghi_kwh_m2": ghi, "days_in_month": days, "month_of_year": moy}


def filled_fields(months: Sequence[WeatherMonth]) -> list[str]:
    """Which months had temperature or irradiation filled in (for data-quality flags)."""
    out: list[str] = []
    for w in months:
        if w.mean_temp_c is None:
            out.append(f"{w.month}:mean_temp_c")
        if w.ghi_kwh_m2 is None:
            out.append(f"{w.month}:ghi_kwh_m2")
    return out


def resolve_weather(area_code: str, months: Optional[Sequence[WeatherMonth]] = None) -> tuple[dict[str, np.ndarray], str]:
    """Arrays plus the weather source label: real months if given, else the synthetic typical year."""
    if months:
        return weather_arrays(months), months[0].source
    return weather_arrays(typical_year(area_code)), "synthetic"
