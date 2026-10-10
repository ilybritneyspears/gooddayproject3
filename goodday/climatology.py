"""
Historical climatology built from the team dataset (data/full_weather_data.csv).

For every location and calendar day we pool all years (1940-2026) and take a
+/- WINDOW_DAYS window around that day, so each "typical day" is based on
roughly 85 years x 15 days ≈ 1,300 historical days.

This module only describes the PAST. The app never calls it directly — it goes
through goodday.forecast.get_expected_conditions(), which is the swap point for
the real predictive model.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "full_weather_data.csv"
WINDOW_DAYS = 7          # +/- days around the target day-of-year
WET_DAY_MM = 1.0         # a day with >= 1 mm of rain counts as a "wet day"


def day_of_year(d: date) -> int:
    """Day of year on a fixed 365-day calendar (29 Feb is treated as 28 Feb)."""
    if d.month == 2 and d.day == 29:
        d = date(d.year, 2, 28)
    return date(2001, d.month, d.day).timetuple().tm_yday  # 2001 is not a leap year


@lru_cache(maxsize=1)
def load_daily_data() -> pd.DataFrame:
    """Load the raw CSV and add a local (NZ) calendar date and derived columns."""
    df = pd.read_csv(DATA_PATH)
    # date_UTC is 11:00 UTC every day = local midnight NZDT, i.e. the start of the
    # NZ calendar day. Shifting by +13 h gives the correct local date year-round.
    utc = pd.to_datetime(df["date_UTC"], utc=True)
    local_date = (utc + pd.Timedelta(hours=13)).dt.date
    df["local_date"] = pd.to_datetime(local_date)
    df["doy"] = [day_of_year(d) for d in local_date]
    df["year"] = df["local_date"].dt.year
    df["sunshine_hours"] = df["sunshine_duration"] / 3600
    df["daylight_hours"] = df["daylight_duration"] / 3600
    return df.dropna(subset=["temperature_2m_mean", "rain_sum", "cloud_cover_mean"])


def _circular_window(doy_values: np.ndarray, centre: int) -> np.ndarray:
    dist = np.abs(doy_values - centre)
    dist = np.minimum(dist, 365 - dist)
    return dist <= WINDOW_DAYS


@lru_cache(maxsize=1)
def build_climatology() -> pd.DataFrame:
    """One row per (location, day-of-year) with typical conditions."""
    df = load_daily_data()
    rows = []
    for loc, g in df.groupby("location"):
        doy = g["doy"].to_numpy()
        for centre in range(1, 366):
            w = g[_circular_window(doy, centre)]
            rows.append({
                "location": loc,
                "doy": centre,
                "n_days": len(w),
                "first_year": int(w["year"].min()),
                "last_year": int(w["year"].max()),
                "temp_mean": w["temperature_2m_mean"].mean(),
                "temp_max": w["temperature_2m_max"].mean(),
                "temp_min": w["temperature_2m_min"].mean(),
                "temp_max_p10": w["temperature_2m_max"].quantile(0.10),
                "temp_max_p90": w["temperature_2m_max"].quantile(0.90),
                "rain_mm": w["rain_sum"].mean(),
                "rain_mm_median": w["rain_sum"].median(),
                "rain_chance": (w["rain_sum"] >= WET_DAY_MM).mean(),
                "heavy_rain_chance": (w["rain_sum"] >= 10).mean(),
                "wind_max_kmh": w["wind_speed_10m_max"].mean(),
                "strong_wind_chance": (w["wind_speed_10m_max"] >= 40).mean(),
                "cloud_cover_pct": w["cloud_cover_mean"].mean(),
                "sunshine_hours": w["sunshine_hours"].mean(),
                "daylight_hours": w["daylight_hours"].mean(),
                "humidity_pct": w["relative_humidity_2m_mean"].mean(),
            })
    return pd.DataFrame(rows).set_index(["location", "doy"]).sort_index()


def locations() -> list[str]:
    return sorted(load_daily_data()["location"].unique())


def typical_day(location: str, d: date) -> pd.Series:
    """Typical conditions for a location on the calendar day of `d`."""
    return build_climatology().loc[(location, day_of_year(d))]


def annual_profile(location: str) -> pd.DataFrame:
    """Full 365-day climatology for a location (used for the seasonal chart)."""
    return build_climatology().loc[location].reset_index()
