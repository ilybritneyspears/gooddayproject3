"""
Where GoodDay gets its expected weather for an event.

    ╔══════════════════════════════════════════════════════════════════════╗
    ║  MODEL SWAP POINT: get_expected_conditions()                         ║
    ║                                                                      ║
    ║  Right now this returns historical climatology (typical conditions   ║
    ║  for that place and time of year). When the predictive model is      ║
    ║  ready, replace the BODY of get_expected_conditions() so it returns  ║
    ║  the model's predictions. Keep the signature and the                 ║
    ║  ExpectedConditions return type the same and nothing else in the     ║
    ║  app needs to change.                                                ║
    ╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from goodday import climatology


@dataclass
class ExpectedConditions:
    """Everything the scoring and UI layers need about one day.

    Required fields drive the match scores. Optional fields are shown as
    supporting detail only if they are not None.
    """
    # --- used for scoring -------------------------------------------------
    sunshine_hours: float     # expected hours of sunshine in the day
    rain_chance: float        # 0-1, probability of a wet day (>= 1 mm)
    cloud_cover_pct: float    # 0-100, mean daily cloud cover
    wind_max_kmh: float       # typical daily maximum wind speed at 10 m

    # --- supporting detail ------------------------------------------------
    temp_mean: float          # °C
    temp_max: float           # °C
    temp_min: float           # °C
    rain_mm: float            # expected daily rainfall, mm

    temp_max_range: Optional[tuple[float, float]] = None  # (10th, 90th percentile) high
    heavy_rain_chance: Optional[float] = None   # 0-1, chance of >= 10 mm
    strong_wind_chance: Optional[float] = None  # 0-1, chance of gusts/max >= 40 km/h
    daylight_hours: Optional[float] = None
    humidity_pct: Optional[float] = None

    # --- provenance (shown in the UI so users know what they're looking at)
    source: str = "Historical climatology"
    source_note: str = ""
    extra: dict = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════
#  MODEL SWAP POINT — replace the body of this function with the real model
# ═════════════════════════════════════════════════════════════════════════
def get_expected_conditions(location: str, event_date: date) -> ExpectedConditions:
    """Return the expected weather for `location` on `event_date`.

    PLACEHOLDER: typical conditions from the historical dataset, pooled across
    all years within +/- 7 days of the same calendar day.

    How to swap in the real model (it predicts temperature, rainfall and wind):
    keep climatology as the baseline for sunshine/cloud, which the model does
    not predict, and overwrite the fields the model does predict, e.g.

        base = _climatology_conditions(location, event_date)
        pred = model.predict(location, event_date)   # your teammate's code
        base.temp_mean = pred["temp_mean"]
        base.temp_max = pred["temp_max"]
        base.temp_min = pred["temp_min"]
        base.rain_mm = pred["rain_mm"]
        base.rain_chance = pred.get("rain_chance", float(pred["rain_mm"] >= 1.0))
        base.wind_max_kmh = pred["wind_max_kmh"]
        base.source = "GoodDay predictive model"
        base.source_note = "Sunshine and cloud still from climatology."
        return base
    """
    return _climatology_conditions(location, event_date)
# ═════════════════════════════════════════════════════════════════════════


def _climatology_conditions(location: str, event_date: date) -> ExpectedConditions:
    t = climatology.typical_day(location, event_date)
    return ExpectedConditions(
        sunshine_hours=float(t.sunshine_hours),
        rain_chance=float(t.rain_chance),
        cloud_cover_pct=float(t.cloud_cover_pct),
        wind_max_kmh=float(t.wind_max_kmh),
        temp_mean=float(t.temp_mean),
        temp_max=float(t.temp_max),
        temp_min=float(t.temp_min),
        rain_mm=float(t.rain_mm),
        temp_max_range=(float(t.temp_max_p10), float(t.temp_max_p90)),
        heavy_rain_chance=float(t.heavy_rain_chance),
        strong_wind_chance=float(t.strong_wind_chance),
        daylight_hours=float(t.daylight_hours),
        humidity_pct=float(t.humidity_pct),
        source="Historical climatology",
        source_note=(
            f"Typical conditions for {event_date:%d %B} ±{climatology.WINDOW_DAYS} days, "
            f"based on {int(t.n_days):,} historical days ({int(t.first_year)}–{int(t.last_year)}). "
            "This is what the weather is usually like, not a forecast."
        ),
    )
