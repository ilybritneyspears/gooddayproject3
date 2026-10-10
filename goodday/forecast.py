"""
Where GoodDay gets its expected weather for an event.

This is the app's port of the notebook's `predict_event_weather()`
(Group5Project3.ipynb, section 10). It loads the exported artifacts in
artifacts/ (built by build_artifacts.py) and reproduces the notebook exactly:

  * Maximum temperature  -> final_temperature_model (GradientBoostingRegressor)
                            on [year, day_sin, day_cos, location dummies],
                            day_sin/cos = sin/cos(2*pi*day_of_year/365.25)
  * Comfortable          -> share of (prediction + validation residuals) in 16–28 °C
  * Dry / Calm / Sunny / Clear -> ±30-day seasonal climatology lookup
                            (final_climatology, by location and day_of_year 1–366)

The app calls get_expected_conditions(), which wraps predict_event_weather() and
adds supporting detail (typical high/low, rainfall etc.) from climatology.py.
To retrain, run build_artifacts.py — nothing in here needs to change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from goodday import climatology

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
COMFORTABLE_MIN_C = 16
COMFORTABLE_MAX_C = 28


# ───────────────────────────────────────────────────────────── artifacts
@dataclass(frozen=True)
class ModelArtifacts:
    temperature_model: object           # fitted GradientBoostingRegressor
    feature_columns: list[str]          # training column order
    temperature_residuals: np.ndarray   # validation residuals (actual - predicted)
    final_climatology: pd.DataFrame     # location, day_of_year, dry, calm, sunny, clear
    metadata: dict


@lru_cache(maxsize=1)
def load_artifacts() -> ModelArtifacts:
    meta_path = ARTIFACT_DIR / "metadata.json"
    return ModelArtifacts(
        temperature_model=joblib.load(ARTIFACT_DIR / "temperature_model.joblib"),
        feature_columns=json.loads((ARTIFACT_DIR / "feature_columns.json").read_text()),
        temperature_residuals=np.load(ARTIFACT_DIR / "temperature_residuals.npy"),
        # round_trip parsing keeps every float bit-identical to the notebook's table;
        # the default parser can be off in the last bit, which can flip a score sitting
        # exactly on a band boundary (e.g. 75.0 -> 74.999…, Strong -> Good).
        final_climatology=pd.read_csv(ARTIFACT_DIR / "final_climatology.csv",
                                      float_precision="round_trip"),
        metadata=json.loads(meta_path.read_text()) if meta_path.exists() else {},
    )


def model_locations() -> list[str]:
    return sorted(load_artifacts().final_climatology["location"].unique())


# ─────────────────────────────────────────── notebook functions (ported 1:1)
def comfortable_temperature_probability(predicted_temperature: float) -> float:
    """Notebook cell 98."""
    possible_temperatures = predicted_temperature + load_artifacts().temperature_residuals
    comfortable = (
        (possible_temperatures >= COMFORTABLE_MIN_C) &
        (possible_temperatures <= COMFORTABLE_MAX_C)
    )
    return comfortable.mean()


def predict_event_weather(location: str, event_date) -> dict:
    """Notebook cell 99 — same inputs, same dict keys, same numbers."""
    a = load_artifacts()
    if location not in model_locations():
        # The notebook silently treats an unknown location as Auckland (all dummies
        # False). Fail loudly instead so a typo can't produce Auckland numbers.
        raise ValueError(f"Unknown location {location!r}; expected one of {model_locations()}")

    event_date = pd.Timestamp(event_date)

    year = event_date.year
    day_of_year = event_date.dayofyear

    # Seasonal features
    day_sin = np.sin(2 * np.pi * day_of_year / 365.25)
    day_cos = np.cos(2 * np.pi * day_of_year / 365.25)

    # Temperature model input
    temperature_input = pd.DataFrame({
        "year": [year],
        "day_sin": [day_sin],
        "day_cos": [day_cos],
        "location_Christchurch": [location == "Christchurch"],
        "location_Dunedin": [location == "Dunedin"],
        "location_Wellington": [location == "Wellington"],
    })
    temperature_input = temperature_input.reindex(columns=a.feature_columns, fill_value=0)

    # Predict daily maximum temperature
    predicted_temperature = a.temperature_model.predict(temperature_input)[0]

    # Probability that maximum temperature is within 16–28°C
    comfortable_probability = comfortable_temperature_probability(predicted_temperature)

    # Seasonal climatology probabilities
    clim = a.final_climatology
    climate_row = clim[
        (clim["location"] == location) &
        (clim["day_of_year"] == day_of_year)
    ].iloc[0]

    return {
        "Location": location,
        "Date": event_date.date(),
        "Maximum Temperature": predicted_temperature,
        "Comfortable Probability": comfortable_probability,
        "Dry Probability": climate_row["dry"],
        "Calm Probability": climate_row["calm"],
        "Sunny Probability": climate_row["sunny"],
        "Clear Probability": climate_row["clear"],
    }


# ─────────────────────────────────────────────────────── app-facing wrapper
@dataclass
class ExpectedConditions:
    """What the app needs about one day.

    `weather` is exactly the notebook's predict_event_weather() dict and is what
    the scoring functions consume. The other fields are display-only.
    """
    weather: dict

    # Spread of plausible maximum temperatures (prediction + residuals)
    max_temp_p10: float = float("nan")
    max_temp_p90: float = float("nan")

    # Supporting detail from the ±7-day historical averages (climatology.py)
    typical_high: Optional[float] = None
    typical_low: Optional[float] = None
    typical_mean: Optional[float] = None
    rain_mm: Optional[float] = None
    wind_max_kmh: Optional[float] = None
    cloud_cover_pct: Optional[float] = None
    sunshine_hours: Optional[float] = None
    daylight_hours: Optional[float] = None
    humidity_pct: Optional[float] = None
    history_note: str = ""

    source: str = "GoodDay model"
    extra: dict = field(default_factory=dict)

    @property
    def max_temperature(self) -> float:
        return float(self.weather["Maximum Temperature"])

    def probability(self, condition: str) -> float:
        """condition in Comfortable / Dry / Calm / Sunny / Clear."""
        return float(self.weather[f"{condition} Probability"])


def get_expected_conditions(location: str, event_date: date) -> ExpectedConditions:
    """Model prediction for one location/date, plus historical supporting detail."""
    weather = predict_event_weather(location, event_date)
    spread = weather["Maximum Temperature"] + load_artifacts().temperature_residuals
    t = climatology.typical_day(location, event_date)
    return ExpectedConditions(
        weather=weather,
        max_temp_p10=float(np.percentile(spread, 10)),
        max_temp_p90=float(np.percentile(spread, 90)),
        typical_high=float(t.temp_max),
        typical_low=float(t.temp_min),
        typical_mean=float(t.temp_mean),
        rain_mm=float(t.rain_mm),
        wind_max_kmh=float(t.wind_max_kmh),
        cloud_cover_pct=float(t.cloud_cover_pct),
        sunshine_hours=float(t.sunshine_hours),
        daylight_hours=float(t.daylight_hours),
        humidity_pct=float(t.humidity_pct),
        history_note=(
            f"Historical averages for {event_date:%d %B} ±{climatology.WINDOW_DAYS} days, "
            f"{int(t.first_year)}–{int(t.last_year)} ({int(t.n_days):,} days)."
        ),
    )
