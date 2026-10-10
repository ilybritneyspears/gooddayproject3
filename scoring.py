"""
Suitability scoring — a direct port of the notebook (Group5Project3.ipynb,
section 10): calculate_suitability(), suitability_rating(), preference_summary()
and assess_event().

Each of the five conditions has a probability (0–1). The user marks each one
Not important (0) / Preferred (1) / Essential (2). The score is the
weighted average of the probabilities x 100, and the bands are:

    Strong Match 75+  ·  Good Match 50+  ·  Weak Match 25+  ·  Poor Match < 25

If every condition is "Not important" the score is None.
"""

from __future__ import annotations

import pandas as pd

from goodday.forecast import predict_event_weather

# ───────────────────────────────────────────── notebook constants (cell 101)
importance_weights = {
    "Not important": 0,
    "Preferred": 1,
    "Essential": 2,
}
IMPORTANCE_LEVELS = list(importance_weights)

# Preference keys, in the notebook's order, with display metadata for the app
CONDITIONS = {
    "Comfortable": {"label": "Comfortable temperature", "icon": "🌡️",
                    "definition": "Max temperature between 16 and 28 °C"},
    "Dry": {"label": "Dry", "icon": "🌂",
            "definition": "1 mm of rain or less"},
    "Calm": {"label": "Calm", "icon": "🍃",
             "definition": "Max wind speed of 30 km/h or less"},
    "Sunny": {"label": "Sunny", "icon": "☀️",
              "definition": "Sun for at least 70% of daylight hours"},
    "Clear": {"label": "Clear skies", "icon": "🌤️",
              "definition": "Average cloud cover of 40% or less"},
}

# (minimum score, rating, colour) — thresholds from suitability_rating()
RATING_BANDS = [
    (75, "Strong Match", "#1e8e3e"),
    (50, "Good Match", "#7cb342"),
    (25, "Weak Match", "#ef6c00"),
    (0, "Poor Match", "#c62828"),
]
NO_PREFERENCES = "No preferences selected"

# Traffic light shown in the app (app-only; the notebook has no traffic light).
# Strong = green, Good = amber, Weak/Poor = red. Change here if the team prefers
# Good Match to show green.
TRAFFIC_LIGHT = {
    "Strong Match": ("green", "Great fit for your event"),
    "Good Match": ("amber", "Decent odds — keep an eye on the forecast"),
    "Weak Match": ("red", "Risky — have a backup plan"),
    "Poor Match": ("red", "Not ideal — consider another date"),
}


# ───────────────────────────────────────────────── notebook functions (1:1)
def calculate_suitability(weather: dict, preferences: dict) -> float | None:
    """Notebook cell 101."""
    probabilities = {
        "Comfortable": weather["Comfortable Probability"],
        "Dry": weather["Dry Probability"],
        "Calm": weather["Calm Probability"],
        "Sunny": weather["Sunny Probability"],
        "Clear": weather["Clear Probability"],
    }

    weighted_total = 0
    total_weight = 0

    for condition, probability in probabilities.items():
        weight = importance_weights[preferences[condition]]
        weighted_total += probability * weight
        total_weight += weight

    # Avoid division by zero if everything is Not important
    if total_weight == 0:
        return None

    return (weighted_total / total_weight) * 100


def suitability_rating(score: float | None) -> str:
    """Notebook cell 103."""
    if score is None:
        return NO_PREFERENCES
    if score >= 75:
        return "Strong Match"
    elif score >= 50:
        return "Good Match"
    elif score >= 25:
        return "Weak Match"
    else:
        return "Poor Match"


def preference_summary(weather: dict, preferences: dict) -> pd.DataFrame:
    """Notebook cell 105."""
    probabilities = {
        "Comfortable Temperature": weather["Comfortable Probability"],
        "Dry": weather["Dry Probability"],
        "Calm": weather["Calm Probability"],
        "Sunny": weather["Sunny Probability"],
        "Clear": weather["Clear Probability"],
    }
    preference_keys = {
        "Comfortable Temperature": "Comfortable",
        "Dry": "Dry",
        "Calm": "Calm",
        "Sunny": "Sunny",
        "Clear": "Clear",
    }

    summary = []
    for condition, probability in probabilities.items():
        importance = preferences[preference_keys[condition]]
        if importance != "Not important":
            summary.append({
                "Condition": condition,
                "Importance": importance,
                "Estimated Likelihood (%)": probability * 100,
            })

    return pd.DataFrame(summary)


def assess_event(location, event_date, preferences: dict) -> dict:
    """Notebook cell 107."""
    weather = predict_event_weather(location, event_date)
    score = calculate_suitability(weather, preferences)
    match = suitability_rating(score)
    summary = preference_summary(weather, preferences)
    return {
        "Location": weather["Location"],
        "Date": weather["Date"],
        "Maximum Temperature": weather["Maximum Temperature"],
        "Suitability Score": score,
        "Preference Match": match,
        "Preference Summary": summary,
    }


# ───────────────────────────────────────────────────────── display helpers
def rating_colour(rating: str) -> str:
    for _, name, colour in RATING_BANDS:
        if name == rating:
            return colour
    return "#9e9e9e"


def probability_colour(probability: float) -> str:
    """Colour a single condition's likelihood using the same 75/50/25 bands."""
    return rating_colour(suitability_rating(probability * 100))


def traffic_light(rating: str) -> tuple[str, str]:
    return TRAFFIC_LIGHT.get(rating, ("off", "Choose at least one preference"))
