"""
Turns expected conditions + the user's 1-5 preferences into match scores.

Method (kept deliberately simple and explainable):

1. Each weather type gets an *intensity* from 0 to 1 — how much of it to expect —
   by rescaling between two anchors (see INTENSITY_SCALES). The anchors bracket
   the range of typical conditions across Auckland, Wellington, Christchurch and
   Dunedin in the dataset, so the scores actually separate places and seasons.
     sunshine  : 5 h -> 0   ...  13 h -> 1      (sunshine hours)
     rain      : 15% -> 0   ...  60% -> 1       (chance of a wet day)
     cloud     : 40% -> 0   ...  80% -> 1       (mean cloud cover)
     wind      : 12 -> 0    ...  36 km/h -> 1   (daily max wind)
   Values beyond the anchors are clipped to 0 or 1.

2. The preference becomes a *desired* intensity: 1 -> 0.0, 3 -> 0.5, 5 -> 1.0.

3. Match = 100 x (1 - |intensity - desired|).

4. Overall = weighted average of the four matches. Strong opinions (1 or 5)
   carry weight 3, mild opinions (2 or 4) weight 2, neutral (3) weight 1.

This layer only reads ExpectedConditions, so it works unchanged whether the
numbers come from climatology or the predictive model.
"""

from __future__ import annotations

from dataclasses import dataclass

from goodday.forecast import ExpectedConditions

WEATHER_TYPES = {
    "sunshine": {"label": "Sunshine", "icon": "☀️"},
    "rain": {"label": "Rain", "icon": "🌧️"},
    "cloud": {"label": "Cloud / overcast", "icon": "☁️"},
    "wind": {"label": "Wind", "icon": "💨"},
}

# (ExpectedConditions field, value meaning "none of it", value meaning "lots of it")
INTENSITY_SCALES = {
    "sunshine": ("sunshine_hours", 5.0, 13.0),
    "rain": ("rain_chance", 0.15, 0.60),
    "cloud": ("cloud_cover_pct", 40.0, 80.0),
    "wind": ("wind_max_kmh", 12.0, 36.0),
}

# (minimum score, label, colour) — checked top to bottom
MATCH_BANDS = [
    (80, "Strong match", "#1e8e3e"),
    (65, "Good match", "#7cb342"),
    (50, "Average", "#f9a825"),
    (35, "Weak match", "#ef6c00"),
    (0, "Poor match", "#c62828"),
]

# Traffic light for the overall suitability
TRAFFIC_LIGHT = [
    (65, "green", "Good day for it", "#1e8e3e"),
    (50, "amber", "Could go either way", "#f9a825"),
    (0, "red", "Not ideal — consider another date", "#c62828"),
]


@dataclass
class MatchResult:
    key: str
    preference: int
    intensity: float
    desired: float
    score: float
    weight: int

    @property
    def band(self) -> tuple[str, str]:
        return match_band(self.score)


def intensities(c: ExpectedConditions) -> dict[str, float]:
    out = {}
    for key, (attr, low, high) in INTENSITY_SCALES.items():
        out[key] = _clip((getattr(c, attr) - low) / (high - low))
    return out


def score_matches(c: ExpectedConditions, prefs: dict[str, int]) -> dict[str, MatchResult]:
    out = {}
    for key, intensity in intensities(c).items():
        p = int(prefs[key])
        desired = (p - 1) / 4
        out[key] = MatchResult(
            key=key,
            preference=p,
            intensity=intensity,
            desired=desired,
            score=100 * (1 - abs(intensity - desired)),
            weight=1 + abs(p - 3),
        )
    return out


def overall_score(matches: dict[str, MatchResult]) -> float:
    total_w = sum(m.weight for m in matches.values())
    return sum(m.score * m.weight for m in matches.values()) / total_w


def match_band(score: float) -> tuple[str, str]:
    for threshold, label, colour in MATCH_BANDS:
        if score >= threshold:
            return label, colour
    return MATCH_BANDS[-1][1], MATCH_BANDS[-1][2]


def traffic_light(score: float) -> tuple[str, str, str]:
    for threshold, light, message, colour in TRAFFIC_LIGHT:
        if score >= threshold:
            return light, message, colour
    return TRAFFIC_LIGHT[-1][1:]


def _clip(x: float) -> float:
    return max(0.0, min(1.0, float(x)))
