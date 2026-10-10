"""
End-to-end check that the app reproduces the notebook.

Run:  python tests/test_against_notebook.py        (≈1 minute, retrains in memory)

Three checks:
 1. Saved notebook outputs: the app's numbers for the notebook's own example
    (Auckland, 2027-02-15) match what the notebook printed.
 2. Notebook code vs app code: the notebook's function cells (98, 99, 101, 103,
    105, 107) are executed verbatim against freshly trained in-memory objects,
    and compared with the app modules (which load the exported artifact files)
    for every location across many dates and preference combinations.
 3. Training sanity numbers: row counts and residual stats match notebook output.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import build_artifacts  # noqa: E402
from goodday import forecast, scoring  # noqa: E402

NOTEBOOK = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Group5Project3.ipynb"
NOTEBOOK_FUNCTION_CELLS = [98, 99, 101, 103, 105, 107]
EXAMPLE_PREFS = {"Comfortable": "Essential", "Dry": "Essential", "Calm": "Preferred",
                 "Sunny": "Preferred", "Clear": "Not important"}

failures = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures.append(name)


# ---------------------------------------------------------------- 1. saved outputs
app = scoring.assess_event("Auckland", "2027-02-15", EXAMPLE_PREFS)
w = forecast.predict_event_weather("Auckland", "2027-02-15")
summary = app["Preference Summary"].round(1)
expected_notebook = {  # printed in notebook cells 100, 102, 104, 106, 108
    "Maximum Temperature": (round(w["Maximum Temperature"], 1), 22.9),
    "Comfortable %": (round(w["Comfortable Probability"] * 100, 1), 93.3),
    "Dry %": (round(w["Dry Probability"] * 100, 1), 69.2),
    "Calm %": (round(w["Calm Probability"] * 100, 1), 79.6),
    "Sunny %": (round(w["Sunny Probability"] * 100, 1), 84.9),
    "Suitability Score": (round(app["Suitability Score"], 1), 81.6),
}
for k, (got, want) in expected_notebook.items():
    check(f"notebook output · {k}", got == want, f"app {got} vs notebook {want}")
check("notebook output · Preference Match", app["Preference Match"] == "Strong Match",
      app["Preference Match"])
check("notebook output · summary table",
      summary["Estimated Likelihood (%)"].tolist() == [93.3, 69.2, 79.6, 84.9]
      and summary["Condition"].tolist() == ["Comfortable Temperature", "Dry", "Calm", "Sunny"])

# ---------------------------------------------------------------- 3. training sanity
fitted = build_artifacts.fit_all()
check("rows · model_data 126736", len(fitted["model_data"]) == 126736)
check("rows · train/val/test 109572/8768/8396",
      (len(fitted["train_data"]), len(fitted["val_data"]), len(fitted["test_data"])) == (109572, 8768, 8396))
r = fitted["temperature_residuals"]
check("residuals · mean 0.746, std 2.777, RMSE 2.876",
      (round(r.mean(), 3), round(r.std(), 3), round(float(np.sqrt(np.mean(r ** 2))), 3)) == (0.746, 2.777, 2.876))

# ---------------------------------------------------------------- 2. notebook code vs app
nb_globals = {
    "pd": pd, "np": np,
    "final_temperature_model": fitted["final_temperature_model"],
    "X_final_temperature": fitted["X_final_temperature"],
    "temperature_residuals": fitted["temperature_residuals"],
    "final_climatology": fitted["final_climatology"],
}
if NOTEBOOK.exists():
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    for i in NOTEBOOK_FUNCTION_CELLS:
        exec("".join(cells[i]["source"]), nb_globals)
    source = f"notebook cells {NOTEBOOK_FUNCTION_CELLS}"
else:
    print(f"(notebook not found at {NOTEBOOK}; comparing against app port only)")
    nb_globals.update(
        predict_event_weather=forecast.predict_event_weather,
        calculate_suitability=scoring.calculate_suitability,
        suitability_rating=scoring.suitability_rating,
        preference_summary=scoring.preference_summary,
    )
    source = "app functions (in-memory model)"

a = forecast.load_artifacts()
check("artifact · feature order",
      a.feature_columns == list(fitted["X_final_temperature"].columns), str(a.feature_columns))
check("artifact · residuals identical", np.array_equal(a.temperature_residuals, fitted["temperature_residuals"]))
check("artifact · climatology identical",
      np.array_equal(a.final_climatology[["dry", "calm", "sunny", "clear"]].to_numpy(),
                     fitted["final_climatology"][["dry", "calm", "sunny", "clear"]].to_numpy())
      and a.final_climatology[["location", "day_of_year"]].equals(
          fitted["final_climatology"][["location", "day_of_year"]]))

dates = pd.date_range("2026-10-01", "2028-12-31", freq="17D").tolist() + [
    pd.Timestamp("2028-02-29"), pd.Timestamp("2028-12-31"), pd.Timestamp("2027-01-01")]
pref_grid = [dict(zip(scoring.CONDITIONS, combo))
             for combo in itertools.product(scoring.IMPORTANCE_LEVELS, repeat=5)]
max_temp_diff = max_prob_diff = max_score_diff = 0.0
rating_mismatch = summary_mismatch = n = 0
for loc in forecast.model_locations():
    for d in dates:
        nb_w = nb_globals["predict_event_weather"](loc, d)
        app_w = forecast.predict_event_weather(loc, d)
        max_temp_diff = max(max_temp_diff, abs(nb_w["Maximum Temperature"] - app_w["Maximum Temperature"]))
        for c in scoring.CONDITIONS:
            max_prob_diff = max(max_prob_diff, abs(nb_w[f"{c} Probability"] - app_w[f"{c} Probability"]))
        for p in pref_grid[::7]:
            s_nb = nb_globals["calculate_suitability"](nb_w, p)
            s_app = scoring.calculate_suitability(app_w, p)
            if (s_nb is None) != (s_app is None):
                max_score_diff = float("inf")
            elif s_nb is not None:
                max_score_diff = max(max_score_diff, abs(s_nb - s_app))
            rating_mismatch += nb_globals["suitability_rating"](s_nb) != scoring.suitability_rating(s_app)
            t_nb = nb_globals["preference_summary"](nb_w, p)
            t_app = scoring.preference_summary(app_w, p)
            summary_mismatch += not (t_nb.round(9).equals(t_app.round(9)))
            n += 1

print(f"\nCompared {n:,} location/date/preference cases against {source}")
check("max temperature identical", max_temp_diff == 0, f"max diff {max_temp_diff:.2e} °C")
check("probabilities identical", max_prob_diff == 0, f"max diff {max_prob_diff:.2e}")
check("suitability scores identical", max_score_diff == 0, f"max diff {max_score_diff:.2e}")
check("ratings identical", rating_mismatch == 0, f"{rating_mismatch} mismatches")
check("preference summaries identical", summary_mismatch == 0, f"{summary_mismatch} mismatches")
check("all Not important -> None / 'No preferences selected'",
      scoring.calculate_suitability(app_w, dict.fromkeys(scoring.CONDITIONS, "Not important")) is None
      and scoring.suitability_rating(None) == "No preferences selected")
try:
    forecast.predict_event_weather("Auckand", "2027-01-01")
    check("unknown location rejected", False)
except ValueError:
    check("unknown location rejected", True)

print("\nALL CHECKS PASSED" if not failures else f"\n{len(failures)} CHECK(S) FAILED: {failures}")
sys.exit(1 if failures else 0)
