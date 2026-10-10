"""
Rebuild GoodDay's model artifacts from data/full_weather_data.csv.

This is a line-for-line extraction of the modelling steps in Group5Project3.ipynb
(sections 3, 6, 7 and 10) — nothing here changes the method:

  Section 3   cells 20–21  load data, NZ local date (date_UTC + 13 h)
  Section 6   cells 49–55  dropna, targets, date features, chronological split
  Section 7   cell 68      X_train_trend / X_val_trend (year + seasonal features)
              cells 81–83  final_train (train + val) -> final_temperature_model
              cell 84      final_climatology (±30-day window, days 1–366)
  Section 10  cell 97      temperature_validation_model -> temperature_residuals

Outputs (in artifacts/):
  temperature_model.joblib   fitted GradientBoostingRegressor (final_temperature_model)
  feature_columns.json       column order the model was trained on
  temperature_residuals.npy  validation residuals (actual - predicted, 2015–2020)
  final_climatology.csv      one row per (location, day_of_year 1–366)
  metadata.json              versions, row counts, sanity numbers

Run:  python build_artifacts.py      (takes ~1 minute)

Note: the .joblib file is tied to the scikit-learn version that wrote it.
If you change scikit-learn version, re-run this script.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "full_weather_data.csv"
OUT = ROOT / "artifacts"


def load_model_data() -> pd.DataFrame:
    # --- cells 18, 20, 21 ---------------------------------------------------
    allweather = pd.read_csv(DATA_PATH)
    workingdata = allweather.drop(columns=["latitude", "longitude"])
    for col in ["date_UTC", "sunrise_UTC", "sunset_UTC"]:
        workingdata[col] = pd.to_datetime(workingdata[col], utc=True)
    workingdata["date_NZDT"] = (
        (workingdata["date_UTC"] + pd.Timedelta(hours=13)).dt.tz_localize(None).dt.normalize()
    )
    workingdata.drop(columns=["sunrise_UTC", "sunset_UTC", "date_UTC"], inplace=True)

    # --- cell 49 ------------------------------------------------------------
    model_data = workingdata.dropna(
        subset=["rain_sum", "sunshine_duration", "cloud_cover_mean"]
    ).copy()

    # --- cell 51: weather preference targets --------------------------------
    model_data["comfortable"] = model_data["temperature_2m_max"].between(16, 28).astype(int)
    model_data["dry"] = (model_data["rain_sum"] <= 1).astype(int)
    model_data["calm"] = (model_data["wind_speed_10m_max"] <= 30).astype(int)
    model_data["sunny"] = (
        model_data["sunshine_duration"] / model_data["daylight_duration"] >= 0.70
    ).astype(int)
    model_data["clear"] = (model_data["cloud_cover_mean"] <= 40).astype(int)

    # --- cell 53: date features ---------------------------------------------
    model_data["year"] = model_data["date_NZDT"].dt.year
    model_data["month"] = model_data["date_NZDT"].dt.month
    model_data["day_of_year"] = model_data["date_NZDT"].dt.dayofyear
    model_data["day_sin"] = np.sin(2 * np.pi * model_data["day_of_year"] / 365.25)
    model_data["day_cos"] = np.cos(2 * np.pi * model_data["day_of_year"] / 365.25)
    return model_data


def fit_all() -> dict:
    """Run the notebook's modelling steps and return the fitted objects (in memory)."""
    model_data = load_model_data()

    # --- cell 55: chronological split ---------------------------------------
    train_data = model_data[model_data["year"] < 2015].copy()
    val_data = model_data[(model_data["year"] >= 2015) & (model_data["year"] <= 2020)].copy()
    test_data = model_data[model_data["year"] >= 2021].copy()

    # --- cell 68: trend feature matrices ------------------------------------
    trend_features = ["year", "day_sin", "day_cos", "location"]
    X_train_trend = pd.get_dummies(train_data[trend_features], columns=["location"], drop_first=True)
    X_val_trend = pd.get_dummies(val_data[trend_features], columns=["location"], drop_first=True)
    X_val_trend = X_val_trend.reindex(columns=X_train_trend.columns, fill_value=0)

    # --- cell 97: validation residuals --------------------------------------
    temperature_validation_model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=2, random_state=42
    )
    temperature_validation_model.fit(X_train_trend, train_data["temperature_2m_max"])
    temperature_val_pred = temperature_validation_model.predict(X_val_trend)
    temperature_residuals = val_data["temperature_2m_max"].values - temperature_val_pred

    # --- cells 81–83: final temperature model -------------------------------
    final_train = pd.concat([train_data, val_data], ignore_index=True)
    X_final_temperature = pd.get_dummies(
        final_train[trend_features], columns=["location"], drop_first=True
    )
    final_temperature_model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=2, random_state=42
    )
    final_temperature_model.fit(X_final_temperature, final_train["temperature_2m_max"])

    # --- cell 84: final ±30-day seasonal climatology ------------------------
    climate_window = 30
    final_climatology = []
    for location in final_train["location"].unique():
        location_data = final_train[final_train["location"] == location]
        for day in range(1, 367):
            distance = abs(location_data["day_of_year"] - day)
            distance = distance.clip(upper=365 - distance)
            nearby_days = location_data[distance <= climate_window]
            final_climatology.append({
                "location": location,
                "day_of_year": day,
                "dry": nearby_days["dry"].mean(),
                "calm": nearby_days["calm"].mean(),
                "sunny": nearby_days["sunny"].mean(),
                "clear": nearby_days["clear"].mean(),
            })
    final_climatology = pd.DataFrame(final_climatology)

    return {
        "model_data": model_data, "train_data": train_data, "val_data": val_data,
        "test_data": test_data, "final_train": final_train,
        "X_final_temperature": X_final_temperature,
        "final_temperature_model": final_temperature_model,
        "temperature_residuals": temperature_residuals,
        "final_climatology": final_climatology,
        "climate_window": climate_window,
    }


def main() -> None:
    f = fit_all()
    model_data, train_data, val_data, test_data, final_train = (
        f["model_data"], f["train_data"], f["val_data"], f["test_data"], f["final_train"])
    X_final_temperature = f["X_final_temperature"]
    final_temperature_model = f["final_temperature_model"]
    temperature_residuals = f["temperature_residuals"]
    final_climatology = f["final_climatology"]
    climate_window = f["climate_window"]

    # --- export -------------------------------------------------------------
    OUT.mkdir(exist_ok=True)
    joblib.dump(final_temperature_model, OUT / "temperature_model.joblib")
    (OUT / "feature_columns.json").write_text(json.dumps(list(X_final_temperature.columns), indent=2))
    np.save(OUT / "temperature_residuals.npy", temperature_residuals.astype(np.float64))
    final_climatology.to_csv(OUT / "final_climatology.csv", index=False, float_format="%.17g")

    meta = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_notebook": "Group5Project3.ipynb (sections 6, 7, 10)",
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "model_params": final_temperature_model.get_params(),
        "feature_columns": list(X_final_temperature.columns),
        "rows": {
            "model_data": len(model_data), "train": len(train_data),
            "validation": len(val_data), "test": len(test_data),
            "final_train": len(final_train), "climatology": len(final_climatology),
        },
        "residuals": {
            "n": int(len(temperature_residuals)),
            "mean": float(temperature_residuals.mean()),
            "std": float(temperature_residuals.std()),
            "rmse": float(np.sqrt(np.mean(temperature_residuals ** 2))),
        },
        "comfortable_range_c": [16, 28],
        "climate_window_days": climate_window,
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    print("Rows:", meta["rows"])
    print("Residuals:", {k: round(v, 3) for k, v in meta["residuals"].items()})
    print("Features:", meta["feature_columns"])
    print(final_climatology.head())
    print("Artifacts written to", OUT)


if __name__ == "__main__":
    main()
