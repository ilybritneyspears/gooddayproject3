# GoodDay 🌤️

Weather suitability for outdoor event planning in New Zealand (Auckland, Wellington,
Christchurch, Dunedin). Pick a location and date (or range), mark each condition as
**Not important / Preferred / Essential**, and GoodDay shows how likely each condition
is, an overall suitability score (0–100) with a match rating and traffic light, plus
supporting detail.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project layout

```
app.py                         UI only
goodday/forecast.py            predict_event_weather()  (notebook cell 99) + artifact loading
goodday/scoring.py             calculate_suitability / suitability_rating /
                               preference_summary / assess_event  (cells 101–107)
goodday/climatology.py         ±7-day historical averages for the "usually like" panel
build_artifacts.py             retrains the model and rewrites artifacts/ (sections 6, 7, 10)
artifacts/
  temperature_model.joblib     final_temperature_model (GradientBoostingRegressor)
  feature_columns.json         training column order
  temperature_residuals.npy    validation residuals (2015–2020), 8,768 values
  final_climatology.csv        ±30-day climatology, 4 locations × days 1–366
  metadata.json                versions, row counts, residual stats
tests/test_against_notebook.py end-to-end check against the notebook
data/full_weather_data.csv     team dataset
```

## Method (identical to Group5Project3.ipynb)

| Condition | Definition | Source |
|---|---|---|
| Comfortable | max temp 16–28 °C | GB model prediction + validation residuals → share within 16–28 °C |
| Dry | rain ≤ 1 mm | ±30-day seasonal climatology |
| Calm | max wind ≤ 30 km/h | ±30-day seasonal climatology |
| Sunny | sunshine ≥ 70% of daylight | ±30-day seasonal climatology |
| Clear | cloud cover ≤ 40% | ±30-day seasonal climatology |

Model features, in order: `year, day_sin, day_cos, location_Christchurch,
location_Dunedin, location_Wellington` with `day_sin/cos = sin/cos(2π·day_of_year/365.25)`.

Suitability = weighted average of the five probabilities × 100 (Essential 2,
Preferred 1, Not important 0). Strong Match 75+, Good Match 50+, Weak Match 25+,
Poor Match < 25. All Not important → "No preferences selected".

App-only additions (not in the notebook): date ranges (score = mean of the daily
scores), traffic light (Strong = green, Good = amber, Weak/Poor = red; set in
`scoring.TRAFFIC_LIGHT`), likely max-temp range (10th–90th percentile of
prediction + residuals), and the historical-averages panel.

## Retraining / verifying

```bash
python build_artifacts.py                                   # ~30 s, rewrites artifacts/
python tests/test_against_notebook.py path/to/Group5Project3.ipynb
```

The test checks the app against the notebook's printed example (Auckland
2027-02-15: 22.9 °C, 93.3% comfortable, score 81.6, Strong Match), then runs the
notebook's own function cells against a freshly trained model and compares every
location over ~60 dates and many preference combinations. Results must be
bit-identical.

The `.joblib` model is tied to the scikit-learn version pinned in
requirements.txt. If you change that version, re-run `build_artifacts.py`.
