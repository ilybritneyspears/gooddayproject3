# GoodDay 🌤️

Weather suitability for outdoor event planning in New Zealand (Auckland, Wellington,
Christchurch, Dunedin). Pick a location and date (or range), rate how much you want
sunshine, rain, cloud and wind (1 = least preferred, 5 = preferred), and GoodDay shows
a match rating per weather type, an overall suitability score with a traffic light,
and supporting detail such as typical temperatures.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first load takes ~6 seconds while it builds the climatology from the dataset;
after that it's instant.

## Project layout

```
app.py                    UI only (inputs, cards, traffic light, charts)
goodday/forecast.py       ★ MODEL SWAP POINT: get_expected_conditions()
goodday/scoring.py        preferences + expected conditions -> match scores
goodday/climatology.py    typical conditions per location & day-of-year
data/full_weather_data.csv  team dataset (1940–2026, daily)
```

## Swapping in the predictive model

Only edit the body of `get_expected_conditions(location, event_date)` in
`goodday/forecast.py`. It must return an `ExpectedConditions` object; the docstring
shows a ready-made pattern: start from climatology (for sunshine and cloud, which the
model doesn't predict) and overwrite temperature, rainfall, rain chance and wind with
the model's predictions. Set `source` / `source_note` so the UI labels the numbers
correctly. Nothing in `app.py` or `scoring.py` needs to change.

## Placeholder method (climatology)

- Dates are converted to NZ local calendar days.
- For each location and calendar day, all years within ±7 days are pooled
  (~1,300 historical days per estimate).
- Wet day = ≥ 1 mm rain; heavy rain = ≥ 10 mm; strong wind = daily max ≥ 40 km/h.

## Scoring

1. Each weather type becomes an amount from 0–1 by rescaling between anchors that
   bracket typical NZ conditions in the dataset: sunshine 5→13 h, wet-day chance
   15→60 %, cloud cover 40→80 %, max wind 12→36 km/h (clipped outside).
2. The 1–5 rating becomes the wanted amount: 1 → 0, 3 → 0.5, 5 → 1.
3. Match = 100 × (1 − |expected − wanted|).
4. Overall = weighted mean; ratings of 1/5 weigh 3, 2/4 weigh 2, 3 weighs 1.
5. Bands: Strong 80+, Good 65+, Average 50+, Weak 35+, Poor < 35.
   Traffic light: green 65+, amber 50–64, red < 50.

All thresholds live as constants at the top of `goodday/scoring.py`.

Note: the dataset's sunshine-duration values (Open-Meteo style) run higher than
official NZ sunshine-hour records, so treat them as relative rather than absolute.
