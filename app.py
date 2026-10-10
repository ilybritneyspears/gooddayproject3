"""
GoodDay — weather suitability for outdoor event planning in New Zealand.

Run with:  streamlit run app.py

This file is UI only:
  * goodday.forecast  — the trained model (predict_event_weather from the notebook)
  * goodday.scoring   — calculate_suitability / suitability_rating / preference_summary
  * goodday.climatology — ±7-day historical averages for the supporting-detail panel
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from goodday import climatology, scoring
from goodday.forecast import (ExpectedConditions, get_expected_conditions,
                              load_artifacts, model_locations, predict_event_weather)

st.set_page_config(page_title="GoodDay", page_icon="🌤️", layout="wide")

MAX_RANGE_DAYS = 31
# Defaults = the example preferences used in the notebook / report
DEFAULT_PREFS = {"Comfortable": "Essential", "Dry": "Essential", "Calm": "Preferred",
                 "Sunny": "Preferred", "Clear": "Not important"}

st.markdown(
    """
    <style>
      .gd-card {border:1px solid rgba(128,128,128,.25); border-radius:12px;
                padding:14px 16px; margin-bottom:12px;}
      .gd-card.off {opacity:.55;}
      .gd-big {font-size:3rem; font-weight:700; line-height:1;}
      .gd-badge {display:inline-block; padding:2px 10px; border-radius:999px;
                 color:#fff; font-size:.8rem; font-weight:600; white-space:nowrap;}
      .gd-chip {display:inline-block; padding:1px 8px; border-radius:999px; font-size:.75rem;
                border:1px solid rgba(128,128,128,.45);}
      .gd-bar {height:8px; border-radius:4px; background:rgba(128,128,128,.2);}
      .gd-bar > div {height:8px; border-radius:4px;}
      .gd-light {display:flex; flex-direction:column; gap:8px; padding:10px;
                 background:#222; border-radius:14px; width:52px; align-items:center;}
      .gd-light span {width:32px; height:32px; border-radius:50%; opacity:.18;}
      .gd-light span.on {opacity:1; box-shadow:0 0 14px currentColor;}
      .gd-muted {opacity:.7; font-size:.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading the GoodDay model and weather history…")
def warm_up() -> list[str]:
    load_artifacts()
    climatology.build_climatology()
    return model_locations()


@st.cache_data(show_spinner=False)
def predicted_max_for_year(location: str, year: int) -> pd.DataFrame:
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    temps = [predict_event_weather(location, d)["Maximum Temperature"] for d in days]
    return pd.DataFrame({"date": days, "pred_max": temps})


def averaged_weather(days: list[ExpectedConditions]) -> dict:
    """Average each probability across a date range, in the notebook's dict format.

    Because calculate_suitability() is a weighted average, scoring this averaged dict
    gives exactly the mean of the daily scores.
    """
    first = days[0].weather
    if len(days) == 1:
        return first
    out = dict(first)
    for key in ["Maximum Temperature"] + [f"{c} Probability" for c in scoring.CONDITIONS]:
        out[key] = sum(d.weather[key] for d in days) / len(days)
    return out


def traffic_light_html(light: str) -> str:
    colours = {"red": "#e53935", "amber": "#fbc02d", "green": "#43a047"}
    spans = "".join(
        f'<span class="{"on" if light == k else ""}" style="background:{v}; color:{v};"></span>'
        for k, v in colours.items()
    )
    return f'<div class="gd-light" role="img" aria-label="{light} light">{spans}</div>'


def mean_of(days: list[ExpectedConditions], attr: str) -> float:
    return sum(getattr(d, attr) for d in days) / len(days)


# ---------------------------------------------------------------- inputs
locations = warm_up()

st.title("🌤️ GoodDay")
st.caption("Will the weather suit your outdoor event? Tell us what matters to you.")

left, right = st.columns([1, 1.7], gap="large")

with left:
    st.subheader("Your event")
    location = st.selectbox("Location", locations,
                            index=locations.index("Christchurch") if "Christchurch" in locations else 0)
    mode = st.radio("Date", ["Single date", "Date range"], horizontal=True,
                    label_visibility="collapsed")
    today = date.today()
    if mode == "Single date":
        picked = st.date_input("Event date", value=today + timedelta(days=14), format="DD/MM/YYYY")
        start = end = picked
    else:
        picked = st.date_input("Event date range",
                               value=(today + timedelta(days=14), today + timedelta(days=20)),
                               format="DD/MM/YYYY")
        if isinstance(picked, tuple) and len(picked) == 2:
            start, end = picked
        else:  # only the first date clicked so far
            start = end = picked[0] if isinstance(picked, tuple) else picked
        if (end - start).days + 1 > MAX_RANGE_DAYS:
            st.warning(f"Showing the first {MAX_RANGE_DAYS} days of that range.")
            end = start + timedelta(days=MAX_RANGE_DAYS - 1)

    st.subheader("What matters for your event?")
    st.caption("Essential counts double, Preferred counts once, Not important is ignored.")
    prefs = {}
    for key, meta in scoring.CONDITIONS.items():
        prefs[key] = st.radio(
            f"{meta['icon']} {meta['label']}",
            scoring.IMPORTANCE_LEVELS,
            index=scoring.IMPORTANCE_LEVELS.index(DEFAULT_PREFS[key]),
            horizontal=True,
            help=meta["definition"],
            key=f"pref_{key}",
        )

# ---------------------------------------------------------------- compute
dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
daily = []
for d in dates:
    cond = get_expected_conditions(location, d)
    score = scoring.calculate_suitability(cond.weather, prefs)
    daily.append({"date": d, "cond": cond, "score": score,
                  "rating": scoring.suitability_rating(score)})

conds = [r["cond"] for r in daily]
weather = averaged_weather(conds)
overall = scoring.calculate_suitability(weather, prefs)
rating = scoring.suitability_rating(overall)
light, light_msg = scoring.traffic_light(rating)
when = (f"{start:%a %d %b %Y}" if start == end
        else f"{start:%d %b} – {end:%d %b %Y} ({len(dates)} days)")

# ---------------------------------------------------------------- results
with right:
    st.subheader(f"{location} · {when}")

    # Overall suitability + traffic light
    c1, c2 = st.columns([0.18, 0.82])
    with c1:
        st.markdown(traffic_light_html(light), unsafe_allow_html=True)
    with c2:
        if overall is None:
            st.markdown(
                """<div class="gd-card"><div class="gd-muted">Overall suitability</div>
                   <div class="gd-big">–</div>
                   <div style="margin-top:6px"><b>No preferences selected</b> · mark at least one
                   condition as Preferred or Essential.</div></div>""",
                unsafe_allow_html=True)
        else:
            colour = scoring.rating_colour(rating)
            st.markdown(
                f"""<div class="gd-card">
                      <div class="gd-muted">Overall suitability{' (average across your dates)' if len(dates) > 1 else ''}</div>
                      <div class="gd-big" style="color:{colour}">{overall:.0f}<span style="font-size:1.4rem">/100</span></div>
                      <div style="margin-top:6px">
                        <span class="gd-badge" style="background:{colour}">{rating}</span>
                        &nbsp;<b>{light_msg}</b>
                      </div>
                    </div>""",
                unsafe_allow_html=True,
            )

    # Likelihood of each condition
    st.markdown("##### How likely is each condition?")
    cols = st.columns(2)
    for i, (key, meta) in enumerate(scoring.CONDITIONS.items()):
        p = float(weather[f"{key} Probability"])
        importance = prefs[key]
        counted = importance != "Not important"
        colour = scoring.probability_colour(p) if counted else "#9e9e9e"
        extra = ""
        if key == "Comfortable":
            extra = f" · predicted max {weather['Maximum Temperature']:.1f} °C"
        with cols[i % 2]:
            st.markdown(
                f"""<div class="gd-card{'' if counted else ' off'}">
                      <div style="display:flex; justify-content:space-between; align-items:center; gap:8px">
                        <b>{meta['icon']} {meta['label']}</b>
                        <span class="gd-chip">{importance}{'' if counted else ' · not counted'}</span>
                      </div>
                      <div style="font-size:1.6rem; font-weight:700; margin:6px 0">{p:.0%}</div>
                      <div class="gd-bar"><div style="width:{p * 100:.0f}%; background:{colour}"></div></div>
                      <div class="gd-muted" style="margin-top:8px">{meta['definition']}{extra}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    if overall is not None:
        with st.expander("Preference summary table"):
            st.dataframe(scoring.preference_summary(weather, prefs).round(1),
                         hide_index=True, width="stretch")

    # Range view: score per day + best day
    if len(daily) > 1 and overall is not None:
        st.markdown("##### Day by day")
        best = max(daily, key=lambda r: r["score"])
        st.success(f"Best day in your range: **{best['date']:%A %d %B}** "
                   f"(score {best['score']:.0f}, {best['rating']})")
        df_days = pd.DataFrame({"date": [r["date"] for r in daily],
                                "score": [r["score"] for r in daily],
                                "rating": [r["rating"] for r in daily]})
        fig = go.Figure(go.Bar(
            x=df_days.date, y=df_days.score,
            marker_color=[scoring.rating_colour(r) for r in df_days.rating],
            customdata=df_days.rating,
            hovertemplate="%{x|%a %d %b}<br>Score %{y:.0f} — %{customdata}<extra></extra>",
        ))
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), bargap=0.25,
                          yaxis=dict(range=[0, 100], title="Suitability",
                                     gridcolor="rgba(128,128,128,.15)"),
                          xaxis=dict(tickformat="%d %b"))
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, width="stretch")

    # Temperature detail
    st.markdown("##### Temperature")
    m1, m2, m3 = st.columns(3)
    m1.metric("🌡️ Predicted max (model)", f"{weather['Maximum Temperature']:.1f} °C")
    m2.metric("Likely range", f"{mean_of(conds, 'max_temp_p10'):.0f}–{mean_of(conds, 'max_temp_p90'):.0f} °C",
              help="10th–90th percentile of the prediction plus the model's validation errors.")
    m3.metric("Comfortable (16–28 °C)", f"{weather['Comfortable Probability']:.0%}")

    # Historical supporting detail
    st.markdown("##### What it's usually like (historical averages)")
    h1, h2, h3 = st.columns(3)
    h1.metric("Average high", f"{mean_of(conds, 'typical_high'):.1f} °C")
    h2.metric("Average low", f"{mean_of(conds, 'typical_low'):.1f} °C")
    h3.metric("Average rainfall", f"{mean_of(conds, 'rain_mm'):.1f} mm")
    h4, h5, h6 = st.columns(3)
    h4.metric("💨 Average max wind", f"{mean_of(conds, 'wind_max_kmh'):.0f} km/h")
    h5.metric("☁️ Average cloud cover", f"{mean_of(conds, 'cloud_cover_pct'):.0f}%")
    h6.metric("Daylight", f"{mean_of(conds, 'daylight_hours'):.1f} h")

    # Seasonal context chart: model prediction vs historical averages
    prof = climatology.annual_profile(location)
    hist_dates = pd.date_range("2001-01-01", periods=365)
    pred = predicted_max_for_year(location, start.year)
    shift = pd.Timestamp(f"{start.year}-01-01") - pd.Timestamp("2001-01-01")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=pred.date, y=pred.pred_max, mode="lines", line=dict(width=2, color="#c2410c"),
        name=f"Predicted max ({start.year}, model)",
        hovertemplate="%{x|%d %b}: %{y:.1f} °C<extra>Predicted max</extra>"))
    fig2.add_trace(go.Scatter(
        x=hist_dates + shift, y=prof.temp_max, mode="lines",
        line=dict(width=2, color="#c2410c", dash="dot"), name="Average high (historical)",
        hovertemplate="%{x|%d %b}: %{y:.1f} °C<extra>Average high</extra>"))
    fig2.add_trace(go.Scatter(
        x=hist_dates + shift, y=prof.temp_min, mode="lines",
        line=dict(width=2, color="#1d4ed8", dash="dot"), name="Average low (historical)",
        hovertemplate="%{x|%d %b}: %{y:.1f} °C<extra>Average low</extra>"))
    fig2.add_hrect(y0=16, y1=28, fillcolor="rgba(30,142,62,.08)", line_width=0)
    fig2.add_vrect(x0=pd.Timestamp(start), x1=pd.Timestamp(end) + pd.Timedelta(days=1),
                   fillcolor="rgba(128,128,128,.25)", line_width=0,
                   annotation_text="Your event", annotation_position="top left")
    fig2.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0),
                       title=dict(text=f"Temperatures through the year — {location} "
                                       "(green band = comfortable 16–28 °C)", font=dict(size=14)),
                       yaxis=dict(title="°C", gridcolor="rgba(128,128,128,.15)"),
                       xaxis=dict(tickformat="%b"),
                       legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig2, width="stretch")

    meta = load_artifacts().metadata
    st.caption(
        "**Source: GoodDay model.** Maximum temperature from a Gradient Boosting model "
        f"(validation RMSE {meta.get('residuals', {}).get('rmse', float('nan')):.2f} °C); "
        "Dry, Calm, Sunny and Clear from a ±30-day seasonal climatology (1940–2020). "
        f"{conds[0].history_note} These are likelihoods for planning, not a forecast."
    )

    with st.expander("How the score works"):
        st.markdown(
            """
- **Comfortable temperature**: the model predicts the day's maximum temperature. Its past
  errors are added to that prediction to see how often the result lands in **16–28 °C**.
- **Dry, Calm, Sunny, Clear**: the share of historical days within ±30 days of your date
  that met each condition (≤1 mm rain · max wind ≤30 km/h · sunshine ≥70% of daylight ·
  cloud cover ≤40%).
- **Suitability** = weighted average of those likelihoods × 100, where **Essential = 2**,
  **Preferred = 1** and **Not important = 0**.
- **Strong Match** 75+ · **Good Match** 50+ · **Weak Match** 25+ · **Poor Match** below 25.
- Traffic light: 🟢 Strong · 🟡 Good · 🔴 Weak or Poor.
            """
        )
