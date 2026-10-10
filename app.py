"""
GoodDay — weather suitability for outdoor event planning in New Zealand.

Run with:  streamlit run app.py

This file is UI only. Expected weather comes from
goodday.forecast.get_expected_conditions() (the model swap point), and the
match maths lives in goodday.scoring.
"""

from __future__ import annotations

from dataclasses import fields
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from goodday import climatology, scoring
from goodday.forecast import ExpectedConditions, get_expected_conditions

st.set_page_config(page_title="GoodDay", page_icon="🌤️", layout="wide")

MAX_RANGE_DAYS = 31
PREF_LABELS = {1: "1 · Avoid", 2: "2", 3: "3 · Don't mind", 4: "4", 5: "5 · Want it"}
DEFAULT_PREFS = {"sunshine": 5, "rain": 1, "cloud": 2, "wind": 1}

st.markdown(
    """
    <style>
      .gd-card {border:1px solid rgba(128,128,128,.25); border-radius:12px;
                padding:14px 16px; margin-bottom:12px;}
      .gd-big {font-size:3rem; font-weight:700; line-height:1;}
      .gd-badge {display:inline-block; padding:2px 10px; border-radius:999px;
                 color:#fff; font-size:.8rem; font-weight:600;}
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
@st.cache_resource(show_spinner="Crunching 85 years of weather history…")
def warm_up() -> list[str]:
    climatology.build_climatology()
    return climatology.locations()


def average_conditions(days: list[ExpectedConditions]) -> ExpectedConditions:
    """Mean of every numeric field across a date range (for range summaries)."""
    if len(days) == 1:
        return days[0]
    kwargs = {}
    for f in fields(ExpectedConditions):
        vals = [getattr(d, f.name) for d in days]
        if all(isinstance(v, (int, float)) for v in vals):
            kwargs[f.name] = sum(vals) / len(vals)
        elif all(isinstance(v, tuple) for v in vals):
            kwargs[f.name] = tuple(sum(x) / len(x) for x in zip(*vals))
        else:
            kwargs[f.name] = vals[0]
    kwargs["source_note"] = f"Averaged across {len(days)} days. " + days[0].source_note
    return ExpectedConditions(**kwargs)


def expectation_text(key: str, c: ExpectedConditions) -> str:
    return {
        "sunshine": f"~{c.sunshine_hours:.1f} h of sunshine",
        "rain": f"{c.rain_chance:.0%} chance of a wet day",
        "cloud": f"~{c.cloud_cover_pct:.0f}% cloud cover",
        "wind": f"max wind ~{c.wind_max_kmh:.0f} km/h",
    }[key]


def traffic_light_html(light: str) -> str:
    colours = {"red": "#e53935", "amber": "#fbc02d", "green": "#43a047"}
    spans = "".join(
        f'<span class="{"on" if light == k else ""}" '
        f'style="background:{v}; color:{v};"></span>'
        for k, v in colours.items()
    )
    return f'<div class="gd-light" role="img" aria-label="{light} light">{spans}</div>'


# ---------------------------------------------------------------- layout
locations = warm_up()

st.title("🌤️ GoodDay")
st.caption("Will the weather suit your outdoor event? Tell us what you're hoping for.")

left, right = st.columns([1, 1.7], gap="large")

with left:
    st.subheader("Your event")
    location = st.selectbox("Location", locations,
                            index=locations.index("Christchurch") if "Christchurch" in locations else 0)
    mode = st.radio("Date", ["Single date", "Date range"], horizontal=True,
                    label_visibility="collapsed")
    today = date.today()
    if mode == "Single date":
        picked = st.date_input("Event date", value=today + timedelta(days=14),
                               format="DD/MM/YYYY")
        start = end = picked
    else:
        picked = st.date_input("Event date range",
                               value=(today + timedelta(days=14), today + timedelta(days=20)),
                               format="DD/MM/YYYY")
        if isinstance(picked, tuple) and len(picked) == 2:
            start, end = picked
        else:  # user has clicked only the first date so far
            start = end = picked[0] if isinstance(picked, tuple) else picked
        if (end - start).days + 1 > MAX_RANGE_DAYS:
            st.warning(f"Showing the first {MAX_RANGE_DAYS} days of that range.")
            end = start + timedelta(days=MAX_RANGE_DAYS - 1)

    st.subheader("Your weather preferences")
    st.caption("1 = least preferred, 5 = preferred")
    prefs = {}
    for key, meta in scoring.WEATHER_TYPES.items():
        prefs[key] = st.select_slider(
            f"{meta['icon']} {meta['label']}",
            options=[1, 2, 3, 4, 5],
            value=DEFAULT_PREFS[key],
            format_func=PREF_LABELS.get,
            key=f"pref_{key}",
        )

# ---------------------------------------------------------------- compute
dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
daily = []
for d in dates:
    cond = get_expected_conditions(location, d)          # <- model swap point
    matches = scoring.score_matches(cond, prefs)
    daily.append({"date": d, "cond": cond, "matches": matches,
                  "overall": scoring.overall_score(matches)})

summary_cond = average_conditions([r["cond"] for r in daily])
summary_matches = {
    k: sum(r["matches"][k].score for r in daily) / len(daily) for k in scoring.WEATHER_TYPES
}
overall = sum(r["overall"] for r in daily) / len(daily)
light, light_msg, light_colour = scoring.traffic_light(overall)
when = (f"{start:%a %d %b %Y}" if start == end
        else f"{start:%d %b} – {end:%d %b %Y} ({len(dates)} days)")

# ---------------------------------------------------------------- results
with right:
    st.subheader(f"{location} · {when}")

    # Overall suitability + traffic light
    band_label, _ = scoring.match_band(overall)
    c1, c2 = st.columns([0.18, 0.82])
    with c1:
        st.markdown(traffic_light_html(light), unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"""<div class="gd-card">
                  <div class="gd-muted">Overall suitability</div>
                  <div class="gd-big" style="color:{light_colour}">{overall:.0f}<span style="font-size:1.4rem">/100</span></div>
                  <div style="margin-top:6px"><b>{light_msg}</b> · {band_label}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    # Match rating for each weather type
    st.markdown("##### Match for each weather type")
    cols = st.columns(2)
    for i, (key, meta) in enumerate(scoring.WEATHER_TYPES.items()):
        score = summary_matches[key]
        label, colour = scoring.match_band(score)
        with cols[i % 2]:
            st.markdown(
                f"""<div class="gd-card">
                      <div style="display:flex; justify-content:space-between; align-items:center">
                        <b>{meta['icon']} {meta['label']}</b>
                        <span class="gd-badge" style="background:{colour}">{label}</span>
                      </div>
                      <div style="font-size:1.6rem; font-weight:700; margin:6px 0">{score:.0f}</div>
                      <div class="gd-bar"><div style="width:{score:.0f}%; background:{colour}"></div></div>
                      <div class="gd-muted" style="margin-top:8px">
                        Expect {expectation_text(key, summary_cond)} · you said {PREF_LABELS[prefs[key]]}
                      </div>
                    </div>""",
                unsafe_allow_html=True,
            )

    # Range view: score per day + best day
    if len(daily) > 1:
        st.markdown("##### Day by day")
        best = max(daily, key=lambda r: r["overall"])
        st.success(f"Best day in your range: **{best['date']:%A %d %B}** "
                   f"(score {best['overall']:.0f})")
        df_days = pd.DataFrame({"date": [r["date"] for r in daily],
                                "score": [r["overall"] for r in daily]})
        df_days["light"] = [scoring.traffic_light(s)[1] for s in df_days.score]
        fig = go.Figure(go.Bar(
            x=df_days.date, y=df_days.score,
            marker_color=[scoring.traffic_light(s)[2] for s in df_days.score],
            customdata=df_days.light,
            hovertemplate="%{x|%a %d %b}<br>Score %{y:.0f} — %{customdata}<extra></extra>",
        ))
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), bargap=0.25,
                          yaxis=dict(range=[0, 100], title="Suitability",
                                     gridcolor="rgba(128,128,128,.15)"),
                          xaxis=dict(tickformat="%d %b"))
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, width="stretch")

    # Supporting detail
    st.markdown("##### All the extra information")
    c = summary_cond
    m1, m2, m3 = st.columns(3)
    m1.metric("🌡️ Typical high", f"{c.temp_max:.1f} °C")
    m2.metric("Typical low", f"{c.temp_min:.1f} °C")
    m3.metric("Daily mean", f"{c.temp_mean:.1f} °C")
    if c.temp_max_range:
        st.caption(f"Highs usually fall between {c.temp_max_range[0]:.0f} °C and "
                   f"{c.temp_max_range[1]:.0f} °C (8 in 10 days).")
    m4, m5, m6 = st.columns(3)
    m4.metric("🌧️ Chance of a wet day", f"{c.rain_chance:.0%}")
    m5.metric("Average rainfall", f"{c.rain_mm:.1f} mm")
    if c.heavy_rain_chance is not None:
        m6.metric("Chance of heavy rain (≥10 mm)", f"{c.heavy_rain_chance:.0%}")
    m7, m8, m9 = st.columns(3)
    m7.metric("💨 Max wind", f"{c.wind_max_kmh:.0f} km/h")
    if c.strong_wind_chance is not None:
        m8.metric("Chance of strong wind (≥40 km/h)", f"{c.strong_wind_chance:.0%}")
    m9.metric("☁️ Cloud cover", f"{c.cloud_cover_pct:.0f}%")
    m10, m11, m12 = st.columns(3)
    m10.metric("☀️ Sunshine", f"{c.sunshine_hours:.1f} h")
    if c.daylight_hours is not None:
        m11.metric("Daylight", f"{c.daylight_hours:.1f} h")
    if c.humidity_pct is not None:
        m12.metric("Humidity", f"{c.humidity_pct:.0f}%")

    # Seasonal context chart
    prof = climatology.annual_profile(location)
    year_dates = pd.date_range("2001-01-01", periods=365)
    event_doys = [climatology.day_of_year(d) for d in dates]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=year_dates, y=prof.temp_max, mode="lines", line=dict(width=2, color="#e8710a"),
        name="Typical high", hovertemplate="%{x|%d %b}: %{y:.1f} °C<extra>Typical high</extra>"))
    fig2.add_trace(go.Scatter(
        x=year_dates, y=prof.temp_min, mode="lines", line=dict(width=2, color="#1a73e8"),
        name="Typical low", hovertemplate="%{x|%d %b}: %{y:.1f} °C<extra>Typical low</extra>"))
    fig2.add_vrect(x0=year_dates[min(event_doys) - 1],
                   x1=year_dates[max(event_doys) - 1] + pd.Timedelta(days=1),
                   fillcolor="rgba(128,128,128,.25)", line_width=0,
                   annotation_text="Your event", annotation_position="top left")
    fig2.update_layout(height=260, margin=dict(l=0, r=0, t=30, b=0),
                       title=dict(text=f"Typical temperatures through the year — {location}",
                                  font=dict(size=14)),
                       yaxis=dict(title="°C", gridcolor="rgba(128,128,128,.15)"),
                       xaxis=dict(tickformat="%b"),
                       legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig2, width="stretch")

    st.caption(f"**Source: {c.source}.** {c.source_note}")

    with st.expander("How the scores work"):
        st.markdown(
            """
- Each weather type is turned into an **amount from 0 to 1** — how much of it to expect
  (sunshine hours 5→13 h, wet-day chance 15→60 %, cloud cover 40→80 %, max wind 12→36 km/h).
- Your rating becomes the amount you'd **like**: 1 → none, 3 → some, 5 → lots.
- **Match = 100 × (1 − gap between expected and wanted).**
- **Overall** is a weighted average: strong opinions (1 or 5) count 3×, mild ones (2 or 4) 2×,
  "don't mind" (3) 1×.
- Traffic light: 🟢 65+ · 🟡 50–64 · 🔴 under 50. Match labels: Strong 80+, Good 65+, Average 50+,
  Weak 35+, Poor below 35.
            """
        )
