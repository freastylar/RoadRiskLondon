from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pydeck as pdk
import streamlit as st

from roadrisk.ui.tables import APP_DIR
from roadrisk.ui.theme import configure_page, show_limitations
from roadrisk.utils.validation import read_required_json, read_required_parquet

TABLE_PATH = APP_DIR / "road_risk_table.parquet"
META_PATH = APP_DIR / "road_risk_meta.json"

configure_page("RoadRisk London - Trip Risk")
st.header("Trip Risk by Road")
st.caption(
    "How risky is a London road compared with the typical road, given the hour, day, weather, and "
    "how you travel? A relative risk index built from historical collisions — not an absolute "
    "per-trip probability."
)

if not TABLE_PATH.exists() or not META_PATH.exists():
    st.warning(
        "Road-risk artifacts are not built. Run:\n\n"
        "```\n"
        "python scripts/14_fetch_london_roads.py\n"
        "python scripts/15_build_road_risk.py --mode mvp --start-year 2020 --end-year 2024\n"
        "```"
    )
    st.stop()

table = read_required_parquet(TABLE_PATH)
meta = read_required_json(META_PATH)

st.warning(
    "How to read this: DfT records only reported injury collisions and provides no per-trip "
    "exposure, so trips are estimated from traffic counts shaped by standard hourly profiles and "
    "fixed mode shares. Treat the result as a relative guide with the confidence label, not precise odds.",
    icon="⚠️",
)

col1, col2, col3, col4 = st.columns(4)
mode = col1.selectbox("How do you travel?", meta["modes"],
                      index=meta["modes"].index("Bicycle") if "Bicycle" in meta["modes"] else 0)
hour_bucket = col2.selectbox("Time of day", meta["hour_buckets"])
day_type = col3.selectbox("Day", ["Weekday", "Weekend"])
weather = col4.selectbox("Weather", meta["weather_categories"])

subset = table[
    (table["mode"] == mode)
    & (table["hour_bucket"] == hour_bucket)
    & (table["day_type"] == day_type)
    & (table["weather"] == weather)
].copy()

if subset.empty:
    st.info("No historical collisions match these exact conditions. Try a broader time or weather.")
    st.stop()

# Relative risk index: how this road compares with the TYPICAL London road for the
# same conditions. The data supports a comparison far better than an absolute
# per-trip probability, so we present risk as a multiple of the median road.
_median_risk = float(subset["risk_per_trip"].median()) or 1.0
subset["risk_index"] = subset["risk_per_trip"] / _median_risk

# A road label can repeat across OSM fragments in the same borough, producing
# duplicate road_area rows. Keep the single highest-risk row per area so the
# dropdown, lookups, and ordering are unambiguous.
subset = (
    subset.sort_values("risk_per_trip", ascending=False)
    .drop_duplicates("road_area", keep="first")
    .reset_index(drop=True)
)


def _risk_color(index: float) -> list[int]:
    """Colour by the ACTUAL risk multiple (vs the typical road), not percentile rank.

    This makes the colour honest: red means genuinely much riskier than typical, so
    a calm condition where nothing stands out stays yellow rather than forcing some
    road to red. Anchored at: <=1.3x -> yellow, 1.3x-1.8x -> yellow->orange,
    1.8x+ -> deep red.
    """
    if index <= 1.3:
        return [255, 225, 40, 200]  # at/near typical -> yellow
    if index <= 1.8:
        t = (index - 1.3) / 0.5  # yellow -> orange
        return [255, int(225 - 75 * t), 35, 215]
    t = min((index - 1.8) / 0.4, 1.0)  # orange -> deep red (caps at ~2.2x)
    return [255, int(150 - 150 * t), 30, 235]


st.subheader("Risk map for these conditions")
st.markdown(
    "**What each line means.** Every major London road is drawn. The DfT data covers all reported "
    "injury collisions, so **pale-green roads had no recorded injury collisions** in these "
    "conditions over the period (a positive sign — though very quiet roads can read green simply "
    "from low traffic). **Line colour shows how risky a road is versus the typical road**: yellow ≈ "
    "typical (≤1.3×), orange from 1.3×, deep red = 1.8×+ — so a calm map with little red genuinely means "
    "few roads stand out. The **red ⚪-ringed dots mark danger spots** (well above typical, with "
    "enough crashes to be credible). Hover any line for stats; click to select it."
)
lc1, lc2, lc3, lc4, lc5 = st.columns(5)
lc1.markdown("🟩 No recorded collisions")
lc2.markdown("🟡 ~Typical (≤1.3×)")
lc3.markdown("🟠 Elevated (1.3×+)")
lc4.markdown("🔴 High (1.8×+)")
lc5.markdown("🔴⚪ **Danger spot**")

min_show = st.slider(
    "Declutter: only show roads at least this risky (× typical)",
    min_value=1.0, max_value=2.0, value=1.0, step=0.1,
    help="Raise this to hide ordinary roads and focus on the elevated/dangerous ones — "
         "useful when zoomed out over all of London.",
)

map_df = subset.copy()
# Rank is computed on the FULL set (before the declutter filter) so the top-1% danger
# logic is unaffected by what is hidden.
map_df["risk_rank"] = map_df["risk_per_trip"].rank(pct=True)
map_df["fill_color"] = map_df["risk_index"].map(_risk_color)
map_df["risk_index_str"] = map_df["risk_index"].map(lambda v: f"{v:.1f}x")
# A danger marker must be genuinely worse than a typical road, not merely in the
# top percentile of a flat distribution. It requires ALL of:
#   - clearly above typical in absolute terms (>= 1.5x the median road), and
#   - enough real crashes behind it (not a high ratio off 1-2 collisions), and
#   - within the top 1% for the chosen conditions.
# This keeps the marking reliable: a road never shows "danger" while reading ~1.0x.
DANGER_RANK = 0.99
MIN_DANGER_CRASHES = 5
MIN_DANGER_INDEX = 1.5
map_df["is_danger"] = (
    (map_df["risk_rank"] >= DANGER_RANK)
    & (map_df["crash_count"] >= MIN_DANGER_CRASHES)
    & (map_df["risk_index"] >= MIN_DANGER_INDEX)
)
map_df["line_width"] = map_df["risk_index"].map(lambda i: 2 + min(i, 3.0) * 1.8)  # thicker = riskier

# Apply the declutter filter, but always keep flagged danger spots visible.
if min_show > 1.0:
    map_df = map_df[(map_df["risk_index"] >= min_show) | map_df["is_danger"]]

# Explode the per-road line paths into PathLayer rows (one row per line segment),
# each carrying its road's risk colour and stats for the tooltip.
path_rows = []
for _, r in map_df.iterrows():
    try:
        line_paths = json.loads(r["road_path_json"]) if r["road_path_json"] else []
    except (TypeError, ValueError):
        line_paths = []
    for path in line_paths:
        if len(path) >= 2:
            path_rows.append(
                {
                    "path": path,
                    "color": r["fill_color"],
                    "width": float(r["line_width"]),
                    "road_area": r["road_area"],
                    "road_label": r["road_label"],
                    "risk_index_str": r["risk_index_str"],
                    "crash_count": int(r["crash_count"]),
                    "confidence": r["confidence"],
                }
            )

layers = []

# Base layer: ALL major/named London roads, so every road is visible. The DfT data
# covers all reported injury collisions, so a road with none over the period is shown
# green as "no recorded injury collisions" (a positive signal, though very low-traffic
# roads can also read green simply from little exposure).
BASE_PATH = APP_DIR / "road_base_geometry.parquet"
if BASE_PATH.exists():
    base_roads = read_required_parquet(BASE_PATH)
    base_rows = [
        {"path": json.loads(p), "road_label": lbl}
        for p, lbl in zip(base_roads["path_json"], base_roads["road_label"], strict=True)
        if p
    ]
    layers.append(
        pdk.Layer(
            "PathLayer",
            data=base_rows,
            get_path="path",
            get_color=[150, 205, 150, 90],  # fainter green so it recedes behind risk lines
            get_width=1,
            width_min_pixels=1,
            width_max_pixels=2,
            pickable=False,
        )
    )

if path_rows:
    layers.append(
        pdk.Layer(
            "PathLayer",
            data=path_rows,
            get_path="path",
            get_color="color",
            get_width="width",
            width_min_pixels=1,
            width_max_pixels=6,
            pickable=True,
            auto_highlight=True,
        )
    )

# Clickable selection targets: small, semi-transparent dots ONLY on roads that
# actually have risk colour (not every road), kept subtle so they don't blanket the
# map when zoomed out. The coloured lines are the main visualization.
layers.append(
    pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[road_lon, road_lat]",
        get_fill_color="fill_color",
        get_radius=40,
        radius_min_pixels=0,
        radius_max_pixels=4,
        opacity=0.35,
        pickable=True,
        auto_highlight=True,
    )
)

# Danger markers: a bold red ringed dot on the top-risk roads, drawn LAST so they sit
# on top. Sized so they stand out at street zoom but shrink politely when zoomed out.
danger = map_df[map_df["is_danger"]]
if not danger.empty:
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=danger,
            get_position="[road_lon, road_lat]",
            get_fill_color=[170, 0, 0, 240],
            get_line_color=[255, 255, 255, 255],
            line_width_min_pixels=1,
            stroked=True,
            get_radius=200,
            radius_min_pixels=4,
            radius_max_pixels=14,
            pickable=True,
            auto_highlight=True,
        )
    )
deck = pdk.Deck(
    map_style="light",
    layers=layers,
    initial_view_state=pdk.ViewState(latitude=51.5074, longitude=-0.1278, zoom=10),
    tooltip={
        "html": "<b>{road_label}</b><br/>"
        "Risk: {risk_index_str} typical road<br/>"
        "Crashes (this slice): {crash_count}<br/>"
        "Confidence: {confidence}<br/><i>click to select</i>",
    },
)
event = st.pydeck_chart(deck, on_select="rerun", selection_mode="single-object", key="road_map")

clicked_area = None
try:
    objects = event.selection.get("objects", {})  # type: ignore[union-attr]
    for _layer_id, picked in objects.items():
        if picked:
            clicked_area = picked[0].get("road_area")
            break
except (AttributeError, KeyError, IndexError):
    clicked_area = None

st.subheader("⚠️ Most dangerous roads for these conditions")
st.caption(
    "The top roads by relative risk, restricted to those with enough reported collisions to be "
    "credible (so a high ratio off one or two crashes is not mistaken for real danger)."
)
credible = subset[
    (subset["crash_count"] >= MIN_DANGER_CRASHES) & (subset["risk_index"] >= MIN_DANGER_INDEX)
]
top_danger = (credible if not credible.empty else subset).head(8).copy()
top_danger_table = top_danger[["road_label", "risk_index", "crash_count", "confidence"]].rename(
    columns={
        "road_label": "Road",
        "risk_index": "Risk vs typical",
        "crash_count": "Reported collisions",
        "confidence": "Confidence",
    }
)
top_danger_table["Risk vs typical"] = top_danger_table["Risk vs typical"].map(lambda v: f"{v:.1f}×")
st.dataframe(top_danger_table, width="stretch", hide_index=True)

options = subset["road_area"].tolist()
labels = subset.set_index("road_area")["road_label"].to_dict()
indices = subset.set_index("road_area")["risk_index"].to_dict()
default_index = options.index(clicked_area) if clicked_area in options else 0
chosen = st.selectbox(
    "Or pick a road from the list (ordered by risk for these conditions)",
    options,
    index=default_index,
    format_func=lambda a: f"{labels.get(a, a)}  -  {indices.get(a, 0):.1f}x",
)
if clicked_area and clicked_area in options:
    st.success(f"Selected from map: {labels.get(clicked_area, clicked_area)}")
row = subset[subset["road_area"] == chosen].iloc[0]


def _risk_word(index: float) -> str:
    if index >= 2.0:
        return "much higher than typical"
    if index >= 1.3:
        return "higher than typical"
    if index >= 0.77:
        return "about typical"
    return "lower than typical"


st.subheader(f"Relative risk on {row['road_label']}")
m1, m2, m3 = st.columns(3)
m1.metric("Risk vs typical road", f"{row['risk_index']:.1f}×")
m2.metric("Reported collisions", f"{int(row['crash_count']):,}")
m3.metric("Confidence", row["confidence"])

st.markdown("**What this means:**")
st.markdown(
    f"- **Risk index {row['risk_index']:.1f}× — {_risk_word(row['risk_index'])}.** For "
    f"**{mode.lower()}** travel on a **{day_type.lower()}**, in the "
    f"**{hour_bucket.split(' ')[0].lower()}**, in **{weather.lower()}** weather, "
    f"**{row['road_label']}** has about **{row['risk_index']:.1f} times** the collision risk of the "
    "typical London road in the same conditions. (1.0× = average; higher = more dangerous.)\n"
    f"- **Confidence — {row['confidence']}.** Based on **{int(row['crash_count'])} matching reported "
    f"collisions** over {meta['year_range'][0]}–{meta['year_range'][1]}. "
    + ("Few crashes here, so treat as indicative only.\n" if row["confidence"] == "Low"
       else "Enough crashes for a steadier comparison.\n")
    + "- **Why relative, not absolute?** The data records collisions but not how many trips actually "
    "pass along each road, so an exact 'per-trip' probability would rest on heavy assumptions. A "
    "comparison against the typical road is what the data genuinely supports."
)

with st.expander("Full numbers for this road"):
    st.table(
        {
            "Statistic": [
                "Road", "Mode", "Time of day", "Day", "Weather",
                "Reported collisions (this slice)", "Risk index (vs typical road)", "Confidence",
            ],
            "Value": [
                row["road_label"], mode, hour_bucket, day_type, weather,
                f"{int(row['crash_count']):,}", f"{row['risk_index']:.2f}×", row["confidence"],
            ],
        }
    )

st.caption(
    "Risk index = this road's collision rate ÷ the median London road's rate for the same "
    "conditions, on crashes snapped to the nearest real road. A relative comparison, not an "
    "absolute per-trip probability. Reported personal-injury collisions only."
)

show_limitations()
