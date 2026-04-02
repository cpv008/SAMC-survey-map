# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random

st.set_page_config(page_title="Parish Survey Map", layout="wide")
st.title("St. Anthony Mary Claret Parish Survey Map")

# =============================
# LOAD ZIP DATA
# =============================
@st.cache_data
def load_zip_data(path: str) -> pd.DataFrame:
    zips = pd.read_csv(path, sep="|", dtype={"GEOID": str})
    zips = zips[["GEOID", "INTPTLAT", "INTPTLONG"]]
    zips.columns = ["zip_code", "lat", "lon"]
    zips["zip_code"] = zips["zip_code"].astype(str).str.zfill(5)
    return zips

# =============================
# LOAD LANDMARKS DATA
# =============================
@st.cache_data
def load_landmarks(path: str) -> pd.DataFrame:
    lm = pd.read_csv(path, dtype={"lat": float, "lon": float})
    lm = lm[["title", "type", "lat", "lon"]].dropna(subset=["lat", "lon"])
    return lm

# =============================
# ZIP FILE PATH
# =============================
zip_file_path = st.text_input(
    "Path to zip_lat_lon.txt",
    value="zip_lat_lon.txt",
    help="Enter the full path to your local zip_lat_lon.txt file, e.g. C:/Users/you/Downloads/zip_lat_lon.txt"
)

if not zip_file_path:
    st.info("Enter the path to your ZIP lat/lon file to begin.")
    st.stop()

try:
    zip_df = load_zip_data(zip_file_path)
except FileNotFoundError:
    st.error(f"File not found: `{zip_file_path}`. Please check the path and try again.")
    st.stop()
except Exception as e:
    st.error(f"Error loading ZIP file: {e}")
    st.stop()

# =============================
# LANDMARKS FILE PATH
# =============================
landmarks_file_path = st.text_input(
    "Path to landmarks.csv",
    value="landmarks.csv",
    help="Enter the full path to your local landmarks.csv file, e.g. C:/Users/you/Downloads/landmarks.csv"
)

landmarks_df = None
if landmarks_file_path:
    try:
        landmarks_df = load_landmarks(landmarks_file_path)
        st.sidebar.success(f"Loaded {len(landmarks_df)} landmarks")
    except FileNotFoundError:
        st.warning(f"Landmarks file not found: `{landmarks_file_path}`. Map will render without landmarks.")
    except Exception as e:
        st.warning(f"Could not load landmarks: {e}. Map will render without landmarks.")

# =============================
# FILE UPLOAD
# =============================
uploaded_file = st.file_uploader("Upload your survey CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload your survey data to begin.")
    st.stop()

# =============================
# LOAD DATA
# =============================
df = pd.read_csv(uploaded_file)

# Clean ZIP codes robustly
df["zip_code"] = (
    df["zip_code"]
    .astype(str)
    .str.extract(r"(\d{5})")[0]
)

# Merge ZIP data
df = df.merge(zip_df, on="zip_code", how="left")

# =============================
# DEBUG: ZIP MATCHING
# =============================
st.subheader("ZIP Merge Check")
st.write("Unique ZIPs:", df["zip_code"].dropna().unique()[:20])
st.dataframe(df[["zip_code", "lat", "lon"]].head(20))
st.write("Missing lat:", df["lat"].isna().sum())
st.write("Missing lon:", df["lon"].isna().sum())

# =============================
# JITTER
# =============================
def jitter(x):
    if pd.isna(x):
        return x
    return x + random.uniform(-0.05, 0.05)

df["lat"] = df["lat"].apply(jitter)
df["lon"] = df["lon"].apply(jitter)

# =============================
# MAPPINGS
# =============================
mappings = {
    "mass_attendance": {
        "Less than once a month": 1,
        "Once a month": 2,
        "Once every two weeks": 3,
        "Every Sunday": 4,
        "Every mass that is offered": 5
    },
    "drive_distance": {
        "Less than 10 miles": 1,
        "10 to 29 miles": 2,
        "30 to 49 miles": 3,
        "Over 50 miles": 4
    },
    "willing_distance": {
        "Less than 10 miles": 1,
        "10 to 29 miles": 2,
        "30 to 49 miles": 3,
        "Over 50 miles": 4
    },
    "parishioner_length": {
        "Less than 1 year": 1,
        "1 to 5 years": 2,
        "6 to 10 years": 3,
        "More than 10 years": 4
    },
    "household_size": {
        "1": 1,
        "2": 2,
        "3 to 4": 3,
        "5 to 6": 4,
        "7 or more": 5
    }
}

for col, mapping in mappings.items():
    if col in df.columns:
        df[col + "_num"] = df[col].map(mapping)

# =============================
# FILTERS
# =============================
st.sidebar.header("Filters")

valid_columns = [c for c in df.columns if c.endswith("_num")]

if not valid_columns:
    st.error("No valid numeric columns found.")
    st.stop()

filter_column = st.sidebar.selectbox("Select question", valid_columns)

min_val = float(df[filter_column].min())
max_val = float(df[filter_column].max())
selected_range = st.sidebar.slider("Range", min_val, max_val, (min_val, max_val))

filtered_df = df[(df[filter_column] >= selected_range[0]) & (df[filter_column] <= selected_range[1])]

# =============================
# AGGREGATION
# =============================
aggregate = st.sidebar.checkbox("Aggregate by ZIP", True)

if aggregate:
    agg_df = filtered_df.groupby("zip_code").agg(
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        count=("zip_code", "count"),
        value=(filter_column, "mean")
    ).reset_index()
else:
    agg_df = filtered_df.copy()
    agg_df["count"] = 1
    agg_df["value"] = agg_df[filter_column]

# Drop bad rows
agg_df = agg_df.dropna(subset=["lat", "lon"])

# =============================
# WEIGHTED MIDPOINT
# =============================
# Weight each participant by household_size if available, otherwise weight = 1.
# The weighted midpoint is the household-weighted geographic center of all
# filtered participants with valid coordinates.
midpoint_df = filtered_df.dropna(subset=["lat", "lon"]).copy()

if "household_size_num" in midpoint_df.columns:
    midpoint_df["weight"] = midpoint_df["household_size_num"].fillna(1)
else:
    midpoint_df["weight"] = 1

total_weight = midpoint_df["weight"].sum()

if total_weight > 0:
    mid_lat = (midpoint_df["lat"] * midpoint_df["weight"]).sum() / total_weight
    mid_lon = (midpoint_df["lon"] * midpoint_df["weight"]).sum() / total_weight
    weighted_midpoint = (mid_lat, mid_lon)
else:
    weighted_midpoint = None

# =============================
# FINAL DEBUG
# =============================
st.subheader("Final Map Data Check")
st.write("Total rows:", len(df))
st.write("Filtered rows:", len(filtered_df))
st.write("Mapped rows:", len(agg_df))
st.dataframe(agg_df.head(20))

# =============================
# MAP
# =============================
st.subheader("Map")

# Base survey layer
fig = px.scatter_mapbox(
    agg_df,
    lat="lat",
    lon="lon",
    size="count",
    color="value",
    color_continuous_scale="YlOrRd",
    hover_data=["zip_code", "count"],
    zoom=6,
    height=650
)

# Landmark layer
# Only "airport" and "circle" are reliably supported on open-street-map without a token.
# Hospitals use a large bright green circle — distinct from the YlOrRd survey dots
# in both color and that they don't scale with count.
if landmarks_df is not None and not landmarks_df.empty:
    type_config = {
        "Airport":  {"symbol": "airport", "color": "blue",        "size": 20},
        "Hospital": {"symbol": "circle",  "color": "limegreen",   "size": 16},
    }
    default_config = {"symbol": "circle", "color": "purple", "size": 14}

    for lm_type, group in landmarks_df.groupby("type"):
        cfg = type_config.get(lm_type, default_config)

        fig.add_trace(go.Scattermapbox(
            lat=group["lat"],
            lon=group["lon"],
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=cfg["size"],
                symbol=cfg["symbol"],
                color=cfg["color"],
            ),
            name=lm_type,
            customdata=group[["title"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"Type: {lm_type}<extra></extra>"
            ),
            showlegend=True,
        ))

# Weighted midpoint marker — solid black circle
if weighted_midpoint:
    weight_label = "household-weighted" if "household_size_num" in filtered_df.columns else "equal-weighted"
    fig.add_trace(go.Scattermapbox(
        lat=[weighted_midpoint[0]],
        lon=[weighted_midpoint[1]],
        mode="markers",
        marker=dict(size=18, color="black", opacity=1.0),
        name="Weighted Midpoint",
        hovertemplate=(
            f"<b>Weighted Midpoint</b><br>"
            f"Weighting: {weight_label}<br>"
            f"Lat: {weighted_midpoint[0]:.4f}<br>"
            f"Lon: {weighted_midpoint[1]:.4f}<br>"
            f"Total weight: {total_weight:.0f}<extra></extra>"
        ),
        showlegend=True,
    ))

fig.update_layout(
    mapbox_style="open-street-map",
    legend=dict(
        title=dict(text="Legend", font=dict(color="black", size=13)),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="gray",
        borderwidth=1,
        x=0.01,
        y=0.99,
        xanchor="left",
        yanchor="top",
        font=dict(size=12, color="black"),
    ),
    coloraxis_colorbar=dict(
        title=dict(text="Value", font=dict(color="black")),
        tickfont=dict(color="black"),
        thickness=15,
        len=0.5,
        x=0.99,
        xanchor="right",
        y=0.5,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="gray",
        borderwidth=1,
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="white",
)

st.plotly_chart(fig, use_container_width=True)

# Show midpoint info below the map
if weighted_midpoint:
    weight_label = "household-weighted" if "household_size_num" in filtered_df.columns else "equal-weighted"
    st.info(
        f"**Weighted Midpoint ({weight_label}):** "
        f"Lat `{weighted_midpoint[0]:.4f}`, Lon `{weighted_midpoint[1]:.4f}` — "
        f"Total weight: `{total_weight:.0f}`"
    )

# =============================
# TEST POINT (for sanity check)
# =============================
st.subheader("Test Point (Should Always Show)")

test = pd.DataFrame({
    "lat": [30.5],
    "lon": [-86.5],
    "count": [10],
    "value": [3]
})

fig2 = px.scatter_mapbox(test, lat="lat", lon="lon", size="count", color="value", zoom=6)
fig2.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig2)

# =============================
# NOTES
# =============================
st.markdown("""
### How to Debug
- If test point shows but map doesn't → ZIP issue
- If 'Mapped rows' = 0 → filtering or merge issue
- If many missing lat/lon → ZIP format problem
- If landmarks don't appear → check path and that lat/lon columns are numeric
""")
