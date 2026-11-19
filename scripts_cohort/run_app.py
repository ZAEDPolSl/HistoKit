import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ARTIFACT_COLUMNS = {
    "Tissue Fold": "Tissue_Fold_pct",
    "Dark Spot / Foreign Object": "Dark_Spot_Foreign_Object_pct",
    "Pen": "Pen_pct",
    "Edge / Air Bubble": "Edge_Air_Bubble_pct",
    "Out of Focus": "Out_of_Focus_pct",
}

# Adjust these hex colors to match your segmentation-mask palette.

ARTIFACT_COLORS = {
    "Tissue Fold": "#FF6347",                  # (255, 99, 71)
    "Dark Spot / Foreign Object": "#00FF00",  # (0, 255, 0)
    "Pen": "#FF0000",                          # (255, 0, 0)
    "Edge / Air Bubble": "#FF00FF",            # (255, 0, 255)
    "Out of Focus": "#4B0082",                 # (75, 0, 130)
    "Total artifacts": "#03dbfc",
}

def parse_args():
    parser = argparse.ArgumentParser(description="HistoKit cohort quality dashboard")
    parser.add_argument("--csv", type=Path, default=Path("C:/Users/julia/Desktop/HistoKit_10_10_2025/slide_stats.csv"))
    parser.add_argument("--thumbnails", type=Path, default=Path("C:/Users/julia/Desktop/HistoKit_10_10_2025/tissue_detection/thumbnails/"))
    parser.add_argument("--masks", type=Path, default=Path("C:/Users/julia/Desktop/artifact_detection_map/"))

    args, _ = parser.parse_known_args()
    return args


@st.cache_data(show_spinner=False)
def load_stats(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def image_index(directory: str | None) -> dict[str, Path]:
    if not directory:
        return {}
    folder = Path(directory)
    if not folder.is_dir():
        return {}
    paths = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    return {p.stem.lower(): p for p in paths}


def find_image(index: dict[str, Path], slide_name: str) -> Path | None:
    key = slide_name.lower()
    if key in index:
        return index[key]
    matches = [p for stem, p in index.items() if stem.startswith(key + "_") or stem.startswith(key + "-")]
    return matches[0] if matches else None


def show_slide_images(slide_id: str, thumbs: dict[str, Path], masks: dict[str, Path]):
    left, right = st.columns(2)
    for column, title, index in [(left, "Thumbnail", thumbs), (right, "Artifact mask", masks)]:
        with column:
            st.markdown(f"**{title}**")
            image = find_image(index, slide_id)
            if image:
                st.image(str(image), width="stretch")
            else:
                st.info(f"{title} not found for {slide_id}.")


def presentation_style(fig):
    """Larger, slide-friendly Plotly fonts without changing the chart data."""
    fig.update_layout(
        font=dict(size=24),
        title_font=dict(size=32),
        xaxis_title_font=dict(size=26),
        yaxis_title_font=dict(size=26),
        xaxis_tickfont=dict(size=22),
        yaxis_tickfont=dict(size=22),
        legend=dict(font=dict(size=24), title_font=dict(size=26)),
    )
    return fig


def sync_threshold(source_key, target_key):
    """Keep threshold sliders in Artifact Trends and Data Explorer synchronized."""
    st.session_state[target_key] = st.session_state[source_key]


def main():
    st.set_page_config(page_title="HistoKit Dashboard", page_icon="🔬", layout="wide")
    args = parse_args()
    st.markdown("""
    <style>
    html, body, [class*="st-"] { font-size: 26px; }
    /* Large page and section headings for presentation screenshots */
    [data-testid="stHeadingWithActionElements"] h1,
    [data-testid="stHeadingWithActionElements"] h1 span,
    h1 { font-size: 64px !important; line-height: 1.15 !important; }
    [data-testid="stHeadingWithActionElements"] h2,
    h2 { font-size: 48px !important; line-height: 1.2 !important; }
    [data-testid="stHeadingWithActionElements"] h3,
    h3 { font-size: 42px !important; line-height: 1.2 !important; }
    [data-baseweb="tab-list"] button,
    [data-baseweb="tab"] {
        font-size: 34px !important;
        font-weight: 750 !important;
        padding: 16px 20px !important;
    }
    [data-baseweb="tab"] p,
    [data-baseweb="tab"] span {
        font-size: 34px !important;
        font-weight: 750 !important;
    }
    p, label, .stCaption, [data-testid="stCaptionContainer"] p {
        font-size: 24px !important;
    }
    button, [role="tab"], [data-baseweb="tab"] {
        font-size: 26px !important;
    }
    [data-testid="stMetric"] {
        background: var(--secondary-background-color, transparent);
        border: 1px solid var(--border-color, rgba(128,128,128,.25));
        border-radius: 14px;
        padding: 24px;
        color: var(--text-color, inherit);
    }
    [data-testid="stMetricLabel"] p { font-size: 26px !important; }
    [data-testid="stMetricValue"] { font-size: 48px !important; }
    [data-testid="stSidebar"] { font-size: 22px; }
    [data-testid="stSidebar"] p { font-size: 20px !important; }
    [data-baseweb="select"] *, [data-baseweb="input"] input {
        font-size: 24px !important;
    }
    [data-testid="stSlider"] * { font-size: 22px; }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:

        st.image(str("C:/Repos/HistoKit_current/HistoKit/docs/assets/images/logo.png"), width=190)
        st.title("HistoKit")
        st.caption("Whole-slide quality control")
        st.divider()
        st.caption(f"Statistics: {args.csv}")
        st.caption(f"Thumbnails: {args.thumbnails or 'not configured'}")
        st.caption(f"Masks: {args.masks or 'not configured'}")

    st.title("Cohort Quality Dashboard")
    if not args.csv.is_file():
        st.error(f"CSV not found: {args.csv}")
        st.stop()
    df = load_stats(str(args.csv), args.csv.stat().st_mtime).copy().reset_index(drop=True)
    if df.empty:
        st.warning("The CSV contains no slides.")
        st.stop()
    if "filepath" in df.columns:
        df["slide_id"] = df["filepath"].astype(str).map(lambda x: Path(x.replace("\\", "/")).stem)
    elif "basename" in df.columns:
        df["slide_id"] = df["basename"].astype(str).map(lambda x: Path(x.replace("\\", "/")).stem)
    else:
        st.error("The CSV must contain a 'filepath' or 'basename' column.")
        st.stop()

    numeric = ["artifact_free_pct", "artifact_pct", "non_background_pixels", "artifact_pixels", *ARTIFACT_COLUMNS.values()]
    for column in numeric:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "artifact_free_pct" not in df and "artifact_pct" in df:
        df["artifact_free_pct"] = 100 - df["artifact_pct"]
    available = {name: col for name, col in ARTIFACT_COLUMNS.items() if col in df.columns}
    thumbs = image_index(str(args.thumbnails) if args.thumbnails else None)
    masks = image_index(str(args.masks) if args.masks else None)

    # Both tabs use the same threshold; changes in either slider update the other.
    if "threshold_trends" not in st.session_state:
        st.session_state["threshold_trends"] = 10.0
    if "threshold_data" not in st.session_state:
        st.session_state["threshold_data"] = st.session_state["threshold_trends"]

    tab_overview, tab_trends, tab_slides, tab_data = st.tabs(
        ["Cohort Overview", "Artifact Trends", "Slide Explorer", "Download CSV"]
    )

    with tab_overview:
        n = len(df)
        avg_free = df["artifact_free_pct"].mean() if "artifact_free_pct" in df else float("nan")
        avg_artifact = df["artifact_pct"].mean() if "artifact_pct" in df else float("nan")
        total_artifact = df["artifact_pixels"].sum() if "artifact_pixels" in df else 0
        a, b, c = st.columns(3)
        a.metric("Slides", f"{n:,}")
        b.metric("Mean artifact-free tissue", f"{avg_free:.2f}%")
        c.metric("Mean artifacts", f"{avg_artifact:.2f}%")
        st.caption("Mean percentages are unweighted across slides; each slide has equal weight.")
        if available:
            long_df = df[["slide_id", *available.values()]].melt(id_vars="slide_id", var_name="column", value_name="percent")
            long_df["Artifact"] = long_df["column"].map({v: k for k, v in available.items()})
            fig = px.box(long_df, x="Artifact", y="percent", color="Artifact", points="outliers",
                         color_discrete_map=ARTIFACT_COLORS,
                         category_orders={"Artifact": list(available)},
                         title="Artifact distribution across slides (non-background pixels)",
                         labels={"percent": "Percent"})
            fig.update_layout(xaxis_tickangle=-18, margin=dict(t=65, b=35), showlegend=False)
            fig = presentation_style(fig)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No artifact percentage columns found.")
        left, right = st.columns(2)
        with left:
            cols = [c for c in ("artifact_free_pct", "artifact_pct") if c in df.columns]
            if cols:
                melted = df[["slide_id", *cols]].melt(id_vars="slide_id", var_name="Type", value_name="Percent")
                melted["Type"] = melted["Type"].replace({"artifact_free_pct": "Artifact-free tissue", "artifact_pct": "Total artifacts"})
                fig = px.box(melted, x="Type", y="Percent", color="Type", points="outliers",
                             color_discrete_map=ARTIFACT_COLORS,
                             title="Artifact-free vs artifact tissue")
                fig.update_layout(showlegend=False)
                fig = presentation_style(fig)
                st.plotly_chart(fig, width="stretch")
        with right:
            if "artifact_pct" in df:
                fig = px.histogram(df, x="artifact_pct", nbins=25,
                                   title="Distribution of artifact burden",
                                   labels={"artifact_pct": "Artifact pixels (% of non-background)"})
                fig.update_traces(marker_color=ARTIFACT_COLORS["Total artifacts"])
                fig = presentation_style(fig)
                st.plotly_chart(fig, width="stretch")

    with tab_trends:
        st.subheader("Artifact percentages by slide")
        series = {**available}
        if "artifact_pct" in df:
            series["Total artifacts"] = "artifact_pct"
        if not series:
            st.info("No artifact percentage columns available.")
        else:
            selected = st.multiselect("Visible lines", options=list(series), default=list(series),
                                      help="You can also click legend items to temporarily hide/show traces.")
            all_values = df[list(series.values())].to_numpy(dtype=float)
            finite_values = all_values[pd.notna(all_values)]
            data_max = float(finite_values.max()) if finite_values.size else 0.0
            slider_max = max(100.0, float(int(data_max + 1)))
            threshold = st.slider(
                "Artifact threshold (%)", min_value=0.0, max_value=slider_max,
                step=0.5, key="threshold_trends",
                on_change=sync_threshold,
                args=("threshold_trends", "threshold_data"),
            )
            # Exclusion uses the overall artifact percentage, not individual artifact types.
            if "artifact_pct" in df.columns:
                total_pct = df["artifact_pct"]
                valid = total_pct.notna()
                excluded = int((valid & (total_pct > threshold)).sum())
                retained = int((valid & (total_pct <= threshold)).sum())
                missing = int((~valid).sum())
                c1, c2, c3 = st.columns(3)
                c1.metric("Excluded slides", f"{excluded:,}")
                c2.metric("Slides remaining", f"{retained:,}")
                c3.metric("Total slides", f"{len(df):,}")
                st.caption(
                    f"Exclusion criterion: artifact_pct > {threshold:.1f}%. "
                    f"Slides at exactly the threshold are retained. "
                    f"{missing} slides with missing artifact_pct are not counted as retained or excluded. "
                    "The threshold only previews exclusion; it does not modify the CSV."
                )
            else:
                st.warning("The artifact_pct column is required to count excluded slides.")
            fig = go.Figure()
            # Numeric positions preserve CSV row order and handle duplicate slide names.
            x = list(range(len(df)))
            for name in selected:
                fig.add_trace(go.Scatter(
                    x=x, y=df[series[name]], mode="lines+markers", name=name,
                    line=dict(color=ARTIFACT_COLORS[name], width=3 if name == "Total artifacts" else 2,
                              dash="dash" if name == "Total artifacts" else "solid"),
                    marker=dict(size=6),
                    customdata=df[["slide_id"]].to_numpy(),
                    hovertemplate="Slide: %{customdata[0]}<br>Artifact: %{y:.2f}%<extra>" + name + "</extra>",
                ))
            fig.add_hline(y=threshold, line_color="#EF4444", line_dash="dash", line_width=2,
                          annotation_text=f"Threshold: {threshold:.1f}%", annotation_position="top left")
            # Every slide gets an X-axis label. Grow the figure horizontally rather
            # than skipping labels when the cohort contains many slides.
            ticks = x
            chart_width = max(1100, len(df) * 95 + 250)
            fig.update_layout(
                width=chart_width,
                height=1000,
                autosize=False,
                hovermode="x unified",
                legend_title_text="Artifact type",
                xaxis=dict(
                    title="Slide", tickmode="array", tickvals=ticks,
                    ticktext=df["slide_id"].astype(str).tolist(),
                    tickangle=-65, tickfont=dict(size=11),
                    rangeslider=dict(visible=len(df) > 30),
                ),
                yaxis=dict(title="Artifact percentage (%)", rangemode="tozero"),
                margin=dict(l=65, r=35, t=45, b=180),
            )
            # Horizontal scrolling preserves legibility for large cohorts.
            with st.container(horizontal=True, horizontal_alignment="left"):
                fig = presentation_style(fig)
                st.plotly_chart(fig, width="content", key="artifact_trends_chart")
            st.caption("Every slide is labeled on the X-axis. Scroll horizontally for large cohorts; hover over points for exact values.")
            st.divider()
            st.subheader("Slide preview")
            chosen_trend = st.selectbox("Choose a slide to preview", options=df.index.tolist(),
                                        format_func=lambda i: f"{df.at[i, 'slide_id']} (row {i + 1})",
                                        key="trend_slide")
            trend_slide = df.loc[chosen_trend]
            show_slide_images(str(trend_slide["slide_id"]), thumbs, masks)
            if "artifact_pct" in df:
                st.metric("Total artifacts", f"{trend_slide['artifact_pct']:.2f}%")
            if available:
                vals = pd.DataFrame({"Artifact": list(available),
                                     "Percent": [trend_slide[col] for col in available.values()]})
                fig_bar = px.bar(vals, x="Percent", y="Artifact", color="Artifact", orientation="h",
                                 color_discrete_map=ARTIFACT_COLORS,
                                 title="Artifact composition for selected slide")
                fig_bar.update_layout(showlegend=False, yaxis=dict(categoryorder="array", categoryarray=list(available)[::-1]))
                fig = presentation_style(fig)
                st.plotly_chart(fig_bar, width="stretch")

    with tab_slides:
        chosen = st.selectbox("Choose a slide", options=df.index.tolist(),
                              format_func=lambda i: f"{df.at[i, 'slide_id']} (row {i + 1})")
        slide = df.loc[chosen]
        slide_id = str(slide["slide_id"])
        st.subheader(slide_id)
        show_slide_images(slide_id, thumbs, masks)
        p, q = st.columns(2)
        p.metric("Artifact-free tissue", f"{slide.get('artifact_free_pct', float('nan')):.2f}%")
        q.metric("Artifacts", f"{slide.get('artifact_pct', float('nan')):.2f}%")
        if available:
            values = pd.DataFrame({"Artifact": list(available),
                                   "Percent": [slide.get(col, float('nan')) for col in available.values()]})
            fig = px.bar(values, x="Percent", y="Artifact", color="Artifact", orientation="h",
                         color_discrete_map=ARTIFACT_COLORS,
                         title="Artifact composition (% of non-background)")
            fig.update_layout(showlegend=False, yaxis=dict(categoryorder="array", categoryarray=list(available)[::-1]))
            fig = presentation_style(fig)
            st.plotly_chart(fig, width="stretch")
        with st.expander("All slide metadata"):
            metadata = pd.DataFrame({"Field": slide.index.astype(str),
                                     "Value": [str(v) for v in slide.values]})
            st.dataframe(metadata, hide_index=True, width="stretch")

    with tab_data:
        subset = df.copy()
        if "artifact_pct" in subset:
            # Use the same maximum as in Artifact Trends to keep slider values valid.
            series_values = [c for c in ARTIFACT_COLUMNS.values() if c in df.columns]
            series_values += ["artifact_pct"]
            all_values = df[series_values].to_numpy(dtype=float)
            finite_values = all_values[pd.notna(all_values)]
            data_max = float(finite_values.max()) if finite_values.size else 0.0
            max_value = max(100.0, float(int(data_max + 1)))
            threshold = st.slider(
                "Maximum artifact percentage", min_value=0.0, max_value=max_value,
                step=0.5, key="threshold_data",
                on_change=sync_threshold,
                args=("threshold_data", "threshold_trends"),
            )
            subset = subset[subset["artifact_pct"] <= threshold]
        query = st.text_input("Filter by slide name", placeholder="e.g. C3N-02274")
        if query:
            subset = subset[subset["slide_id"].str.contains(query, case=False, regex=False, na=False)]
        st.caption(f"Showing {len(subset)} of {len(df)} slides")
        st.dataframe(subset, width="stretch", hide_index=True)
        st.download_button("Download filtered CSV",
                           subset.drop(columns=["slide_id"]).to_csv(index=False).encode("utf-8-sig"),
                           file_name="histokit_filtered_stats.csv", mime="text/csv")


if __name__ == "__main__":
    main()
