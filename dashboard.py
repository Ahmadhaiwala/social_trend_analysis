"""
╔══════════════════════════════════════════════════════════════╗
║         Instagram Analytics Dashboard  ·  dashboard.py      ║
║  Streamlit · Plotly · Scikit-learn · Kaggle dataset          ║
║  Run:  streamlit run dashboard.py                            ║
╚══════════════════════════════════════════════════════════════╝
"""

# ──────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ──────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# 1.  PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Instagram Analytics Dashboard",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# 2.  GLOBAL STYLES
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"], .stApp {
  font-family: 'Inter', sans-serif !important;
  background: #070B11 !important;
  color: #E8ECF1 !important;
}
header[data-testid="stHeader"] { display: none !important; }
section[data-testid="stSidebar"] > div:first-child {
  background: #0D1320 !important;
  border-right: 1px solid #1E2A3A;
}

/* ── Metric cards ── */
.ig-card {
  background: linear-gradient(145deg, #111827, #0F1A2A);
  border: 1px solid #1E2D42;
  border-radius: 16px;
  padding: 22px 24px;
  text-align: center;
  transition: transform .2s, border-color .2s;
  position: relative;
  overflow: hidden;
}
.ig-card::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(255,255,255,.02), transparent);
  pointer-events: none;
}
.ig-card:hover { transform: translateY(-3px); border-color: #2D4A6A; }
.ig-card .card-icon { font-size: 28px; margin-bottom: 8px; }
.ig-card .card-value {
  font-size: 2rem; font-weight: 800;
  background: linear-gradient(135deg, var(--c1), var(--c2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 4px 0;
}
.ig-card .card-label { font-size: 11px; color: #5A7590; text-transform: uppercase; letter-spacing: .08em; }
.ig-card .card-delta { font-size: 12px; margin-top: 6px; color: #4ADE80; font-weight: 600; }

/* ── Section headers ── */
.sec-header {
  font-size: 13px; font-weight: 700; color: #4A6A8A;
  letter-spacing: .1em; text-transform: uppercase;
  padding-bottom: 10px; margin-bottom: 16px;
  border-bottom: 1px solid #1A2535;
}

/* ── Insight cards ── */
.insight-card {
  background: linear-gradient(135deg, #0E1A2B, #111F30);
  border: 1px solid #1E3048;
  border-left: 4px solid var(--accent);
  border-radius: 12px;
  padding: 16px 20px;
  margin: 8px 0;
  font-size: 14px; line-height: 1.6;
}
.insight-card .insight-title { font-size: 11px; color: #4A6A8A; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 6px; }
.insight-card .insight-value { font-size: 15px; font-weight: 700; color: #E8ECF1; }
.insight-card .insight-sub { font-size: 12px; color: #5A7590; margin-top: 4px; }

/* ── Prediction result ── */
.pred-box {
  background: linear-gradient(135deg, #091422, #0D1E30);
  border: 1px solid #1A3D5C;
  border-radius: 14px;
  padding: 28px;
  text-align: center;
}
.pred-box .pred-label { font-size: 12px; color: #4A7AA0; text-transform: uppercase; letter-spacing: .1em; }
.pred-box .pred-value { font-size: 3.2rem; font-weight: 900; background: linear-gradient(135deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 8px 0; }
.pred-box .pred-tier { font-size: 14px; font-weight: 600; margin-top: 4px; }

/* ── Top-5 table ── */
.top5-row {
  background: #0D1625;
  border: 1px solid #1A2D42;
  border-radius: 10px;
  padding: 12px 16px;
  margin: 6px 0;
  display: flex;
  align-items: center;
  gap: 14px;
  font-size: 13px;
}
.top5-rank { width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 12px; flex-shrink: 0; }

/* ── A/B compare ── */
.ab-card {
  background: #0D1625;
  border: 1px solid #1A2D42;
  border-radius: 14px;
  padding: 20px;
}
.ab-winner { border-color: #10B981 !important; }

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, #2563EB, #7C3AED) !important;
  color: white !important; border: none !important;
  border-radius: 10px !important; font-weight: 600 !important;
  padding: 10px 28px !important; font-size: 14px !important;
  transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .85 !important; }

/* ── Inputs ── */
.stSelectbox > div > div, .stSlider > div, .stNumberInput > div > div {
  background: #0D1625 !important;
  border-color: #1E2D42 !important;
  border-radius: 10px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: #0D1320; border-radius: 12px; padding: 6px;
  border: 1px solid #1A2535; gap: 6px;
}
.stTabs [data-baseweb="tab"] {
  background: transparent; border-radius: 8px; color: #5A7590;
  font-weight: 500; font-size: 14px; padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, #1E3A5F22, #2D1B6922) !important;
  color: #38BDF8 !important; border: 1px solid #38BDF855 !important;
}

/* ── Progress bars ── */
.viral-bar { height: 10px; border-radius: 999px; background: #1A2535; overflow: hidden; margin: 4px 0 12px; }
.viral-bar-fill { height: 100%; border-radius: 999px; }

/* ── Divider ── */
hr { border-color: #1A2535 !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 3.  DATA LOADING + MODEL TRAINING  (cached)
# ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading dataset & training model…")
def load_and_train():
    """
    Loads the Kaggle Instagram Analytics dataset via kagglehub,
    preprocesses it, and trains a Random Forest pipeline.
    Returns (raw_df, processed_df, model_pipeline, feature_names, metrics).
    """
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    import os

    dataset_id = "kundanbedmutha/instagram-analytics-dataset"

    # ── Load raw dataframe ──────────────────────────────────
    path = kagglehub.dataset_download(dataset_id)
    csv_file = next((f for f in os.listdir(path) if f.endswith(".csv")), None)
    if not csv_file:
        raise FileNotFoundError("No CSV found in Kaggle dataset download.")

    raw_df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, dataset_id, csv_file)

    # ── Normalise column names ──────────────────────────────
    raw_df.columns = raw_df.columns.str.strip().str.lower()

    # ── Parse datetime & extract time features ──────────────
    raw_df["post_datetime"] = pd.to_datetime(raw_df["post_datetime"], errors="coerce")
    raw_df["hour"]  = raw_df["post_datetime"].dt.hour
    raw_df["day"]   = raw_df["post_datetime"].dt.day_name()
    raw_df["month"] = raw_df["post_datetime"].dt.month_name()
    raw_df["month_num"] = raw_df["post_datetime"].dt.month

    # ── Viral score (likes + 2×shares + 3×saves) ───────────
    if all(c in raw_df.columns for c in ["likes", "shares", "saves"]):
        raw_df["viral_score"] = raw_df["likes"] + 2 * raw_df["shares"] + 3 * raw_df["saves"]
    else:
        raw_df["viral_score"] = 0

    # ── Keep a clean analytics copy (before dropping cols) ──
    analytics_df = raw_df.copy()
    analytics_df = analytics_df.dropna(subset=["engagement_rate"])

    # ── Model training copy ─────────────────────────────────
    drop_cols = ["post_id", "account_id", "post_datetime", "post_date",
                 "performance_bucket_label", "day", "month", "viral_score"]
    model_df = raw_df.drop(columns=drop_cols, errors="ignore").dropna()

    y = model_df["engagement_rate"]
    X = model_df.drop(columns=["engagement_rate"])

    numeric_features   = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    metrics = {
        "mae":  round(mean_absolute_error(y_test, y_pred_test), 5),
        "r2":   round(r2_score(y_test, y_pred_test), 4),
        "features": X.columns.tolist(),
        "num_features": numeric_features,
        "cat_features": categorical_features,
    }

    return analytics_df, model, metrics


# ── Load everything ─────────────────────────────────────────
try:
    df, model, model_metrics = load_and_train()
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()


# ──────────────────────────────────────────────────────────────
# 4.  PLOTLY THEME HELPER
# ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#8BA4BF"),
    margin=dict(t=30, b=10, l=10, r=10),
    xaxis=dict(gridcolor="#111F32", linecolor="#1A2D42", tickfont=dict(color="#6A8AA8")),
    yaxis=dict(gridcolor="#111F32", linecolor="#1A2D42", tickfont=dict(color="#6A8AA8")),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8BA4BF")),
)

PALETTE = ["#38BDF8", "#818CF8", "#34D399", "#F472B6", "#FB923C",
           "#A78BFA", "#22D3EE", "#4ADE80", "#FBBF24", "#F87171"]

def apply_layout(fig, height=320, **kwargs):
    layout = {**PLOTLY_LAYOUT, "height": height, **kwargs}
    fig.update_layout(**layout)
    return fig


# ──────────────────────────────────────────────────────────────
# 5.  SIDEBAR  — GLOBAL FILTERS
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 16px;">
      <div style="font-size:26px;">📸</div>
      <div style="font-size:16px;font-weight:800;background:linear-gradient(135deg,#38BDF8,#818CF8);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
           margin-top:4px;">Instagram Analytics</div>
      <div style="font-size:11px;color:#3A5570;margin-top:2px;">Performance Intelligence</div>
    </div>
    <hr style="border-color:#1A2535;margin:0 0 16px;">
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-header">🎛️ Filters</div>', unsafe_allow_html=True)

    # Content category
    all_cats = sorted(df["content_category"].dropna().unique())
    sel_cats = st.multiselect("Content Category", all_cats, default=all_cats,
                               help="Filter by content category")

    # Media type
    all_media = sorted(df["media_type"].dropna().unique())
    sel_media = st.multiselect("Media Type", all_media, default=all_media)

    # Account type
    all_acct = sorted(df["account_type"].dropna().unique())
    sel_acct = st.multiselect("Account Type", all_acct, default=all_acct)

    # Follower count range
    fc_min, fc_max = int(df["follower_count"].min()), int(df["follower_count"].max())
    fc_range = st.slider("Follower Count Range",
                          min_value=fc_min, max_value=fc_max,
                          value=(fc_min, fc_max), step=500,
                          format="%d")

    # Date range
    if "post_datetime" in df.columns:
        d_min = df["post_datetime"].min().date()
        d_max = df["post_datetime"].max().date()
        date_range = st.date_input("Date Range", value=(d_min, d_max),
                                   min_value=d_min, max_value=d_max)
    else:
        date_range = None

    st.markdown("---")
    st.markdown('<div class="sec-header">🤖 Model Performance</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex;gap:10px;flex-wrap:wrap;">
      <div style="background:#0D1625;border:1px solid #1A2D42;border-radius:10px;padding:10px 14px;flex:1;text-align:center;">
        <div style="font-size:10px;color:#4A6A8A;text-transform:uppercase;">MAE</div>
        <div style="font-size:18px;font-weight:800;color:#38BDF8;">{model_metrics['mae']}</div>
      </div>
      <div style="background:#0D1625;border:1px solid #1A2D42;border-radius:10px;padding:10px 14px;flex:1;text-align:center;">
        <div style="font-size:10px;color:#4A6A8A;text-transform:uppercase;">R²</div>
        <div style="font-size:18px;font-weight:800;color:#818CF8;">{model_metrics['r2']}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 6.  APPLY FILTERS
# ──────────────────────────────────────────────────────────────
fdf = df.copy()
if sel_cats:
    fdf = fdf[fdf["content_category"].isin(sel_cats)]
if sel_media:
    fdf = fdf[fdf["media_type"].isin(sel_media)]
if sel_acct:
    fdf = fdf[fdf["account_type"].isin(sel_acct)]
fdf = fdf[(fdf["follower_count"] >= fc_range[0]) & (fdf["follower_count"] <= fc_range[1])]
if date_range and len(date_range) == 2:
    fdf = fdf[(fdf["post_datetime"].dt.date >= date_range[0]) &
              (fdf["post_datetime"].dt.date <= date_range[1])]

if fdf.empty:
    st.warning("⚠️ No data matches your current filters. Please adjust them in the sidebar.")
    st.stop()


# ──────────────────────────────────────────────────────────────
# 7.  HERO HEADER
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:36px 0 28px;text-align:center;">
  <div style="font-size:2.4rem;font-weight:900;
       background:linear-gradient(135deg,#38BDF8 0%,#818CF8 50%,#F472B6 100%);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
       letter-spacing:-0.02em;line-height:1.1;">
    Instagram Analytics Dashboard
  </div>
  <div style="color:#3A5A7A;font-size:14px;margin-top:8px;font-weight:400;">
    Data-driven insights · ML-powered predictions · Real-time trend analysis
  </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 8.  KPI METRIC CARDS
# ──────────────────────────────────────────────────────────────
avg_eng   = fdf["engagement_rate"].mean()
avg_likes = fdf["likes"].mean() if "likes" in fdf.columns else 0
tot_reach = fdf["reach"].sum() if "reach" in fdf.columns else 0
tot_impr  = fdf["impressions"].sum() if "impressions" in fdf.columns else 0
avg_viral = fdf["viral_score"].mean() if "viral_score" in fdf.columns else 0

c1, c2, c3, c4, c5 = st.columns(5)

def kpi_html(icon, value, label, c1="#38BDF8", c2="#818CF8", delta=""):
    delta_html = f'<div class="card-delta">{delta}</div>' if delta else ""
    return f"""
    <div class="ig-card" style="--c1:{c1};--c2:{c2};">
      <div class="card-icon">{icon}</div>
      <div class="card-value">{value}</div>
      <div class="card-label">{label}</div>
      {delta_html}
    </div>
    """

with c1:
    st.markdown(kpi_html("📊", f"{avg_eng:.2%}", "Avg Engagement Rate",
                          "#38BDF8", "#818CF8"), unsafe_allow_html=True)
with c2:
    st.markdown(kpi_html("❤️", f"{avg_likes:,.0f}", "Avg Likes",
                          "#F472B6", "#FB7185"), unsafe_allow_html=True)
with c3:
    st.markdown(kpi_html("👁️", f"{tot_reach/1e6:.1f}M", "Total Reach",
                          "#34D399", "#10B981"), unsafe_allow_html=True)
with c4:
    st.markdown(kpi_html("📣", f"{tot_impr/1e6:.1f}M", "Total Impressions",
                          "#A78BFA", "#7C3AED"), unsafe_allow_html=True)
with c5:
    st.markdown(kpi_html("🔥", f"{avg_viral:,.0f}", "Avg Viral Score",
                          "#FB923C", "#F59E0B"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 9.  SMART INSIGHTS STRIP
# ──────────────────────────────────────────────────────────────
best_hour_row = fdf.groupby("hour")["engagement_rate"].mean().idxmax()
best_hour_rate = fdf.groupby("hour")["engagement_rate"].mean().max()
best_cat  = fdf.groupby("content_category")["engagement_rate"].mean().idxmax()
best_cat_rate = fdf.groupby("content_category")["engagement_rate"].mean().max()
best_media = fdf.groupby("media_type")["engagement_rate"].mean().idxmax()
best_media_rate = fdf.groupby("media_type")["engagement_rate"].mean().max()
best_day  = fdf.groupby("day")["engagement_rate"].mean().idxmax() if "day" in fdf.columns else "N/A"

st.markdown('<div class="sec-header">💡 Smart Insights</div>', unsafe_allow_html=True)
ins1, ins2, ins3, ins4 = st.columns(4)

def insight_html(title, value, sub, accent="#38BDF8"):
    return f"""
    <div class="insight-card" style="--accent:{accent};">
      <div class="insight-title">{title}</div>
      <div class="insight-value">{value}</div>
      <div class="insight-sub">{sub}</div>
    </div>
    """

with ins1:
    hour_label = f"{best_hour_row}:00" if best_hour_row < 12 else f"{best_hour_row}:00"
    am_pm = "AM" if best_hour_row < 12 else "PM"
    st.markdown(insight_html("🕐 Best Time to Post",
                              f"{best_hour_row:02d}:00 {am_pm}",
                              f"Avg {best_hour_rate:.2%} engagement", "#38BDF8"),
                unsafe_allow_html=True)
with ins2:
    st.markdown(insight_html("🗂️ Best Content Category",
                              best_cat,
                              f"Avg {best_cat_rate:.2%} engagement", "#818CF8"),
                unsafe_allow_html=True)
with ins3:
    st.markdown(insight_html("🎬 Best Media Format",
                              best_media.title(),
                              f"Avg {best_media_rate:.2%} engagement", "#34D399"),
                unsafe_allow_html=True)
with ins4:
    st.markdown(insight_html("📅 Best Day to Post",
                              best_day,
                              "Highest avg engagement rate", "#F472B6"),
                unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 10.  TABS
# ──────────────────────────────────────────────────────────────
tab_trend, tab_content, tab_predict, tab_ab, tab_top5 = st.tabs([
    "📈 Trend Analysis",
    "🎯 Content Performance",
    "🤖 ML Prediction",
    "🆚 A/B Compare",
    "🏆 Top Posts",
])


# ════════════════════════════════════════════════════════════════
# TAB 1 — TREND ANALYSIS
# ════════════════════════════════════════════════════════════════
with tab_trend:
    st.markdown('<div class="sec-header">📈 Engagement Trends</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    # ── Engagement by Hour ──────────────────────────────────
    with col_left:
        hour_data = (fdf.groupby("hour")["engagement_rate"]
                     .mean().reset_index()
                     .sort_values("hour"))
        best_h = hour_data.loc[hour_data["engagement_rate"].idxmax(), "hour"]
        colors = ["#F59E0B" if h == best_h else "#38BDF8" for h in hour_data["hour"]]

        fig = go.Figure(go.Bar(
            x=hour_data["hour"],
            y=hour_data["engagement_rate"],
            marker_color=colors,
            text=[f"{v:.2%}" for v in hour_data["engagement_rate"]],
            textposition="outside",
            textfont=dict(color="#E8ECF1", size=10),
            hovertemplate="Hour %{x}:00<br>Engagement: %{y:.3f}<extra></extra>",
        ))
        apply_layout(fig, height=310,
                     title="Engagement Rate by Posting Hour",
                     xaxis_title="Hour of Day",
                     yaxis_title="Avg Engagement Rate")
        fig.update_traces(marker_line_color="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Monthly Trend ───────────────────────────────────────
    with col_right:
        if "month_num" in fdf.columns and "month" in fdf.columns:
            month_data = (fdf.groupby(["month_num", "month"])["engagement_rate"]
                          .mean().reset_index()
                          .sort_values("month_num"))
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=month_data["month"],
                y=month_data["engagement_rate"],
                mode="lines+markers",
                line=dict(color="#818CF8", width=3),
                marker=dict(size=9, color="#818CF8",
                            line=dict(color="#070B11", width=2)),
                fill="tozeroy",
                fillcolor="rgba(129,140,248,0.08)",
                hovertemplate="%{x}<br>Engagement: %{y:.3f}<extra></extra>",
            ))
            apply_layout(fig2, height=310,
                         title="Monthly Engagement Trend",
                         xaxis_title="Month",
                         yaxis_title="Avg Engagement Rate")
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ── Hashtag Count vs Engagement ─────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        hash_data = (fdf.groupby("hashtags_count")["engagement_rate"]
                     .mean().reset_index().sort_values("hashtags_count"))
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=hash_data["hashtags_count"],
            y=hash_data["engagement_rate"],
            mode="lines+markers",
            line=dict(color="#34D399", width=3),
            marker=dict(size=8, color="#34D399",
                        line=dict(color="#070B11", width=2)),
            fill="tozeroy",
            fillcolor="rgba(52,211,153,0.08)",
            hovertemplate="Hashtags: %{x}<br>Engagement: %{y:.3f}<extra></extra>",
        ))
        apply_layout(fig3, height=310,
                     title="Hashtag Count vs Engagement Rate",
                     xaxis_title="Number of Hashtags",
                     yaxis_title="Avg Engagement Rate")
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # ── Followers vs Likes scatter ──────────────────────────
    with col4:
        sample_scatter = fdf.sample(min(2000, len(fdf)), random_state=42)
        fig4 = go.Figure(go.Scatter(
            x=sample_scatter["follower_count"],
            y=sample_scatter["likes"] if "likes" in sample_scatter.columns else sample_scatter["engagement_rate"],
            mode="markers",
            marker=dict(
                color=sample_scatter["engagement_rate"],
                colorscale=[[0, "#111F32"], [0.5, "#818CF8"], [1, "#38BDF8"]],
                size=5,
                opacity=0.7,
                colorbar=dict(title="Eng Rate", tickfont=dict(color="#6A8AA8")),
                showscale=True,
            ),
            hovertemplate="Followers: %{x:,}<br>Likes: %{y:,}<br>Engagement: %{marker.color:.3f}<extra></extra>",
        ))
        apply_layout(fig4, height=310,
                     title="Followers vs Likes (colour = engagement)",
                     xaxis_title="Follower Count",
                     yaxis_title="Likes")
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    # ── Engagement by Day of Week ───────────────────────────
    if "day" in fdf.columns:
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        day_data = (fdf.groupby("day")["engagement_rate"].mean()
                    .reindex(day_order).dropna().reset_index())
        fig5 = go.Figure(go.Bar(
            x=day_data["day"],
            y=day_data["engagement_rate"],
            marker=dict(
                color=day_data["engagement_rate"],
                colorscale=[[0,"#0D1625"],[0.5,"#818CF8"],[1,"#38BDF8"]],
                line=dict(color="rgba(0,0,0,0)"),
            ),
            text=[f"{v:.2%}" for v in day_data["engagement_rate"]],
            textposition="outside",
            textfont=dict(color="#E8ECF1", size=11),
            hovertemplate="%{x}<br>Engagement: %{y:.3f}<extra></extra>",
        ))
        apply_layout(fig5, height=290,
                     title="Engagement Rate by Day of Week",
                     xaxis_title="Day", yaxis_title="Avg Engagement Rate")
        st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════════
# TAB 2 — CONTENT PERFORMANCE
# ════════════════════════════════════════════════════════════════
with tab_content:
    st.markdown('<div class="sec-header">🎯 Content Performance Analysis</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    # ── By Content Category ─────────────────────────────────
    with col_a:
        cat_data = (fdf.groupby("content_category")["engagement_rate"]
                    .mean().sort_values(ascending=True).reset_index())
        fig_cat = go.Figure(go.Bar(
            x=cat_data["engagement_rate"],
            y=cat_data["content_category"],
            orientation="h",
            marker=dict(
                color=PALETTE[:len(cat_data)],
                line=dict(color="rgba(0,0,0,0)"),
            ),
            text=[f"{v:.2%}" for v in cat_data["engagement_rate"]],
            textposition="outside",
            textfont=dict(color="#E8ECF1", size=10),
            hovertemplate="%{y}<br>Engagement: %{x:.3f}<extra></extra>",
        ))
        apply_layout(fig_cat, height=340,
                     title="Avg Engagement Rate by Content Category",
                     xaxis_title="Avg Engagement Rate", yaxis_title="")
        st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar": False})

    # ── By Media Type ───────────────────────────────────────
    with col_b:
        media_data = (fdf.groupby("media_type")["engagement_rate"]
                      .mean().sort_values(ascending=False).reset_index())
        fig_media = go.Figure(go.Bar(
            x=media_data["media_type"],
            y=media_data["engagement_rate"],
            marker=dict(
                color=["#38BDF8","#818CF8","#34D399","#F472B6"][:len(media_data)],
                line=dict(color="rgba(0,0,0,0)"),
            ),
            text=[f"{v:.2%}" for v in media_data["engagement_rate"]],
            textposition="outside",
            textfont=dict(color="#E8ECF1", size=11),
            hovertemplate="%{x}<br>Engagement: %{y:.3f}<extra></extra>",
        ))
        apply_layout(fig_media, height=340,
                     title="Avg Engagement Rate by Media Type",
                     xaxis_title="Media Type", yaxis_title="Avg Engagement Rate")
        st.plotly_chart(fig_media, use_container_width=True, config={"displayModeBar": False})

    # ── Traffic Source breakdown ────────────────────────────
    col_c, col_d = st.columns(2)
    with col_c:
        if "traffic_source" in fdf.columns:
            ts_data = (fdf.groupby("traffic_source")["engagement_rate"]
                       .mean().sort_values(ascending=False).reset_index())
            fig_ts = go.Figure(go.Bar(
                x=ts_data["traffic_source"],
                y=ts_data["engagement_rate"],
                marker=dict(color=PALETTE[:len(ts_data)],
                            line=dict(color="rgba(0,0,0,0)")),
                text=[f"{v:.2%}" for v in ts_data["engagement_rate"]],
                textposition="outside",
                textfont=dict(color="#E8ECF1", size=10),
            ))
            apply_layout(fig_ts, height=310,
                         title="Engagement by Traffic Source",
                         xaxis_title="Traffic Source", yaxis_title="Avg Engagement Rate")
            st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": False})

    # ── CTA impact ──────────────────────────────────────────
    with col_d:
        if "has_call_to_action" in fdf.columns:
            cta_data = (fdf.groupby("has_call_to_action")["engagement_rate"]
                        .mean().reset_index())
            cta_data["label"] = cta_data["has_call_to_action"].map({0: "No CTA", 1: "Has CTA"})
            fig_cta = go.Figure(go.Pie(
                labels=cta_data["label"],
                values=cta_data["engagement_rate"],
                hole=0.55,
                marker=dict(colors=["#1E2D42","#38BDF8"],
                            line=dict(color="#070B11", width=3)),
                textfont=dict(color="#E8ECF1"),
                hovertemplate="%{label}<br>Avg Engagement: %{value:.3f}<extra></extra>",
            ))
            apply_layout(fig_cta, height=310,
                         title="CTA Impact on Engagement")
            fig_cta.update_layout(showlegend=True)
            st.plotly_chart(fig_cta, use_container_width=True, config={"displayModeBar": False})

    # ── Account type comparison ─────────────────────────────
    if "account_type" in fdf.columns:
        acct_data = (fdf.groupby(["account_type","media_type"])["engagement_rate"]
                     .mean().reset_index())
        fig_acct = px.bar(
            acct_data, x="media_type", y="engagement_rate",
            color="account_type",
            barmode="group",
            color_discrete_sequence=PALETTE,
            labels={"engagement_rate":"Avg Engagement Rate","media_type":"Media Type","account_type":"Account Type"},
        )
        apply_layout(fig_acct, height=290,
                     title="Engagement by Account Type × Media Type")
        fig_acct.update_traces(marker_line_color="rgba(0,0,0,0)")
        st.plotly_chart(fig_acct, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════════
# TAB 3 — ML PREDICTION
# ════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown('<div class="sec-header">🤖 Predict Engagement Rate</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#4A6A8A;font-size:13px;margin-bottom:24px;">Enter post attributes below and the Random Forest model will predict the expected engagement rate.</div>', unsafe_allow_html=True)

    # ── Build input form matching model training features ───
    all_num_cols  = model_metrics["num_features"]
    all_cat_cols  = model_metrics["cat_features"]
    all_feat_cols = model_metrics["features"]

    with st.form("predict_form"):
        fc1, fc2 = st.columns(2)

        with fc1:
            st.markdown("**📊 Numeric Inputs**")
            follower_count  = st.number_input("Follower Count", min_value=100, max_value=500_000,
                                               value=10000, step=500)
            likes_val       = st.number_input("Likes", min_value=0, max_value=100_000, value=500, step=10) \
                               if "likes" in all_num_cols else 500
            comments_val    = st.number_input("Comments", min_value=0, max_value=10_000, value=20, step=1)
            shares_val      = st.number_input("Shares", min_value=0, max_value=5_000, value=10, step=1) \
                               if "shares" in all_num_cols else 10
            saves_val       = st.number_input("Saves", min_value=0, max_value=5_000, value=30, step=1) \
                               if "saves" in all_num_cols else 30
            reach_val       = st.number_input("Reach", min_value=0, max_value=500_000, value=8000, step=100) \
                               if "reach" in all_num_cols else 8000
            impressions_val = st.number_input("Impressions", min_value=0, max_value=1_000_000, value=12000, step=100) \
                               if "impressions" in all_num_cols else 12000

        with fc2:
            st.markdown("**🏷️ Categorical Inputs**")
            hashtags_count  = st.slider("Hashtag Count", 0, 30, 7)
            caption_length  = st.slider("Caption Length (chars)", 0, 300, 120) \
                               if "caption_length" in all_num_cols else 120
            hour_val        = st.slider("Post Hour (0–23)", 0, 23, 12) \
                               if "hour" in all_num_cols else 12
            day_val         = st.slider("Day of Month", 1, 31, 15) \
                               if "day" in all_num_cols else 15
            month_val_num   = st.slider("Month", 1, 12, 6) \
                               if "month_num" in all_num_cols else 6

            media_type_val  = st.selectbox("Media Type", sorted(df["media_type"].dropna().unique()))
            content_cat_val = st.selectbox("Content Category", sorted(df["content_category"].dropna().unique()))
            account_type_val= st.selectbox("Account Type", sorted(df["account_type"].dropna().unique())) \
                               if "account_type" in all_cat_cols else "creator"
            traffic_src_val = st.selectbox("Traffic Source", sorted(df["traffic_source"].dropna().unique())) \
                               if "traffic_source" in all_cat_cols else "Home Feed"
            cta_val         = st.selectbox("Has Call to Action", [0, 1], format_func=lambda x: "Yes" if x else "No") \
                               if "has_call_to_action" in all_num_cols else 1
            followers_gained= st.number_input("Followers Gained", 0, 5000, 200, 10) \
                               if "followers_gained" in all_num_cols else 200

        predict_btn = st.form_submit_button("🔮 Predict Engagement Rate", use_container_width=True)

    if predict_btn:
        # Build input dict matching all features
        input_dict = {}
        for col in all_feat_cols:
            val_map = {
                "follower_count":    follower_count,
                "likes":             likes_val,
                "comments":          comments_val,
                "shares":            shares_val,
                "saves":             saves_val,
                "reach":             reach_val,
                "impressions":       impressions_val,
                "hashtags_count":    hashtags_count,
                "caption_length":    caption_length,
                "hour":              hour_val,
                "day":               day_val,
                "month_num":         month_val_num,
                "has_call_to_action":cta_val,
                "followers_gained":  followers_gained,
                "media_type":        media_type_val,
                "content_category":  content_cat_val,
                "account_type":      account_type_val,
                "traffic_source":    traffic_src_val,
            }
            input_dict[col] = val_map.get(col, 0)

        input_df = pd.DataFrame([input_dict])
        # Ensure correct dtypes for categorical columns
        for c in all_cat_cols:
            if c in input_df.columns:
                input_df[c] = input_df[c].astype(str)

        try:
            pred = model.predict(input_df)[0]
            pred = max(0.0, pred)

            # Tier classification
            if pred >= 0.08:
                tier, tier_color = "🔥 VIRAL", "#F59E0B"
                tips = "Exceptional! This post has viral potential. Consider boosting it."
            elif pred >= 0.05:
                tier, tier_color = "🚀 HIGH", "#34D399"
                tips = "Strong engagement expected. Great content strategy!"
            elif pred >= 0.03:
                tier, tier_color = "📈 MEDIUM", "#38BDF8"
                tips = "Above-average performance. Try adding a stronger CTA or question."
            elif pred >= 0.01:
                tier, tier_color = "📊 LOW", "#818CF8"
                tips = "Below average. Consider a different media type or posting time."
            else:
                tier, tier_color = "⚠️ VERY LOW", "#F87171"
                tips = "Very low predicted engagement. Rethink content strategy."

            # Viral score calculation
            v_score = likes_val + 2 * shares_val + 3 * saves_val

            p1, p2, p3 = st.columns([2, 1, 1])
            with p1:
                st.markdown(f"""
                <div class="pred-box">
                  <div class="pred-label">Predicted Engagement Rate</div>
                  <div class="pred-value">{pred:.4f}</div>
                  <div class="pred-label">({pred:.2%})</div>
                  <div class="pred-tier" style="color:{tier_color};margin-top:12px;">{tier}</div>
                  <div style="font-size:12px;color:#3A5A7A;margin-top:10px;line-height:1.5;">{tips}</div>
                </div>
                """, unsafe_allow_html=True)

            with p2:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pred * 100,
                    number={"suffix": "%", "font": {"size": 26, "color": "#E8ECF1", "family": "Inter"}},
                    title={"text": "Engagement %", "font": {"size": 12, "color": "#4A6A8A"}},
                    gauge={
                        "axis": {"range": [0, 15], "tickcolor": "#1A2D42",
                                 "tickfont": {"color": "#4A6A8A"}},
                        "bar": {"color": "#38BDF8", "thickness": 0.7},
                        "bgcolor": "#0D1625",
                        "bordercolor": "#1A2D42",
                        "steps": [
                            {"range": [0, 3],  "color": "#0D1625"},
                            {"range": [3, 8],  "color": "#0F1A2A"},
                            {"range": [8, 15], "color": "#111F30"},
                        ],
                    },
                ))
                gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=220,
                    margin=dict(t=30, b=0, l=20, r=20),
                    font={"family": "Inter"},
                )
                st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

            with p3:
                st.markdown(f"""
                <div style="background:#0D1625;border:1px solid #1A2D42;border-radius:12px;padding:18px;height:220px;display:flex;flex-direction:column;justify-content:center;gap:10px;">
                  <div>
                    <div style="font-size:10px;color:#4A6A8A;text-transform:uppercase;letter-spacing:.08em;">Viral Score</div>
                    <div style="font-size:22px;font-weight:800;color:#FB923C;">{v_score:,}</div>
                  </div>
                  <div>
                    <div style="font-size:10px;color:#4A6A8A;text-transform:uppercase;letter-spacing:.08em;">vs Dataset Avg</div>
                    <div style="font-size:22px;font-weight:800;color:{'#34D399' if pred >= avg_eng else '#F87171'};">
                      {'▲' if pred >= avg_eng else '▼'} {abs(pred - avg_eng):.4f}
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as ex:
            st.error(f"Prediction error: {ex}")


# ════════════════════════════════════════════════════════════════
# TAB 4 — A/B COMPARE
# ════════════════════════════════════════════════════════════════
with tab_ab:
    st.markdown('<div class="sec-header">🆚 A/B Content Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#4A6A8A;font-size:13px;margin-bottom:20px;">Compare two content strategies side-by-side to see which performs better on your selected metric.</div>', unsafe_allow_html=True)

    ab1, ab2 = st.columns(2)
    all_cats_ab = sorted(fdf["content_category"].dropna().unique())
    all_media_ab = sorted(fdf["media_type"].dropna().unique())

    with ab1:
        st.markdown("**Strategy A**")
        cat_a  = st.selectbox("Content Category A", all_cats_ab, key="cat_a")
        med_a  = st.selectbox("Media Type A", all_media_ab, key="med_a")

    with ab2:
        st.markdown("**Strategy B**")
        cat_b  = st.selectbox("Content Category B", all_cats_ab,
                               index=min(1, len(all_cats_ab)-1), key="cat_b")
        med_b  = st.selectbox("Media Type B", all_media_ab,
                               index=min(1, len(all_media_ab)-1), key="med_b")

    run_ab = st.button("▶ Run Comparison")

    if run_ab:
        group_a = fdf[(fdf["content_category"]==cat_a) & (fdf["media_type"]==med_a)]
        group_b = fdf[(fdf["content_category"]==cat_b) & (fdf["media_type"]==med_b)]

        def safe_mean(grp, col):
            return grp[col].mean() if len(grp) > 0 and col in grp.columns else 0

        metrics_list = ["engagement_rate", "likes", "shares", "saves", "reach", "viral_score"]
        labels_nice  = ["Engagement Rate", "Likes", "Shares", "Saves", "Reach", "Viral Score"]

        a_vals = [safe_mean(group_a, m) for m in metrics_list]
        b_vals = [safe_mean(group_b, m) for m in metrics_list]

        eng_a, eng_b = a_vals[0], b_vals[0]
        winner = "A" if eng_a >= eng_b else "B"
        lift   = abs(eng_b - eng_a) / max(eng_a, 0.0001) * 100

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0D1625,#111F30);border:1px solid {'#10B981' if winner=='B' else '#38BDF8'};
             border-radius:12px;padding:16px 22px;text-align:center;margin:16px 0;">
          <span style="font-size:15px;font-weight:700;color:{'#10B981' if winner=='B' else '#38BDF8'};">
            🏆 Strategy {winner} wins — {lift:.1f}% higher engagement rate
          </span>
        </div>
        """, unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        for col_idx, (col_widget, label, a_val, b_val, is_win) in enumerate(zip(
            [r1, r2],
            [f"Strategy A: {cat_a} · {med_a}", f"Strategy B: {cat_b} · {med_b}"],
            [a_vals, b_vals],
            [b_vals, a_vals],
            [winner=="A", winner=="B"],
        )):
            with col_widget:
                win_cls = "ab-winner" if is_win else ""
                badge = "🏆 WINNER" if is_win else ""
                data_html = "".join([
                    f"""<div style="display:flex;justify-content:space-between;padding:7px 0;
                    border-bottom:1px solid #1A2535;font-size:13px;">
                    <span style="color:#5A7590;">{nm}</span>
                    <span style="font-weight:700;color:{'#34D399' if av>=bv else '#E8ECF1'};">
                      {av:,.3f if nm=='Engagement Rate' else av:,.0f}
                    </span></div>"""
                    for nm, av, bv in zip(labels_nice, a_val, b_val)
                ])
                st.markdown(f"""
                <div class="ab-card {win_cls}">
                  <div style="font-size:13px;font-weight:700;color:#E8ECF1;margin-bottom:12px;">
                    {label}&nbsp;&nbsp;<span style="color:#10B981;font-size:12px;">{badge}</span>
                  </div>
                  {data_html}
                  <div style="font-size:11px;color:#3A5570;margin-top:10px;">n = {len(group_a if col_idx==0 else group_b):,} posts</div>
                </div>
                """, unsafe_allow_html=True)

        # Radar chart comparison
        fig_radar = go.Figure()
        norm_a = [v / max(max(a_vals[i], b_vals[i]), 0.0001) * 100 for i, v in enumerate(a_vals)]
        norm_b = [v / max(max(a_vals[i], b_vals[i]), 0.0001) * 100 for i, v in enumerate(b_vals)]

        for vals, name, color in [(norm_a, f"A: {cat_a}·{med_a}", "#38BDF8"),
                                   (norm_b, f"B: {cat_b}·{med_b}", "#34D399")]:
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=labels_nice + [labels_nice[0]],
                fill="toself",
                fillcolor=f"rgba({','.join(['56,189,248' if color=='#38BDF8' else '52,211,153'])},0.12)",
                line=dict(color=color, width=2),
                name=name,
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100],
                                color="#4A6A8A", gridcolor="#1A2D42"),
                angularaxis=dict(color="#4A6A8A", gridcolor="#1A2D42"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8BA4BF")),
            font=dict(family="Inter", color="#8BA4BF"),
            height=340,
            margin=dict(t=30, b=20, l=40, r=40),
            title="Normalized Performance Radar",
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════════
# TAB 5 — TOP 5 POSTS
# ════════════════════════════════════════════════════════════════
with tab_top5:
    st.markdown('<div class="sec-header">🏆 Top 5 Performing Posts</div>', unsafe_allow_html=True)

    rank_by = st.radio("Rank by", ["Engagement Rate", "Viral Score", "Likes", "Reach"],
                        horizontal=True)
    rank_col_map = {
        "Engagement Rate": "engagement_rate",
        "Viral Score":     "viral_score",
        "Likes":           "likes",
        "Reach":           "reach",
    }
    rank_col = rank_col_map[rank_by]
    if rank_col not in fdf.columns:
        rank_col = "engagement_rate"

    top5 = fdf.nlargest(5, rank_col).reset_index(drop=True)
    rank_colors = ["#F59E0B","#9CA3AF","#CD7F32","#38BDF8","#818CF8"]
    rank_medals = ["🥇","🥈","🥉","4️⃣","5️⃣"]

    for i, row in top5.iterrows():
        media_icon = {"reel":"🎬","image":"🖼️","carousel":"🎠"}.get(
            str(row.get("media_type","")).lower(), "📄")
        eng_pct = f"{row.get('engagement_rate',0):.2%}"
        vir_s   = f"{row.get('viral_score',0):,.0f}"
        reach_s = f"{row.get('reach',0):,.0f}"
        likes_s = f"{row.get('likes',0):,.0f}"
        cat_s   = str(row.get("content_category","N/A"))
        media_s = str(row.get("media_type","N/A"))
        hour_s  = f"{int(row['hour'])}:00" if pd.notna(row.get("hour")) else "N/A"
        acct_s  = str(row.get("account_type","N/A"))

        st.markdown(f"""
        <div class="top5-row">
          <div class="top5-rank" style="background:{rank_colors[i]}22;color:{rank_colors[i]};">
            {rank_medals[i]}
          </div>
          <div style="flex:1;">
            <div style="font-weight:700;color:#E8ECF1;font-size:14px;">
              {media_icon} {cat_s} · {media_s.title()}
            </div>
            <div style="font-size:11px;color:#4A6A8A;margin-top:3px;">
              {acct_s} account · posted at {hour_s}
            </div>
          </div>
          <div style="text-align:right;">
            <div style="font-size:18px;font-weight:800;color:{rank_colors[i]};">{eng_pct}</div>
            <div style="font-size:11px;color:#4A6A8A;">engagement rate</div>
          </div>
          <div style="text-align:right;min-width:90px;">
            <div style="font-size:13px;font-weight:700;color:#F472B6;">❤️ {likes_s}</div>
            <div style="font-size:11px;color:#4A6A8A;">👁️ {reach_s}</div>
            <div style="font-size:11px;color:#FB923C;">🔥 {vir_s}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Viral Score distribution chart ──────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-header">📊 Viral Score Distribution</div>', unsafe_allow_html=True)
    fig_vir = go.Figure(go.Histogram(
        x=fdf["viral_score"].clip(upper=fdf["viral_score"].quantile(0.99)),
        nbinsx=50,
        marker=dict(
            color="#818CF8",
            line=dict(color="rgba(0,0,0,0)"),
        ),
        hovertemplate="Viral Score: %{x}<br>Count: %{y}<extra></extra>",
    ))
    apply_layout(fig_vir, height=280,
                 title="Viral Score Distribution (99th percentile cap)",
                 xaxis_title="Viral Score", yaxis_title="Count")
    st.plotly_chart(fig_vir, use_container_width=True, config={"displayModeBar": False})

    # ── Performance bucket breakdown ─────────────────────────
    if "performance_bucket_label" in df.columns:
        bucket_df_raw = df.copy()
        bucket_data = (bucket_df_raw.groupby("performance_bucket_label")
                       .size().reset_index(name="count"))
        bucket_order = {"viral":0,"high":1,"medium":2,"low":3}
        bucket_data["rank"] = bucket_data["performance_bucket_label"].map(bucket_order).fillna(99)
        bucket_data = bucket_data.sort_values("rank")

        fig_bucket = go.Figure(go.Bar(
            x=bucket_data["performance_bucket_label"].str.title(),
            y=bucket_data["count"],
            marker=dict(color=["#F59E0B","#34D399","#38BDF8","#F87171"],
                        line=dict(color="rgba(0,0,0,0)")),
            text=bucket_data["count"],
            textposition="outside",
            textfont=dict(color="#E8ECF1", size=12),
        ))
        apply_layout(fig_bucket, height=280,
                     title="Post Count by Performance Bucket (full dataset)",
                     xaxis_title="Performance Tier", yaxis_title="Number of Posts")
        st.plotly_chart(fig_bucket, use_container_width=True, config={"displayModeBar": False})


# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:36px 0 20px;color:#1E3048;font-size:12px;">
  Instagram Analytics Dashboard &nbsp;·&nbsp;
  Powered by Random Forest &amp; Streamlit &nbsp;·&nbsp;
  Dataset: Kaggle · kundanbedmutha/instagram-analytics-dataset
</div>
""", unsafe_allow_html=True)
