"""
Data-Driven Social Engagement Initiative
main_dashboard.py  —  Premium Analytics Dashboard
Run:  streamlit run main_dashboard.py
"""

# ─────────────────────────────────────────────────────────────
# 0.  BOOTSTRAP
# ─────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Social Engagement Initiative",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# 1.  DESIGN SYSTEM — CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
  --bg:      #0E1117;
  --surface: #161B27;
  --card:    #1A2133;
  --border:  #1F2937;
  --border2: #2D3F55;
  --text:    #E2E8F0;
  --muted:   #4B6280;
  --muted2:  #6B82A0;
  --accent:  #4F46E5;
  --blue:    #3B82F6;
  --violet:  #7C3AED;
  --teal:    #0EA5E9;
  --green:   #10B981;
  --amber:   #F59E0B;
  --rose:    #F43F5E;
  --purple:  #8B5CF6;
}

html, body, [class*="css"], .stApp {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}

/* Hide default header */
header[data-testid="stHeader"] { display: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] > div:first-child {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stMarkdown p {
  color: var(--muted2) !important;
  font-size: 12px !important;
}

/* ── KPI Card ── */
.kpi-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 18px;
  text-align: center;
  transition: border-color 0.2s ease, transform 0.2s ease;
  position: relative;
  overflow: hidden;
}
.kpi-card::after {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--c1), var(--c2));
  opacity: 0.7;
}
.kpi-card:hover {
  border-color: var(--border2);
  transform: translateY(-2px);
}
.kpi-icon { font-size: 20px; margin-bottom: 8px; opacity: 0.85; }
.kpi-value {
  font-size: 1.75rem;
  font-weight: 800;
  background: linear-gradient(135deg, var(--c1), var(--c2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.1;
  margin: 4px 0;
}
.kpi-label {
  font-size: 10px;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-top: 4px;
}
.kpi-delta {
  font-size: 11px;
  color: var(--green);
  font-weight: 600;
  margin-top: 6px;
}

/* ── Section header ── */
.section-header {
  font-size: 10px;
  font-weight: 700;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.14em;
  padding-bottom: 10px;
  margin-bottom: 18px;
  border-bottom: 1px solid var(--border);
}

/* ── Insight card ── */
.insight-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent-clr);
  border-radius: 10px;
  padding: 14px 16px;
  transition: border-color 0.2s ease;
}
.insight-card:hover { border-color: var(--border2); }
.insight-label {
  font-size: 9px;
  font-weight: 700;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-bottom: 5px;
}
.insight-value {
  font-size: 15px;
  font-weight: 700;
  color: var(--text);
  line-height: 1.2;
}
.insight-sub {
  font-size: 11px;
  color: var(--muted2);
  margin-top: 4px;
}

/* ── Module badge ── */
.mod-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.05em;
  border: 1px solid;
  margin: 2px;
}

/* ── Info banner ── */
.info-banner {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 16px;
  font-size: 12px;
  color: var(--muted2);
  margin-bottom: 20px;
  line-height: 1.6;
}

/* ── Comment card ── */
.comment-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 16px;
  margin: 6px 0;
  font-size: 13px;
  line-height: 1.6;
  color: var(--text);
  transition: border-color 0.2s;
}
.comment-card:hover { border-color: var(--border2); }

/* ── Recommendation card ── */
.rec-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px 20px;
  margin: 8px 0;
  transition: border-color 0.2s, transform 0.2s;
}
.rec-card:hover { border-color: var(--border2); transform: translateY(-1px); }
.rec-title {
  font-size: 11px;
  font-weight: 700;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 8px;
}
.rec-value {
  font-size: 16px;
  font-weight: 700;
  color: var(--text);
}
.rec-sub {
  font-size: 12px;
  color: var(--muted2);
  margin-top: 5px;
  line-height: 1.5;
}

/* ── Trend keyword card ── */
.trend-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
  margin: 6px 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  transition: border-color 0.2s;
}
.trend-card:hover { border-color: var(--border2); }
.trend-keyword { font-size: 14px; font-weight: 600; color: var(--text); }
.trend-category { font-size: 10px; color: var(--muted); margin-top: 2px; }
.trend-velocity-up { font-size: 13px; font-weight: 700; color: var(--green); }
.trend-velocity-down { font-size: 13px; font-weight: 700; color: var(--rose); }

/* ── AB card ── */
.ab-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px;
}
.ab-winner-card {
  background: var(--card);
  border: 1px solid rgba(16, 185, 129, 0.4);
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 0 0 1px rgba(16, 185, 129, 0.1);
}
.ab-label {
  font-size: 9px;
  font-weight: 700;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-bottom: 6px;
}
.ab-value {
  font-size: 2rem;
  font-weight: 800;
  color: var(--text);
}
.winner-badge {
  display: inline-block;
  background: rgba(16, 185, 129, 0.12);
  border: 1px solid rgba(16, 185, 129, 0.4);
  color: #10B981;
  border-radius: 6px;
  padding: 3px 10px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

/* ── Prediction box ── */
.pred-box {
  background: var(--surface);
  border: 1px solid var(--border2);
  border-radius: 14px;
  padding: 32px;
  text-align: center;
}
.pred-value {
  font-size: 3rem;
  font-weight: 900;
  background: linear-gradient(135deg, var(--blue), var(--purple));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 10px 0;
}
.pred-label {
  font-size: 10px;
  font-weight: 700;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.12em;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface);
  border-radius: 10px;
  padding: 5px;
  border: 1px solid var(--border);
  gap: 3px;
  flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
  background: transparent;
  border-radius: 7px;
  color: var(--muted2);
  font-weight: 500;
  font-size: 12px;
  padding: 7px 14px;
  transition: all 0.15s ease;
  border: 1px solid transparent;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--text);
  background: var(--card);
}
.stTabs [aria-selected="true"] {
  background: var(--card) !important;
  color: var(--blue) !important;
  border-color: var(--border) !important;
  font-weight: 600 !important;
}

/* ── Buttons ── */
.stButton > button {
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  padding: 9px 22px !important;
  transition: opacity 0.15s !important;
  letter-spacing: 0.02em !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Inputs ── */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stSlider > div,
.stNumberInput > div > div {
  background: var(--surface) !important;
  border-color: var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
}
.stTextArea textarea {
  background: var(--surface) !important;
  border-color: var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-size: 13px !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
.stDataFrame [data-testid="stDataFrameResizable"] {
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 16px 0 !important; }

/* ── Metric ── */
[data-testid="stMetric"] {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 11px !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 1.4rem !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 2.  PLOTLY THEME
# ─────────────────────────────────────────────────────────────
_PLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#6B82A0", size=11),
    margin=dict(t=40, b=16, l=16, r=16),
    xaxis=dict(
        gridcolor="#1A2133",
        linecolor="#1F2937",
        tickfont=dict(color="#4B6280", size=10),
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor="#1A2133",
        linecolor="#1F2937",
        tickfont=dict(color="#4B6280", size=10),
        zeroline=False,
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6B82A0", size=11),
        bordercolor="rgba(0,0,0,0)",
    ),
    hoverlabel=dict(
        bgcolor="#1A2133",
        bordercolor="#2D3F55",
        font=dict(color="#E2E8F0", size=12),
    ),
)

PAL = [
    "#3B82F6", "#8B5CF6", "#10B981", "#F43F5E", "#F59E0B",
    "#0EA5E9", "#EC4899", "#14B8A6", "#A78BFA", "#FB923C",
]


def pf(fig, h=300, title="", **kw):
    layout = {**_PLY_BASE, "height": h}
    if title:
        layout["title"] = dict(
            text=title,
            font=dict(color="#94A3B8", size=12, family="Inter"),
            x=0, pad=dict(l=4),
        )
    layout.update(kw)
    fig.update_layout(**layout)
    return fig


# ─────────────────────────────────────────────────────────────
# 3.  REUSABLE COMPONENT HELPERS
# ─────────────────────────────────────────────────────────────

def kpi_card(icon, value, label, c1="#3B82F6", c2="#8B5CF6", delta=""):
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card" style="--c1:{c1};--c2:{c2};">
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-label">{label}</div>
      {delta_html}
    </div>"""


def insight_card(label, value, sub, color="#3B82F6"):
    return f"""
    <div class="insight-card" style="--accent-clr:{color};">
      <div class="insight-label">{label}</div>
      <div class="insight-value">{value}</div>
      <div class="insight-sub">{sub}</div>
    </div>"""


def section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def info_banner(html_content):
    st.markdown(f'<div class="info-banner">{html_content}</div>', unsafe_allow_html=True)


def divider():
    st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 4.  LOAD DATA
# ─────────────────────────────────────────────────────────────
from modules.data_engine import get_dataframe, get_model, compute_performance_tiers, summarise_dataset
from modules.virality_engine import compute_viral_coefficients, topic_viral_ranking, passive_vs_highvalue_breakdown
from modules.sentiment_analyzer import analyze_sentiment_corpus
from modules.ab_testing import run_ab_test, batch_ab_tests
from modules.recommender import generate_recommendations
from modules.trend_forecaster import (
    compute_trend_velocities, forecast_engagement,
    hashtag_effectiveness, load_keyword_trends,
)

try:
    raw_df = get_dataframe()
    raw_df = compute_performance_tiers(raw_df)
except Exception as e:
    st.error(f"Dataset load failed: {e}")
    st.stop()

try:
    model, model_metrics = get_model(raw_df)
except Exception as e:
    st.error(f"Model training failed: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────
# 5.  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    # Project title
    st.markdown("""
    <div style="padding: 24px 0 20px; text-align: center;">
      <div style="
        font-size: 13px;
        font-weight: 800;
        color: #E2E8F0;
        letter-spacing: -0.01em;
        line-height: 1.3;
      ">Data-Driven Social<br>Engagement Initiative</div>
      <div style="
        font-size: 9px;
        font-weight: 600;
        color: #4B6280;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-top: 6px;
      ">Analytics Ecosystem</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Navigation label
    st.markdown("""
    <div style="font-size:9px;font-weight:700;color:#4B6280;
         text-transform:uppercase;letter-spacing:0.12em;
         margin-bottom:10px;">Navigation</div>
    """, unsafe_allow_html=True)

    page = st.radio(
        label="Navigation",
        options=[
            "Overview",
            "Content Performance",
            "Virality Engine",
            "Sentiment Analysis",
            "A/B Testing",
            "Recommendations",
            "Trend Forecasting",
            "Scrape & Train",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Filters
    st.markdown("""
    <div style="font-size:9px;font-weight:700;color:#4B6280;
         text-transform:uppercase;letter-spacing:0.12em;
         margin-bottom:10px;">Filters</div>
    """, unsafe_allow_html=True)

    all_cats  = sorted(raw_df["content_category"].dropna().unique())
    all_media = sorted(raw_df["media_type"].dropna().unique())
    all_acct  = sorted(raw_df["account_type"].dropna().unique())

    sel_cats  = st.multiselect("Category",     all_cats,  default=all_cats)
    sel_media = st.multiselect("Media Type",   all_media, default=all_media)
    sel_acct  = st.multiselect("Account Type", all_acct,  default=all_acct)

    fc_min = int(raw_df["follower_count"].min())
    fc_max = int(raw_df["follower_count"].max())
    fc_range = st.slider("Follower Count", fc_min, fc_max, (fc_min, fc_max), step=500)

    if "post_datetime" in raw_df.columns:
        d_min = raw_df["post_datetime"].min().date()
        d_max = raw_df["post_datetime"].max().date()
        date_range = st.date_input("Date Range", (d_min, d_max), min_value=d_min, max_value=d_max)
    else:
        date_range = None

    st.markdown("<hr>", unsafe_allow_html=True)

    # Model stats
    st.markdown("""
    <div style="font-size:9px;font-weight:700;color:#4B6280;
         text-transform:uppercase;letter-spacing:0.12em;
         margin-bottom:10px;">RF Model</div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">
      <div style="background:#1A2133;border:1px solid #1F2937;border-radius:8px;
           padding:10px;text-align:center;">
        <div style="font-size:9px;color:#4B6280;text-transform:uppercase;
             letter-spacing:0.1em;margin-bottom:3px;">MAE</div>
        <div style="font-size:18px;font-weight:800;color:#3B82F6;">{model_metrics['mae']}</div>
      </div>
      <div style="background:#1A2133;border:1px solid #1F2937;border-radius:8px;
           padding:10px;text-align:center;">
        <div style="font-size:9px;color:#4B6280;text-transform:uppercase;
             letter-spacing:0.1em;margin-bottom:3px;">R²</div>
        <div style="font-size:18px;font-weight:800;color:#8B5CF6;">{model_metrics['r2']}</div>
      </div>
    </div>
    <div style="font-size:10px;color:#4B6280;text-align:center;">
      Trained on {model_metrics['n_train']:,} posts
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 6.  APPLY FILTERS
# ─────────────────────────────────────────────────────────────
fdf = raw_df.copy()
if sel_cats:  fdf = fdf[fdf["content_category"].isin(sel_cats)]
if sel_media: fdf = fdf[fdf["media_type"].isin(sel_media)]
if sel_acct:  fdf = fdf[fdf["account_type"].isin(sel_acct)]
fdf = fdf[(fdf["follower_count"] >= fc_range[0]) & (fdf["follower_count"] <= fc_range[1])]
if date_range and len(date_range) == 2:
    fdf = fdf[
        (fdf["post_datetime"].dt.date >= date_range[0]) &
        (fdf["post_datetime"].dt.date <= date_range[1])
    ]

if fdf.empty:
    st.warning("No data matches the current filters. Adjust the sidebar settings.")
    st.stop()

fdf = compute_viral_coefficients(fdf)
stats = summarise_dataset(fdf)

# ─────────────────────────────────────────────────────────────
# 7.  PAGE HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding: 36px 0 28px;">
  <div style="
    font-size: 1.9rem;
    font-weight: 800;
    color: #E2E8F0;
    letter-spacing: -0.025em;
    line-height: 1.15;
  ">Data-Driven Social Engagement Initiative</div>
  <div style="
    font-size: 13px;
    color: #4B6280;
    margin-top: 8px;
    font-weight: 400;
  ">{page} &nbsp;·&nbsp; {len(fdf):,} posts &nbsp;·&nbsp; {len(fdf.columns)} features</div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═════════════════════════════════════════════════════════════
if page == "Overview":

    # KPI strip
    section_header("Key Performance Indicators")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpis = [
        (k1, "Engagement Rate",    f"{stats['avg_engagement']:.2%}",       "#3B82F6", "#8B5CF6"),
        (k2, "Avg Likes",          f"{stats['avg_likes']:,.0f}",            "#F43F5E", "#FB923C"),
        (k3, "Total Reach",        f"{stats['total_reach']/1e6:.1f}M",      "#10B981", "#0EA5E9"),
        (k4, "Total Impressions",  f"{stats['total_impr']/1e6:.1f}M",       "#8B5CF6", "#7C3AED"),
        (k5, "Avg Viral Coeff",    f"{fdf['vc_raw'].mean():.2f}",           "#F59E0B", "#F43F5E"),
        (k6, "Total Posts",        f"{stats['total_posts']:,}",             "#0EA5E9", "#10B981"),
    ]
    for col, label, val, c1, c2 in kpis:
        with col:
            st.markdown(kpi_card("", val, label, c1, c2), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Smart insights
    section_header("Smart Insights")
    bh   = fdf.groupby("hour")["engagement_rate"].mean().idxmax()
    bc   = fdf.groupby("content_category")["engagement_rate"].mean().idxmax()
    bm   = fdf.groupby("media_type")["engagement_rate"].mean().idxmax()
    bday = fdf.groupby("day")["engagement_rate"].mean().idxmax() if "day" in fdf.columns else "N/A"
    rank_df = topic_viral_ranking(fdf)
    bvc  = rank_df.iloc[0]["content_category"] if not rank_df.empty else "N/A"

    i1, i2, i3, i4, i5 = st.columns(5)
    insights = [
        (i1, "Best Post Time",    f"{bh:02d}:00 {'AM' if bh < 12 else 'PM'}",
             f"Avg {fdf.groupby('hour')['engagement_rate'].mean()[bh]:.2%} engagement", "#3B82F6"),
        (i2, "Best Category",     bc,
             f"Avg {fdf.groupby('content_category')['engagement_rate'].mean()[bc]:.2%} engagement", "#8B5CF6"),
        (i3, "Best Media Format", bm.title(),
             f"Avg {fdf.groupby('media_type')['engagement_rate'].mean()[bm]:.2%} engagement", "#10B981"),
        (i4, "Best Day",          bday,
             "Highest avg engagement rate", "#F43F5E"),
        (i5, "Top Viral Topic",   bvc,
             "Highest viral coefficient", "#F59E0B"),
    ]
    for col, label, val, sub, clr in insights:
        with col:
            st.markdown(insight_card(label, val, sub, clr), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    divider()

    # Growth trend + top topics
    section_header("Growth Trend & Top Topics")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        if "month_num" in fdf.columns and "month" in fdf.columns:
            md = (fdf.groupby(["month_num", "month"])["engagement_rate"]
                  .mean().reset_index().sort_values("month_num"))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=md["month"], y=md["engagement_rate"],
                mode="lines+markers",
                line=dict(color="#3B82F6", width=2.5),
                marker=dict(size=7, color="#3B82F6", line=dict(color="#0E1117", width=2)),
                fill="tozeroy",
                fillcolor="rgba(59,130,246,0.06)",
                hovertemplate="%{x}<br>Engagement: %{y:.4f}<extra></extra>",
                name="Engagement Rate",
            ))
            pf(fig, 300, "Monthly Engagement Trend",
               xaxis_title="Month", yaxis_title="Avg Engagement Rate")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_r:
        cat_data = (fdf.groupby("content_category")["engagement_rate"]
                    .mean().sort_values(ascending=True).reset_index())
        fig2 = go.Figure(go.Bar(
            x=cat_data["engagement_rate"],
            y=cat_data["content_category"],
            orientation="h",
            marker=dict(
                color=PAL[:len(cat_data)],
                line=dict(color="rgba(0,0,0,0)"),
            ),
            text=[f"{v:.2%}" for v in cat_data["engagement_rate"]],
            textposition="outside",
            textfont=dict(color="#94A3B8", size=10),
            hovertemplate="%{y}<br>Engagement: %{x:.4f}<extra></extra>",
        ))
        pf(fig2, 300, "Top Performing Topics",
           xaxis_title="Avg Engagement Rate")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    divider()

    # Performance tier funnel
    section_header("Post Performance Distribution")
    col_a, col_b = st.columns(2)

    with col_a:
        if "tier" in fdf.columns:
            tier_cnt = fdf["tier"].value_counts()
            tier_ord = ["Viral", "High", "Medium", "Low"]
            tier_clr = {"Viral": "#F59E0B", "High": "#10B981", "Medium": "#3B82F6", "Low": "#8B5CF6"}
            fig3 = go.Figure(go.Funnel(
                y=[t for t in tier_ord if t in tier_cnt],
                x=[tier_cnt.get(t, 0) for t in tier_ord if t in tier_cnt],
                textposition="inside",
                textinfo="value+percent initial",
                marker_color=[tier_clr[t] for t in tier_ord if t in tier_cnt],
                connector=dict(line=dict(color="#1F2937", width=1)),
            ))
            pf(fig3, 280, "Performance Tier Funnel")
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        # Engagement by day of week
        if "day" in fdf.columns:
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dod = (fdf.groupby("day")["engagement_rate"]
                   .mean().reindex(day_order).dropna().reset_index())
            fig4 = go.Figure(go.Bar(
                x=dod["day"],
                y=dod["engagement_rate"],
                marker=dict(
                    color=dod["engagement_rate"],
                    colorscale=[[0, "#1A2133"], [0.5, "#3B82F6"], [1, "#8B5CF6"]],
                    line=dict(color="rgba(0,0,0,0)"),
                ),
                text=[f"{v:.2%}" for v in dod["engagement_rate"]],
                textposition="outside",
                textfont=dict(color="#94A3B8", size=10),
                hovertemplate="%{x}<br>Engagement: %{y:.4f}<extra></extra>",
            ))
            pf(fig4, 280, "Engagement by Day of Week",
               xaxis_title="Day", yaxis_title="Avg Engagement Rate")
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})


# ═════════════════════════════════════════════════════════════
# PAGE: CONTENT PERFORMANCE
# ═════════════════════════════════════════════════════════════
elif page == "Content Performance":

    section_header("Content Performance Analysis")

    # Filters row
    f1, f2, f3 = st.columns(3)
    with f1:
        post_type_filter = st.selectbox("Post Type", ["All"] + sorted(fdf["media_type"].dropna().unique().tolist()))
    with f2:
        if "post_datetime" in fdf.columns:
            min_d = fdf["post_datetime"].min().date()
            max_d = fdf["post_datetime"].max().date()
            date_f = st.date_input("Date Filter", (min_d, max_d), min_value=min_d, max_value=max_d, key="cp_date")
        else:
            date_f = None
    with f3:
        metric_choice = st.selectbox("Primary Metric", ["engagement_rate", "likes", "shares", "saves"])

    # Apply local filters
    cdf = fdf.copy()
    if post_type_filter != "All":
        cdf = cdf[cdf["media_type"] == post_type_filter]
    if date_f and len(date_f) == 2:
        cdf = cdf[(cdf["post_datetime"].dt.date >= date_f[0]) & (cdf["post_datetime"].dt.date <= date_f[1])]

    st.markdown("<br>", unsafe_allow_html=True)

    # Engagement metrics visualized
    section_header("Engagement Metrics")
    col_a, col_b = st.columns(2)

    with col_a:
        med = (cdf.groupby("media_type")["engagement_rate"]
               .mean().sort_values(ascending=False).reset_index())
        fig = go.Figure(go.Bar(
            x=med["media_type"],
            y=med["engagement_rate"],
            marker=dict(color=PAL[:len(med)], line=dict(color="rgba(0,0,0,0)")),
            text=[f"{v:.2%}" for v in med["engagement_rate"]],
            textposition="outside",
            textfont=dict(color="#94A3B8", size=11),
            hovertemplate="%{x}<br>Engagement: %{y:.4f}<extra></extra>",
        ))
        pf(fig, 300, "Engagement Rate by Media Type",
           xaxis_title="Media Type", yaxis_title="Avg Engagement Rate")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        # Likes / Shares / Saves by media type
        if all(c in cdf.columns for c in ["likes", "shares", "saves"]):
            lss = cdf.groupby("media_type")[["likes", "shares", "saves"]].mean().reset_index()
            fig2 = go.Figure()
            for col_name, clr in [("likes", "#3B82F6"), ("shares", "#10B981"), ("saves", "#F59E0B")]:
                fig2.add_trace(go.Bar(
                    name=col_name.title(),
                    x=lss["media_type"],
                    y=lss[col_name],
                    marker_color=clr,
                    hovertemplate=f"{col_name.title()}: %{{y:,.0f}}<extra></extra>",
                ))
            fig2.update_layout(barmode="group")
            pf(fig2, 300, "Avg Likes / Shares / Saves by Media Type",
               xaxis_title="Media Type", yaxis_title="Avg Count")
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    divider()

    # Retention rate line graph (engagement over time as proxy)
    section_header("Retention Rate Over Time")
    if "month_num" in cdf.columns and "month" in cdf.columns:
        ret = (cdf.groupby(["month_num", "month"])["engagement_rate"]
               .mean().reset_index().sort_values("month_num"))
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=ret["month"], y=ret["engagement_rate"],
            mode="lines+markers",
            line=dict(color="#10B981", width=2.5),
            marker=dict(size=7, color="#10B981", line=dict(color="#0E1117", width=2)),
            fill="tozeroy",
            fillcolor="rgba(16,185,129,0.06)",
            hovertemplate="%{x}<br>Engagement: %{y:.4f}<extra></extra>",
            name="Retention Proxy",
        ))
        pf(fig3, 260, "Monthly Retention Rate (Engagement Proxy)",
           xaxis_title="Month", yaxis_title="Avg Engagement Rate")
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    divider()

    # Data table
    section_header("Post Data Table")
    display_cols = [c for c in [
        "content_category", "media_type", "account_type",
        "engagement_rate", "likes", "shares", "saves",
        "reach", "impressions", "hashtags_count", "hour", "day", "tier",
    ] if c in cdf.columns]

    table_df = cdf[display_cols].copy()
    for col in ["engagement_rate"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].round(4)

    st.dataframe(
        table_df.head(500),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Showing up to 500 of {len(cdf):,} filtered posts")


# ═════════════════════════════════════════════════════════════
# PAGE: VIRALITY ENGINE
# ═════════════════════════════════════════════════════════════
elif page == "Virality Engine":

    section_header("Virality Prediction Engine")

    info_banner("""
    <strong style="color:#3B82F6;">Viral Coefficient (VC)</strong> =
    (Saves &times; 3 + Shares &times; 2 + Comments &times; 1.5) / Reach &times; 1000
    &nbsp;&middot;&nbsp;
    Weights high-intent actions over passive engagement.
    Higher VC = stronger organic distribution potential.
    """)

    rank_df = topic_viral_ranking(fdf)
    hv_df   = passive_vs_highvalue_breakdown(fdf)

    # KPIs
    v1, v2, v3, v4 = st.columns(4)
    avg_vc   = fdf["vc_raw"].mean()
    peak_vc  = fdf["vc_raw"].max()
    top_cat  = rank_df.iloc[0]["content_category"] if not rank_df.empty else "N/A"
    mega_cnt = (fdf["vc_tier"] == "Mega-Viral").sum() if "vc_tier" in fdf.columns else 0

    with v1: st.markdown(kpi_card("", f"{avg_vc:.2f}",    "Avg Viral Coefficient", "#F59E0B", "#FB923C"), unsafe_allow_html=True)
    with v2: st.markdown(kpi_card("", f"{peak_vc:.1f}",   "Peak VC",               "#F43F5E", "#F59E0B"), unsafe_allow_html=True)
    with v3: st.markdown(kpi_card("", top_cat[:16],        "Top Viral Category",    "#10B981", "#0EA5E9"), unsafe_allow_html=True)
    with v4: st.markdown(kpi_card("", f"{mega_cnt:,}",    "Mega-Viral Posts",      "#8B5CF6", "#3B82F6"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    divider()

    # Topic ranking + high-value breakdown
    section_header("Topic Viral Ranking")
    col_l, col_r = st.columns(2)

    with col_l:
        if not rank_df.empty and "vc_raw" in rank_df.columns:
            fig = go.Figure(go.Bar(
                y=rank_df["content_category"],
                x=rank_df["vc_raw"],
                orientation="h",
                marker=dict(color=PAL[:len(rank_df)], line=dict(color="rgba(0,0,0,0)")),
                text=[f"VC: {v:.2f}" for v in rank_df["vc_raw"]],
                textposition="outside",
                textfont=dict(color="#94A3B8", size=10),
                hovertemplate="%{y}<br>Viral Coefficient: %{x:.3f}<extra></extra>",
            ))
            pf(fig, 320, "Avg Viral Coefficient by Topic")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_r:
        if not hv_df.empty:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name="High-Value (Saves + Shares)",
                y=hv_df["content_category"],
                x=hv_df["high_value"],
                orientation="h",
                marker_color="#10B981",
                hovertemplate="%{y}<br>High-Value: %{x:.1f}<extra></extra>",
            ))
            fig2.add_trace(go.Bar(
                name="Passive (Likes)",
                y=hv_df["content_category"],
                x=hv_df["passive"],
                orientation="h",
                marker_color="#1A2133",
                hovertemplate="%{y}<br>Passive: %{x:.1f}<extra></extra>",
            ))
            fig2.update_layout(barmode="stack")
            pf(fig2, 320, "High-Value vs Passive Actions by Category")
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    divider()

    # Scatter: Shares vs Saves
    section_header("Shares vs Saves Scatter")
    if "shares" in fdf.columns and "saves" in fdf.columns:
        sample = fdf.sample(min(2000, len(fdf)), random_state=42)
        fig3 = go.Figure(go.Scatter(
            x=sample["shares"],
            y=sample["saves"],
            mode="markers",
            marker=dict(
                color=sample["vc_raw"],
                colorscale=[[0, "#1A2133"], [0.5, "#3B82F6"], [1, "#F59E0B"]],
                size=5,
                opacity=0.65,
                colorbar=dict(
                    title=dict(text="VC", font=dict(color="#4B6280")),
                    tickfont=dict(color="#4B6280"),
                ),
                showscale=True,
                line=dict(color="rgba(0,0,0,0)"),
            ),
            hovertemplate="Shares: %{x:,}<br>Saves: %{y:,}<br>VC: %{marker.color:.2f}<extra></extra>",
        ))
        pf(fig3, 320, "Shares vs Saves (colour = Viral Coefficient)",
           xaxis_title="Shares", yaxis_title="Saves")
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    divider()

    # Top viral posts cards
    section_header("Top Viral Posts")
    top_posts = fdf.nlargest(6, "vc_raw")[
        [c for c in ["content_category", "media_type", "vc_raw", "engagement_rate",
                     "likes", "shares", "saves", "tier"] if c in fdf.columns]
    ].reset_index(drop=True)

    cols = st.columns(3)
    for i, row in top_posts.iterrows():
        with cols[i % 3]:
            vc_val = f"{row['vc_raw']:.2f}" if "vc_raw" in row else "N/A"
            eng    = f"{row['engagement_rate']:.2%}" if "engagement_rate" in row else "N/A"
            cat    = row.get("content_category", "N/A")
            media  = str(row.get("media_type", "N/A")).title()
            tier   = row.get("tier", "")
            tier_clr = {"Viral": "#F59E0B", "High": "#10B981", "Medium": "#3B82F6", "Low": "#8B5CF6"}.get(tier, "#4B6280")
            st.markdown(f"""
            <div class="rec-card">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <div style="font-size:10px;font-weight:700;color:#4B6280;
                     text-transform:uppercase;letter-spacing:0.1em;">#{i+1} Top Viral</div>
                <div style="background:{tier_clr}18;border:1px solid {tier_clr}44;
                     color:{tier_clr};border-radius:5px;padding:2px 8px;
                     font-size:9px;font-weight:700;text-transform:uppercase;">{tier}</div>
              </div>
              <div style="font-size:1.4rem;font-weight:800;color:#E2E8F0;margin-bottom:4px;">VC {vc_val}</div>
              <div style="font-size:12px;color:#6B82A0;">{cat} &middot; {media}</div>
              <div style="font-size:11px;color:#4B6280;margin-top:6px;">Engagement: {eng}</div>
            </div>
            """, unsafe_allow_html=True)

    divider()

    # VC distribution
    section_header("Viral Coefficient Distribution")
    fig4 = go.Figure(go.Histogram(
        x=fdf["vc_raw"].clip(upper=fdf["vc_raw"].quantile(0.99)),
        nbinsx=60,
        marker_color="#3B82F6",
        marker_line=dict(color="rgba(0,0,0,0)"),
        opacity=0.85,
        hovertemplate="VC: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))
    pf(fig4, 240, "Viral Coefficient Distribution (99th percentile cap)",
       xaxis_title="Viral Coefficient", yaxis_title="Post Count")
    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    divider()

    # Full ranking table
    section_header("Full Topic Ranking Table")
    if not rank_df.empty:
        display_cols = [c for c in [
            "rank", "content_category", "vc_raw", "save_to_share_ratio",
            "organic_growth_score", "engagement_rate", "high_value_ratio",
        ] if c in rank_df.columns]
        rd = rank_df[display_cols].copy()
        rd.columns = [
            {"rank": "Rank", "content_category": "Category", "vc_raw": "Avg VC",
             "save_to_share_ratio": "Save:Share", "organic_growth_score": "Organic Score",
             "engagement_rate": "Avg Engagement", "high_value_ratio": "HV Ratio"}.get(c, c)
            for c in rd.columns
        ]
        for col in ["Avg VC", "Save:Share", "Organic Score", "Avg Engagement", "HV Ratio"]:
            if col in rd.columns:
                rd[col] = rd[col].round(4)
        st.dataframe(rd, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════
# PAGE: SENTIMENT ANALYSIS
# ═════════════════════════════════════════════════════════════
elif page == "Sentiment Analysis":

    section_header("Audience Sentiment Analyzer")

    info_banner("""
    Using <strong style="color:#F43F5E;">VADER</strong> (Valence Aware Dictionary and sEntiment Reasoner)
    + custom linguistic trigger detection to classify comments as
    <strong style="color:#10B981;">Relatable</strong> /
    <strong style="color:#3B82F6;">Positive</strong> /
    <strong style="color:#F43F5E;">Negative</strong> /
    <strong style="color:#8B5CF6;">Neutral</strong>.
    Problem Awareness levels are computed per content category.
    """)

    with st.spinner("Running NLP sentiment analysis on comment corpus..."):
        nlp = analyze_sentiment_corpus(fdf)

    comments_df  = nlp["comments_df"]
    cat_summary  = nlp["category_summary"]
    label_counts = nlp["label_counts"]
    top_relatable = nlp["top_relatable"]

    # KPIs
    total_comments = len(comments_df)
    pct_relatable  = label_counts.get("Relatable", 0) / max(total_comments, 1) * 100
    pct_negative   = label_counts.get("Negative",  0) / max(total_comments, 1) * 100
    avg_vader      = comments_df["vader_compound"].mean()

    s1, s2, s3, s4 = st.columns(4)
    with s1: st.markdown(kpi_card("", f"{total_comments:,}",    "Comments Analyzed",  "#3B82F6", "#8B5CF6"), unsafe_allow_html=True)
    with s2: st.markdown(kpi_card("", f"{pct_relatable:.1f}%",  "Relatable Rate",     "#10B981", "#0EA5E9"), unsafe_allow_html=True)
    with s3: st.markdown(kpi_card("", f"{pct_negative:.1f}%",   "Negative Rate",      "#F43F5E", "#F59E0B"), unsafe_allow_html=True)
    with s4: st.markdown(kpi_card("", f"{avg_vader:+.3f}",      "Avg VADER Score",    "#8B5CF6", "#3B82F6"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    divider()

    # Pie chart + category breakdown
    section_header("Sentiment Distribution")
    col_l, col_r = st.columns([1, 2])

    with col_l:
        label_clrs = {
            "Relatable": "#10B981",
            "Positive":  "#3B82F6",
            "Negative":  "#F43F5E",
            "Neutral":   "#8B5CF6",
        }
        labels = list(label_counts.keys())
        values = list(label_counts.values())
        colors = [label_clrs.get(l, "#4B6280") for l in labels]

        fig = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            marker=dict(
                colors=colors,
                line=dict(color="#0E1117", width=3),
            ),
            textfont=dict(color="#E2E8F0", size=11),
            hovertemplate="%{label}<br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        ))
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="v",
                font=dict(color="#6B82A0", size=11),
                bgcolor="rgba(0,0,0,0)",
            ),
            annotations=[dict(
                text=f"<b>{total_comments:,}</b><br><span style='font-size:10px'>comments</span>",
                x=0.5, y=0.5,
                font=dict(color="#E2E8F0", size=13),
                showarrow=False,
            )],
        )
        pf(fig, 300, "Sentiment Label Distribution")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_r:
        if not cat_summary.empty:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name="Relatable %",
                x=cat_summary["content_category"],
                y=cat_summary["pct_relatable"],
                marker_color="#10B981",
                hovertemplate="%{x}<br>Relatable: %{y:.1f}%<extra></extra>",
            ))
            fig2.add_trace(go.Bar(
                name="Negative %",
                x=cat_summary["content_category"],
                y=cat_summary["pct_negative"],
                marker_color="#F43F5E",
                hovertemplate="%{x}<br>Negative: %{y:.1f}%<extra></extra>",
            ))
            fig2.update_layout(barmode="group")
            pf(fig2, 300, "Relatable vs Negative Rate by Category",
               xaxis_title="Category", yaxis_title="Percentage (%)")
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    divider()

    # Word frequency / common phrases (top relatable phrases)
    section_header("Common Phrases — Top Relatable Comments")
    if not top_relatable.empty:
        for _, row in top_relatable.head(8).iterrows():
            score = row.get("relatability_score", 0)
            cat   = row.get("content_category", "")
            text  = row.get("comment_text", "")
            vader = row.get("vader_compound", 0)
            bar_w = int(min(score, 100))
            st.markdown(f"""
            <div class="comment-card">
              <div style="display:flex;justify-content:space-between;
                   align-items:flex-start;margin-bottom:8px;">
                <div style="font-size:10px;font-weight:700;color:#4B6280;
                     text-transform:uppercase;letter-spacing:0.1em;">{cat}</div>
                <div style="font-size:10px;color:#10B981;font-weight:600;">
                  Relatability {score:.0f}/100
                </div>
              </div>
              <div style="font-size:13px;color:#CBD5E1;line-height:1.6;margin-bottom:8px;">
                "{text}"
              </div>
              <div style="height:3px;background:#1F2937;border-radius:2px;overflow:hidden;">
                <div style="height:100%;width:{bar_w}%;background:linear-gradient(90deg,#10B981,#3B82F6);
                     border-radius:2px;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    divider()

    # Category summary table
    section_header("Category Sentiment Summary")
    if not cat_summary.empty:
        display = cat_summary[[c for c in [
            "content_category", "avg_relatability", "avg_vader",
            "pct_relatable", "pct_negative", "n_comments", "problem_awareness",
        ] if c in cat_summary.columns]].copy()
        display.columns = [
            {"content_category": "Category", "avg_relatability": "Avg Relatability",
             "avg_vader": "Avg VADER", "pct_relatable": "Relatable %",
             "pct_negative": "Negative %", "n_comments": "Comments",
             "problem_awareness": "Problem Awareness"}.get(c, c)
            for c in display.columns
        ]
        for col in ["Avg Relatability", "Avg VADER", "Relatable %", "Negative %"]:
            if col in display.columns:
                display[col] = display[col].round(2)
        st.dataframe(display, use_container_width=True, hide_index=True)

    divider()

    # Model info + retrain
    section_header("Sentiment Model")
    from modules.sentiment_analyzer import retrain_sentiment_model, SENTIMENT_MODEL_PKL
    import os as _os

    mi = nlp.get("model_info", {})
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div style="background:#1A2133;border:1px solid #1F2937;border-radius:8px;
             padding:12px;text-align:center;">
          <div style="font-size:9px;color:#4B6280;text-transform:uppercase;
               letter-spacing:.1em;margin-bottom:3px;">Classifier</div>
          <div style="font-size:13px;font-weight:700;color:#3B82F6;">
            {mi.get('classifier','LogisticRegression')}
          </div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div style="background:#1A2133;border:1px solid #1F2937;border-radius:8px;
             padding:12px;text-align:center;">
          <div style="font-size:9px;color:#4B6280;text-transform:uppercase;
               letter-spacing:.1em;margin-bottom:3px;">Vectorizer</div>
          <div style="font-size:13px;font-weight:700;color:#8B5CF6;">
            TF-IDF {mi.get('ngram_range','')}
          </div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div style="background:#1A2133;border:1px solid #1F2937;border-radius:8px;
             padding:12px;text-align:center;">
          <div style="font-size:9px;color:#4B6280;text-transform:uppercase;
               letter-spacing:.1em;margin-bottom:3px;">Max Features</div>
          <div style="font-size:13px;font-weight:700;color:#10B981;">
            {mi.get('n_features', 5000):,}
          </div>
        </div>""", unsafe_allow_html=True)
    with m4:
        model_status = "Saved" if mi.get("model_exists") else "In-memory"
        status_clr   = "#10B981" if mi.get("model_exists") else "#F59E0B"
        st.markdown(f"""
        <div style="background:#1A2133;border:1px solid #1F2937;border-radius:8px;
             padding:12px;text-align:center;">
          <div style="font-size:9px;color:#4B6280;text-transform:uppercase;
               letter-spacing:.1em;margin-bottom:3px;">Model Status</div>
          <div style="font-size:13px;font-weight:700;color:{status_clr};">
            {model_status}
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    info_banner(f"""
    Classes: {', '.join(mi.get('classes', ['Relatable','Positive','Negative','Neutral']))}
    &nbsp;&middot;&nbsp;
    Model: TF-IDF (unigrams + bigrams) + Logistic Regression with balanced class weights.
    Labels are predicted by the trained classifier — no hardcoded rules.
    """)

    if st.button("Retrain Sentiment Model"):
        with st.spinner("Retraining TF-IDF + Logistic Regression on seed corpus..."):
            try:
                result = retrain_sentiment_model()
                st.success(
                    f"Retrained. CV Accuracy: {result['cv_accuracy_mean']:.2%} "
                    f"(+/- {result['cv_accuracy_std']:.2%})  "
                    f"Samples: {result['n_samples']}"
                )
                st.json(result["label_distribution"])
            except Exception as e:
                st.error(f"Retraining failed: {e}")


# ═════════════════════════════════════════════════════════════
# PAGE: A/B TESTING
# ═════════════════════════════════════════════════════════════
elif page == "A/B Testing":

    section_header("A/B Testing Framework")

    info_banner("""
    Welch's two-sample t-test (unequal variance) with Cohen's d effect size.
    Compares two groups on any engagement metric.
    Results include p-value, confidence intervals, lift percentage, and winner declaration.
    """)

    # Custom test
    section_header("Custom Experiment")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ab_variable = st.selectbox("Test Variable", [
            "media_type", "content_category", "account_type",
            "has_call_to_action", "hour",
        ])
    with c2:
        unique_vals = sorted(fdf[ab_variable].dropna().unique().astype(str).tolist())
        group_a = st.selectbox("Group A", unique_vals, index=0)
    with c3:
        group_b = st.selectbox("Group B", unique_vals, index=min(1, len(unique_vals) - 1))
    with c4:
        ab_metric = st.selectbox("Metric", ["engagement_rate", "likes", "shares", "saves"])

    run_btn = st.button("Run Experiment")

    if run_btn or True:  # auto-run on load
        result = run_ab_test(fdf, ab_variable, group_a, group_b, metric=ab_metric)

        if "error" in result:
            st.warning(result["error"])
        else:
            st.markdown("<br>", unsafe_allow_html=True)

            # Side-by-side comparison cards
            section_header("Side-by-Side Comparison")
            winner = result["winner"]
            a_is_winner = (winner == result["group_a"])

            col_a, col_sep, col_b = st.columns([5, 1, 5])

            with col_a:
                card_cls = "ab-winner-card" if a_is_winner else "ab-card"
                winner_badge = '<div class="winner-badge">WINNER</div>' if a_is_winner else ""
                st.markdown(f"""
                <div class="{card_cls}">
                  <div style="display:flex;justify-content:space-between;
                       align-items:center;margin-bottom:14px;">
                    <div class="ab-label">Group A</div>
                    {winner_badge}
                  </div>
                  <div style="font-size:1.1rem;font-weight:700;color:#E2E8F0;
                       margin-bottom:12px;">{result['group_a']}</div>
                  <div class="ab-label">Avg {ab_metric.replace('_',' ').title()}</div>
                  <div class="ab-value">{result['mean_a']:.4f}</div>
                  <div style="font-size:11px;color:#4B6280;margin-top:6px;">
                    n = {result['n_a']:,} &nbsp;&middot;&nbsp; SD = {result['std_a']:.4f}
                  </div>
                  <div style="font-size:11px;color:#4B6280;margin-top:3px;">
                    95% CI: [{result['ci_a'][0]:.4f}, {result['ci_a'][1]:.4f}]
                  </div>
                </div>
                """, unsafe_allow_html=True)

            with col_sep:
                st.markdown("""
                <div style="display:flex;align-items:center;justify-content:center;
                     height:100%;padding-top:60px;">
                  <div style="font-size:18px;font-weight:700;color:#4B6280;">vs</div>
                </div>
                """, unsafe_allow_html=True)

            with col_b:
                card_cls = "ab-winner-card" if not a_is_winner else "ab-card"
                winner_badge = '<div class="winner-badge">WINNER</div>' if not a_is_winner else ""
                st.markdown(f"""
                <div class="{card_cls}">
                  <div style="display:flex;justify-content:space-between;
                       align-items:center;margin-bottom:14px;">
                    <div class="ab-label">Group B</div>
                    {winner_badge}
                  </div>
                  <div style="font-size:1.1rem;font-weight:700;color:#E2E8F0;
                       margin-bottom:12px;">{result['group_b']}</div>
                  <div class="ab-label">Avg {ab_metric.replace('_',' ').title()}</div>
                  <div class="ab-value">{result['mean_b']:.4f}</div>
                  <div style="font-size:11px;color:#4B6280;margin-top:6px;">
                    n = {result['n_b']:,} &nbsp;&middot;&nbsp; SD = {result['std_b']:.4f}
                  </div>
                  <div style="font-size:11px;color:#4B6280;margin-top:3px;">
                    95% CI: [{result['ci_b'][0]:.4f}, {result['ci_b'][1]:.4f}]
                  </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Metrics difference visualization
            section_header("Statistical Results")
            m1, m2, m3, m4 = st.columns(4)
            sig_color = "#10B981" if result["significant"] else "#F43F5E"
            sig_label = "Significant" if result["significant"] else "Not Significant"

            with m1: st.markdown(kpi_card("", f"{result['p_value']:.4f}", "p-value", "#3B82F6", "#8B5CF6"), unsafe_allow_html=True)
            with m2: st.markdown(kpi_card("", f"{result['lift_pct']:.1f}%", "Lift", "#10B981", "#0EA5E9"), unsafe_allow_html=True)
            with m3: st.markdown(kpi_card("", f"{result['cohens_d']:.3f}", "Cohen's d", "#F59E0B", "#FB923C"), unsafe_allow_html=True)
            with m4: st.markdown(kpi_card("", sig_label, "Result", sig_color, sig_color), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Interpretation
            interp_color = "#10B981" if result["significant"] else "#F43F5E"
            st.markdown(f"""
            <div style="background:#1A2133;border:1px solid #1F2937;
                 border-left:3px solid {interp_color};border-radius:10px;
                 padding:14px 18px;font-size:13px;color:#CBD5E1;line-height:1.6;">
              {result['interpretation']}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # CI comparison chart
            section_header("Confidence Interval Comparison")
            fig = go.Figure()
            for grp, mean, ci, clr in [
                (result["group_a"], result["mean_a"], result["ci_a"], "#3B82F6"),
                (result["group_b"], result["mean_b"], result["ci_b"], "#8B5CF6"),
            ]:
                fig.add_trace(go.Scatter(
                    x=[grp], y=[mean],
                    error_y=dict(
                        type="data",
                        array=[ci[1] - mean],
                        arrayminus=[mean - ci[0]],
                        color=clr,
                        thickness=2,
                        width=12,
                    ),
                    mode="markers",
                    marker=dict(size=12, color=clr, line=dict(color="#0E1117", width=2)),
                    name=grp,
                    hovertemplate=f"{grp}<br>Mean: {mean:.4f}<br>CI: [{ci[0]:.4f}, {ci[1]:.4f}]<extra></extra>",
                ))
            pf(fig, 260, "95% Confidence Intervals",
               xaxis_title="Group", yaxis_title=ab_metric.replace("_", " ").title())
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    divider()

    # Batch tests
    section_header("Batch Test Results")
    with st.spinner("Running batch A/B tests..."):
        batch_df = batch_ab_tests(fdf, metric="engagement_rate")

    if not batch_df.empty:
        st.dataframe(batch_df, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════
# PAGE: RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════
elif page == "Recommendations":

    section_header("Engagement Optimization Recommender")

    info_banner("""
    AI-generated content strategy recommendations based on historical performance data.
    Factors analyzed: best posting hour, best content category, best media type,
    optimal hashtag count, and CTA impact.
    """)

    with st.spinner("Generating recommendations..."):
        recs = generate_recommendations(fdf)

    fs   = recs["factor_scores"]
    strats = recs["top_strategies"]
    cal  = recs["strategy_table"]
    cta  = recs["cta_impact"]
    bdays = recs["best_days_df"]

    # Top 3 strategy cards
    section_header("Top Recommended Strategies")
    if strats:
        cols = st.columns(len(strats))
        rank_colors = ["#F59E0B", "#8B5CF6", "#3B82F6"]
        for i, s in enumerate(strats):
            with cols[i]:
                clr = rank_colors[i % len(rank_colors)]
                conf = s.get("confidence", 0)
                st.markdown(f"""
                <div class="rec-card" style="border-top:2px solid {clr};">
                  <div style="display:flex;justify-content:space-between;
                       align-items:center;margin-bottom:12px;">
                    <div style="font-size:9px;font-weight:700;color:{clr};
                         text-transform:uppercase;letter-spacing:0.1em;">
                      Strategy #{s['rank']}
                    </div>
                    <div style="font-size:10px;color:#4B6280;">
                      {conf}% confidence
                    </div>
                  </div>
                  <div style="font-size:1.1rem;font-weight:700;color:#E2E8F0;margin-bottom:8px;">
                    {s['cat_emoji']} {s['category']}
                  </div>
                  <div style="font-size:12px;color:#6B82A0;line-height:1.7;">
                    {s['media_emoji']} {s['media'].title()}<br>
                    Post at {s['hour']}<br>
                    #{s['hashtags']} hashtags<br>
                    CTA: {'Yes' if s['cta'] == 1 else 'No'}
                  </div>
                  <div style="margin-top:12px;padding-top:10px;border-top:1px solid #1F2937;">
                    <div style="font-size:9px;color:#4B6280;text-transform:uppercase;
                         letter-spacing:0.1em;margin-bottom:3px;">Predicted Engagement</div>
                    <div style="font-size:1.3rem;font-weight:800;color:{clr};">
                      {s['pred_engagement']:.2%}
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    divider()

    # Best posting time
    section_header("Best Posting Time")
    col_l, col_r = st.columns(2)

    with col_l:
        hour_data = fdf.groupby("hour")["engagement_rate"].mean().reset_index().sort_values("hour")
        best_h = hour_data.loc[hour_data["engagement_rate"].idxmax(), "hour"]
        fig = go.Figure(go.Bar(
            x=hour_data["hour"],
            y=hour_data["engagement_rate"],
            marker_color=["#F59E0B" if h == best_h else "#3B82F6" for h in hour_data["hour"]],
            text=[f"{v:.2%}" for v in hour_data["engagement_rate"]],
            textposition="outside",
            textfont=dict(color="#94A3B8", size=9),
            hovertemplate="Hour %{x}:00<br>Engagement: %{y:.4f}<extra></extra>",
        ))
        pf(fig, 280, "Engagement Rate by Hour (gold = best)",
           xaxis_title="Hour of Day", yaxis_title="Avg Engagement Rate")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_r:
        if not bdays.empty and "day" in bdays.columns:
            fig2 = go.Figure(go.Bar(
                x=bdays["day"],
                y=bdays["avg_metric"],
                marker=dict(
                    color=bdays["avg_metric"],
                    colorscale=[[0, "#1A2133"], [0.5, "#3B82F6"], [1, "#8B5CF6"]],
                    line=dict(color="rgba(0,0,0,0)"),
                ),
                text=[f"{v:.2%}" for v in bdays["avg_metric"]],
                textposition="outside",
                textfont=dict(color="#94A3B8", size=10),
                hovertemplate="%{x}<br>Engagement: %{y:.4f}<extra></extra>",
            ))
            pf(fig2, 280, "Best Days to Post",
               xaxis_title="Day", yaxis_title="Avg Engagement Rate")
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    divider()

    # Best content type + CTA impact
    section_header("Best Content Type & CTA Impact")
    col_a, col_b = st.columns(2)

    with col_a:
        cat_eng = fdf.groupby("content_category")["engagement_rate"].mean().sort_values(ascending=False).reset_index()
        fig3 = go.Figure(go.Bar(
            x=cat_eng["content_category"],
            y=cat_eng["engagement_rate"],
            marker=dict(color=PAL[:len(cat_eng)], line=dict(color="rgba(0,0,0,0)")),
            text=[f"{v:.2%}" for v in cat_eng["engagement_rate"]],
            textposition="outside",
            textfont=dict(color="#94A3B8", size=10),
            hovertemplate="%{x}<br>Engagement: %{y:.4f}<extra></extra>",
        ))
        pf(fig3, 280, "Best Content Categories",
           xaxis_title="Category", yaxis_title="Avg Engagement Rate")
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        if cta:
            cta_labels = {0: "No CTA", 1: "Has CTA"}
            fig4 = go.Figure(go.Bar(
                x=[cta_labels.get(k, str(k)) for k in cta.keys()],
                y=list(cta.values()),
                marker=dict(color=["#1A2133", "#3B82F6"], line=dict(color="rgba(0,0,0,0)")),
                text=[f"{v:.4f}" for v in cta.values()],
                textposition="outside",
                textfont=dict(color="#94A3B8", size=11),
                hovertemplate="%{x}<br>Avg Engagement: %{y:.4f}<extra></extra>",
            ))
            pf(fig4, 280, "CTA Impact on Engagement",
               xaxis_title="CTA Presence", yaxis_title="Avg Engagement Rate")
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    divider()

    # 7-day content calendar
    section_header("7-Day Content Calendar")
    if not cal.empty:
        display_cal = cal[[c for c in [
            "Day", "Content Category", "Media Type", "Post Time",
            "Hashtags", "CTA", "Predicted Engagement", "Confidence",
        ] if c in cal.columns]].copy()
        st.dataframe(display_cal, use_container_width=True, hide_index=True)

    divider()

    # Suggested topics cards
    section_header("Suggested Topics")
    top_topics = fdf.groupby("content_category")["engagement_rate"].mean().sort_values(ascending=False).head(4)
    topic_cols = st.columns(4)
    topic_colors = ["#3B82F6", "#8B5CF6", "#10B981", "#F59E0B"]
    for i, (cat, eng) in enumerate(top_topics.items()):
        with topic_cols[i]:
            clr = topic_colors[i % len(topic_colors)]
            media_best = (fdf[fdf["content_category"] == cat]
                          .groupby("media_type")["engagement_rate"].mean().idxmax())
            hour_best  = (fdf[fdf["content_category"] == cat]
                          .groupby("hour")["engagement_rate"].mean().idxmax())
            st.markdown(f"""
            <div class="rec-card">
              <div style="font-size:9px;font-weight:700;color:{clr};
                   text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">
                Suggested Topic #{i+1}
              </div>
              <div style="font-size:1rem;font-weight:700;color:#E2E8F0;margin-bottom:8px;">{cat}</div>
              <div style="font-size:11px;color:#6B82A0;line-height:1.8;">
                Best format: {media_best.title()}<br>
                Best hour: {hour_best:02d}:00 {'AM' if hour_best < 12 else 'PM'}<br>
                Avg engagement: {eng:.2%}
              </div>
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE: TREND FORECASTING
# ═════════════════════════════════════════════════════════════
elif page == "Trend Forecasting":

    section_header("Trend Forecasting Module")

    info_banner("""
    Linear regression slope on weekly engagement data to compute trend velocity.
    Rising topics show positive velocity; declining topics show negative velocity.
    14-day forward forecasts with 95% confidence intervals.
    """)

    with st.spinner("Computing trend velocities..."):
        trend_df = compute_trend_velocities(fdf)
        kw_data  = load_keyword_trends()
        hash_df  = hashtag_effectiveness(fdf)

    # KPIs
    if not trend_df.empty:
        rising_cnt   = (trend_df["velocity_pct"] > 0.5).sum()
        declining_cnt = (trend_df["velocity_pct"] < -0.5).sum()
        top_rising   = trend_df.iloc[0]["content_category"] if len(trend_df) > 0 else "N/A"
        top_vel      = trend_df.iloc[0]["velocity_pct"] if len(trend_df) > 0 else 0

        t1, t2, t3, t4 = st.columns(4)
        with t1: st.markdown(kpi_card("", f"{rising_cnt}",       "Rising Topics",    "#10B981", "#0EA5E9"), unsafe_allow_html=True)
        with t2: st.markdown(kpi_card("", f"{declining_cnt}",    "Declining Topics", "#F43F5E", "#F59E0B"), unsafe_allow_html=True)
        with t3: st.markdown(kpi_card("", top_rising[:16],        "Fastest Rising",   "#3B82F6", "#8B5CF6"), unsafe_allow_html=True)
        with t4: st.markdown(kpi_card("", f"{top_vel:+.1f}%/wk", "Top Velocity",     "#F59E0B", "#FB923C"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        divider()

    # Trend velocity chart
    section_header("Category Trend Velocities")
    if not trend_df.empty:
        fig = go.Figure(go.Bar(
            x=trend_df["velocity_pct"],
            y=trend_df["content_category"],
            orientation="h",
            marker=dict(
                color=["#10B981" if v > 0 else "#F43F5E" for v in trend_df["velocity_pct"]],
                line=dict(color="rgba(0,0,0,0)"),
            ),
            text=[f"{v:+.1f}%/wk" for v in trend_df["velocity_pct"]],
            textposition="outside",
            textfont=dict(color="#94A3B8", size=10),
            hovertemplate="%{y}<br>Velocity: %{x:+.2f}%/week<extra></extra>",
        ))
        fig.add_vline(x=0, line_color="#1F2937", line_width=1)
        pf(fig, 320, "Trend Velocity by Category (% change per week)",
           xaxis_title="Velocity (%/week)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    divider()

    # Forecast section
    section_header("14-Day Engagement Forecast")
    all_cats = sorted(fdf["content_category"].dropna().unique().tolist())
    sel_cat  = st.selectbox("Select Category to Forecast", all_cats)

    with st.spinner(f"Forecasting {sel_cat}..."):
        fc_df = forecast_engagement(fdf, sel_cat, n_forecast=14)

    if not fc_df.empty:
        hist = fc_df[fc_df["type"] == "historical"]
        fore = fc_df[fc_df["type"] == "forecast"]

        fig2 = go.Figure()

        # CI band
        if not fore.empty:
            fig2.add_trace(go.Scatter(
                x=list(fore["date"]) + list(fore["date"])[::-1],
                y=list(fore["upper_ci"]) + list(fore["lower_ci"])[::-1],
                fill="toself",
                fillcolor="rgba(139,92,246,0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
                name="95% CI",
            ))

        # Historical
        if not hist.empty:
            fig2.add_trace(go.Scatter(
                x=hist["date"], y=hist["actual"],
                mode="lines+markers",
                line=dict(color="#3B82F6", width=2.5),
                marker=dict(size=6, color="#3B82F6", line=dict(color="#0E1117", width=2)),
                name="Historical",
                hovertemplate="%{x}<br>Engagement: %{y:.4f}<extra></extra>",
            ))

        # Forecast line
        if not fore.empty:
            fig2.add_trace(go.Scatter(
                x=fore["date"], y=fore["forecast"],
                mode="lines+markers",
                line=dict(color="#8B5CF6", width=2.5, dash="dot"),
                marker=dict(size=5, color="#8B5CF6"),
                name="Forecast",
                hovertemplate="%{x}<br>Forecast: %{y:.4f}<extra></extra>",
            ))

        pf(fig2, 320, f"14-Day Forecast — {sel_cat}",
           xaxis_title="Date", yaxis_title="Engagement Rate")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    divider()

    # Trending keywords
    section_header("Trending Keywords & Hashtags")
    col_l, col_r = st.columns(2)

    rising_kws   = kw_data.get("rising",   [])
    declining_kws = kw_data.get("declining", [])

    with col_l:
        st.markdown("""
        <div style="font-size:10px;font-weight:700;color:#10B981;
             text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;">
          Rising Keywords
        </div>
        """, unsafe_allow_html=True)
        for kw in rising_kws:
            vel = kw.get("velocity", 0)
            cat = kw.get("category", "")
            keyword = kw.get("keyword", "")
            bar_w = min(int(vel / 4), 100)
            st.markdown(f"""
            <div class="trend-card">
              <div>
                <div class="trend-keyword">#{keyword}</div>
                <div class="trend-category">{cat}</div>
              </div>
              <div>
                <div class="trend-velocity-up">+{vel}</div>
                <div style="height:3px;width:60px;background:#1F2937;border-radius:2px;
                     margin-top:4px;overflow:hidden;">
                  <div style="height:100%;width:{bar_w}%;background:#10B981;border-radius:2px;"></div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div style="font-size:10px;font-weight:700;color:#F43F5E;
             text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;">
          Declining Keywords
        </div>
        """, unsafe_allow_html=True)
        for kw in declining_kws:
            vel = kw.get("velocity", 0)
            cat = kw.get("category", "")
            keyword = kw.get("keyword", "")
            bar_w = min(int(abs(vel) / 2), 100)
            st.markdown(f"""
            <div class="trend-card">
              <div>
                <div class="trend-keyword">#{keyword}</div>
                <div class="trend-category">{cat}</div>
              </div>
              <div>
                <div class="trend-velocity-down">{vel}</div>
                <div style="height:3px;width:60px;background:#1F2937;border-radius:2px;
                     margin-top:4px;overflow:hidden;">
                  <div style="height:100%;width:{bar_w}%;background:#F43F5E;border-radius:2px;"></div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    divider()

    # Hashtag effectiveness
    section_header("Hashtag Count Effectiveness")
    if not hash_df.empty:
        optimal_h = int(hash_df.loc[hash_df["avg_engagement"].idxmax(), "hashtags_count"])
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=hash_df["hashtags_count"],
            y=hash_df["avg_engagement"],
            mode="lines+markers",
            line=dict(color="#3B82F6", width=2.5),
            marker=dict(
                size=[12 if h == optimal_h else 7 for h in hash_df["hashtags_count"]],
                color=["#F59E0B" if h == optimal_h else "#3B82F6" for h in hash_df["hashtags_count"]],
                line=dict(color="#0E1117", width=2),
            ),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.05)",
            hovertemplate="Hashtags: %{x}<br>Avg Engagement: %{y:.4f}<extra></extra>",
            name="Avg Engagement",
        ))
        fig3.add_vline(
            x=optimal_h,
            line_color="#F59E0B",
            line_width=1.5,
            line_dash="dot",
            annotation_text=f"Optimal: {optimal_h}",
            annotation_font_color="#F59E0B",
            annotation_font_size=11,
        )
        pf(fig3, 280, "Hashtag Count vs Engagement Rate (gold = optimal)",
           xaxis_title="Number of Hashtags", yaxis_title="Avg Engagement Rate")
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    divider()

    # Rising topics highlight cards
    section_header("Rising Topics")
    if not trend_df.empty:
        rising = trend_df[trend_df["velocity_pct"] > 0].head(4)
        if not rising.empty:
            r_cols = st.columns(len(rising))
            for i, (_, row) in enumerate(rising.iterrows()):
                with r_cols[i]:
                    vel = row["velocity_pct"]
                    clr = "#10B981" if vel > 2 else "#3B82F6"
                    st.markdown(f"""
                    <div class="rec-card" style="border-top:2px solid {clr};">
                      <div style="font-size:9px;font-weight:700;color:{clr};
                           text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">
                        {row['trend_label']}
                      </div>
                      <div style="font-size:1rem;font-weight:700;color:#E2E8F0;margin-bottom:8px;">
                        {row['content_category']}
                      </div>
                      <div style="font-size:11px;color:#6B82A0;line-height:1.8;">
                        Velocity: <strong style="color:{clr};">{vel:+.1f}%/wk</strong><br>
                        Avg Engagement: {row['avg_engagement']:.2%}<br>
                        R²: {row['r2']:.3f}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

    # Trend velocity table
    if not trend_df.empty:
        divider()
        section_header("Full Trend Velocity Table")
        display_trend = trend_df[[c for c in [
            "content_category", "trend_label", "velocity_pct",
            "avg_engagement", "slope", "r2", "n_weeks",
        ] if c in trend_df.columns]].copy()
        display_trend.columns = [
            {"content_category": "Category", "trend_label": "Trend",
             "velocity_pct": "Velocity (%/wk)", "avg_engagement": "Avg Engagement",
             "slope": "Slope", "r2": "R²", "n_weeks": "Weeks"}.get(c, c)
            for c in display_trend.columns
        ]
        for col in ["Velocity (%/wk)", "Avg Engagement", "Slope", "R²"]:
            if col in display_trend.columns:
                display_trend[col] = display_trend[col].round(4)
        st.dataframe(display_trend, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════
# PAGE: SCRAPE & TRAIN
# ═════════════════════════════════════════════════════════════
elif page == "Scrape & Train":

    from modules.data_engine import (
        append_scraped_data, retrain_model,
        load_training_log, get_dataframe, AUGMENTED_CSV,
    )
    import plotly.graph_objects as go

    section_header("Scrape & Train — Incremental Learning")

    info_banner("""
    Scrape real Instagram reels by hashtag, predict their engagement with the current model,
    then append the scraped rows to the training pool and retrain.
    Each run improves the model with fresh real-world data.
    <br><br>
    <strong style="color:#F59E0B;">Note:</strong>
    Because Instagram does not expose <code>reach</code>, <code>saves</code>, or
    <code>shares</code> publicly, those values are estimated from scraped signals
    (likes, comments, views, follower count). The predicted engagement rate is used
    as the training label — so the model learns from its own predictions on real posts,
    progressively refining itself with each scrape run.
    """)

    # ── Current model stats ───────────────────────────────────
    history = load_training_log()
    last    = history[-1] if history else None

    aug_count = 0
    if os.path.exists(AUGMENTED_CSV):
        try:
            aug_count = len(pd.read_csv(AUGMENTED_CSV))
        except Exception:
            aug_count = 0

    section_header("Current Model State")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(kpi_card("", f"{last['mae']:.5f}" if last else "Base",
                             "MAE", "#3B82F6", "#8B5CF6"), unsafe_allow_html=True)
    with s2:
        st.markdown(kpi_card("", f"{last['r2']:.4f}" if last else "Base",
                             "R²", "#8B5CF6", "#3B82F6"), unsafe_allow_html=True)
    with s3:
        st.markdown(kpi_card("", f"{last['n_train']:,}" if last else "—",
                             "Training Rows", "#10B981", "#0EA5E9"), unsafe_allow_html=True)
    with s4:
        st.markdown(kpi_card("", f"{aug_count:,}",
                             "Augmented Rows", "#F59E0B", "#FB923C"), unsafe_allow_html=True)

    divider()

    # ── Scraper config form ───────────────────────────────────
    section_header("Scraper Configuration")

    with st.form("scrape_train_form", clear_on_submit=False):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown('<div style="font-size:11px;color:#6B82A0;margin-bottom:6px;">Instagram Credentials</div>',
                        unsafe_allow_html=True)
            ig_user = st.text_input("Username", value="your_username",
                                    placeholder="Instagram username")
            ig_pass = st.text_input("Password", value="",
                                    type="password", placeholder="Instagram password")

        with col_b:
            st.markdown('<div style="font-size:11px;color:#6B82A0;margin-bottom:6px;">Scrape Target</div>',
                        unsafe_allow_html=True)
            hashtags_input = st.text_input(
                "Hashtags (comma-separated, no #)",
                value="coding, fitness, travel",
                placeholder="coding, fitness, food",
                help="Enter one or more hashtags. The scraper will cycle through each.",
            )
            max_per_tag = st.number_input(
                "Reels per hashtag", min_value=1, max_value=30,
                value=5, step=1,
                help="How many reels to scrape per hashtag. Keep low to avoid rate limits.",
            )

        with col_c:
            st.markdown('<div style="font-size:11px;color:#6B82A0;margin-bottom:6px;">Options</div>',
                        unsafe_allow_html=True)
            scroll_passes = st.number_input("Scroll passes", min_value=1, max_value=8,
                                            value=3, step=1)
            headless      = st.checkbox("Headless browser", value=False,
                                        help="Run Chrome without opening a visible window")
            retrain_after = st.checkbox("Retrain model after scraping", value=True,
                                        help="Immediately retrain on base + new data after scraping")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Start Scraping")

    # ── Retrain-only button (outside form) ────────────────────
    col_btn1, col_btn2, _ = st.columns([2, 2, 6])
    with col_btn1:
        retrain_only_btn = st.button("Retrain Only (skip scrape)")
    with col_btn2:
        clear_aug_btn = st.button("Clear Augmented Data")

    if clear_aug_btn:
        if os.path.exists(AUGMENTED_CSV):
            os.remove(AUGMENTED_CSV)
            st.success("Augmented data cleared. Next retrain uses base Kaggle data only.")
            st.rerun()
        else:
            st.info("No augmented data file found.")

    if retrain_only_btn:
        with st.spinner("Retraining on base + augmented data..."):
            try:
                base_df     = get_dataframe()
                new_metrics = retrain_model(base_df)
                st.cache_resource.clear()
                st.success(
                    f"Retrained. MAE: {new_metrics['mae']}  "
                    f"R²: {new_metrics['r2']}  "
                    f"Trees: {new_metrics['n_trees']}  "
                    f"Augmented rows used: {new_metrics['n_augmented']}"
                )
                if history:
                    prev = history[-1]
                    mae_d = new_metrics["mae"] - prev["mae"]
                    r2_d  = new_metrics["r2"]  - prev["r2"]
                    mae_clr = "#10B981" if mae_d < 0 else "#F43F5E"
                    r2_clr  = "#10B981" if r2_d  > 0 else "#F43F5E"
                    st.markdown(f"""
                    <div style="background:#1A2133;border:1px solid #1F2937;
                         border-radius:10px;padding:12px 16px;font-size:13px;color:#CBD5E1;">
                      MAE change: <strong style="color:{mae_clr};">{mae_d:+.5f}</strong>
                      &nbsp;&nbsp;
                      R² change: <strong style="color:{r2_clr};">{r2_d:+.4f}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Retraining failed: {e}")

    # ── Main scrape + train pipeline ──────────────────────────
    if submitted:
        # Validate inputs
        if not ig_user or not ig_pass:
            st.error("Enter your Instagram username and password.")
            st.stop()

        hashtag_list = [h.strip().lstrip("#") for h in hashtags_input.split(",") if h.strip()]
        if not hashtag_list:
            st.error("Enter at least one hashtag.")
            st.stop()

        # Import scraper pieces from checktrend
        try:
            from modules.checktrend import (
                InstagramScraper, train_model_standalone, compute_defaults,
                build_feature_vector, classify_engagement_tier,
                compute_viral_coefficient,
            )
        except ImportError as e:
            st.error(f"Could not import scraper: {e}")
            st.stop()

        total_tags   = len(hashtag_list)
        all_results  = []
        progress_bar = st.progress(0, text="Initialising...")
        status_box   = st.empty()
        results_box  = st.empty()

        # Step 1 — load prediction model
        status_box.info("Loading model for predictions...")
        try:
            pipeline, model_stats, df_defaults = train_model_standalone()
            defaults  = compute_defaults(df_defaults)
            all_feats = model_stats["features"]
            cat_feats = model_stats["cat_feats"]
        except Exception as e:
            st.error(f"Model load failed: {e}")
            st.stop()

        # Step 2 — launch browser once, loop over hashtags
        cfg = {
            "username":     ig_user,
            "password":     ig_pass,
            "hashtag":      hashtag_list[0],   # will be overridden per tag
            "max_posts":    int(max_per_tag),
            "scroll_passes": int(scroll_passes),
            "page_delay":   2.5,
            "headless":     headless,
            "output_csv":   "data/real_reel_predictions.csv",
            "category_keywords": {
                "fitness": "Fitness", "workout": "Fitness", "gym": "Fitness",
                "health": "Fitness", "beauty": "Beauty", "skincare": "Beauty",
                "makeup": "Beauty", "travel": "Travel", "food": "Food",
                "recipe": "Food", "cooking": "Food", "tech": "Technology",
                "coding": "Technology", "programming": "Technology",
                "software": "Technology", "music": "Music",
                "photo": "Photography", "comedy": "Comedy",
                "funny": "Comedy", "meme": "Comedy",
            },
        }

        scraper = InstagramScraper(cfg)
        try:
            status_box.info("Logging in to Instagram...")
            scraper.login()
            progress_bar.progress(10, text="Logged in.")

            for tag_idx, tag in enumerate(hashtag_list):
                tag_pct_start = 10 + int(tag_idx / total_tags * 70)
                tag_pct_end   = 10 + int((tag_idx + 1) / total_tags * 70)

                status_box.info(f"Scraping #{tag}  ({tag_idx+1}/{total_tags})...")
                scraper.cfg["hashtag"] = tag
                links = scraper.collect_reel_links()

                if not links:
                    st.warning(f"No links found for #{tag} — skipping.")
                    continue

                for post_idx, link in enumerate(links):
                    post = scraper.scrape_post(link)
                    if post.get("username"):
                        fc = scraper.get_follower_count(post["username"])
                        post["follower_count"] = fc if fc > 0 else int(defaults["follower_count"])
                    else:
                        post["follower_count"] = int(defaults["follower_count"])

                    feat_vec, category = build_feature_vector(
                        scraped=post, defaults=defaults,
                        all_features=all_feats, cat_features=cat_feats,
                        hashtag_for_category=tag,
                    )

                    input_df = pd.DataFrame([feat_vec])
                    for c in cat_feats:
                        if c in input_df.columns:
                            input_df[c] = input_df[c].astype(str)

                    try:
                        pred = max(0.0, float(pipeline.predict(input_df)[0]))
                    except Exception:
                        pred = float(defaults.get("engagement_rate_median", 0.03))

                    vc         = compute_viral_coefficient(feat_vec)
                    tier_label = classify_engagement_tier(pred)[0]

                    row = {
                        # identifiers
                        "link":               post.get("link", ""),
                        "username":           post.get("username", ""),
                        "hashtag":            tag,
                        "scraped_at":         datetime.datetime.now().isoformat(timespec="seconds"),
                        # scraped signals
                        "follower_count":     post.get("follower_count", 0),
                        "likes":              post.get("likes", 0),
                        "comments":           post.get("comments_count", 0),
                        "views":              post.get("views", 0),
                        "caption_length":     post.get("caption_length", 0),
                        "hashtags_count":     post.get("hashtags_count", 0),
                        "has_call_to_action": post.get("has_cta", 0),
                        "media_type":         feat_vec.get("media_type", "reel"),
                        "content_category":   category,
                        "hour":               post.get("hour", 9),
                        "day_num":            post.get("day_num", 1),
                        "month_num":          post.get("month_num", 4),
                        # estimated (needed as model features)
                        "reach":              feat_vec.get("reach", 0),
                        "impressions":        feat_vec.get("impressions", 0),
                        "shares":             int(defaults.get("shares", 0)),
                        "saves":              int(defaults.get("saves", 0)),
                        "followers_gained":   int(defaults.get("followers_gained", 0)),
                        "account_type":       defaults.get("account_type", "creator"),
                        "traffic_source":     "Hashtags",
                        # label for training
                        "engagement_rate":    round(pred, 6),
                        # extra info
                        "predicted_engagement": round(pred, 6),
                        "viral_coefficient":    vc,
                        "engagement_tier":      tier_label,
                    }
                    all_results.append(row)

                    # live progress
                    done_posts = tag_idx * int(max_per_tag) + post_idx + 1
                    total_posts = total_tags * int(max_per_tag)
                    pct = tag_pct_start + int((post_idx + 1) / len(links) * (tag_pct_end - tag_pct_start))
                    progress_bar.progress(min(pct, 79), text=f"#{tag} — post {post_idx+1}/{len(links)}")

                    # live preview table
                    preview_df = pd.DataFrame(all_results)[[
                        "hashtag", "username", "content_category", "media_type",
                        "predicted_engagement", "viral_coefficient", "engagement_tier",
                        "likes", "comments", "views", "hashtags_count",
                    ]]
                    results_box.dataframe(preview_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Scraping error: {e}")
        finally:
            scraper.close()

        if not all_results:
            st.error("No posts were scraped. Check credentials and hashtags.")
            st.stop()

        progress_bar.progress(80, text=f"Scraped {len(all_results)} posts. Saving...")

        # Step 3 — append to augmented data
        rows_added, total_aug = append_scraped_data(all_results)
        progress_bar.progress(85, text=f"Saved {rows_added} new rows ({total_aug} total augmented).")

        st.markdown("<br>", unsafe_allow_html=True)
        divider()
        section_header("Scrape Results")

        results_df = pd.DataFrame(all_results)
        display_cols = [c for c in [
            "hashtag", "username", "content_category", "media_type",
            "predicted_engagement", "viral_coefficient", "engagement_tier",
            "likes", "comments", "views", "hashtags_count", "has_call_to_action",
            "hour", "content_category",
        ] if c in results_df.columns]
        # deduplicate display cols
        seen_dc = []
        for c in display_cols:
            if c not in seen_dc:
                seen_dc.append(c)

        st.dataframe(results_df[seen_dc], use_container_width=True, hide_index=True)
        st.caption(f"{rows_added} new rows added  ·  {total_aug} total augmented rows  ·  {len(all_results) - rows_added} duplicates skipped")

        # Step 4 — retrain
        if retrain_after:
            progress_bar.progress(88, text="Retraining model...")
            status_box.info(f"Retraining on Kaggle base + {total_aug} augmented rows...")
            try:
                base_df     = get_dataframe()
                new_metrics = retrain_model(base_df)
                st.cache_resource.clear()
                progress_bar.progress(100, text="Done.")
                status_box.empty()

                divider()
                section_header("Retraining Results")

                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    st.markdown(kpi_card("", f"{new_metrics['mae']:.5f}", "New MAE",
                                         "#3B82F6", "#8B5CF6"), unsafe_allow_html=True)
                with r2:
                    st.markdown(kpi_card("", f"{new_metrics['r2']:.4f}", "New R²",
                                         "#8B5CF6", "#3B82F6"), unsafe_allow_html=True)
                with r3:
                    st.markdown(kpi_card("", f"{new_metrics['n_train']:,}", "Training Rows",
                                         "#10B981", "#0EA5E9"), unsafe_allow_html=True)
                with r4:
                    st.markdown(kpi_card("", f"{new_metrics['n_augmented']:,}", "Augmented Rows",
                                         "#F59E0B", "#FB923C"), unsafe_allow_html=True)

                # Delta vs previous run
                history = load_training_log()
                if len(history) >= 2:
                    prev  = history[-2]
                    mae_d = new_metrics["mae"] - prev["mae"]
                    r2_d  = new_metrics["r2"]  - prev["r2"]
                    mae_clr = "#10B981" if mae_d < 0 else "#F43F5E"
                    r2_clr  = "#10B981" if r2_d  > 0 else "#F43F5E"
                    mae_note = "lower is better" if mae_d < 0 else "needs more data"
                    r2_note  = "improved"        if r2_d  > 0 else "slight drop — normal"
                    st.markdown(f"""
                    <div style="background:#1A2133;border:1px solid #1F2937;
                         border-left:3px solid #3B82F6;border-radius:10px;
                         padding:14px 18px;font-size:13px;color:#CBD5E1;margin-top:12px;">
                      <div style="font-size:9px;font-weight:700;color:#4B6280;
                           text-transform:uppercase;letter-spacing:.12em;margin-bottom:8px;">
                        vs Previous Run
                      </div>
                      MAE: <strong style="color:{mae_clr};">{mae_d:+.5f}</strong>
                      <span style="color:{mae_clr};font-size:10px;"> ({mae_note})</span>
                      &nbsp;&nbsp;&nbsp;
                      R²: <strong style="color:{r2_clr};">{r2_d:+.4f}</strong>
                      <span style="color:{r2_clr};font-size:10px;"> ({r2_note})</span>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Retraining failed: {e}")
        else:
            progress_bar.progress(100, text="Done. Retrain skipped.")
            status_box.empty()

    divider()

    # ── Training history chart ────────────────────────────────
    history = load_training_log()
    if len(history) > 0:
        section_header("Training History")

        hist_df = pd.DataFrame(history)
        hist_df["run"] = range(1, len(hist_df) + 1)

        if len(history) > 1:
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(
                x=hist_df["run"], y=hist_df["mae"],
                mode="lines+markers", name="MAE",
                line=dict(color="#3B82F6", width=2.5),
                marker=dict(size=7, color="#3B82F6", line=dict(color="#0E1117", width=2)),
                yaxis="y1",
                hovertemplate="Run %{x}<br>MAE: %{y:.5f}<extra></extra>",
            ))
            fig_h.add_trace(go.Scatter(
                x=hist_df["run"], y=hist_df["r2"],
                mode="lines+markers", name="R²",
                line=dict(color="#8B5CF6", width=2.5),
                marker=dict(size=7, color="#8B5CF6", line=dict(color="#0E1117", width=2)),
                yaxis="y2",
                hovertemplate="Run %{x}<br>R²: %{y:.4f}<extra></extra>",
            ))
            fig_h.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#6B82A0"),
                height=260,
                margin=dict(t=36, b=10, l=10, r=10),
                xaxis=dict(
                    title="Training Run", gridcolor="#1A2133",
                    tickfont=dict(color="#4B6280"), linecolor="#1F2937",
                ),
                yaxis=dict(
                    title="MAE", gridcolor="#1A2133", linecolor="#1F2937",
                    tickfont=dict(color="#3B82F6"),
                    title_font=dict(color="#3B82F6"),
                ),
                yaxis2=dict(
                    title="R²", overlaying="y", side="right",
                    tickfont=dict(color="#8B5CF6"),
                    title_font=dict(color="#8B5CF6"),
                    gridcolor="rgba(0,0,0,0)",
                ),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#6B82A0")),
                hoverlabel=dict(bgcolor="#1A2133", bordercolor="#2D3F55",
                                font=dict(color="#E2E8F0")),
                title=dict(text="Model Accuracy Over Training Runs",
                           font=dict(color="#94A3B8", size=12), x=0),
            )
            st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})

        # History table
        show_cols = [c for c in ["run", "timestamp", "mae", "r2",
                                  "n_train", "n_augmented", "n_trees"] if c in hist_df.columns]
        st.dataframe(hist_df[show_cols], use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div style="background:#1A2133;border:1px solid #1F2937;border-radius:10px;
             padding:20px;text-align:center;color:#4B6280;font-size:13px;">
          No training runs yet. Run a scrape above to start building history.
        </div>
        """, unsafe_allow_html=True)

    divider()

    # ── Augmented data preview ────────────────────────────────
    if os.path.exists(AUGMENTED_CSV):
        section_header("Augmented Data Pool")
        try:
            aug_df = pd.read_csv(AUGMENTED_CSV)
            st.caption(f"{len(aug_df):,} rows in augmented pool")
            preview_cols = [c for c in [
                "hashtag", "scraped_at", "username", "content_category",
                "media_type", "engagement_rate", "likes", "comments",
                "views", "hashtags_count",
            ] if c in aug_df.columns]
            st.dataframe(aug_df[preview_cols].tail(50), use_container_width=True, hide_index=True)
            st.caption("Showing last 50 rows")
        except Exception as e:
            st.warning(f"Could not read augmented data: {e}")
