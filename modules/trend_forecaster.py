"""
Module 7 — Trend Forecasting Module
======================================
Analyzes engagement trends by content category over time.
Computes "trend velocity" (linear regression slope) to identify
rising vs declining topics.
Projects 14-day forward engagement forecasts.
Also analyses hashtag effectiveness and keyword trend signals.
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def _linear_trend(y: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Fit linear regression y ~ a*x + b.
    Returns (slope, r_squared, y_fitted).
    """
    n = len(y)
    if n < 3:
        return 0.0, 0.0, y
    x = np.arange(n, dtype=float)
    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]
    y_fit = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(r2), y_fit


def compute_trend_velocities(df: pd.DataFrame, metric: str = "engagement_rate") -> pd.DataFrame:
    """
    Compute per-category trend velocity (weekly rolling mean slope).

    Returns DataFrame with columns:
      content_category, weeks, avg_engagement,
      slope (velocity), r2, trend_label, trend_color
    """
    if "week" not in df.columns or "year" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["week_key"] = df["year"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2)

    # Sort globally to get ordered week list
    week_order = (
        df.drop_duplicates("week_key")
        .sort_values(["year", "week"])[["year", "week", "week_key"]]
        .reset_index(drop=True)
    )
    week_rank = {wk: i for i, wk in enumerate(week_order["week_key"])}

    rows = []
    for cat, grp in df.groupby("content_category"):
        weekly = (
            grp.groupby("week_key")[metric]
            .mean()
            .reset_index()
        )
        weekly["week_rank"] = weekly["week_key"].map(week_rank)
        weekly = weekly.sort_values("week_rank")

        y   = weekly[metric].values
        slope, r2, _ = _linear_trend(y)

        # Normalised velocity: slope / mean engagement
        mean_eng = y.mean() if y.mean() > 0 else 1
        velocity = slope / mean_eng * 100  # % change per week

        if velocity >= 2.0:
            label, color = "🚀 Rising",   "#34D399"
        elif velocity >= 0.5:
            label, color = "📈 Growing",  "#38BDF8"
        elif velocity >= -0.5:
            label, color = "➡️ Stable",   "#818CF8"
        elif velocity >= -2.0:
            label, color = "📉 Declining","#F59E0B"
        else:
            label, color = "⚠️ Falling",  "#F87171"

        rows.append({
            "content_category": cat,
            "n_weeks":          len(weekly),
            "avg_engagement":   float(y.mean()),
            "slope":            float(slope),
            "velocity_pct":     float(round(velocity, 2)),
            "r2":               float(round(r2, 3)),
            "trend_label":      label,
            "trend_color":      color,
        })

    return pd.DataFrame(rows).sort_values("velocity_pct", ascending=False).reset_index(drop=True)


def forecast_engagement(
    df: pd.DataFrame,
    category: str,
    n_forecast: int = 14,
    metric: str = "engagement_rate",
) -> pd.DataFrame:
    """
    14-day forward forecast for a specific content category.
    Uses the last 12 weeks of data, fits a linear trend,
    and extrapolates forward.

    Returns DataFrame: date | actual | forecast | lower_ci | upper_ci
    """
    if "week" not in df.columns:
        return pd.DataFrame()

    cat_df = df[df["content_category"] == category].copy() if category in df["content_category"].values else df.copy()

    if "year" not in cat_df.columns:
        cat_df["year"] = 2025

    weekly = (
        cat_df.groupby(["year", "week"])[metric]
        .mean()
        .reset_index()
        .sort_values(["year", "week"])
        .tail(12)
        .reset_index(drop=True)
    )

    if len(weekly) < 3:
        return pd.DataFrame()

    y = weekly[metric].values
    slope, r2, y_fit = _linear_trend(y)
    residuals = y - y_fit
    std_resid = residuals.std() if len(residuals) > 1 else 0.01

    # Build output date-indexed series (daily from last data point)
    last_year, last_week = int(weekly["year"].iloc[-1]), int(weekly["week"].iloc[-1])
    try:
        import datetime
        anchor = datetime.date.fromisocalendar(last_year, last_week, 1)
    except Exception:
        anchor = pd.Timestamp("2025-01-01").date()

    records = []
    # Historical points (one per week resampled to daily for smooth chart)
    for i, (_, row) in enumerate(weekly.iterrows()):
        records.append({
            "day_offset": i * 7,
            "actual":     row[metric],
            "forecast":   float(y_fit[i]),
            "lower_ci":   float(y_fit[i] - 1.96 * std_resid),
            "upper_ci":   float(y_fit[i] + 1.96 * std_resid),
            "type":       "historical",
        })

    # Forecast points
    last_x  = len(weekly) - 1
    for j in range(1, n_forecast + 1):
        x_new = last_x + j / 7  # daily resolution
        f_val = np.polyval([slope, y_fit[-1] - slope * last_x], x_new)
        f_val = max(0, f_val)
        records.append({
            "day_offset": len(weekly) * 7 + (j - 1),
            "actual":     None,
            "forecast":   float(round(f_val, 5)),
            "lower_ci":   float(round(max(0, f_val - 1.96 * std_resid), 5)),
            "upper_ci":   float(round(f_val + 1.96 * std_resid, 5)),
            "type":       "forecast",
        })

    result = pd.DataFrame(records)
    import datetime
    result["date"] = result["day_offset"].apply(
        lambda d: anchor + datetime.timedelta(days=int(d))
    )
    return result[["date", "actual", "forecast", "lower_ci", "upper_ci", "type"]]


def hashtag_effectiveness(df: pd.DataFrame, metric: str = "engagement_rate") -> pd.DataFrame:
    """
    Compute average engagement per hashtag count and
    identify the optimal range (sweet spot).
    """
    if "hashtags_count" not in df.columns:
        return pd.DataFrame()
    result = (
        df.groupby("hashtags_count")[metric]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_engagement", "count": "n_posts"})
        .query("n_posts >= 10")
        .sort_values("hashtags_count")
    )
    optimal = int(result.loc[result["avg_engagement"].idxmax(), "hashtags_count"])
    result["is_optimal"] = result["hashtags_count"] == optimal
    return result


def load_keyword_trends() -> Dict:
    """Load the keyword trend seed data from data/."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "data", "comments_sample.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("trend_keywords", {})
    return {}
