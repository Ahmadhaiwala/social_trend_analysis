"""
Module 2 — Virality Prediction Engine
=======================================
Computes Viral Coefficients (VC) per post and per content topic.
Formula: VC = (shares × 2 + saves × 3) / max(reach, 1) × 1000
High-value actions (shares, saves) are weighted above passive actions (likes).
"""

import pandas as pd
import numpy as np


# ── Weights for each action ──────────────────────────────────
ACTION_WEIGHTS = {
    "saves":  3.0,   # highest value: saves = strong intent to revisit
    "shares": 2.0,   # high value:   shares = organic distribution
    "likes":  1.0,   # baseline:     likes  = passive acknowledgement
    "comments": 1.5, # medium:       comments = active engagement
}


def compute_viral_coefficients(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of df with added columns:
      - vc_raw          : raw viral coefficient per post
      - vc_percentile   : percentile rank (0–100)
      - vc_tier         : Low / Rising / Viral / Mega-Viral
      - high_value_ratio: (saves+shares) / total_actions
    """
    df = df.copy()

    saves    = df["saves"].fillna(0)   if "saves"    in df.columns else pd.Series(0, index=df.index)
    shares   = df["shares"].fillna(0)  if "shares"   in df.columns else pd.Series(0, index=df.index)
    likes    = df["likes"].fillna(0)   if "likes"    in df.columns else pd.Series(0, index=df.index)
    comments = df["comments"].fillna(0) if "comments" in df.columns else pd.Series(0, index=df.index)
    reach    = df["reach"].fillna(1).replace(0, 1) if "reach" in df.columns else pd.Series(1, index=df.index)

    # Core formula
    df["vc_raw"] = (
        saves    * ACTION_WEIGHTS["saves"]    +
        shares   * ACTION_WEIGHTS["shares"]   +
        comments * ACTION_WEIGHTS["comments"]
    ) / reach * 1000

    # High-value ratio = share of actions that are saves or shares
    total_actions = saves + shares + likes + comments
    df["high_value_ratio"] = (saves + shares) / total_actions.replace(0, 1)

    # Percentile rank
    df["vc_percentile"] = df["vc_raw"].rank(pct=True) * 100

    # Tier classification
    p50, p80, p95 = df["vc_raw"].quantile([0.5, 0.8, 0.95])
    def _vc_tier(v):
        if v >= p95: return "Mega-Viral"
        if v >= p80: return "Viral"
        if v >= p50: return "Rising"
        return "Low"
    df["vc_tier"] = df["vc_raw"].apply(_vc_tier)

    return df


def topic_viral_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate VC by content_category.
    Returns a ranked DataFrame with columns:
      content_category, avg_vc, avg_saves, avg_shares, avg_likes,
      save_to_share_ratio, organic_growth_score, rank
    """
    metrics = {
        "vc_raw":             "mean",
        "saves":              "mean",
        "shares":             "mean",
        "likes":              "mean",
        "high_value_ratio":   "mean",
        "engagement_rate":    "mean",
        "post_id" if "post_id" in df.columns else "engagement_rate": "count",
    }
    # safe aggregation
    agg_cols = {k: v for k, v in metrics.items() if k in df.columns}
    ranked = df.groupby("content_category").agg(agg_cols).reset_index()

    # Save-to-share ratio
    if "saves" in ranked.columns and "shares" in ranked.columns:
        ranked["save_to_share_ratio"] = ranked["saves"] / ranked["shares"].replace(0, np.nan)
        ranked["save_to_share_ratio"] = ranked["save_to_share_ratio"].fillna(ranked["saves"])

    # Organic Growth Score: combines VC + save_to_share
    if "save_to_share_ratio" in ranked.columns:
        ranked["organic_growth_score"] = (
            ranked["vc_raw"] * 0.6 + ranked["save_to_share_ratio"] * 0.4
        )
    else:
        ranked["organic_growth_score"] = ranked["vc_raw"]

    ranked = ranked.sort_values("organic_growth_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    return ranked


def passive_vs_highvalue_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per-category breakdown of passive (likes) vs high-value (saves+shares) actions.
    Useful for the stacked bar chart showing action quality.
    """
    saves  = df["saves"].fillna(0)  if "saves"  in df.columns else pd.Series(0, index=df.index)
    shares = df["shares"].fillna(0) if "shares" in df.columns else pd.Series(0, index=df.index)
    likes  = df["likes"].fillna(0)  if "likes"  in df.columns else pd.Series(0, index=df.index)
    df = df.copy()
    df["_high_value"] = saves + shares
    df["_passive"]    = likes

    summary = df.groupby("content_category").agg(
        high_value=("_high_value", "mean"),
        passive=("_passive",    "mean"),
    ).reset_index()
    summary["high_value_pct"] = summary["high_value"] / (summary["high_value"] + summary["passive"]) * 100
    return summary.sort_values("high_value_pct", ascending=False)
