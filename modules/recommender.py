"""
Module 5 — Engagement Optimization Recommender
================================================
A prescriptive analytics engine that recommends the optimal content
strategy for the coming 7 days based on historical performance data.
Factors: best hour × best category × best media type × best CTA × best hashtag count.
"""

import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, List


DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"]

MEDIA_EMOJIS = {"reel": "🎬", "image": "🖼️", "carousel": "🎠", "video": "📹"}
CATEGORY_EMOJIS = {
    "Technology": "💻", "Fitness": "💪", "Music": "🎵",
    "Beauty": "💄", "Travel": "✈️", "Food": "🍜",
    "Photography": "📷", "Comedy": "😂",
}


def _best_n(df: pd.DataFrame, group_col: str, metric: str, n: int = 3) -> pd.DataFrame:
    """Return the top-n values for a given grouping column by metric mean."""
    if group_col not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby(group_col)[metric]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_metric", "count": "n_posts"})
        .query("n_posts >= 5")
        .sort_values("avg_metric", ascending=False)
        .head(n)
    )


def _compute_optimal_hashtags(df: pd.DataFrame, metric: str = "engagement_rate") -> int:
    """Find the hashtag count with the highest average engagement."""
    if "hashtags_count" not in df.columns:
        return 7
    res = df.groupby("hashtags_count")[metric].mean()
    return int(res.idxmax())


def _confidence_score(n_posts: int, effect_size: float) -> int:
    """
    Heuristic confidence 1–100 based on sample size and effect magnitude.
    """
    base = min(50, n_posts / 10 * 10)
    effect_bonus = min(50, effect_size * 100)
    return int(min(100, base + effect_bonus))


def generate_recommendations(
    df: pd.DataFrame,
    metric: str = "engagement_rate",
    n_strategies: int = 3,
) -> Dict:
    """
    Main entry point for Module 5.
    Returns a dict with:
      - strategy_table   : per-day content calendar DataFrame
      - top_strategies   : top-3 overall strategies
      - factor_scores    : dict of best values per factor
      - confidence_chart : list of (strategy, confidence) for chart
    """
    df = df.copy()

    # ── Best factors ─────────────────────────────────────────
    best_hours    = _best_n(df, "hour",             metric, 3)
    best_cats     = _best_n(df, "content_category", metric, 3)
    best_media    = _best_n(df, "media_type",        metric, 3)
    optimal_hashes= _compute_optimal_hashtags(df, metric)

    # CTA impact
    cta_impact = {}
    if "has_call_to_action" in df.columns:
        cta_agg = df.groupby("has_call_to_action")[metric].mean()
        cta_impact = {int(k): round(v, 5) for k, v in cta_agg.items()}
    best_cta = max(cta_impact, key=cta_impact.get) if cta_impact else 1

    # Day of week impact
    best_days = _best_n(df, "day", metric, 7) if "day" in df.columns else pd.DataFrame()

    factor_scores = {
        "best_hour":       int(best_hours.iloc[0][best_hours.columns[0]]) if not best_hours.empty else 9,
        "best_category":   str(best_cats.iloc[0]["content_category"]) if not best_cats.empty else "Fitness",
        "best_media":      str(best_media.iloc[0]["media_type"])      if not best_media.empty else "reel",
        "optimal_hashtags":optimal_hashes,
        "best_cta":        best_cta,
        "top_categories":  best_cats["content_category"].tolist() if not best_cats.empty else [],
        "top_hours":       best_hours["hour"].tolist()            if not best_hours.empty else [],
        "top_media":       best_media["media_type"].tolist()      if not best_media.empty else [],
    }

    # ── 7-day content calendar ───────────────────────────────
    cats_cycle    = (best_cats["content_category"].tolist()   or ["Fitness"]) * 3
    media_cycle   = (best_media["media_type"].tolist()        or ["reel"])    * 4
    hours_cycle   = (best_hours["hour"].astype(int).tolist()  or [9])         * 4

    calendar_rows = []
    for i, day in enumerate(DAYS_OF_WEEK):
        cat   = cats_cycle[i % len(cats_cycle)]
        media = media_cycle[i % len(media_cycle)]
        hour  = hours_cycle[i % len(hours_cycle)]

        # Predicted engagement for this combo
        combo_df = df[
            (df.get("content_category", pd.Series("", index=df.index)) == cat) &
            (df.get("media_type",       pd.Series("", index=df.index)) == media)
        ]
        pred_eng = combo_df[metric].mean() if len(combo_df) >= 5 else df[metric].mean()
        n_obs    = len(combo_df)

        confidence = _confidence_score(n_obs, pred_eng)

        am_pm = "AM" if hour < 12 else "PM"
        hour12 = hour if hour <= 12 else hour - 12
        hour12 = 12 if hour12 == 0 else hour12

        cat_emoji   = CATEGORY_EMOJIS.get(cat,   "📌")
        media_emoji = MEDIA_EMOJIS.get(media, "📄")

        calendar_rows.append({
            "Day":              day,
            "Content Category": f"{cat_emoji} {cat}",
            "Media Type":       f"{media_emoji} {media.title()}",
            "Post Time":        f"{hour12}:00 {am_pm}",
            "Hashtags":         f"#{optimal_hashes}",
            "CTA":              "✅ Yes" if best_cta == 1 else "❌ No",
            "Predicted Engagement": f"{pred_eng:.2%}",
            "Confidence":       confidence,
            "_pred_raw":        pred_eng,
            "_hour":            hour,
        })

    strategy_table = pd.DataFrame(calendar_rows)

    # ── Top-3 universal strategies ───────────────────────────
    top_strategies = []
    if not best_cats.empty and not best_media.empty and not best_hours.empty:
        for idx in range(min(n_strategies, len(best_cats))):
            cat   = best_cats.iloc[idx]["content_category"] if idx < len(best_cats) else factor_scores["best_category"]
            media = best_media.iloc[idx]["media_type"]       if idx < len(best_media) else factor_scores["best_media"]
            hour  = int(best_hours.iloc[idx]["hour"])         if idx < len(best_hours) else factor_scores["best_hour"]
            n_obs = int(best_cats.iloc[idx]["n_posts"])       if idx < len(best_cats) else 100

            combo_df  = df[(df.get("content_category", pd.Series()) == cat) &
                           (df.get("media_type",       pd.Series()) == media)]
            pred_eng  = combo_df[metric].mean() if len(combo_df) >= 5 else df[metric].mean()
            confidence = _confidence_score(n_obs, pred_eng)
            cat_emoji  = CATEGORY_EMOJIS.get(cat,  "📌")
            media_emoji= MEDIA_EMOJIS.get(media,   "📄")

            am_pm  = "AM" if hour < 12 else "PM"
            hour12 = hour if hour <= 12 else hour - 12
            hour12 = 12 if hour12 == 0 else hour12

            top_strategies.append({
                "rank":         idx + 1,
                "category":     cat,
                "cat_emoji":    cat_emoji,
                "media":        media,
                "media_emoji":  media_emoji,
                "hour":         f"{hour12}:00 {am_pm}",
                "hashtags":     optimal_hashes,
                "cta":          best_cta,
                "pred_engagement": pred_eng,
                "confidence":   confidence,
                "n_obs":        n_obs,
            })

    return {
        "strategy_table":  strategy_table,
        "top_strategies":  top_strategies,
        "factor_scores":   factor_scores,
        "cta_impact":      cta_impact,
        "best_days_df":    best_days,
    }
