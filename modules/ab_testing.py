"""
Module 4 — A/B Testing Framework
===================================
Runs controlled experiments comparing two groups on engagement metrics.
Uses Welch's two-sample t-test (unequal variance) from scipy.stats.
Reports: p-value, Cohen's d effect size, confidence interval, winner.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict, Any


EFFECT_SIZE_LABELS = {
    (0.0, 0.2):   ("Negligible", "#6A8AA8"),
    (0.2, 0.5):   ("Small",      "#818CF8"),
    (0.5, 0.8):   ("Medium",     "#38BDF8"),
    (0.8, 1.2):   ("Large",      "#34D399"),
    (1.2, float("inf")): ("Very Large", "#F59E0B"),
}


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-std Cohen's d."""
    n1, n2  = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(((n1 - 1) * a.std(ddof=1)**2 + (n2 - 1) * b.std(ddof=1)**2) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0


def _effect_label(d: float) -> Tuple[str, str]:
    abs_d = abs(d)
    for (lo, hi), (label, color) in EFFECT_SIZE_LABELS.items():
        if lo <= abs_d < hi:
            return label, color
    return "Very Large", "#F59E0B"


def _mean_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    n = len(data)
    if n < 2:
        return data.mean(), data.mean()
    se = stats.sem(data)
    h  = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return data.mean() - h, data.mean() + h


def run_ab_test(
    df: pd.DataFrame,
    variable: str,
    group_a_val: str,
    group_b_val: str,
    metric: str = "engagement_rate",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Compare two groups on a metric using Welch's t-test.

    Parameters
    ----------
    df          : filtered analytics DataFrame
    variable    : column to group on (e.g. 'media_type', 'hour')
    group_a_val : value defining group A (e.g. 'reel')
    group_b_val : value defining group B (e.g. 'image')
    metric      : numeric target column (default 'engagement_rate')
    alpha       : significance threshold (default 0.05)

    Returns
    -------
    dict with all test statistics and human-readable interpretations
    """
    a_series = df[df[variable] == group_a_val][metric].dropna()
    b_series = df[df[variable] == group_b_val][metric].dropna()

    a = a_series.values
    b = b_series.values

    if len(a) < 5 or len(b) < 5:
        return {
            "error": f"Insufficient data: Group A n={len(a)}, Group B n={len(b)}. Need ≥5 each.",
            "significant": False,
        }

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    d               = _cohens_d(a, b)
    effect_label, effect_color = _effect_label(d)

    ci_a_lo, ci_a_hi = _mean_confidence_interval(a)
    ci_b_lo, ci_b_hi = _mean_confidence_interval(b)

    significant = p_value < alpha
    winner      = group_a_val if a.mean() > b.mean() else group_b_val
    lift_pct    = abs(a.mean() - b.mean()) / max(min(a.mean(), b.mean()), 1e-9) * 100

    return {
        # Groups
        "variable":     variable,
        "group_a":      group_a_val,
        "group_b":      group_b_val,
        "metric":       metric,
        "n_a":          len(a),
        "n_b":          len(b),
        # Means
        "mean_a":       float(a.mean()),
        "mean_b":       float(b.mean()),
        "std_a":        float(a.std(ddof=1)),
        "std_b":        float(b.std(ddof=1)),
        # CI
        "ci_a":         (float(ci_a_lo), float(ci_a_hi)),
        "ci_b":         (float(ci_b_lo), float(ci_b_hi)),
        # Statistics
        "t_stat":       float(t_stat),
        "p_value":      float(p_value),
        "cohens_d":     float(d),
        "effect_label": effect_label,
        "effect_color": effect_color,
        # Interpretation
        "significant":  significant,
        "alpha":        alpha,
        "winner":       winner,
        "lift_pct":     float(lift_pct),
        "interpretation": _interpret(
            group_a_val, group_b_val, a.mean(), b.mean(),
            p_value, alpha, effect_label, lift_pct
        ),
    }


def _interpret(a_name, b_name, mean_a, mean_b, p, alpha, effect, lift_pct) -> str:
    winner = a_name if mean_a > mean_b else b_name
    loser  = b_name if mean_a > mean_b else a_name
    if p < alpha:
        return (
            f"✅ Statistically significant (p={p:.4f} < α={alpha}). "
            f"**{winner}** outperforms {loser} by {lift_pct:.1f}% "
            f"with a {effect.lower()} effect size."
        )
    else:
        return (
            f"❌ Not statistically significant (p={p:.4f} ≥ α={alpha}). "
            f"The {lift_pct:.1f}% difference between {a_name} and {b_name} "
            f"could be due to chance — collect more data."
        )


def batch_ab_tests(df: pd.DataFrame, metric: str = "engagement_rate") -> pd.DataFrame:
    """
    Run a pre-defined battery of A/B tests across the most common variables.
    Returns a summary DataFrame for the quick-overview table.
    """
    tests = [
        ("media_type",    "reel",    "image"),
        ("media_type",    "reel",    "carousel"),
        ("media_type",    "image",   "carousel"),
        ("has_call_to_action", 1,    0),
    ]

    # Dynamic: add hour-based tests if hour column exists
    if "hour" in df.columns:
        # Peak hours (6–9 AM) vs late night (22–2 AM)
        df = df.copy()
        df["_hour_group"] = df["hour"].apply(
            lambda h: "morning_peak" if 6 <= h <= 9 else
                      ("evening_peak" if 18 <= h <= 21 else "off_peak")
        )
        tests.append(("_hour_group", "morning_peak", "evening_peak"))
        tests.append(("_hour_group", "morning_peak", "off_peak"))

    rows = []
    for variable, a_val, b_val in tests:
        if variable not in df.columns:
            continue
        res = run_ab_test(df, variable, str(a_val), str(b_val), metric=metric)
        if "error" in res:
            continue
        rows.append({
            "Test":         f"{variable}: {a_val} vs {b_val}",
            "Winner":       res["winner"],
            "Lift %":       round(res["lift_pct"], 1),
            "p-value":      round(res["p_value"], 4),
            "Significant":  "✅ Yes" if res["significant"] else "❌ No",
            "Effect Size":  res["effect_label"],
            "Cohen's d":    round(res["cohens_d"], 3),
            "n(A)":         res["n_a"],
            "n(B)":         res["n_b"],
        })

    return pd.DataFrame(rows)
