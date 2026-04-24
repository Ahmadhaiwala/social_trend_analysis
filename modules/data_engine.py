"""
Module 1 — Content Performance Tracker (Data Extraction Engine)
================================================================
Loads the Kaggle Instagram Analytics dataset, cleans it, and
computes all derived metrics used by every other module.

Incremental learning:
  - append_scraped_data()  merges new scraped rows into data/augmented_data.csv
  - retrain_model()        retrains on base + augmented data, saves model.pkl
                           and appends a record to data/training_log.json
  - get_model()            loads from model.pkl if it exists, otherwise trains fresh
"""

import os
import json
import pickle
import datetime
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── File paths ───────────────────────────────────────────────
_ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUGMENTED_CSV = os.path.join(_ROOT, "data", "augmented_data.csv")
MODEL_PKL     = os.path.join(_ROOT, "data", "model.pkl")
TRAINING_LOG  = os.path.join(_ROOT, "data", "training_log.json")

# Columns to drop before training (metadata / derived / non-feature)
_MODEL_DROP = [
    "post_id", "account_id", "post_datetime", "post_date",
    "performance_bucket_label", "day", "month",
    "viral_score", "save_to_share",
    "tier", "vc_raw", "vc_percentile", "vc_tier", "high_value_ratio",
    "year", "week", "link", "username", "caption",
    "at_mentions", "views", "comments_count", "scraped_ok",
]


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading dataset...")
def get_dataframe() -> pd.DataFrame:
    """
    Download and clean the Kaggle Instagram Analytics dataset.
    Cached for the Streamlit session.
    """
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    dataset_id = "kundanbedmutha/instagram-analytics-dataset"
    path = kagglehub.dataset_download(dataset_id)
    csv_file = next((f for f in os.listdir(path) if f.endswith(".csv")), None)
    if not csv_file:
        raise FileNotFoundError("No CSV in Kaggle dataset download.")

    raw = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, dataset_id, csv_file)
    raw.columns = raw.columns.str.strip().str.lower()

    raw["post_datetime"] = pd.to_datetime(raw["post_datetime"], errors="coerce")
    raw["hour"]      = raw["post_datetime"].dt.hour
    raw["day"]       = raw["post_datetime"].dt.day_name()
    raw["day_num"]   = raw["post_datetime"].dt.dayofweek
    raw["month"]     = raw["post_datetime"].dt.month_name()
    raw["month_num"] = raw["post_datetime"].dt.month
    raw["week"]      = raw["post_datetime"].dt.isocalendar().week.astype(int)
    raw["year"]      = raw["post_datetime"].dt.year

    raw["viral_score"] = (
        raw.get("likes", 0)
        + raw.get("shares", pd.Series(0, index=raw.index)) * 2
        + raw.get("saves",  pd.Series(0, index=raw.index)) * 3
    )

    if "saves" in raw.columns and "shares" in raw.columns:
        raw["save_to_share"] = raw["saves"] / raw["shares"].replace(0, np.nan)
        raw["save_to_share"] = raw["save_to_share"].fillna(raw["saves"])
    else:
        raw["save_to_share"] = 0

    return raw.dropna(subset=["engagement_rate"]).copy()


# ─────────────────────────────────────────────────────────────
# MODEL — LOAD OR TRAIN
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Training model...")
def get_model(analytics_df: pd.DataFrame):
    """
    Load the saved model from disk if available, otherwise train fresh.
    Returns (pipeline, metrics_dict).
    """
    if os.path.exists(MODEL_PKL):
        try:
            with open(MODEL_PKL, "rb") as f:
                saved = pickle.load(f)
            return saved["pipeline"], saved["metrics"]
        except Exception:
            pass  # corrupt file — fall through to retrain

    return _train_fresh(analytics_df)


def _build_pipeline(X: pd.DataFrame, n_estimators: int = 150) -> Pipeline:
    """Construct a fresh preprocessor + RF pipeline matched to X's dtypes."""
    num_feats = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_feats = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    pre = ColumnTransformer([
        ("num", StandardScaler(),                    num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
    ])
    return Pipeline([
        ("preprocessor", pre),
        ("regressor",    RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def _train_fresh(analytics_df: pd.DataFrame):
    """Train from scratch without touching disk (used as fallback)."""
    model_df = analytics_df.drop(columns=_MODEL_DROP, errors="ignore").dropna(
        subset=["engagement_rate"]
    )
    y = model_df["engagement_rate"]
    X = model_df.drop(columns=["engagement_rate"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = _build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    return pipeline, {
        "mae":          round(float(mean_absolute_error(y_test, y_pred)), 5),
        "r2":           round(float(r2_score(y_test, y_pred)), 4),
        "features":     X.columns.tolist(),
        "num_features": X.select_dtypes(include=["int64", "float64"]).columns.tolist(),
        "cat_features": X.select_dtypes(include=["object", "bool"]).columns.tolist(),
        "n_train":      int(len(X_train)),
        "n_test":       int(len(X_test)),
        "n_augmented":  0,
        "n_trees":      100,
        "timestamp":    datetime.datetime.now().isoformat(timespec="seconds"),
    }


# ─────────────────────────────────────────────────────────────
# INCREMENTAL LEARNING
# ─────────────────────────────────────────────────────────────

def append_scraped_data(new_rows: list) -> tuple:
    """
    Merge a list of scraped post dicts into data/augmented_data.csv.
    Deduplicates on 'link' column if present.

    Returns (rows_added, total_augmented_rows).
    """
    os.makedirs(os.path.dirname(AUGMENTED_CSV), exist_ok=True)
    new_df = pd.DataFrame(new_rows)

    if os.path.exists(AUGMENTED_CSV):
        existing = pd.read_csv(AUGMENTED_CSV)
        if "link" in new_df.columns and "link" in existing.columns:
            new_df = new_df[~new_df["link"].isin(existing["link"])]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(AUGMENTED_CSV, index=False)
    return int(len(new_df)), int(len(combined))


def retrain_model(base_df: pd.DataFrame = None) -> dict:
    """
    Retrain the Random Forest on base Kaggle data + any augmented rows.
    Saves the fitted pipeline to data/model.pkl.
    Appends a timestamped record to data/training_log.json.

    Returns metrics dict: mae, r2, n_train, n_test, n_augmented, n_trees, timestamp.
    """
    os.makedirs(os.path.dirname(MODEL_PKL), exist_ok=True)

    if base_df is None:
        base_df = get_dataframe()

    frames = [base_df]
    n_augmented = 0

    if os.path.exists(AUGMENTED_CSV):
        aug = pd.read_csv(AUGMENTED_CSV)
        shared_cols = [c for c in aug.columns if c in base_df.columns]
        aug = aug[shared_cols].dropna(subset=["engagement_rate"])
        if len(aug) > 0:
            frames.append(aug)
            n_augmented = len(aug)

    combined = pd.concat(frames, ignore_index=True)

    model_df = combined.drop(columns=_MODEL_DROP, errors="ignore").dropna(
        subset=["engagement_rate"]
    )
    y = model_df["engagement_rate"]
    X = model_df.drop(columns=["engagement_rate"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale tree count with data volume
    n_trees = min(200, 100 + n_augmented // 10)
    pipeline = _build_pipeline(X_train, n_estimators=n_trees)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "mae":          round(float(mean_absolute_error(y_test, y_pred)), 5),
        "r2":           round(float(r2_score(y_test, y_pred)), 4),
        "n_train":      int(len(X_train)),
        "n_test":       int(len(X_test)),
        "n_augmented":  int(n_augmented),
        "n_trees":      int(n_trees),
        "features":     X.columns.tolist(),
        "num_features": X.select_dtypes(include=["int64", "float64"]).columns.tolist(),
        "cat_features": X.select_dtypes(include=["object", "bool"]).columns.tolist(),
        "timestamp":    datetime.datetime.now().isoformat(timespec="seconds"),
    }

    with open(MODEL_PKL, "wb") as f:
        pickle.dump({"pipeline": pipeline, "metrics": metrics}, f)

    # Log entry (no feature lists — keep log compact)
    log_entry = {k: v for k, v in metrics.items()
                 if k not in ("features", "num_features", "cat_features")}
    _append_training_log(log_entry)

    return metrics


def _append_training_log(entry: dict):
    """Append one record to data/training_log.json."""
    os.makedirs(os.path.dirname(TRAINING_LOG), exist_ok=True)
    history = []
    if os.path.exists(TRAINING_LOG):
        try:
            with open(TRAINING_LOG, "r") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(entry)
    with open(TRAINING_LOG, "w") as f:
        json.dump(history, f, indent=2)


def load_training_log() -> list:
    """Return the full training history list (oldest first)."""
    if not os.path.exists(TRAINING_LOG):
        return []
    try:
        with open(TRAINING_LOG, "r") as f:
            return json.load(f)
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────

def compute_performance_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Assign Low / Medium / High / Viral tier based on engagement_rate quartiles."""
    df = df.copy()
    q1, q2, q3 = df["engagement_rate"].quantile([0.25, 0.50, 0.75])

    def _tier(v):
        if v >= q3: return "Viral"
        if v >= q2: return "High"
        if v >= q1: return "Medium"
        return "Low"

    df["tier"] = df["engagement_rate"].apply(_tier)
    return df


def summarise_dataset(df: pd.DataFrame) -> dict:
    """Quick summary stats for the KPI panel."""
    return {
        "total_posts":    len(df),
        "avg_engagement": df["engagement_rate"].mean(),
        "avg_likes":      df["likes"].mean()       if "likes"       in df.columns else 0,
        "total_reach":    df["reach"].sum()         if "reach"       in df.columns else 0,
        "total_impr":     df["impressions"].sum()   if "impressions" in df.columns else 0,
        "avg_viral":      df["viral_score"].mean()  if "viral_score" in df.columns else 0,
        "categories":     sorted(df["content_category"].dropna().unique().tolist()),
        "media_types":    sorted(df["media_type"].dropna().unique().tolist()),
    }
