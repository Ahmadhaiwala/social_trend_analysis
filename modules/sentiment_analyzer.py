"""
Module 3 — Audience Sentiment Analyzer (ML Model)
===================================================
Uses a TF-IDF + Logistic Regression classifier trained on the seed
comment corpus to label comments as:
  Relatable | Positive | Negative | Neutral

The model is trained once and persisted to data/sentiment_model.pkl.
On subsequent calls it is loaded from disk — no hardcoded rules.

VADER is still used as a secondary signal (compound score) but the
final label comes from the trained classifier.

Public API (unchanged):
  analyze_sentiment_corpus(df) -> dict
"""

import json
import os
import pickle
import random
import re
import numpy as np
import pandas as pd

# ── NLTK / VADER (secondary signal) ─────────────────────────
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("punkt",         quiet=True)
    nltk.download("stopwords",     quiet=True)
    _vader = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False

# ── Scikit-learn ─────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────
_ROOT              = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMMENTS_JSON     = os.path.join(_ROOT, "data", "comments_sample.json")
SENTIMENT_MODEL_PKL = os.path.join(_ROOT, "data", "sentiment_model.pkl")


# ─────────────────────────────────────────────────────────────
# SEED CORPUS LOADER
# ─────────────────────────────────────────────────────────────

def _load_corpus() -> dict:
    if os.path.exists(_COMMENTS_JSON):
        with open(_COMMENTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"struggle_comments": {}, "neutral_comments": []}


def _build_training_corpus() -> tuple:
    """
    Build (texts, labels) from the seed JSON.

    Label assignment:
      - struggle_comments  -> "Relatable"  (high-empathy, problem-aware)
      - neutral_comments   -> "Neutral"
      - VADER-positive augmentation -> "Positive"
      - VADER-negative augmentation -> "Negative"
    """
    corpus = _load_corpus()
    struggle_bank = corpus.get("struggle_comments", {})
    neutral_bank  = corpus.get("neutral_comments", [])

    texts, labels = [], []

    # Relatable — all struggle comments across categories
    for cat_comments in struggle_bank.values():
        for text in cat_comments:
            texts.append(text)
            labels.append("Relatable")

    # Neutral
    for text in neutral_bank:
        texts.append(text)
        labels.append("Neutral")

    # Positive — short affirmative phrases (augmented)
    positive_seeds = [
        "This is amazing content, love it so much",
        "Absolutely brilliant, keep it up",
        "Best video I have seen all week",
        "So helpful and well explained, thank you",
        "You are so talented, this is incredible",
        "Wow this changed my perspective completely",
        "Fantastic work, really inspiring",
        "This made my day, thank you so much",
        "Incredible value in this post",
        "Loved every second of this",
        "This is exactly what I needed today",
        "You always deliver such great content",
        "Brilliant as always, keep going",
        "This deserves way more views",
        "Sharing this with everyone I know",
    ]
    for text in positive_seeds:
        texts.append(text)
        labels.append("Positive")

    # Negative — critical / disappointed phrases (augmented)
    negative_seeds = [
        "This is completely wrong and misleading",
        "Terrible advice, do not follow this",
        "Waste of time, nothing useful here",
        "This is so bad I cannot believe it",
        "Disappointing content, expected better",
        "This is harmful and irresponsible",
        "Clickbait, nothing in this video is real",
        "Awful, just awful",
        "This made things worse for me",
        "Completely disagree with everything here",
        "This is dangerous misinformation",
        "Unsubscribed, this is garbage",
        "How is this allowed on the platform",
        "Reported for misleading content",
        "This is the worst advice I have ever heard",
    ]
    for text in negative_seeds:
        texts.append(text)
        labels.append("Negative")

    return texts, labels


# ─────────────────────────────────────────────────────────────
# MODEL TRAINING & PERSISTENCE
# ─────────────────────────────────────────────────────────────

def _train_sentiment_model() -> Pipeline:
    """
    Train TF-IDF + Logistic Regression on the seed corpus.
    Saves to data/sentiment_model.pkl and returns the pipeline.
    """
    texts, labels = _build_training_corpus()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),       # unigrams + bigrams
            max_features=5000,
            sublinear_tf=True,        # log-scale TF
            min_df=1,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",  # handles label imbalance
            solver="lbfgs",
        )),
    ])

    pipeline.fit(texts, labels)

    os.makedirs(os.path.dirname(SENTIMENT_MODEL_PKL), exist_ok=True)
    with open(SENTIMENT_MODEL_PKL, "wb") as f:
        pickle.dump(pipeline, f)

    return pipeline


def _load_or_train_model() -> Pipeline:
    """Load from disk if available, otherwise train fresh."""
    if os.path.exists(SENTIMENT_MODEL_PKL):
        try:
            with open(SENTIMENT_MODEL_PKL, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return _train_sentiment_model()


# Module-level singleton — loaded once per process
_sentiment_pipeline: Pipeline = None


def get_sentiment_pipeline() -> Pipeline:
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = _load_or_train_model()
    return _sentiment_pipeline


def retrain_sentiment_model(extra_texts: list = None, extra_labels: list = None) -> dict:
    """
    Retrain the sentiment model, optionally adding new labelled examples.
    Returns a dict with cross-val accuracy and label distribution.

    extra_texts  : list of comment strings
    extra_labels : list of labels ("Relatable"|"Positive"|"Negative"|"Neutral")
    """
    global _sentiment_pipeline

    base_texts, base_labels = _build_training_corpus()

    if extra_texts and extra_labels:
        assert len(extra_texts) == len(extra_labels)
        base_texts  = base_texts  + list(extra_texts)
        base_labels = base_labels + list(extra_labels)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            min_df=1,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])

    # Cross-val accuracy (3-fold, safe for small corpus)
    cv_scores = cross_val_score(pipeline, base_texts, base_labels, cv=3, scoring="accuracy")
    pipeline.fit(base_texts, base_labels)

    os.makedirs(os.path.dirname(SENTIMENT_MODEL_PKL), exist_ok=True)
    with open(SENTIMENT_MODEL_PKL, "wb") as f:
        pickle.dump(pipeline, f)

    _sentiment_pipeline = pipeline

    from collections import Counter
    label_dist = dict(Counter(base_labels))
    return {
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std":  round(float(cv_scores.std()),  4),
        "n_samples":        len(base_texts),
        "label_distribution": label_dist,
    }


# ─────────────────────────────────────────────────────────────
# COMMENT SCORING
# ─────────────────────────────────────────────────────────────

def classify_text(text: str) -> dict:
    """
    Classify a single comment string.
    Returns dict: label, confidence, vader_compound, relatability_score.
    """
    pipeline = get_sentiment_pipeline()

    # ML label + probability
    label = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    confidence = float(proba[list(classes).index(label)])

    # VADER compound as secondary signal
    if VADER_AVAILABLE:
        vader_compound = round(_vader.polarity_scores(text)["compound"], 3)
    else:
        vader_compound = 0.0

    # Relatability score (0-100) — still useful as a continuous feature
    relatability = _score_relatability(text)

    return {
        "label":              label,
        "confidence":         round(confidence, 3),
        "vader_compound":     vader_compound,
        "relatability_score": round(relatability, 1),
    }


# ─────────────────────────────────────────────────────────────
# RELATABILITY SCORER (continuous feature, not the label)
# ─────────────────────────────────────────────────────────────

_STRUGGLE_WORDS = [
    "struggle", "fail", "failed", "anxiety", "anxious", "overwhelm",
    "overwhelmed", "alone", "lonely", "lost", "fear", "scared", "doubt",
    "imposter", "quit", "gave up", "exhausted", "burned out",
    "comparison", "behind", "not enough", "worthless", "hopeless",
    "cry", "cried", "pressure", "toxic", "trauma", "hate myself",
    "cannot", "couldn't", "was impossible",
]

_RELATABLE_PHRASES = [
    "anyone else", "who else", "same", "me too", "not just me",
    "relatable", "felt seen", "felt understood", "exactly this",
    "waiting for this", "needed this", "why did nobody",
    "nobody talks about", "finally", "saved this", "sharing this",
    "sending this", "preach", "been waiting", "this is me", "literally me",
    "i thought i was", "i felt", "i've been",
]

_EMPATHY_MARKERS = [
    "thank you", "saved my", "changed my", "gave me courage", "gave me hope",
    "made me feel", "made me cry", "needed to hear", "healing", "feel seen",
    "feel normal", "not alone", "validation", "helps so much",
]


def _score_relatability(text: str) -> float:
    t = text.lower()
    score = 0.0
    score += min(sum(1 for w in _STRUGGLE_WORDS   if w in t) * 15, 40)
    score += min(sum(1 for p in _RELATABLE_PHRASES if p in t) * 20, 35)
    score += min(sum(1 for e in _EMPATHY_MARKERS   if e in t) * 15, 25)
    if re.search(r"\?|why does|why is|can we talk about|has anyone", t):
        score += 5
    return min(score, 100.0)


def _problem_awareness_level(comments: list) -> str:
    text = " ".join(comments).lower()
    systemic = ["society", "system", "nobody talks", "industry", "culture",
                "always been", "norm", "we all", "everyone feels"]
    shared   = ["who else", "anyone else", "not just me", "me too", "same",
                "same here", "all of us", "so many people"]
    personal = ["i ", "my ", "me ", "i've ", "i was", "i felt", "i cried"]
    if sum(1 for k in systemic if k in text) >= 2: return "Systemic"
    if sum(1 for k in shared   if k in text) >= 2: return "Shared"
    if sum(1 for k in personal if k in text) >= 3: return "Personal"
    return "None"


# ─────────────────────────────────────────────────────────────
# COMMENT DATAFRAME BUILDER
# ─────────────────────────────────────────────────────────────

def generate_comment_dataframe(df: pd.DataFrame, n_per_post: int = 3) -> pd.DataFrame:
    """
    Build a comment DataFrame from the seed corpus, scored by the ML model.
    """
    corpus = _load_corpus()
    struggle_bank = corpus.get("struggle_comments", {})
    neutral_bank  = corpus.get("neutral_comments", [])

    pipeline = get_sentiment_pipeline()

    records = []
    sample_size = min(300, len(df))
    sampled = df.sample(sample_size, random_state=42).reset_index(drop=True)

    for _, row in sampled.iterrows():
        cat  = str(row.get("content_category", "Other"))
        tier = str(row.get("tier", "Low"))

        struggle_weight = 0.8 if tier in ["Viral", "High"] else 0.3
        cat_comments = struggle_bank.get(cat, neutral_bank)

        for _ in range(n_per_post):
            use_struggle = random.random() < struggle_weight
            if use_struggle and cat_comments:
                text = random.choice(cat_comments)
            else:
                text = random.choice(neutral_bank) if neutral_bank else "Great content"

            scored = classify_text(text)

            records.append({
                "content_category":     cat,
                "media_type":           row.get("media_type", ""),
                "engagement_tier":      tier,
                "comment_text":         text,
                "label":                scored["label"],
                "confidence":           scored["confidence"],
                "vader_compound":       scored["vader_compound"],
                "relatability_score":   scored["relatability_score"],
                "has_struggle_word":    int(any(w in text.lower() for w in _STRUGGLE_WORDS)),
                "has_relatable_phrase": int(any(p in text.lower() for p in _RELATABLE_PHRASES)),
                "has_empathy_marker":   int(any(e in text.lower() for e in _EMPATHY_MARKERS)),
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def analyze_sentiment_corpus(df: pd.DataFrame) -> dict:
    """
    Main entry point for Module 3.
    Returns:
      comments_df      : full tagged comment table
      category_summary : per-category sentiment + awareness stats
      label_counts     : overall label distribution
      top_relatable    : top 10 most relatable comments
      model_info       : info about the classifier being used
    """
    comments_df = generate_comment_dataframe(df)

    cat_summary = (
        comments_df.groupby("content_category")
        .agg(
            avg_relatability=("relatability_score", "mean"),
            avg_vader=("vader_compound", "mean"),
            avg_confidence=("confidence", "mean"),
            pct_relatable=("label", lambda x: (x == "Relatable").mean() * 100),
            pct_negative=("label",  lambda x: (x == "Negative").mean()  * 100),
            n_comments=("comment_text", "count"),
        )
        .reset_index()
    )

    def _awareness(cat):
        texts = comments_df[comments_df["content_category"] == cat]["comment_text"].tolist()
        return _problem_awareness_level(texts)

    cat_summary["problem_awareness"] = cat_summary["content_category"].apply(_awareness)
    cat_summary = cat_summary.sort_values("avg_relatability", ascending=False).reset_index(drop=True)

    label_counts = comments_df["label"].value_counts().to_dict()

    top_relatable = (
        comments_df[comments_df["label"] == "Relatable"]
        .sort_values("relatability_score", ascending=False)
        .head(10)[["comment_text", "content_category", "relatability_score",
                   "vader_compound", "confidence"]]
        .reset_index(drop=True)
    )

    pipeline = get_sentiment_pipeline()
    model_info = {
        "classifier":  type(pipeline.named_steps["clf"]).__name__,
        "vectorizer":  type(pipeline.named_steps["tfidf"]).__name__,
        "n_features":  pipeline.named_steps["tfidf"].max_features,
        "ngram_range": str(pipeline.named_steps["tfidf"].ngram_range),
        "classes":     list(pipeline.classes_),
        "model_path":  SENTIMENT_MODEL_PKL,
        "model_exists": os.path.exists(SENTIMENT_MODEL_PKL),
    }

    return {
        "comments_df":      comments_df,
        "category_summary": cat_summary,
        "label_counts":     label_counts,
        "top_relatable":    top_relatable,
        "model_info":       model_info,
    }
