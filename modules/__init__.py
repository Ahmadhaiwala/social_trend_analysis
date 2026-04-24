"""modules package — Data-Driven Social Engagement Initiative"""
from .data_engine import (
    get_dataframe, get_model,
    append_scraped_data, retrain_model, load_training_log,
)
from .virality_engine import compute_viral_coefficients
from .sentiment_analyzer import (
    analyze_sentiment_corpus, classify_text,
    get_sentiment_pipeline, retrain_sentiment_model,
)
from .ab_testing import run_ab_test
from .recommender import generate_recommendations
from .trend_forecaster import compute_trend_velocities, forecast_engagement

__all__ = [
    "get_dataframe", "get_model",
    "append_scraped_data", "retrain_model", "load_training_log",
    "compute_viral_coefficients",
    "analyze_sentiment_corpus", "classify_text",
    "get_sentiment_pipeline", "retrain_sentiment_model",
    "run_ab_test",
    "generate_recommendations",
    "compute_trend_velocities", "forecast_engagement",
]
