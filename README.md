# Data-Driven Social Engagement Initiative

A full-stack analytics dashboard for Instagram content performance, built with Streamlit, Plotly, and scikit-learn. Combines a Kaggle dataset baseline with real-time Instagram scraping to incrementally improve a Random Forest engagement prediction model.

---

## Features

| Module | What it does |
|---|---|
| **Overview** | KPI strip, smart insights, growth trend, performance tier funnel |
| **Content Performance** | Filterable data table, engagement by media type, retention trend |
| **Virality Engine** | Viral Coefficient per post/topic, shares vs saves scatter, top viral post cards |
| **Sentiment Analysis** | TF-IDF + Logistic Regression comment classifier (Relatable / Positive / Negative / Neutral) |
| **A/B Testing** | Welch's t-test with Cohen's d, CI comparison chart, batch test table |
| **Recommendations** | 7-day content calendar, best hour/day/category/CTA charts |
| **Trend Forecasting** | Linear trend velocity per category, 14-day forecast with CI band, keyword cards |
| **Scrape & Train** | Scrape real Instagram reels by hashtag, predict engagement, append to training pool, retrain model |

---

## Project Structure

```
predicting_social_trend/
├── main_dashboard.py          # Main Streamlit app (all 8 pages)
├── dashboard.py               # Legacy standalone dashboard
├── model_train.ipynb          # Exploratory training notebook
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── comments_sample.json   # Seed comment corpus for sentiment model
│   ├── training_log.json      # Accuracy history across retrain runs
│   ├── model.pkl              # Saved RF model (generated, not in git)
│   ├── sentiment_model.pkl    # Saved sentiment classifier (generated)
│   ├── augmented_data.csv     # Scraped rows appended per run (generated)
│   └── real_reel_predictions.csv  # Latest scrape predictions (generated)
│
└── modules/
    ├── data_engine.py         # Data loading, model training, incremental learning
    ├── virality_engine.py     # Viral Coefficient computation
    ├── sentiment_analyzer.py  # TF-IDF + LR sentiment classifier
    ├── ab_testing.py          # Welch's t-test A/B framework
    ├── recommender.py         # Content strategy recommender
    ├── trend_forecaster.py    # Trend velocity + 14-day forecast
    ├── checktrend.py          # Instagram Selenium scraper
    └── __init__.py
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd predicting_social_trend
pip install -r requirements.txt
```

Additional dependencies for the scraper:

```bash
pip install selenium webdriver-manager
```

### 2. Configure Kaggle credentials

The dataset is downloaded automatically via `kagglehub`. You need a Kaggle account and API key:

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token
2. Place the downloaded `kaggle.json` in `~/.kaggle/kaggle.json`

Dataset used: [`kundanbedmutha/instagram-analytics-dataset`](https://www.kaggle.com/datasets/kundanbedmutha/instagram-analytics-dataset)

### 3. Run the dashboard

```bash
streamlit run main_dashboard.py
```

The first run downloads the Kaggle dataset and trains the model — this takes about 30–60 seconds. Subsequent runs load the saved model from `data/model.pkl` instantly.

---

## Incremental Learning (Scrape & Train)

The **Scrape & Train** page lets you feed real Instagram data into the model:

1. Enter your Instagram credentials and one or more hashtags (e.g. `coding, fitness, travel`)
2. Set how many reels to scrape per hashtag (keep it under 10 to avoid rate limits)
3. Click **Start Scraping** — the scraper logs in, collects reels, predicts engagement, and shows a live preview table
4. Scraped rows are appended to `data/augmented_data.csv` (deduplicated by post URL)
5. The model retrains on Kaggle base data + all augmented rows and saves to `data/model.pkl`
6. MAE and R² delta vs the previous run are shown immediately

Each subsequent scrape run adds more real-world signal and the model improves progressively. The training history chart tracks accuracy across all runs.

> **Note on scraped features:** Instagram does not expose `reach`, `saves`, or `shares` publicly. These are estimated from likes, comments, views, and follower count using industry heuristics. The predicted engagement rate is used as the training label for scraped rows.

---

## Sentiment Model

The sentiment classifier is a **TF-IDF (unigram + bigram, 5000 features) + Logistic Regression** pipeline trained on the seed corpus in `data/comments_sample.json`. Labels:

- **Relatable** — high-empathy, struggle-aware comments
- **Positive** — affirmative, encouraging comments
- **Negative** — critical or disappointed comments
- **Neutral** — generic low-signal comments

The model is trained once and saved to `data/sentiment_model.pkl`. You can retrain it with new labelled examples from the Sentiment Analysis page using the **Retrain Sentiment Model** button.

---

## Tech Stack

| Library | Purpose |
|---|---|
| Streamlit | Dashboard UI |
| Plotly | Interactive charts |
| scikit-learn | Random Forest, TF-IDF, Logistic Regression |
| pandas / numpy | Data processing |
| kagglehub | Dataset download |
| NLTK / VADER | Secondary sentiment signal |
| scipy | Welch's t-test for A/B testing |
| Selenium | Instagram scraping |
| webdriver-manager | ChromeDriver auto-management |

---

## Notes

- The scraper uses Selenium with Chrome. Make sure Google Chrome is installed.
- Instagram credentials are entered in the UI at scrape time — they are never stored to disk.
- This project is for educational and research purposes only.
