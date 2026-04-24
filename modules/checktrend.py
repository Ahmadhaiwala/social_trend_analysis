"""
========================================================================
  INSTAGRAM REAL REEL SCRAPER + ENGAGEMENT PREDICTOR
  modules/checktrend.py

  What it does:
   1. Logs into Instagram with Selenium
   2. Scrapes real reel posts from a target hashtag
   3. Extracts: caption, likes, comments, views, timestamp,
               hashtag count, CTA presence, follower count
   4. Feeds scraped data into the trained Random Forest model
   5. Predicts engagement rate + tier + viral score for each reel
   6. Saves results to ../data/real_reel_predictions.csv
   7. Prints a rich console report

  Run:  python modules/checktrend.py
  Run from project root -- NOT from inside modules/
========================================================================
"""

import os
import re
import sys
import io
import time
import json
import logging
import datetime
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows (prevents cp1252 UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np

# ── Selenium imports ──────────────────────────────────────────
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, StaleElementReferenceException
)
from webdriver_manager.chrome import ChromeDriverManager

# ── ML imports (same pipeline as training notebook) ───────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ──────────────────────────────────────────────────────────────
# CONFIG  — edit these before running
# ──────────────────────────────────────────────────────────────
CONFIG = {
    # ── Instagram credentials ──
    "username":           "",
    "password":           "",

    # ── Target hashtag (without #) ──
    "hashtag":            "coding",

    # ── How many reels to scrape ──
    "max_posts":          10,

    # ── Extra scroll passes on hashtag page ──
    "scroll_passes":      4,

    # ── Wait between page navigations (seconds) ──
    "page_delay":         2.5,

    # ── Output CSV path (relative to project root) ──
    "output_csv":         "data/real_reel_predictions.csv",

    # ── Chrome headless mode (False = shows browser window) ──
    "headless":           False,

    # ── Content category classifier: keyword → category ──
    "category_keywords": {
        "fitness":         "Fitness",
        "workout":         "Fitness",
        "gym":             "Fitness",
        "health":          "Fitness",
        "beauty":          "Beauty",
        "skincare":        "Beauty",
        "makeup":          "Beauty",
        "travel":          "Travel",
        "food":            "Food",
        "recipe":          "Food",
        "cooking":         "Food",
        "tech":            "Technology",
        "coding":          "Technology",
        "programming":     "Technology",
        "software":        "Technology",
        "music":           "Music",
        "photo":           "Photography",
        "comedy":          "Comedy",
        "funny":           "Comedy",
        "meme":            "Comedy",
    },
}

# ──────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("checktrend")


# ══════════════════════════════════════════════════════════════
# PART 1 — MODEL TRAINER (standalone, no Streamlit cache)
# ══════════════════════════════════════════════════════════════

def train_model_standalone():
    """
    Downloads the Kaggle dataset and trains the Random Forest pipeline.
    Returns (model_pipeline, feature_names, cat_features, num_features,
             training_stats_dict, df_for_defaults).
    """
    log.info("📡  Loading Kaggle dataset…")
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    dataset_id = "kundanbedmutha/instagram-analytics-dataset"
    path = kagglehub.dataset_download(dataset_id)
    csv_file = next((f for f in os.listdir(path) if f.endswith(".csv")), None)
    if not csv_file:
        raise FileNotFoundError("No CSV found in Kaggle dataset.")

    raw = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, dataset_id, csv_file)
    raw.columns = raw.columns.str.strip().str.lower()

    # Parse datetime & extract features
    raw["post_datetime"] = pd.to_datetime(raw["post_datetime"], errors="coerce")
    raw["hour"]      = raw["post_datetime"].dt.hour
    raw["day"]       = raw["post_datetime"].dt.day_name()
    raw["day_num"]   = raw["post_datetime"].dt.dayofweek
    raw["month"]     = raw["post_datetime"].dt.month_name()
    raw["month_num"] = raw["post_datetime"].dt.month
    raw["week"]      = raw["post_datetime"].dt.isocalendar().week.astype(int)

    # Keep a copy for computing defaults for features we can't scrape
    defaults_df = raw.dropna(subset=["engagement_rate"]).copy()

    # Prepare model data frame
    drop_cols = [
        "post_id", "account_id", "post_datetime", "post_date",
        "performance_bucket_label", "day", "month",
    ]
    model_df = raw.drop(columns=drop_cols, errors="ignore").dropna()

    y = model_df["engagement_rate"]
    X = model_df.drop(columns=["engagement_rate"])

    num_feats = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_feats = X.select_dtypes(include=["object","bool"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(),                    num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor",    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log.info("🤖  Training Random Forest on %d posts…", len(X_train))
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    stats  = {
        "mae":    round(mean_absolute_error(y_test, y_pred), 5),
        "r2":     round(r2_score(y_test, y_pred), 4),
        "n_train": len(X_train),
        "n_test":  len(X_test),
        "features": X.columns.tolist(),
        "num_feats": num_feats,
        "cat_feats": cat_feats,
    }
    log.info("✅  Model trained  MAE=%.5f  R²=%.4f", stats["mae"], stats["r2"])
    return pipeline, stats, defaults_df


# ══════════════════════════════════════════════════════════════
# PART 2 — DEFAULTS CALCULATOR
# ══════════════════════════════════════════════════════════════

def compute_defaults(df: pd.DataFrame) -> dict:
    """
    Compute median / mode values from training data.
    These fill in features we can't directly scrape from Instagram.
    """
    def med(col):
        return float(df[col].median()) if col in df.columns else 0.0

    def mode_val(col):
        return str(df[col].mode().iloc[0]) if col in df.columns else ""

    return {
        # numeric defaults (median from training data)
        "follower_count":    med("follower_count"),
        "likes":             med("likes"),
        "comments":          med("comments"),
        "shares":            med("shares"),
        "saves":             med("saves"),
        "reach":             med("reach"),
        "impressions":       med("impressions"),
        "followers_gained":  med("followers_gained"),
        "caption_length":    med("caption_length"),
        "hashtags_count":    med("hashtags_count"),
        "hour":              9.0,    # sensible default: 9 AM
        "day_num":           1.0,    # Tuesday
        "month_num":         4.0,
        "week":              16.0,
        "has_call_to_action": 1.0,

        # categorical defaults (mode from training)
        "account_type":     mode_val("account_type"),
        "media_type":       "reel",  # we're scraping reels
        "content_category": mode_val("content_category"),
        "traffic_source":   "Hashtags",  # found via hashtag page
    }


# ══════════════════════════════════════════════════════════════
# PART 3 — CAPTION ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════

CTA_PHRASES = [
    "link in bio", "click the link", "order now", "buy now", "shop now",
    "follow", "subscribe", "comment below", "tag a friend", "share this",
    "save this", "dm me", "check out", "visit our", "book now",
    "sign up", "register", "learn more", "swipe up", "tap link",
]

STRUGGLE_WORDS = [
    "struggle", "fail", "anxiety", "alone", "lost", "fear", "scared",
    "overwhelmed", "exhausted", "burned out", "quit", "hopeless", "cry",
    "pressure", "toxic", "trauma", "worried", "nervous", "depressed",
]

def detect_cta(caption: str) -> int:
    caption_lower = caption.lower()
    return int(any(phrase in caption_lower for phrase in CTA_PHRASES))


def count_hashtags(caption: str) -> int:
    return len(re.findall(r"#\w+", caption))


def count_at_mentions(caption: str) -> int:
    return len(re.findall(r"@\w+", caption))


def classify_category(caption: str, hashtag: str) -> str:
    """Map caption text + hashtag to the training dataset's content_category values."""
    combined = (caption + " " + hashtag).lower()
    for keyword, category in CONFIG["category_keywords"].items():
        if keyword in combined:
            return category
    return "Technology"    # safe default (training set has lots of this)


def estimate_reach(likes: int, comments: int, follower_count: int) -> int:
    """
    Conservative reach estimate when we can't get it directly.
    Industry rule-of-thumb: reach ≈ likes × 5 + comments × 10 (capped at followers).
    """
    est = likes * 5 + comments * 10
    return min(max(est, likes + comments), follower_count) if follower_count > 0 else est


def estimate_impressions(reach: int) -> int:
    """Impressions ≈ reach × 1.3 (each user sees it ~1.3 times on average)."""
    return int(reach * 1.3)


def parse_count(text: str) -> int:
    """
    Parse Instagram's shorthand counts: '1.2K' → 1200, '3.5M' → 3500000.
    Returns 0 if unparseable.
    """
    if not text or text in ("N/A", "--", ""):
        return 0
    text = text.replace(",", "").strip()
    match = re.search(r"([\d.]+)\s*([KkMm]?)", text)
    if not match:
        return 0
    num = float(match.group(1))
    suffix = match.group(2).upper()
    if suffix == "K": num *= 1_000
    elif suffix == "M": num *= 1_000_000
    return int(num)


# ══════════════════════════════════════════════════════════════
# PART 4 — INSTAGRAM SCRAPER CLASS
# ══════════════════════════════════════════════════════════════

class InstagramScraper:
    """
    Selenium-based Instagram scraper.
    Handles login, navigation, and structured data extraction.
    """

    WAIT_TIMEOUT = 15
    SHORT_WAIT   = 5

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.driver = self._init_driver()
        self.wait   = WebDriverWait(self.driver, self.WAIT_TIMEOUT)
        log.info("🌐  Chrome driver initialised")

    # ── Driver setup ──────────────────────────────────────────
    def _init_driver(self) -> webdriver.Chrome:
        opts = Options()
        if self.cfg.get("headless"):
            opts.add_argument("--headless=new")

        # ── Stability flags (prevent Windows native crash) ────
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--disable-software-rasterizer")
        opts.add_argument("--disable-extensions")
        opts.add_argument("--disable-infobars")
        opts.add_argument("--disable-notifications")
        opts.add_argument("--disable-popup-blocking")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--window-size=1280,900")
        opts.add_argument("--remote-debugging-port=0")  # avoid port conflicts
        opts.add_argument("--log-level=3")              # suppress Chrome console noise

        # ── Anti-detection ────────────────────────────────────
        opts.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        opts.add_experimental_option("useAutomationExtension", False)
        opts.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/147.0.0.0 Safari/537.36"
        )

        # ── Driver resolution strategy ────────────────────────
        # 1. Try Selenium Manager (built-in since Selenium 4.6) — matches Chrome exactly
        # 2. Fall back to webdriver_manager with explicit 64-bit path fix
        driver = None

        # Strategy 1: let Selenium pick the right driver automatically
        try:
            driver = webdriver.Chrome(options=opts)
            log.info("Driver initialised via Selenium Manager")
        except Exception as e1:
            log.warning("Selenium Manager failed (%s), trying webdriver_manager...", e1)

            # Strategy 2: webdriver_manager — fix the win32 vs win64 path issue
            try:
                from webdriver_manager.chrome import ChromeDriverManager
                from webdriver_manager.core.os_manager import ChromeType
                import shutil

                raw_path = ChromeDriverManager().install()

                # webdriver_manager sometimes returns the win32 binary on win64 systems.
                # The win64 binary sits in the same folder — swap it in if present.
                driver_dir  = os.path.dirname(raw_path)
                parent_dir  = os.path.dirname(driver_dir)
                win64_path  = os.path.join(parent_dir, "chromedriver-win64", "chromedriver.exe")
                win32_path  = os.path.join(parent_dir, "chromedriver-win32", "chromedriver.exe")

                if "win32" in raw_path and os.path.exists(win64_path):
                    log.info("Switching from win32 to win64 driver: %s", win64_path)
                    service_path = win64_path
                else:
                    service_path = raw_path

                service = Service(executable_path=service_path)
                driver  = webdriver.Chrome(service=service, options=opts)
                log.info("Driver initialised via webdriver_manager: %s", service_path)

            except Exception as e2:
                raise RuntimeError(
                    f"Could not start ChromeDriver.\n"
                    f"  Selenium Manager error: {e1}\n"
                    f"  webdriver_manager error: {e2}\n\n"
                    f"Fix: make sure Chrome 147 is installed and run:\n"
                    f"  pip install --upgrade selenium webdriver-manager"
                ) from e2

        # Mask webdriver fingerprint
        driver.execute_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
        )
        return driver

    # ── Safe element getter ───────────────────────────────────
    def _find(self, *locators, timeout=None):
        """Try multiple (By, selector) pairs in order. Return text or None."""
        t = timeout or self.SHORT_WAIT
        for by, selector in locators:
            try:
                el = WebDriverWait(self.driver, t).until(
                    EC.presence_of_element_located((by, selector))
                )
                return el
            except (TimeoutException, NoSuchElementException):
                continue
        return None

    def _text(self, *locators, timeout=None) -> str:
        el = self._find(*locators, timeout=timeout)
        return el.text.strip() if el else ""

    # ── Popup dismisser ───────────────────────────────────────
    def _dismiss_popups(self):
        popup_texts = ["Not Now", "Not now", "Skip", "Later", "Close", "Dismiss"]
        for text in popup_texts:
            for _ in range(2):
                try:
                    btn = WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable(
                            (By.XPATH, f"//button[contains(text(),'{text}')]")
                        )
                    )
                    btn.click()
                    time.sleep(0.5)
                    break
                except:
                    pass

    # ── LOGIN ─────────────────────────────────────────────────
    def login(self):
        log.info("🔑  Logging in as %s…", self.cfg["username"])
        self.driver.get("https://www.instagram.com/accounts/login/")

        username_input = self.wait.until(
            EC.visibility_of_element_located((By.NAME, "username"))
        )
        time.sleep(0.8)
        username_input.clear()
        username_input.send_keys(self.cfg["username"])

        password_input = self.wait.until(
            EC.visibility_of_element_located((By.NAME, "password"))
        )
        password_input.clear()
        password_input.send_keys(self.cfg["password"])
        time.sleep(0.5)
        password_input.send_keys(Keys.RETURN)

        # Wait for home feed or nav icon
        try:
            self.wait.until(lambda d: "/accounts/login/" not in d.current_url)
            log.info("✅  Login successful")
        except TimeoutException:
            log.warning("⚠️  Login may have failed — continuing anyway")

        time.sleep(2)
        self._dismiss_popups()
        time.sleep(1)
        self._dismiss_popups()   # second pass for stacked popups

    # ── COLLECT REEL LINKS FROM HASHTAG PAGE ─────────────────
    def collect_reel_links(self) -> list:
        tag = self.cfg["hashtag"]
        url = f"https://www.instagram.com/explore/tags/{tag}/"
        log.info("🔍  Opening hashtag page: %s", url)
        self.driver.get(url)

        # Wait for post grid
        try:
            self.wait.until(
                EC.presence_of_element_located((By.XPATH, "//a[contains(@href,'/p/') or contains(@href,'/reel/')]"))
            )
        except TimeoutException:
            log.warning("⚠️  No posts found on hashtag page")
            return []

        # Scroll to load more posts
        for i in range(self.cfg.get("scroll_passes", 4)):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            log.info("   scroll %d/%d", i+1, self.cfg["scroll_passes"])
            time.sleep(2.0)

        # Collect links — both /p/ posts and /reel/ links
        anchors = self.driver.find_elements(
            By.XPATH, "//a[contains(@href,'/p/') or contains(@href,'/reel/')]"
        )
        seen = set()
        links = []
        for a in anchors:
            href = a.get_attribute("href") or ""
            if href and href not in seen:
                seen.add(href)
                links.append(href)

        log.info("📋  Found %d unique post/reel links", len(links))
        return links[:self.cfg["max_posts"]]

    # ── GET FOLLOWER COUNT FROM PROFILE ──────────────────────
    def get_follower_count(self, username: str) -> int:
        """Visit the profile page and parse follower count."""
        if not username or username in ("N/A", ""):
            return 0
        try:
            profile_url = f"https://www.instagram.com/{username}/"
            self.driver.get(profile_url)

            # Instagram shows follower count in a <span> inside <header>
            # Multiple selector fallbacks
            selectors = [
                (By.XPATH, "//header//ul//li[2]//span//span"),
                (By.XPATH, "//header//a[contains(@href,'followers')]//span"),
                (By.CSS_SELECTOR,  "header section ul li:nth-child(2) span span"),
                (By.XPATH, "//span[contains(@class,'_ac2a')]"),
            ]
            for by, sel in selectors:
                try:
                    el = WebDriverWait(self.driver, 6).until(
                        EC.presence_of_element_located((by, sel))
                    )
                    raw = el.get_attribute("title") or el.text
                    count = parse_count(raw)
                    if count > 0:
                        log.info("   👤 %s → %d followers", username, count)
                        return count
                except:
                    continue
        except Exception as e:
            log.debug("   ⚠️  Could not fetch followers for %s: %s", username, e)
        return 0

    # ── SCRAPE A SINGLE POST / REEL PAGE ─────────────────────
    def scrape_post(self, link: str) -> dict:
        """
        Navigate to a post/reel URL and extract all available metrics.
        Returns a structured dict.
        """
        log.info("   📄  Scraping: %s", link)
        self.driver.get(link)
        time.sleep(self.cfg.get("page_delay", 2.5))

        result = {
            "link":            link,
            "media_type":      "reel" if "/reel/" in link else "image",
            "username":        "",
            "follower_count":  0,
            "caption":         "",
            "caption_length":  0,
            "hashtags_count":  0,
            "at_mentions":     0,
            "has_cta":         0,
            "likes":           0,
            "comments_count":  0,
            "views":           0,
            "post_datetime":   "",
            "hour":            9,
            "day_num":         1,
            "month_num":       4,
            "week":            16,
            "scraped_ok":      True,
        }

        # ── Username ─────────────────────────────────────────
        username_el = self._find(
            (By.XPATH, "//header//a[contains(@class,'notranslate')]"),
            (By.XPATH, "//header//a[@role='link']"),
            (By.CSS_SELECTOR, "header a._acan"),
            (By.XPATH, "//article//header//a"),
        )
        if username_el:
            result["username"] = username_el.text.strip() or username_el.get_attribute("href","").split("/")[-2]

        # ── Caption ──────────────────────────────────────────
        caption_selectors = [
            (By.XPATH, "//div[@data-testid='post-comment-root']//span[@dir='auto']"),
            (By.CSS_SELECTOR, "h1._ap3a"),
            (By.XPATH, "//article//div//ul//li[1]//div//span"),
            (By.XPATH, "//div[contains(@class,'_a9zs')]//span"),
            (By.XPATH, "//div[@role='button']//span[@dir='auto']"),
        ]
        for by, sel in caption_selectors:
            try:
                el = WebDriverWait(self.driver, 4).until(
                    EC.presence_of_element_located((by, sel))
                )
                text = el.text.strip()
                if len(text) > 5:
                    result["caption"] = text
                    break
            except:
                continue

        # ── Timestamp ────────────────────────────────────────
        try:
            time_el = self.driver.find_element(By.TAG_NAME, "time")
            dt_str  = time_el.get_attribute("datetime") or ""
            if dt_str:
                dt = datetime.datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                result["post_datetime"] = dt_str
                result["hour"]          = dt.hour
                result["day_num"]       = dt.weekday()   # 0=Monday
                result["month_num"]     = dt.month
                result["week"]          = dt.isocalendar()[1]
        except:
            pass

        # ── Like count ───────────────────────────────────────
        like_selectors = [
            # aria-label approach: "1,234 likes"
            (By.XPATH, "//section//span[@class='_ac2a']"),
            (By.XPATH, "//section//a[contains(@href,'/liked_by/')]//span"),
            (By.CSS_SELECTOR, "section span._ac2a"),
            # "View insights" button text may contain count
            (By.XPATH, "//span[contains(text(),'like')]"),
            (By.XPATH, "//button[contains(@aria-label,'like')]"),
        ]
        for by, sel in like_selectors:
            try:
                el = self.driver.find_element(by, sel)
                raw = el.get_attribute("aria-label") or el.text
                cnt = parse_count(raw)
                if cnt > 0:
                    result["likes"] = cnt
                    break
            except:
                continue

        # ── Comment count ─────────────────────────────────────
        comment_selectors = [
            (By.XPATH, "//a[contains(@href,'/comments/')]//span"),
            (By.XPATH, "//span[contains(text(),'comment')]"),
            (By.CSS_SELECTOR, "span._ae5q"),
        ]
        for by, sel in comment_selectors:
            try:
                el = self.driver.find_element(by, sel)
                raw = el.get_attribute("aria-label") or el.text
                cnt = parse_count(raw)
                if cnt >= 0:
                    result["comments_count"] = cnt
                    break
            except:
                continue

        # ── View count (reels) ───────────────────────────────
        view_selectors = [
            (By.XPATH, "//span[contains(text(),'view') or contains(text(),'View')]"),
            (By.CSS_SELECTOR,  "span._ac2a"),
        ]
        for by, sel in view_selectors:
            try:
                el = self.driver.find_element(by, sel)
                raw = el.get_attribute("aria-label") or el.text
                if "view" in raw.lower():
                    result["views"] = parse_count(raw)
                    break
            except:
                continue

        # ── Derived fields from caption ───────────────────────
        cap = result["caption"]
        result["caption_length"] = len(cap)
        result["hashtags_count"] = count_hashtags(cap)
        result["at_mentions"]    = count_at_mentions(cap)
        result["has_cta"]        = detect_cta(cap)

        return result

    def close(self):
        try:
            self.driver.quit()
        except:
            pass
        log.info("🔒  Browser closed")


# ══════════════════════════════════════════════════════════════
# PART 5 — FEATURE VECTOR BUILDER
# ══════════════════════════════════════════════════════════════

def build_feature_vector(
    scraped: dict,
    defaults: dict,
    all_features: list,
    cat_features: list,
    hashtag_for_category: str,
) -> dict:
    """
    Map scraped reel data onto the model's expected feature columns.
    Uses defaults for any feature we couldn't scrape directly.
    """
    cat = classify_category(scraped.get("caption",""), hashtag_for_category)

    # Estimate reach and impressions from what we scraped
    likes    = int(scraped.get("likes", 0))
    comments = int(scraped.get("comments_count", 0))
    views    = int(scraped.get("views", 0))
    fc       = int(scraped.get("follower_count", defaults["follower_count"]))

    if views > 0:
        # Reels: reach ≈ 60% of views; impressions ≈ views
        reach_est       = int(views * 0.6)
        impressions_est = views
    else:
        reach_est       = estimate_reach(likes, comments, fc)
        impressions_est = estimate_impressions(reach_est)

    raw_vec = {
        # Directly scraped
        "follower_count":     fc or defaults["follower_count"],
        "likes":              likes or defaults["likes"],
        "comments":           comments or defaults["comments"],
        "caption_length":     scraped.get("caption_length", defaults["caption_length"]),
        "hashtags_count":     scraped.get("hashtags_count", defaults["hashtags_count"]),
        "has_call_to_action": scraped.get("has_cta",    defaults["has_call_to_action"]),
        "hour":               scraped.get("hour",       defaults["hour"]),
        "day_num":            scraped.get("day_num",    defaults["day_num"]),
        "month_num":          scraped.get("month_num",  defaults["month_num"]),
        "week":               scraped.get("week",       defaults["week"]),

        # Estimated from scraped + formulas
        "reach":              reach_est,
        "impressions":        impressions_est,
        "shares":             int(defaults["shares"]),   # can't scrape shares
        "saves":              int(defaults["saves"]),    # can't scrape saves
        "followers_gained":   int(defaults["followers_gained"]),  # not visible

        # Categorical — scraped / inferred
        "media_type":         scraped.get("media_type", "reel"),
        "account_type":       defaults["account_type"],  # default: creator
        "content_category":   cat,
        "traffic_source":     "Hashtags",                # found via hashtag page
    }

    # Build a row dict aligned to model's exact feature list
    row = {}
    for feat in all_features:
        if feat in raw_vec:
            row[feat] = raw_vec[feat]
        elif feat in defaults:
            row[feat] = defaults[feat]
        else:
            row[feat] = 0  # safe fallback

    return row, cat


# ══════════════════════════════════════════════════════════════
# PART 6 — ENGAGEMENT TIER CLASSIFIER
# ══════════════════════════════════════════════════════════════

def classify_engagement_tier(rate: float) -> tuple:
    """Returns (tier_label, tier_color_ansi, emoji)."""
    if rate >= 0.08:
        return "VIRAL",     "\033[33m",  "🔥"
    if rate >= 0.05:
        return "HIGH",      "\033[32m",  "🚀"
    if rate >= 0.03:
        return "MEDIUM",    "\033[36m",  "📈"
    if rate >= 0.01:
        return "LOW",       "\033[35m",  "📊"
    return     "VERY LOW",  "\033[31m",  "⚠️"


def compute_viral_coefficient(row: dict) -> float:
    """VC = (saves×3 + shares×2 + comments×1.5) / max(reach,1) × 1000"""
    saves    = row.get("saves",    0) or 0
    shares   = row.get("shares",   0) or 0
    comments = row.get("comments", 0) or 0
    reach    = max(row.get("reach", 1) or 1, 1)
    return round((saves * 3 + shares * 2 + comments * 1.5) / reach * 1000, 4)


# ══════════════════════════════════════════════════════════════
# PART 7 — RICH CONSOLE REPORT
# ══════════════════════════════════════════════════════════════

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
PURPLE = "\033[95m"

def print_report(results: list, model_stats: dict):
    sep = "-" * 70
    banner = "=" * 70
    print(f"\n{BOLD}{CYAN}{banner}{RESET}")
    print(f"{BOLD}{CYAN}  INSTAGRAM REAL REEL ENGAGEMENT PREDICTIONS{RESET}")
    print(f"{BOLD}{CYAN}  Hashtag: #{CONFIG['hashtag']}  |  {len(results)} reels analysed{RESET}")
    print(f"{DIM}  Model: Random Forest  MAE={model_stats['mae']}  R2={model_stats['r2']}{RESET}")
    print(f"{BOLD}{CYAN}{banner}{RESET}\n")

    for i, r in enumerate(results, 1):
        tier, tcolor, emj = classify_engagement_tier(r["predicted_engagement"])
        vc   = r.get("viral_coefficient", 0)
        cat  = r.get("content_category", "N/A")
        username = r.get("username", "unknown")

        print(f"{BOLD}{BLUE}[{i:02d}]  @{username}{RESET}")
        print(f"     {DIM}Link:{RESET} {r.get('link','')[:70]}")
        print(f"     {DIM}Category:{RESET} {PURPLE}{cat}{RESET}  "
              f"{DIM}Media:{RESET} {r.get('media_type','reel')}  "
              f"{DIM}Posted:{RESET} Hour {int(r.get('hour',9))}:00")
        print(f"     {DIM}Caption length:{RESET} {int(r.get('caption_length',0))} chars  "
              f"{DIM}Hashtags:{RESET} {int(r.get('hashtags_count',0))}  "
              f"{DIM}CTA:{RESET} {'Yes' if r.get('has_cta') else 'No'}")
        print(f"     {DIM}Scraped Likes:{RESET} {int(r.get('likes',0)):,}  "
              f"{DIM}Comments:{RESET} {int(r.get('comments_count',0)):,}  "
              f"{DIM}Views:{RESET} {int(r.get('views',0)):,}")
        print(f"     {DIM}Est. Reach:{RESET} {int(r.get('reach',0)):,}  "
              f"{DIM}Followers:{RESET} {int(r.get('follower_count',0)):,}")
        print()
        print(f"     {BOLD}Predicted Engagement Rate:{RESET}  "
              f"{tcolor}{BOLD}{r['predicted_engagement']:.4f}  ({r['predicted_engagement']:.2%}){RESET}")
        print(f"     {BOLD}Performance Tier:          "
              f"{tcolor}{emj}  {tier}{RESET}")
        print(f"     {BOLD}Viral Coefficient:         "
              f"{YELLOW}{vc:.4f}{RESET}")
        print(f"\n{DIM}{sep}{RESET}\n")

    # Summary table
    print(f"{BOLD}{CYAN}  SUMMARY TABLE{RESET}")
    print(f"{BOLD}{'#':>3}  {'@Username':<20} {'Category':<14} {'Eng Rate':>10} {'Tier':<10} {'Viral Coeff':>12}{RESET}")
    print(f"{DIM}{'-'*3}  {'-'*20} {'-'*14} {'-'*10} {'-'*10} {'-'*12}{RESET}")
    for i, r in enumerate(results, 1):
        tier, tcolor, emj = classify_engagement_tier(r["predicted_engagement"])
        print(f"{i:>3}  {r.get('username','?'):<20} "
              f"{r.get('content_category','?'):<14} "
              f"{tcolor}{r['predicted_engagement']:>10.4f}{RESET}  "
              f"{tcolor}{emj} {tier:<8}{RESET}  "
              f"{YELLOW}{r.get('viral_coefficient',0):>12.4f}{RESET}")

    avg_eng = np.mean([r["predicted_engagement"] for r in results])
    avg_vc  = np.mean([r.get("viral_coefficient",0) for r in results])
    best    = max(results, key=lambda r: r["predicted_engagement"])

    print(f"\n{BOLD}  Average Predicted Engagement:  {GREEN}{avg_eng:.4f} ({avg_eng:.2%}){RESET}")
    print(f"{BOLD}  Average Viral Coefficient:     {YELLOW}{avg_vc:.4f}{RESET}")
    print(f"{BOLD}  Best Performing Reel:          @{best.get('username','?')} → {GREEN}{best['predicted_engagement']:.2%}{RESET}\n")


# ══════════════════════════════════════════════════════════════
# PART 8 — SAVE RESULTS
# ══════════════════════════════════════════════════════════════

def save_results(results: list, path: str):
    """Save predictions + metrics to CSV in the data/ directory."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    log.info("💾  Results saved → %s  (%d rows)", path, len(df))
    return df


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def main():
    print(f"\n{BOLD}{CYAN}  +----------------------------------------------+{RESET}")
    print(f"{BOLD}{CYAN}  |  Instagram Real Reel Predictor               |{RESET}")
    print(f"{BOLD}{CYAN}  |  Hashtag: #{CONFIG['hashtag']:<35}|{RESET}")
    print(f"{BOLD}{CYAN}  +----------------------------------------------+{RESET}\n")

    # ── Step 1: Train model ───────────────────────────────────
    pipeline, model_stats, df_defaults = train_model_standalone()
    defaults   = compute_defaults(df_defaults)
    all_feats  = model_stats["features"]
    cat_feats  = model_stats["cat_feats"]

    log.info("📐  Model expects %d features: %s", len(all_feats), all_feats)

    # ── Step 2: Scrape Instagram ──────────────────────────────
    scraper = InstagramScraper(CONFIG)
    try:
        scraper.login()
        links = scraper.collect_reel_links()

        if not links:
            log.error("❌  No reel links found. Exiting.")
            scraper.close()
            return

        scraped_posts = []
        for link in links:
            post = scraper.scrape_post(link)

            # Try to get follower count from profile
            if post.get("username") and post["username"] not in ("N/A",""):
                fc = scraper.get_follower_count(post["username"])
                post["follower_count"] = fc if fc > 0 else int(defaults["follower_count"])
            else:
                post["follower_count"] = int(defaults["follower_count"])

            scraped_posts.append(post)
            time.sleep(0.8)   # polite delay between requests

    except KeyboardInterrupt:
        log.warning("⚠️  Interrupted by user")
        scraped_posts = []
    finally:
        scraper.close()

    if not scraped_posts:
        log.error("❌  No data scraped. Exiting.")
        return

    # ── Step 3: Build feature vectors + predict ───────────────
    log.info("🧠  Running predictions on %d scraped reels…", len(scraped_posts))
    results = []

    for post in scraped_posts:
        feat_vec, category = build_feature_vector(
            scraped=post,
            defaults=defaults,
            all_features=all_feats,
            cat_features=cat_feats,
            hashtag_for_category=CONFIG["hashtag"],
        )

        # Ensure categorical columns are strings
        input_df = pd.DataFrame([feat_vec])
        for c in cat_feats:
            if c in input_df.columns:
                input_df[c] = input_df[c].astype(str)

        try:
            pred = max(0.0, float(pipeline.predict(input_df)[0]))
        except Exception as e:
            log.warning("   ⚠️  Prediction error: %s", e)
            pred = 0.0

        vc = compute_viral_coefficient(feat_vec)

        result_row = {
            # Identification
            "link":                  post.get("link", ""),
            "username":              post.get("username", ""),
            "post_datetime":         post.get("post_datetime", ""),
            "scraped_at":            datetime.datetime.now().isoformat(timespec="seconds"),

            # Scraped metrics
            "scraped_likes":         post.get("likes", 0),
            "scraped_comments":      post.get("comments_count", 0),
            "scraped_views":         post.get("views", 0),
            "follower_count":        post.get("follower_count", 0),
            "caption_length":        post.get("caption_length", 0),
            "hashtags_count":        post.get("hashtags_count", 0),
            "has_cta":               post.get("has_cta", 0),
            "at_mentions":           post.get("at_mentions", 0),
            "media_type":            feat_vec.get("media_type", "reel"),
            "content_category":      category,
            "hour":                  post.get("hour", 9),
            "day_num":               post.get("day_num", 1),
            "month_num":             post.get("month_num", 4),

            # Estimated
            "est_reach":             feat_vec.get("reach", 0),
            "est_impressions":       feat_vec.get("impressions", 0),

            # Predictions
            "predicted_engagement":  round(pred, 6),
            "predicted_pct":         f"{pred:.2%}",
            "viral_coefficient":     vc,
            "engagement_tier":       classify_engagement_tier(pred)[0],
        }
        results.append(result_row)

    # ── Step 4: Print + save ──────────────────────────────────
    print_report(results, model_stats)

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        CONFIG["output_csv"]
    )
    saved_df = save_results(results, out_path)

    print(f"\n{BOLD}{GREEN}  ✅  Done!  Results saved to:{RESET}")
    print(f"     {out_path}")
    print(f"\n{DIM}  To load in your notebooks:{RESET}")
    print(f"     import pandas as pd")
    print(f"     df = pd.read_csv('{out_path}')\n")


# ══════════════════════════════════════════════════════════════
# PART 9 — STREAMLIT UI
# Runs when:  streamlit run modules/checktrend.py
# ══════════════════════════════════════════════════════════════

def _run_streamlit_ui():
    """
    Minimal, clean Streamlit control panel for the scraper pipeline.
    Lets you configure credentials, hashtag, and post count, then:
      1. Scrapes Instagram
      2. Predicts engagement with the current model
      3. Appends real data to data/augmented_data.csv
      4. Retrains the model on base + augmented data
      5. Shows accuracy improvement vs previous run
    """
    import streamlit as st

    st.set_page_config(
        page_title="Scraper & Retrainer",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ── Minimal CSS ───────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html,body,[class*="css"],.stApp{
      font-family:'Inter',sans-serif!important;
      background:#0E1117!important;color:#E2E8F0!important;
    }
    header[data-testid="stHeader"]{display:none!important;}
    .panel{background:#1A2133;border:1px solid #1F2937;border-radius:12px;padding:20px 22px;margin-bottom:16px;}
    .stat-box{background:#0E1117;border:1px solid #1F2937;border-radius:8px;
              padding:12px 16px;text-align:center;}
    .stat-val{font-size:1.6rem;font-weight:800;margin:4px 0;}
    .stat-lbl{font-size:9px;font-weight:700;color:#4B6280;text-transform:uppercase;letter-spacing:.1em;}
    .delta-up{color:#10B981;font-size:11px;font-weight:700;}
    .delta-dn{color:#F43F5E;font-size:11px;font-weight:700;}
    .stButton>button{
      background:#4F46E5!important;color:#fff!important;border:none!important;
      border-radius:8px!important;font-weight:600!important;font-size:13px!important;
      padding:10px 24px!important;transition:opacity .15s!important;
    }
    .stButton>button:hover{opacity:.85!important;}
    .stTextInput>div>div,.stNumberInput>div>div,.stSelectbox>div>div{
      background:#0E1117!important;border-color:#1F2937!important;
      border-radius:8px!important;color:#E2E8F0!important;
    }
    hr{border-color:#1F2937!important;}
    </style>
    """, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:28px 0 20px;">
      <div style="font-size:1.5rem;font-weight:800;color:#E2E8F0;letter-spacing:-.02em;">
        Scraper &amp; Incremental Retrainer
      </div>
      <div style="font-size:12px;color:#4B6280;margin-top:6px;">
        Scrape real Instagram reels, predict engagement, append to training data,
        and retrain the model — all in one run.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load training log for history ─────────────────────────
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from modules.data_engine import (
        append_scraped_data, retrain_model, load_training_log,
        get_dataframe, AUGMENTED_CSV,
    )

    history = load_training_log()
    last    = history[-1] if history else None

    # ── Current model stats ───────────────────────────────────
    st.markdown('<div style="font-size:9px;font-weight:700;color:#4B6280;'
                'text-transform:uppercase;letter-spacing:.12em;margin-bottom:10px;">'
                'Current Model</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    aug_count = 0
    if os.path.exists(AUGMENTED_CSV):
        try:
            aug_count = len(pd.read_csv(AUGMENTED_CSV))
        except Exception:
            aug_count = 0

    with c1:
        mae_val = f"{last['mae']:.5f}" if last else "Not trained"
        st.markdown(f'<div class="stat-box"><div class="stat-lbl">MAE</div>'
                    f'<div class="stat-val" style="color:#3B82F6;">{mae_val}</div></div>',
                    unsafe_allow_html=True)
    with c2:
        r2_val = f"{last['r2']:.4f}" if last else "Not trained"
        st.markdown(f'<div class="stat-box"><div class="stat-lbl">R²</div>'
                    f'<div class="stat-val" style="color:#8B5CF6;">{r2_val}</div></div>',
                    unsafe_allow_html=True)
    with c3:
        n_train = f"{last['n_train']:,}" if last else "—"
        st.markdown(f'<div class="stat-box"><div class="stat-lbl">Training Rows</div>'
                    f'<div class="stat-val" style="color:#10B981;">{n_train}</div></div>',
                    unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-box"><div class="stat-lbl">Augmented Rows</div>'
                    f'<div class="stat-val" style="color:#F59E0B;">{aug_count:,}</div></div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Configuration form ────────────────────────────────────
    st.markdown('<div style="font-size:9px;font-weight:700;color:#4B6280;'
                'text-transform:uppercase;letter-spacing:.12em;margin-bottom:10px;">'
                'Scraper Configuration</div>', unsafe_allow_html=True)

    with st.form("scraper_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            username  = st.text_input("Instagram Username", value=CONFIG["username"])
            password  = st.text_input("Instagram Password", value=CONFIG["password"], type="password")
            hashtag   = st.text_input("Target Hashtag (no #)", value=CONFIG["hashtag"])
        with col_b:
            max_posts    = st.number_input("Max Posts to Scrape", min_value=1, max_value=50,
                                           value=CONFIG["max_posts"], step=1)
            scroll_passes = st.number_input("Scroll Passes", min_value=1, max_value=10,
                                            value=CONFIG["scroll_passes"], step=1)
            headless     = st.checkbox("Headless Browser", value=CONFIG["headless"])

        st.markdown("<br>", unsafe_allow_html=True)
        col_run, col_retrain, _ = st.columns([2, 2, 4])
        with col_run:
            run_scraper = st.form_submit_button("Scrape + Predict + Retrain")
        with col_retrain:
            retrain_only = st.form_submit_button("Retrain Only (no scrape)")

    # ── Retrain only ──────────────────────────────────────────
    if retrain_only:
        with st.spinner("Retraining model on base + augmented data..."):
            try:
                base_df = get_dataframe()
                new_metrics = retrain_model(base_df)
                st.success(f"Model retrained. MAE: {new_metrics['mae']}  R2: {new_metrics['r2']}  "
                           f"Trees: {new_metrics['n_trees']}  Augmented rows: {new_metrics['n_augmented']}")
                _show_accuracy_delta(history, new_metrics)
                st.rerun()
            except Exception as e:
                st.error(f"Retraining failed: {e}")

    # ── Full scrape + retrain pipeline ────────────────────────
    if run_scraper:
        cfg = {**CONFIG,
               "username": username, "password": password,
               "hashtag": hashtag, "max_posts": int(max_posts),
               "scroll_passes": int(scroll_passes), "headless": headless}

        progress = st.progress(0, text="Starting scraper...")
        status   = st.empty()
        results_placeholder = st.empty()

        # Step 1 — Train model for predictions
        status.info("Loading model for predictions...")
        try:
            pipeline, model_stats, df_defaults = train_model_standalone()
            defaults  = compute_defaults(df_defaults)
            all_feats = model_stats["features"]
            cat_feats = model_stats["cat_feats"]
        except Exception as e:
            st.error(f"Model load failed: {e}")
            st.stop()

        progress.progress(10, text="Model ready. Launching browser...")

        # Step 2 — Scrape
        scraper = InstagramScraper(cfg)
        scraped_posts = []
        try:
            status.info("Logging in to Instagram...")
            scraper.login()
            progress.progress(20, text="Logged in. Collecting links...")

            links = scraper.collect_reel_links()
            if not links:
                st.warning("No reel links found for that hashtag.")
                scraper.close()
                st.stop()

            status.info(f"Found {len(links)} links. Scraping posts...")
            for idx, link in enumerate(links):
                post = scraper.scrape_post(link)
                if post.get("username"):
                    fc = scraper.get_follower_count(post["username"])
                    post["follower_count"] = fc if fc > 0 else int(defaults["follower_count"])
                else:
                    post["follower_count"] = int(defaults["follower_count"])
                scraped_posts.append(post)
                pct = 20 + int((idx + 1) / len(links) * 40)
                progress.progress(pct, text=f"Scraped {idx+1}/{len(links)} posts...")
        except Exception as e:
            st.error(f"Scraping error: {e}")
        finally:
            scraper.close()

        if not scraped_posts:
            st.error("No posts scraped. Check credentials and hashtag.")
            st.stop()

        # Step 3 — Predict
        progress.progress(65, text="Running engagement predictions...")
        results = []
        for post in scraped_posts:
            feat_vec, category = build_feature_vector(
                scraped=post, defaults=defaults,
                all_features=all_feats, cat_features=cat_feats,
                hashtag_for_category=cfg["hashtag"],
            )
            input_df = pd.DataFrame([feat_vec])
            for c in cat_feats:
                if c in input_df.columns:
                    input_df[c] = input_df[c].astype(str)
            try:
                pred = max(0.0, float(pipeline.predict(input_df)[0]))
            except Exception:
                pred = 0.0

            vc = compute_viral_coefficient(feat_vec)
            tier_label = classify_engagement_tier(pred)[0]

            result_row = {
                "link":                 post.get("link", ""),
                "username":             post.get("username", ""),
                "post_datetime":        post.get("post_datetime", ""),
                "scraped_at":           datetime.datetime.now().isoformat(timespec="seconds"),
                "scraped_likes":        post.get("likes", 0),
                "scraped_comments":     post.get("comments_count", 0),
                "scraped_views":        post.get("views", 0),
                "follower_count":       post.get("follower_count", 0),
                "caption_length":       post.get("caption_length", 0),
                "hashtags_count":       post.get("hashtags_count", 0),
                "has_call_to_action":   post.get("has_cta", 0),
                "media_type":           feat_vec.get("media_type", "reel"),
                "content_category":     category,
                "hour":                 post.get("hour", 9),
                "day_num":              post.get("day_num", 1),
                "month_num":            post.get("month_num", 4),
                "reach":                feat_vec.get("reach", 0),
                "impressions":          feat_vec.get("impressions", 0),
                "likes":                post.get("likes", 0),
                "comments":             post.get("comments_count", 0),
                "shares":               int(defaults.get("shares", 0)),
                "saves":                int(defaults.get("saves", 0)),
                "followers_gained":     int(defaults.get("followers_gained", 0)),
                "account_type":         defaults.get("account_type", "creator"),
                "traffic_source":       "Hashtags",
                "predicted_engagement": round(pred, 6),
                "viral_coefficient":    vc,
                "engagement_tier":      tier_label,
                # engagement_rate = predicted (best estimate for training)
                "engagement_rate":      round(pred, 6),
            }
            results.append(result_row)

        # Step 4 — Show predictions table
        progress.progress(75, text="Saving scraped data...")
        results_df = pd.DataFrame(results)
        display_cols = [c for c in [
            "username", "content_category", "media_type",
            "predicted_engagement", "viral_coefficient", "engagement_tier",
            "scraped_likes", "scraped_comments", "scraped_views",
            "hashtags_count", "has_call_to_action",
        ] if c in results_df.columns]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:9px;font-weight:700;color:#4B6280;'
                    'text-transform:uppercase;letter-spacing:.12em;margin-bottom:10px;">'
                    'Scraped & Predicted Posts</div>', unsafe_allow_html=True)
        st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)

        # Step 5 — Append to augmented data
        rows_added, total_aug = append_scraped_data(results)
        progress.progress(85, text=f"Appended {rows_added} new rows. Retraining model...")

        # Step 6 — Retrain
        status.info(f"Retraining on base data + {total_aug} augmented rows...")
        try:
            base_df = get_dataframe()
            new_metrics = retrain_model(base_df)
            progress.progress(100, text="Done.")
            status.empty()

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div style="font-size:9px;font-weight:700;color:#4B6280;'
                        'text-transform:uppercase;letter-spacing:.12em;margin-bottom:10px;">'
                        'Retraining Results</div>', unsafe_allow_html=True)

            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown(f'<div class="stat-box"><div class="stat-lbl">New MAE</div>'
                            f'<div class="stat-val" style="color:#3B82F6;">{new_metrics["mae"]:.5f}</div></div>',
                            unsafe_allow_html=True)
            with r2:
                st.markdown(f'<div class="stat-box"><div class="stat-lbl">New R²</div>'
                            f'<div class="stat-val" style="color:#8B5CF6;">{new_metrics["r2"]:.4f}</div></div>',
                            unsafe_allow_html=True)
            with r3:
                st.markdown(f'<div class="stat-box"><div class="stat-lbl">Training Rows</div>'
                            f'<div class="stat-val" style="color:#10B981;">{new_metrics["n_train"]:,}</div></div>',
                            unsafe_allow_html=True)
            with r4:
                st.markdown(f'<div class="stat-box"><div class="stat-lbl">Augmented Rows</div>'
                            f'<div class="stat-val" style="color:#F59E0B;">{new_metrics["n_augmented"]:,}</div></div>',
                            unsafe_allow_html=True)

            _show_accuracy_delta(history, new_metrics)

        except Exception as e:
            st.error(f"Retraining failed: {e}")

        # Save predictions CSV
        out_path = os.path.join(project_root, CONFIG["output_csv"])
        results_df.to_csv(out_path, index=False)
        st.caption(f"Predictions saved to {out_path}")

    # ── Training history chart ────────────────────────────────
    history = load_training_log()
    if len(history) > 1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:9px;font-weight:700;color:#4B6280;'
                    'text-transform:uppercase;letter-spacing:.12em;margin-bottom:10px;">'
                    'Training History</div>', unsafe_allow_html=True)

        hist_df = pd.DataFrame(history)
        hist_df["run"] = range(1, len(hist_df) + 1)

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_df["run"], y=hist_df["mae"],
            mode="lines+markers",
            name="MAE",
            line=dict(color="#3B82F6", width=2),
            marker=dict(size=7, color="#3B82F6"),
            yaxis="y1",
        ))
        fig.add_trace(go.Scatter(
            x=hist_df["run"], y=hist_df["r2"],
            mode="lines+markers",
            name="R²",
            line=dict(color="#8B5CF6", width=2),
            marker=dict(size=7, color="#8B5CF6"),
            yaxis="y2",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#6B82A0"),
            height=260,
            margin=dict(t=30, b=10, l=10, r=10),
            xaxis=dict(title="Training Run", gridcolor="#1A2133",
                       tickfont=dict(color="#4B6280")),
            yaxis=dict(title="MAE", gridcolor="#1A2133",
                       tickfont=dict(color="#3B82F6"), titlefont=dict(color="#3B82F6")),
            yaxis2=dict(title="R²", overlaying="y", side="right",
                        tickfont=dict(color="#8B5CF6"), titlefont=dict(color="#8B5CF6"),
                        gridcolor="rgba(0,0,0,0)"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#6B82A0")),
            hoverlabel=dict(bgcolor="#1A2133", bordercolor="#2D3F55",
                            font=dict(color="#E2E8F0")),
            title=dict(text="Model Accuracy Over Training Runs",
                       font=dict(color="#94A3B8", size=12), x=0),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.dataframe(
            hist_df[["run", "timestamp", "mae", "r2", "n_train", "n_augmented", "n_trees"]],
            use_container_width=True,
            hide_index=True,
        )


def _show_accuracy_delta(history: list, new_metrics: dict):
    """Show a delta card comparing new metrics to the previous run."""
    import streamlit as st
    if not history:
        return
    prev = history[-1]
    mae_delta = new_metrics["mae"] - prev["mae"]
    r2_delta  = new_metrics["r2"]  - prev["r2"]

    mae_clr = "#10B981" if mae_delta < 0 else "#F43F5E"
    r2_clr  = "#10B981" if r2_delta  > 0 else "#F43F5E"
    mae_arrow = "down" if mae_delta < 0 else "up"
    r2_arrow  = "up"   if r2_delta  > 0 else "down"

    st.markdown(f"""
    <div style="background:#1A2133;border:1px solid #1F2937;border-radius:10px;
         padding:14px 18px;margin-top:12px;font-size:13px;color:#CBD5E1;">
      <div style="font-size:9px;font-weight:700;color:#4B6280;text-transform:uppercase;
           letter-spacing:.12em;margin-bottom:8px;">vs Previous Run</div>
      <span style="margin-right:20px;">
        MAE: <strong style="color:{mae_clr};">{mae_delta:+.5f}</strong>
        <span style="color:{mae_clr};font-size:10px;">
          ({'lower is better' if mae_delta < 0 else 'higher — more data needed'})
        </span>
      </span>
      <span>
        R²: <strong style="color:{r2_clr};">{r2_delta:+.4f}</strong>
        <span style="color:{r2_clr};font-size:10px;">
          ({'improved' if r2_delta > 0 else 'slight drop — normal with new data distribution'})
        </span>
      </span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Detect if running under Streamlit
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        _in_streamlit = get_script_run_ctx() is not None
    except Exception:
        _in_streamlit = False

    if _in_streamlit:
        _run_streamlit_ui()
    else:
        main()