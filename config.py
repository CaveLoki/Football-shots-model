"""
Central configuration. Every threshold/window the app exposes as an
adjustable control reads its DEFAULT from here — the Streamlit app
overrides these at runtime, this file only sets what a fresh run starts with.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DB_PATH = os.path.join(DATA_DIR, "matches.db")

for d in (DATA_DIR, MODELS_DIR, LOGS_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Season boundary — nothing before this date ever enters the database.
# This is what "only this season" is enforced by; change it once per season.
# ---------------------------------------------------------------------------
SEASON_START_DATE = "2026-08-01"
SEASON_LABEL = "2026-2027"

# ---------------------------------------------------------------------------
# Competitions in scope: FBref numeric competition IDs.
# IMPORTANT: verify these against the current FBref URL before the first
# real run — see README "Before you run anything" section. FBref does not
# change these often, but they are not guaranteed stable forever.
# ---------------------------------------------------------------------------
COMPETITIONS = {
    "Premier League": 9,
    "La Liga": 12,
    "Bundesliga": 20,
    "Serie A": 11,
    "Ligue 1": 13,
    "Champions League": 8,
    "Europa League": 19,
    "Conference League": 882,
}

# ---------------------------------------------------------------------------
# Scraper politeness settings
# ---------------------------------------------------------------------------
MIN_DELAY_SECONDS = 3.5
MAX_DELAY_SECONDS = 6.5
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 20   # first retry waits ~20s, then doubles
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
REQUEST_CACHE_DIR = os.path.join(DATA_DIR, "html_cache")
os.makedirs(REQUEST_CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature engineering defaults (all overridable from the Streamlit sidebar)
# ---------------------------------------------------------------------------
RECENT_N_OVERALL = 5          # rolling window for a player's general form
RECENT_N_OPPONENT = 3         # rolling window for form vs this exact opponent
MIN_MINUTES_TO_COUNT = 45     # ignore appearances shorter than this
MIN_SAMPLE_SIZE_WARNING = 3   # flag predictions built on fewer than N games

# ---------------------------------------------------------------------------
# Prediction thresholds (defaults — adjustable live in the app)
# ---------------------------------------------------------------------------
DEFAULT_SHOT_THRESHOLDS = [1, 2, 3]
DEFAULT_SOT_THRESHOLDS = [1, 2]
DEFAULT_MIN_PERCENTAGE = 0.60   # confidence (predicted probability) required to flag "eligible"

# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------
PSI_WARNING = 0.10
PSI_ALERT = 0.25
DRIFT_BASELINE_WEEKS = 4   # first N weeks of the season = the baseline distribution

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_SHOTS_PATH = os.path.join(MODELS_DIR, "model_shots.joblib")
MODEL_SOT_PATH = os.path.join(MODELS_DIR, "model_sot.joblib")
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "feature_list.json")
TEST_HOLDOUT_WEEKS = 3   # most recent N weeks held out for time-based validation
