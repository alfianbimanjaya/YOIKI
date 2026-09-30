from pathlib import Path

# =============================================================================
# YOIKI H1 V2.1 — SHARED CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

UNIVERSE_FILE = PROJECT_ROOT / "universe_research.csv"
DATA_DIR = PROJECT_ROOT / "data"
RAW_FILE = DATA_DIR / "raw_h1_ohlcv_v2.parquet"
PROCESSED_DIR = DATA_DIR / "processed_v2"
FEATURES_FILE = PROCESSED_DIR / "features_h1_v2.parquet"
LABELED_FILE = PROCESSED_DIR / "dataset_labeled_h1_v2.parquet"
VALIDATION_PRED_FILE = PROCESSED_DIR / "validation_predictions_h1_v2.parquet"
FINAL_TEST_PRED_FILE = PROCESSED_DIR / "final_test_predictions_h1_v2.parquet"

MODELS_DIR = PROJECT_ROOT / "models_v2"
LOGS_DIR = PROJECT_ROOT / "logs_v2"
MODEL_FILE = MODELS_DIR / "lgbm_h1_v2.txt"
CALIBRATOR_FILE = MODELS_DIR / "isotonic_calibrator_h1_v2.joblib"
FEATURES_JSON = MODELS_DIR / "feature_columns_h1_v2.json"
MODEL_META_JSON = MODELS_DIR / "model_metadata_h1_v2.json"
THRESHOLD_JSON = MODELS_DIR / "selected_threshold_h1_v2.json"

TIMEZONE = "Asia/Jakarta"
BAR_HOURS = 1
INITIAL_FETCH_DAYS = 720
OVERLAP_DAYS = 2
MIN_PRICE = 80.0

# Frozen target — do not change while measuring model improvements.
TP_PCT = 0.04
SL_PCT = 0.02
MAX_DRAWDOWN_PCT = 0.012
MAX_HOLDING = 15

# Portfolio / execution baseline.
INITIAL_CAPITAL = 100_000_000.0
POSITION_ALLOCATION_PCT = 0.20
MAX_POSITIONS = 5
LOT_SIZE = 100
BUY_FEE_PCT = 0.0015
SELL_FEE_PCT = 0.0025
SLIPPAGE_PCT = 0.0010
BE_TRIGGER_PCT = 0.025

# Model: optimize ranking first; do not distort probabilities with class weighting.
RANDOM_STATE = 42
N_ESTIMATORS = 1500
LEARNING_RATE = 0.03
NUM_LEAVES = 31
MAX_DEPTH = 7
MIN_CHILD_SAMPLES = 120
SUBSAMPLE = 0.90
COLSAMPLE = 0.80
REG_ALPHA = 0.10
REG_LAMBDA = 1.00
EARLY_STOPPING_ROUNDS = 80
INNER_EARLY_STOP_SHARE = 0.15

# Global chronological split by unique timestamps.
TRAIN_SHARE = 0.60
CALIBRATION_SHARE = 0.10
VALIDATION_SHARE = 0.10
FINAL_TEST_SHARE = 0.20
PURGE_BARS = MAX_HOLDING

# The threshold is NOT hard-coded to 0.70+ anymore. V2.1 builds candidates from
# actual validation score distribution plus a broad probability grid.
THRESHOLD_MIN = 0.10
THRESHOLD_MAX = 0.80
THRESHOLD_STEP = 0.01
THRESHOLD_QUANTILES = [0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99]
MIN_VALIDATION_TRADES = 20
MIN_VALIDATION_WIN_RATE = 0.65

# Live only uses a completed H1 candle.
REQUIRE_CLOSED_BAR = True
