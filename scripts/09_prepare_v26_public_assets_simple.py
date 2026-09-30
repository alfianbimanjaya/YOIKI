# -*- coding: utf-8 -*-
"""Create compact PUBLIC deployment assets from the local YOIKI project."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_h1_ohlcv_v2.parquet"
OUT_UNIVERSE = ROOT / "universe_v26.csv"
OUT_SEED = ROOT / "v26_initial_history.parquet"
MODEL_DIR = ROOT / "models_v2"
KEEP_ROWS = 700

required = [
    RAW,
    MODEL_DIR / "lgbm_h1_v2.txt",
    MODEL_DIR / "isotonic_calibrator_h1_v2.joblib",
    MODEL_DIR / "feature_columns_h1_v2.json",
    MODEL_DIR / "model_metadata_h1_v2.json",
    MODEL_DIR / "selected_threshold_h1_v2.json",
]
for p in required:
    if not p.exists():
        raise FileNotFoundError(str(p))

raw = pd.read_parquet(RAW)
raw["datetime"] = pd.to_datetime(raw["datetime"], errors="coerce")
raw = raw.dropna(subset=["ticker", "datetime", "open", "high", "low", "close", "volume"])

universe = (
    raw["ticker"].astype(str).str.strip().str.upper().drop_duplicates().sort_values()
)
pd.DataFrame({"ticker": universe}).to_csv(OUT_UNIVERSE, index=False)

seed = (
    raw.sort_values(["ticker", "datetime"])
       .groupby("ticker", group_keys=False)
       .tail(KEEP_ROWS)
       .reset_index(drop=True)
)
seed.to_parquet(OUT_SEED, index=False)

print("=" * 90)
print("YOIKI H1 V2.6 — SIMPLE PUBLIC ASSETS")
print("=" * 90)
print(f"Tickers: {len(universe)}")
print(f"Seed rows: {len(seed):,}")
print(f"Universe: {OUT_UNIVERSE}")
print(f"Seed: {OUT_SEED}")
print("Model files in models_v2/ will be PUBLIC in this deployment.")
