# -*- coding: utf-8 -*-
"""
YOIKI H1 V2.6 — SIMPLE PUBLIC GITHUB ACTIONS RUNNER
PATH-FIXED VERSION

Important:
- Does NOT modify h1_v2_config.py.
- Production model paths are resolved explicitly from repository root.
- Standalone: no dependency on the old realtime Telegram bot.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

feature_factory = __import__("02_feature_factory_h1_v2")
compute_ticker_features = feature_factory.compute_ticker_features

# Only import scalar configuration from h1_v2_config.
# Do NOT import its path constants because the public cloud repo layout
# has h1_v2_config.py at repository root.
from h1_v2_config import MIN_PRICE, TIMEZONE

# ============================================================================
# EXPLICIT PUBLIC-REPO PATHS
# ============================================================================
MODEL_DIR = ROOT / "models_v2"

MODEL_FILE = MODEL_DIR / "lgbm_h1_v2.txt"
CALIBRATOR_FILE = MODEL_DIR / "isotonic_calibrator_h1_v2.joblib"
FEATURES_JSON = MODEL_DIR / "feature_columns_h1_v2.json"
MODEL_META_JSON = MODEL_DIR / "model_metadata_h1_v2.json"

UNIVERSE_FILE = ROOT / "universe_v26.csv"
SEED_HISTORY = ROOT / "v26_initial_history.parquet"

RUNTIME_DIR = ROOT / "cloud_runtime"
HISTORY_FILE = RUNTIME_DIR / "history.parquet"
STATE_FILE = RUNTIME_DIR / "state.json"

# ============================================================================
# FROZEN V2.6
# ============================================================================
VERSION = "V2.6"
THRESHOLD = 0.4336
SIGNAL_GAP_MAX = 0.015
ENTRY_GAP_MAX = 0.001
TP_PCT = 0.04
SL_PCT = 0.02
BE_TRIGGER_PCT = 0.025
MAX_HOLDING_H1 = 15
MAX_POSITIONS = 5
ALLOC_PCT = 0.20

FETCH_DAYS = 12
MIN_HISTORY_ROWS = 210
KEEP_HISTORY_ROWS = 700


def now_wib():
    return datetime.now(ZoneInfo(TIMEZONE)).replace(tzinfo=None)


def fmt_price(x):
    return f"Rp {float(x):,.0f}".replace(",", ".")


def fmt_pct(x):
    return f"{float(x) * 100:+.2f}%"


def telegram(method, payload):
    token = os.environ["TELEGRAM_BOT_TOKEN"].strip()

    body = urllib.parse.urlencode(
        {k: str(v) for k, v in payload.items() if v is not None}
    ).encode("utf-8")

    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/{method}",
        data=body,
        headers={"User-Agent": "YOIKI-H1-V2.6-GHA/1.0"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=20) as r:
        result = json.loads(r.read().decode("utf-8"))

    if not result.get("ok"):
        raise RuntimeError(result)

    return result


def send_message(text):
    telegram(
        "sendMessage",
        {
            "chat_id": os.environ["TELEGRAM_CHAT_ID"].strip(),
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        },
    )


def load_universe():
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError(
            f"Universe tidak ditemukan: {UNIVERSE_FILE}"
        )

    df = pd.read_csv(UNIVERSE_FILE)

    if "ticker" not in df.columns:
        raise KeyError(
            "universe_v26.csv wajib mempunyai kolom ticker."
        )

    return (
        df["ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .drop_duplicates()
        .tolist()
    )


def load_model():
    for p in (
        MODEL_FILE,
        CALIBRATOR_FILE,
        FEATURES_JSON,
        MODEL_META_JSON,
    ):
        if not p.exists():
            raise FileNotFoundError(
                f"Production artifact tidak ditemukan: {p}"
            )

    meta = json.loads(
        MODEL_META_JSON.read_text(encoding="utf-8")
    )

    feature_cols = json.loads(
        FEATURES_JSON.read_text(encoding="utf-8")
    )

    count = int(meta.get("feature_count", 0))

    if count and count != len(feature_cols):
        raise RuntimeError(
            f"Feature mismatch metadata={count} "
            f"json={len(feature_cols)}"
        )

    print(f"MODEL FILE : {MODEL_FILE}")
    print(f"CALIBRATOR : {CALIBRATOR_FILE}")
    print(f"FEATURE JSON: {FEATURES_JSON}")
    print(f"MODEL META : {MODEL_META_JSON}")

    return (
        lgb.Booster(model_file=str(MODEL_FILE)),
        joblib.load(CALIBRATOR_FILE),
        feature_cols,
    )


def fetch_recent(tickers):
    now = datetime.now(ZoneInfo(TIMEZONE))
    start = now - timedelta(days=FETCH_DAYS)
    end = now + timedelta(days=1)

    yf_tickers = [
        t if t.endswith(".JK") else f"{t}.JK"
        for t in tickers
    ]

    raw = yf.download(
        tickers=yf_tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1h",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    rows = []

    for t in yf_tickers:
        try:
            if len(yf_tickers) == 1:
                df = raw.copy()
            else:
                if not hasattr(raw, "columns") or t not in raw.columns.levels[0]:
                    continue
                df = raw[t].copy()

            df = df.dropna(subset=["Close"]).reset_index()

            if df.empty:
                continue

            date_col = (
                "Datetime"
                if "Datetime" in df.columns
                else "Date"
            )

            dt = pd.to_datetime(
                df[date_col],
                errors="coerce",
            )

            if getattr(dt.dt, "tz", None) is not None:
                dt = (
                    dt.dt.tz_convert(TIMEZONE)
                    .dt.tz_localize(None)
                )

            rows.append(
                pd.DataFrame(
                    {
                        "ticker": t.replace(".JK", ""),
                        "datetime": dt,
                        "open": df["Open"].to_numpy(),
                        "high": df["High"].to_numpy(),
                        "low": df["Low"].to_numpy(),
                        "close": df["Close"].to_numpy(),
                        "volume": df["Volume"].to_numpy(),
                    }
                )
            )

        except Exception as exc:
            print(f"FETCH ERROR {t}: {exc}")

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def load_history():
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    if HISTORY_FILE.exists():
        return pd.read_parquet(HISTORY_FILE)

    if not SEED_HISTORY.exists():
        raise FileNotFoundError(SEED_HISTORY)

    return pd.read_parquet(SEED_HISTORY)


def save_history(history):
    history["datetime"] = pd.to_datetime(
        history["datetime"],
        errors="coerce",
    )

    history = (
        history.dropna(
            subset=[
                "ticker",
                "datetime",
                "close",
            ]
        )
        .drop_duplicates(
            ["ticker", "datetime"],
            keep="last",
        )
        .sort_values(
            ["ticker", "datetime"]
        )
    )

    history = (
        history.groupby(
            "ticker",
            group_keys=False,
        )
        .tail(KEEP_HISTORY_ROWS)
        .reset_index(drop=True)
    )

    history.to_parquet(
        HISTORY_FILE,
        index=False,
    )


def keep_closed(df):
    now = pd.Timestamp(now_wib())

    dt = pd.to_datetime(
        df["datetime"],
        errors="coerce",
    )

    return df[
        dt + pd.Timedelta(hours=1) <= now
    ].copy()


def latest_features(history):
    closed = keep_closed(history)

    blocks = []

    for ticker, group in closed.groupby(
        "ticker",
        sort=False,
    ):
        if len(group) < MIN_HISTORY_ROWS:
            continue

        try:
            blocks.append(
                compute_ticker_features(group)
            )
        except Exception as exc:
            print(f"FEATURE ERROR {ticker}: {exc}")

    if not blocks:
        return pd.DataFrame()

    features = pd.concat(
        blocks,
        ignore_index=True,
    )

    return (
        features
        .sort_values(
            ["ticker", "datetime"]
        )
        .groupby(
            "ticker",
            as_index=False,
        )
        .tail(1)
        .copy()
    )


def load_state():
    if not STATE_FILE.exists():
        return {"sent": {}}

    try:
        state = json.loads(
            STATE_FILE.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(state, dict):
            return {"sent": {}}

        state.setdefault("sent", {})

        return state

    except Exception:
        return {"sent": {}}


def save_state(state):
    STATE_FILE.write_text(
        json.dumps(
            state,
            indent=2,
        ),
        encoding="utf-8",
    )


def next_bar(
    history,
    ticker,
    signal_dt,
):
    df = history[
        history["ticker"]
        .astype(str)
        .str.upper()
        == ticker
    ].copy()

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
    )

    df = df[
        df["datetime"]
        > pd.Timestamp(signal_dt)
    ].sort_values("datetime")

    if df.empty:
        return None

    return df.iloc[0]


def main():
    print("=" * 100)
    print("YOIKI H1 V2.6 — SIMPLE PUBLIC GITHUB ACTIONS")
    print("=" * 100)

    print(f"REPO ROOT : {ROOT}")
    print(f"MODEL DIR : {MODEL_DIR}")

    universe = load_universe()

    print(f"Universe: {len(universe)}")

    model, calibrator, feature_cols = load_model()

    recent = fetch_recent(universe)

    if recent.empty:
        raise RuntimeError(
            "No Yahoo data returned. "
            "DATA ERROR, not NO SIGNAL."
        )

    got = set(
        recent["ticker"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
    )

    coverage = (
        100.0
        * len(got)
        / max(len(universe), 1)
    )

    print(
        f"DATA HEALTH: {len(got)}/{len(universe)} "
        f"({coverage:.1f}%)"
    )

    if coverage < 90.0:
        raise RuntimeError(
            f"DATA HEALTH too low: {coverage:.1f}%"
        )

    history = pd.concat(
        [
            load_history(),
            recent,
        ],
        ignore_index=True,
    )

    save_history(history)

    latest = latest_features(history)

    if latest.empty:
        print("NO FEATURES — history belum cukup.")
        return

    missing = [
        c
        for c in feature_cols
        if c not in latest.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing production feature(s): "
            + ", ".join(missing)
        )

    raw_score = model.predict(
        latest[feature_cols]
    )

    latest["pred_calibrated"] = np.asarray(
        calibrator.predict(raw_score),
        dtype=float,
    )

    candidates = latest[
        (latest["pred_calibrated"] >= THRESHOLD)
        & (latest["close"] >= MIN_PRICE)
        & (
            pd.to_numeric(
                latest["gap_1h"],
                errors="coerce",
            )
            <= SIGNAL_GAP_MAX
        )
    ].copy()

    candidates = candidates.sort_values(
        "pred_calibrated",
        ascending=False,
    )

    print(
        f"Candidates: {len(candidates)}"
    )

    state = load_state()
    new_count = 0

    for _, row in candidates.iterrows():
        ticker = str(
            row["ticker"]
        ).upper()

        signal_dt = pd.Timestamp(
            row["datetime"]
        )

        key = (
            f"{ticker}|"
            f"{signal_dt.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        if key in state["sent"]:
            continue

        nxt = next_bar(
            history,
            ticker,
            signal_dt,
        )

        if nxt is None:
            continue

        signal_close = float(
            row["close"]
        )

        entry = float(
            nxt["open"]
        )

        if signal_close <= 0:
            continue

        entry_gap = (
            entry
            / signal_close
            - 1.0
        )

        score = float(
            row["pred_calibrated"]
        )

        if abs(entry_gap) <= ENTRY_GAP_MAX:
            msg = (
                "🟢 <b>YOIKI H1 V2.6 — ENTRY READY</b>\n\n"
                f"<b>{ticker}</b>\n"
                f"Signal: <code>{signal_dt:%Y-%m-%d %H:%M}</code>\n"
                f"P: <b>{score:.4f}</b>\n"
                f"Signal close: {fmt_price(signal_close)}\n"
                f"Next open: <b>{fmt_price(entry)}</b>\n"
                f"Execution gap: <b>{fmt_pct(entry_gap)}</b>\n\n"
                f"TP: {fmt_price(entry * 1.04)} (+4%)\n"
                f"SL: {fmt_price(entry * 0.98)} (-2%)\n"
                f"BE: {fmt_price(entry * 1.025)} (+2.5%)\n"
                "⚠️ BOT TIDAK MENEMPATKAN ORDER."
            )
        else:
            msg = (
                "🔴 <b>YOIKI H1 V2.6 — SKIP</b>\n\n"
                f"<b>{ticker}</b>\n"
                f"Signal: <code>{signal_dt:%Y-%m-%d %H:%M}</code>\n"
                f"P: <b>{score:.4f}</b>\n"
                f"Next open: {fmt_price(entry)}\n"
                f"Execution gap: <b>{fmt_pct(entry_gap)}</b>\n\n"
                "❌ Gap > 0.10% — SKIP."
            )

        send_message(msg)

        state["sent"][key] = (
            now_wib().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        new_count += 1

    cutoff = (
        now_wib()
        - timedelta(days=7)
    )

    for key, stamp in list(
        state["sent"].items()
    ):
        try:
            if (
                pd.Timestamp(stamp)
                .to_pydatetime()
                < cutoff
            ):
                del state["sent"][key]
        except Exception:
            pass

    save_state(state)

    print(
        f"[{now_wib():%Y-%m-%d %H:%M:%S}] "
        f"candidates={len(candidates)} "
        f"new_signal={new_count}"
    )

    print(
        "GitHub Actions run finished."
    )


if __name__ == "__main__":
    main()
