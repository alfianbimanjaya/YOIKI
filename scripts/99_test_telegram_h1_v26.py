# -*- coding: utf-8 -*-
"""
YOIKI H1 V2.6 — SIMPLE PUBLIC GITHUB ACTIONS RUNNER
PATH-FIXED VERSION

Important:
- Does NOT modify h1_v2_config.py.
- Production model paths are resolved explicitly from repository root.
- Standalone: no dependency on the old realtime Telegram bot.
- Pending signal state survives between GitHub Actions runs.
- Next-bar execution is resolved only when the next H1 OPEN exists.
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
STATE_FILE = RUNTIME_DIR / "telegram_test_state.json"

# ============================================================================
# FROZEN V2.6
# ============================================================================
VERSION = "V2.6"
# TEST ONLY — deliberately relaxed so Telegram path can be exercised.
THRESHOLD = 0.0
SIGNAL_GAP_MAX = 1.0
TEST_MAX_CANDIDATES = 1
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


def default_state():
    return {
        "version": VERSION,
        "last_run": None,
        "last_completed_bar": None,
        "pending": {},
        "resolved": {},
    }


def load_state():
    if not STATE_FILE.exists():
        return default_state()

    try:
        state = json.loads(
            STATE_FILE.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(state, dict):
            return default_state()

        # ------------------------------------------------------------------
        # Backward compatibility with the previous simple runner, which
        # stored everything in state["sent"]. A legacy sent item is treated
        # as already resolved so it cannot be re-alerted.
        # ------------------------------------------------------------------
        legacy_sent = state.pop("sent", {})

        state.setdefault("version", VERSION)
        state.setdefault("last_run", None)
        state.setdefault("last_completed_bar", None)
        state.setdefault("pending", {})
        state.setdefault("resolved", {})

        if legacy_sent and not state["resolved"]:
            for key, stamp in legacy_sent.items():
                state["resolved"][key] = {
                    "status": "LEGACY_SENT",
                    "resolved_at": stamp,
                }

        if not isinstance(state["pending"], dict):
            state["pending"] = {}
        if not isinstance(state["resolved"], dict):
            state["resolved"] = {}

        return state

    except Exception:
        print("WARNING: runtime state invalid; starting with empty state.")
        return default_state()


def save_state(state):
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            state,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tmp.replace(STATE_FILE)


def signal_key(ticker, signal_dt):
    return (
        f"{ticker}|"
        f"{pd.Timestamp(signal_dt).strftime('%Y-%m-%d %H:%M:%S')}"
    )


def prune_state(state, keep_days=7):
    cutoff = now_wib() - timedelta(days=keep_days)

    for bucket_name in ("pending", "resolved"):
        bucket = state.get(bucket_name, {})
        remove = []

        for key, item in bucket.items():
            try:
                stamp = item.get("signal_datetime")
                if not stamp:
                    # Legacy resolved records may only contain resolved_at.
                    stamp = item.get("resolved_at")

                ts = pd.Timestamp(stamp)

                if not pd.isna(ts) and ts.to_pydatetime() < cutoff:
                    remove.append(key)
            except Exception:
                continue

        for key in remove:
            bucket.pop(key, None)


def register_new_pending(state, candidates):
    """
    Register a newly detected signal even when its NEXT H1 bar is not yet
    available. This is the critical cloud-state fix: the signal must survive
    between GitHub Actions runs.
    """
    created = 0

    for _, row in candidates.iterrows():
        ticker = str(row["ticker"]).upper()
        signal_dt = pd.Timestamp(row["datetime"])
        key = signal_key(ticker, signal_dt)

        if key in state["pending"] or key in state["resolved"]:
            continue

        signal_close = float(row["close"])
        score = float(row["pred_calibrated"])
        gap1h = float(row["gap_1h"])

        state["pending"][key] = {
            "ticker": ticker,
            "signal_datetime": signal_dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "signal_close": signal_close,
            "pred_calibrated": score,
            "gap_1h": gap1h,
            "created_at": now_wib().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }

        created += 1

    return created


def next_bar(
    history,
    ticker,
    signal_dt,
):
    """Return the first available market bar strictly after signal_dt."""
    df = history[
        history["ticker"]
        .astype(str)
        .str.upper()
        == str(ticker).upper()
    ].copy()

    if df.empty:
        return None

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

    row = df.iloc[0]

    try:
        entry = float(row["open"])
    except Exception:
        return None

    if not np.isfinite(entry) or entry <= 0:
        return None

    return row


def resolve_pending(state, history):
    """
    Resolve pending signals using the first available market bar after the
    signal bar. The NEXT bar OPEN is the frozen V2.6 execution price.

    Telegram is sent only after the next bar exists. If Telegram fails, the
    pending record is deliberately left intact so the next run can retry.
    """
    resolved = 0

    for key, pending in list(state["pending"].items()):
        ticker = pending["ticker"]
        signal_dt = pd.Timestamp(pending["signal_datetime"])
        signal_close = float(pending["signal_close"])
        score = float(pending["pred_calibrated"])

        if signal_close <= 0:
            # Invalid production data: resolve and suppress further retries.
            state["resolved"][key] = {
                **pending,
                "status": "INVALID_SIGNAL_CLOSE",
                "resolved_at": now_wib().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
            state["pending"].pop(key, None)
            continue

        nxt = next_bar(
            history,
            ticker,
            signal_dt,
        )

        if nxt is None:
            continue

        entry = float(nxt["open"])
        entry_gap = entry / signal_close - 1.0
        next_dt = pd.Timestamp(nxt["datetime"])

        if abs(entry_gap) <= ENTRY_GAP_MAX:
            msg = (
                "🧪🟢 <b>YOIKI H1 V2.6 — TELEGRAM TEST / ENTRY READY</b>\n\n"
                f"<b>{ticker}</b>\n"
                f"Signal: <code>{signal_dt:%Y-%m-%d %H:%M}</code>\n"
                f"P: <b>{score:.4f}</b>\n"
                f"Signal close: {fmt_price(signal_close)}\n"
                f"Next open: <b>{fmt_price(entry)}</b>\n"
                f"Execution gap: <b>{fmt_pct(entry_gap)}</b>\n\n"
                f"TP: {fmt_price(entry * (1 + TP_PCT))} (+4%)\n"
                f"SL: {fmt_price(entry * (1 - SL_PCT))} (-2%)\n"
                f"BE: {fmt_price(entry * (1 + BE_TRIGGER_PCT))} (+2.5%)\n"
                f"Max hold: {MAX_HOLDING_H1} H1\n"
                f"Portfolio cap: {MAX_POSITIONS} positions × "
                f"{ALLOC_PCT * 100:.0f}%\n\n"
                "⚠️ BOT TIDAK MENEMPATKAN ORDER."
            )
            status = "ENTRY_READY"
        else:
            msg = (
                "🧪🔴 <b>YOIKI H1 V2.6 — TELEGRAM TEST / SKIP</b>\n\n"
                f"<b>{ticker}</b>\n"
                f"Signal: <code>{signal_dt:%Y-%m-%d %H:%M}</code>\n"
                f"P: <b>{score:.4f}</b>\n"
                f"Next open: {fmt_price(entry)}\n"
                f"Execution gap: <b>{fmt_pct(entry_gap)}</b>\n\n"
                "❌ Gap > 0.10% — SKIP."
            )
            status = "SKIP_ENTRY_GAP"

        # Only mark resolved after Telegram succeeds.
        send_message(msg)

        state["resolved"][key] = {
            **pending,
            "status": status,
            "next_bar_datetime": next_dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "entry_open": entry,
            "entry_gap": entry_gap,
            "resolved_at": now_wib().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }
        state["pending"].pop(key, None)
        resolved += 1

    return resolved

def main():
    print("=" * 100)
    print("YOIKI H1 V2.6 — TELEGRAM DELIVERY TEST (NOT PRODUCTION)")
    print("=" * 100)

    print(f"REPO ROOT : {ROOT}")
    print(f"MODEL DIR : {MODEL_DIR}")
    print(f"TEST THRESHOLD: {THRESHOLD:.4f} | TEST SIGNAL GAP MAX: {SIGNAL_GAP_MAX:.2f} | MAX TEST CANDIDATES: {TEST_MAX_CANDIDATES}")

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
    ).head(TEST_MAX_CANDIDATES).copy()

    print(
        f"Candidates: {len(candidates)}"
    )

    # ------------------------------------------------------------------
    # DIRECT TELEGRAM DELIVERY TEST
    # ------------------------------------------------------------------
    # This deliberately sends a test message as soon as a candidate is
    # detected. It does NOT wait for the next H1 bar.
    #
    # Purpose:
    #   prove GitHub Secret -> Telegram API -> user chat works.
    #
    # This is TEST ONLY and does not modify the production V2.6 runner.
    if not candidates.empty:
        test_row = candidates.iloc[0]

        test_ticker = str(test_row["ticker"]).upper()
        test_signal_dt = pd.Timestamp(test_row["datetime"])
        test_score = float(test_row["pred_calibrated"])
        test_close = float(test_row["close"])
        test_gap1h = float(test_row["gap_1h"])

        direct_test_msg = (
            "🧪 <b>YOIKI H1 V2.6 — TELEGRAM DELIVERY TEST</b>\n\n"
            f"<b>{test_ticker}</b>\n"
            f"Signal: <code>{test_signal_dt:%Y-%m-%d %H:%M}</code>\n"
            f"P calibrated: <b>{test_score:.4f}</b>\n"
            f"Signal close: <b>{fmt_price(test_close)}</b>\n"
            f"Gap 1H: <b>{fmt_pct(test_gap1h)}</b>\n\n"
            "✅ Candidate detected by GitHub Actions.\n"
            "✅ Telegram API delivery test reached this stage.\n\n"
            "⚠️ TEST ONLY — bukan production signal.\n"
            "⚠️ BOT TIDAK MENEMPATKAN ORDER."
        )

        send_message(direct_test_msg)
        print(
            f"TELEGRAM DIRECT TEST: SENT | "
            f"{test_ticker} | P={test_score:.4f}"
        )
    else:
        print("TELEGRAM DIRECT TEST: NO CANDIDATE, MESSAGE NOT SENT")

    state = load_state()

    # 1) Resolve old pending signals first. This preserves the exact
    #    signal-bar economics and waits for the next H1 OPEN.
    resolved_count = resolve_pending(
        state,
        history,
    )

    # 2) Register newly detected signal bars even when the next H1 bar is
    #    still unavailable. They will be resolved on a later run.
    new_pending = register_new_pending(
        state,
        candidates,
    )

    latest_bar = None
    closed_history = keep_closed(history)
    if not closed_history.empty:
        latest_bar = pd.to_datetime(
            closed_history["datetime"],
            errors="coerce",
        ).max()

    state["version"] = VERSION
    state["last_run"] = now_wib().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    state["last_completed_bar"] = (
        latest_bar.strftime("%Y-%m-%d %H:%M:%S")
        if pd.notna(latest_bar)
        else None
    )

    prune_state(state)
    save_state(state)

    print(
        f"[{state['last_run']}] "
        f"candidates={len(candidates)} "
        f"new_pending={new_pending} "
        f"resolved={resolved_count} "
        f"pending={len(state['pending'])} "
        f"resolved_total={len(state['resolved'])}"
    )

    print(
        "GitHub Actions run finished."
    )


if __name__ == "__main__":
    main()
