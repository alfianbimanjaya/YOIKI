# -*- coding: utf-8 -*-
"""
YOIKI H1 V2.6 — PUBLIC GITHUB ACTIONS RUNNER
INTRADAY OPEN RESOLVER — BATCH / NARROW WINDOW VERSION

Important:
- H1 is used only for signal generation.
- The actual next-H1 OPEN is resolved from 1m first, 5m fallback.
- Intraday data is fetched only when the expected next-H1 OPEN is due.
- Intraday requests are BATCHED across pending tickers.
- We request a narrow time window around the expected OPEN instead of
  downloading a full day for every ticker.
- No per-ticker 50-second retry loop.
- A short retry is allowed only for the current due batch, so delayed
  publication of the first 1m bar can still be caught by the same run.
- Does NOT place orders.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
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

# ============================================================================
# INTRADAY RESOLVER
# ============================================================================
# The resolver intentionally does NOT download period="1d" per ticker.
# It queries a narrow window around the expected H1 OPEN.
INTRADAY_LOOKBACK_MINUTES = 2
INTRADAY_FUTURE_BUFFER_MINUTES = 2
INTRADAY_RETRY_SECONDS = 15
INTRADAY_MAX_WAIT_SECONDS = 45
EXECUTION_ALERT_GRACE_MINUTES = 2


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
        headers={"User-Agent": "YOIKI-H1-V2.6-GHA/1.1"},
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
        raise FileNotFoundError(f"Universe tidak ditemukan: {UNIVERSE_FILE}")

    df = pd.read_csv(UNIVERSE_FILE)

    if "ticker" not in df.columns:
        raise KeyError("universe_v26.csv wajib mempunyai kolom ticker.")

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
            raise FileNotFoundError(f"Production artifact tidak ditemukan: {p}")

    meta = json.loads(MODEL_META_JSON.read_text(encoding="utf-8"))
    feature_cols = json.loads(FEATURES_JSON.read_text(encoding="utf-8"))
    count = int(meta.get("feature_count", 0))

    if count and count != len(feature_cols):
        raise RuntimeError(
            f"Feature mismatch metadata={count} json={len(feature_cols)}"
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

            date_col = "Datetime" if "Datetime" in df.columns else "Date"
            dt = pd.to_datetime(df[date_col], errors="coerce")

            if getattr(dt.dt, "tz", None) is not None:
                dt = dt.dt.tz_convert(TIMEZONE).dt.tz_localize(None)

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
    history["datetime"] = pd.to_datetime(history["datetime"], errors="coerce")

    history = (
        history.dropna(subset=["ticker", "datetime", "close"])
        .drop_duplicates(["ticker", "datetime"], keep="last")
        .sort_values(["ticker", "datetime"])
    )

    history = (
        history.groupby("ticker", group_keys=False)
        .tail(KEEP_HISTORY_ROWS)
        .reset_index(drop=True)
    )

    history.to_parquet(HISTORY_FILE, index=False)


def keep_closed(df):
    now = pd.Timestamp(now_wib())
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    return df[dt + pd.Timedelta(hours=1) <= now].copy()


def latest_features(history):
    closed = keep_closed(history)
    blocks = []

    for ticker, group in closed.groupby("ticker", sort=False):
        if len(group) < MIN_HISTORY_ROWS:
            continue

        try:
            blocks.append(compute_ticker_features(group))
        except Exception as exc:
            print(f"FEATURE ERROR {ticker}: {exc}")

    if not blocks:
        return pd.DataFrame()

    features = pd.concat(blocks, ignore_index=True)

    return (
        features.sort_values(["ticker", "datetime"])
        .groupby("ticker", as_index=False)
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
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))

        if not isinstance(state, dict):
            return default_state()

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
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(STATE_FILE)


def signal_key(ticker, signal_dt):
    return f"{ticker}|{pd.Timestamp(signal_dt).strftime('%Y-%m-%d %H:%M:%S')}"


def prune_state(state, keep_days=7):
    cutoff = now_wib() - timedelta(days=keep_days)

    for bucket_name in ("pending", "resolved"):
        bucket = state.get(bucket_name, {})
        remove = []

        for key, item in bucket.items():
            try:
                stamp = item.get("signal_datetime") or item.get("resolved_at")
                if not stamp:
                    continue
                ts = pd.Timestamp(stamp)
                if not pd.isna(ts) and ts.to_pydatetime() < cutoff:
                    remove.append(key)
            except Exception:
                continue

        for key in remove:
            bucket.pop(key, None)


def infer_next_h1_clock(history, ticker, signal_dt):
    """
    Infer the next H1 bar clock from historical H1 sequence.
    """
    ticker = str(ticker).upper()
    signal_dt = pd.Timestamp(signal_dt)
    df = history[history["ticker"].astype(str).str.upper() == ticker].copy()

    if df.empty:
        if signal_dt.hour >= 16:
            return (9, 0)
        return ((signal_dt.hour + 1) % 24, signal_dt.minute)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    target_clock = (signal_dt.hour, signal_dt.minute)
    matches = df[
        (df["datetime"].dt.hour == target_clock[0])
        & (df["datetime"].dt.minute == target_clock[1])
    ]

    next_clocks = []

    for _, row in matches.tail(80).iterrows():
        later = df[df["datetime"] > row["datetime"]]
        if later.empty:
            continue
        nxt = later.iloc[0]["datetime"]
        next_clocks.append((int(nxt.hour), int(nxt.minute)))

    if next_clocks:
        counts = {}
        for clock in next_clocks:
            counts[clock] = counts.get(clock, 0) + 1
        return max(counts.items(), key=lambda x: (x[1], x[0]))[0]

    if signal_dt.hour >= 16:
        return (9, 0)

    return ((signal_dt.hour + 1) % 24, signal_dt.minute)


def infer_next_h1_datetime(signal_dt, expected_clock):
    """Return the next weekday occurrence of expected_clock after signal_dt."""
    signal_dt = pd.Timestamp(signal_dt)
    candidate = signal_dt.normalize() + pd.Timedelta(
        hours=int(expected_clock[0]),
        minutes=int(expected_clock[1]),
    )

    # Find the first weekday candidate strictly after the signal bar.
    while candidate <= signal_dt or candidate.weekday() >= 5:
        candidate = candidate + pd.Timedelta(days=1)
        candidate = candidate.normalize() + pd.Timedelta(
            hours=int(expected_clock[0]),
            minutes=int(expected_clock[1]),
        )

    return candidate


def normalize_intraday_download(raw, symbols):
    """Normalize yfinance intraday download to symbol -> OHLCV frames."""
    if raw is None or raw.empty:
        return {}

    result = {}

    # Single symbol can arrive as a normal DataFrame or MultiIndex.
    if len(symbols) == 1:
        symbol = symbols[0]
        df = raw.copy()

        if isinstance(df.columns, pd.MultiIndex):
            field_map = {}
            for col in df.columns:
                parts = [str(x) for x in col if x is not None]
                field = next(
                    (x for x in parts if x in {"Open", "High", "Low", "Close", "Volume"}),
                    None,
                )
                if field:
                    field_map[field] = col
            if field_map:
                df = df[[field_map[x] for x in field_map]]
                df.columns = list(field_map.keys())
        else:
            df.columns = [str(c) for c in df.columns]

        required = {"Open", "High", "Low", "Close", "Volume"}
        if required.issubset(set(df.columns)):
            result[symbol] = df.copy()
        return result

    # Multi-symbol: prefer symbol as outer level, but also handle the reverse.
    if not isinstance(raw.columns, pd.MultiIndex):
        return {}

    for symbol in symbols:
        try:
            if symbol in raw.columns.get_level_values(0):
                df = raw[symbol].copy()
                df.columns = [str(c) for c in df.columns]
                if {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
                    result[symbol] = df
                    continue

            if symbol in raw.columns.get_level_values(1):
                df = raw.xs(symbol, axis=1, level=1).copy()
                df.columns = [str(c) for c in df.columns]
                if {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
                    result[symbol] = df
                    continue
        except Exception:
            continue

    return result


def _extract_window_for_symbol(df, start_local, end_local, signal_dt, expected_clock):
    """Return matching first intraday bar at expected clock within narrow window."""
    if df is None or df.empty:
        return None

    df = df.dropna(subset=["Open"]).reset_index()
    if df.empty:
        return None

    date_col = "Datetime" if "Datetime" in df.columns else "Date"
    dt = pd.to_datetime(df[date_col], errors="coerce")

    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert(TIMEZONE).dt.tz_localize(None)

    df["local_dt"] = dt
    df["open_num"] = pd.to_numeric(df["Open"], errors="coerce")

    mask = (
        df["local_dt"].notna()
        & df["open_num"].notna()
        & (df["local_dt"] >= pd.Timestamp(start_local))
        & (df["local_dt"] <= pd.Timestamp(end_local))
        & (df["local_dt"] > pd.Timestamp(signal_dt))
        & (df["local_dt"].dt.hour == int(expected_clock[0]))
        & (df["local_dt"].dt.minute == int(expected_clock[1]))
    )

    candidates = df.loc[mask].sort_values("local_dt")
    if candidates.empty:
        return None

    r = candidates.iloc[0]
    return {
        "datetime": pd.Timestamp(r["local_dt"]),
        "open": float(r["open_num"]),
    }


def fetch_intraday_open_batch(pending_items):
    """
    Batch resolver for due pending signals.

    Strategy:
      - 1m is primary.
      - 5m is queried ONLY for symbols still unresolved after 1m.
      - Retries operate at batch level.
      - The intraday window is narrow around the expected H1 OPEN.
    """
    outputs = {}
    if not pending_items:
        return outputs

    now = pd.Timestamp(now_wib())

    groups = defaultdict(list)
    for key, pending in pending_items:
        next_dt = pd.Timestamp(pending["next_open_datetime"])
        if now + pd.Timedelta(minutes=1) < next_dt:
            continue
        groups[next_dt].append((key, pending))

    if not groups:
        return outputs

    for target_dt, group in sorted(groups.items(), key=lambda x: x[0]):
        expected_clock = (target_dt.hour, target_dt.minute)

        start_local = target_dt - pd.Timedelta(minutes=INTRADAY_LOOKBACK_MINUTES)
        end_local = target_dt + pd.Timedelta(minutes=6)

        deadline = time.time() + INTRADAY_MAX_WAIT_SECONDS

        while True:
            unresolved = [
                (key, pending)
                for key, pending in group
                if key not in outputs
            ]

            if not unresolved:
                break

            # 1m primary for all currently-unresolved symbols.
            # 5m fallback only for symbols that remain unresolved after 1m.
            for interval in ("1m", "5m"):
                unresolved = [
                    (key, pending)
                    for key, pending in group
                    if key not in outputs
                ]

                if not unresolved:
                    break

                symbols = [
                    (
                        p["ticker"]
                        if p["ticker"].endswith(".JK")
                        else f'{p["ticker"]}.JK'
                    )
                    for _, p in unresolved
                ]
                symbols = list(dict.fromkeys(symbols))

                try:
                    # yfinance in this environment accepts Python datetime
                    # objects more reliably than intraday datetime strings.
                    yahoo_start = (
                        pd.Timestamp(start_local)
                        .tz_localize(TIMEZONE)
                        .tz_convert("UTC")
                        - pd.Timedelta(minutes=1)
                    ).to_pydatetime()

                    yahoo_end = (
                        pd.Timestamp(end_local)
                        .tz_localize(TIMEZONE)
                        .tz_convert("UTC")
                        + pd.Timedelta(minutes=1)
                    ).to_pydatetime()

                    raw = yf.download(
                        tickers=symbols,
                        start=yahoo_start,
                        end=yahoo_end,
                        interval=interval,
                        group_by="ticker",
                        auto_adjust=False,
                        progress=False,
                        threads=True,
                    )

                    frames = normalize_intraday_download(raw, symbols)

                    print(
                        f"INTRADAY BATCH {interval}: "
                        f"tickers={len(symbols)} target={target_dt:%H:%M} "
                        f"window={start_local:%H:%M}-{end_local:%H:%M} "
                        f"frames={len(frames)}"
                    )

                    for key, pending in unresolved:
                        symbol = pending["ticker"]
                        yf_symbol = (
                            symbol
                            if symbol.endswith(".JK")
                            else f"{symbol}.JK"
                        )

                        frame = frames.get(yf_symbol)
                        if frame is None:
                            continue

                        match = _extract_window_for_symbol(
                            frame,
                            start_local,
                            end_local,
                            pending["signal_datetime"],
                            expected_clock,
                        )

                        if match is not None:
                            outputs[key] = {
                                **match,
                                "source": interval,
                            }

                except Exception as exc:
                    print(
                        f"INTRADAY {interval} BATCH ERROR: {exc}"
                    )

            if len(outputs) >= len(group):
                break

            if time.time() >= deadline:
                break

            if (
                pd.Timestamp(now_wib()) - target_dt
                > pd.Timedelta(minutes=EXECUTION_ALERT_GRACE_MINUTES)
            ):
                break

            time.sleep(INTRADAY_RETRY_SECONDS)

    return outputs


def register_new_pending(state, candidates, history):
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

        next_clock = infer_next_h1_clock(history, ticker, signal_dt)
        next_dt = infer_next_h1_datetime(signal_dt, next_clock)

        state["pending"][key] = {
            "ticker": ticker,
            "signal_datetime": signal_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "signal_close": signal_close,
            "pred_calibrated": score,
            "gap_1h": gap1h,
            "next_open_datetime": next_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "next_open_hour": int(next_clock[0]),
            "next_open_minute": int(next_clock[1]),
            "created_at": now_wib().strftime("%Y-%m-%d %H:%M:%S"),
        }

        print(
            f"PENDING {ticker}: signal={signal_dt:%Y-%m-%d %H:%M} "
            f"next_h1_open={next_dt:%Y-%m-%d %H:%M}"
        )
        created += 1

    return created


def resolve_pending(state):
    resolved = 0
    now = pd.Timestamp(now_wib())

    # Only send due pending items into the intraday resolver.
    due = []
    for key, pending in state["pending"].items():
        next_dt = pd.Timestamp(pending.get("next_open_datetime"))
        # Do not fetch data for future OPENs.
        if now + pd.Timedelta(minutes=1) >= next_dt:
            due.append((key, pending))

    if not due:
        return 0

    print(f"INTRADAY DUE PENDING: {len(due)}")
    resolved_map = fetch_intraday_open_batch(due)

    for key, pending in due:
        intraday = resolved_map.get(key)
        if intraday is None:
            continue

        ticker = pending["ticker"]
        signal_dt = pd.Timestamp(pending["signal_datetime"])
        signal_close = float(pending["signal_close"])
        score = float(pending["pred_calibrated"])
        entry = float(intraday["open"])
        next_dt = pd.Timestamp(intraday["datetime"])
        source = intraday["source"]

        entry_gap = entry / signal_close - 1.0
        age_minutes = (pd.Timestamp(now_wib()) - next_dt).total_seconds() / 60.0

        if abs(entry_gap) > ENTRY_GAP_MAX:
            msg = (
                "🔴 <b>YOIKI H1 V2.6 — SKIP</b>\n\n"
                f"<b>{ticker}</b>\n"
                f"Signal: <code>{signal_dt:%Y-%m-%d %H:%M}</code>\n"
                f"Next H1 open: <code>{next_dt:%Y-%m-%d %H:%M}</code>\n"
                f"Actual open ({source}): <b>{fmt_price(entry)}</b>\n"
                f"Execution gap: <b>{fmt_pct(entry_gap)}</b>\n\n"
                "❌ Gap > 0.10% — SKIP.\n"
                "Jangan mengejar harga."
            )
            status = "SKIP_ENTRY_GAP"

        elif age_minutes <= EXECUTION_ALERT_GRACE_MINUTES:
            msg = (
                "🟢 <b>YOIKI H1 V2.6 — ENTER NOW</b>\n\n"
                f"<b>{ticker}</b>\n"
                f"Signal: <code>{signal_dt:%Y-%m-%d %H:%M}</code>\n"
                f"P: <b>{score:.4f}</b>\n"
                f"Actual H1 open: <b>{fmt_price(entry)}</b>\n"
                f"Open time: <code>{next_dt:%H:%M}</code>\n"
                f"Execution gap: <b>{fmt_pct(entry_gap)}</b>\n"
                f"Source: <b>{source}</b>\n\n"
                f"TP: {fmt_price(entry * (1 + TP_PCT))} (+4%)\n"
                f"SL: {fmt_price(entry * (1 - SL_PCT))} (-2%)\n"
                f"BE: {fmt_price(entry * (1 + BE_TRIGGER_PCT))} (+2.5%)\n"
                f"Max hold: {MAX_HOLDING_H1} H1\n"
                f"Portfolio cap: {MAX_POSITIONS} × {ALLOC_PCT * 100:.0f}%\n\n"
                "⚠️ BOT TIDAK MENEMPATKAN ORDER."
            )
            status = "ENTRY_READY_INTRADAY"

        else:
            msg = (
                "⚪ <b>YOIKI H1 V2.6 — MISSED OPEN</b>\n\n"
                f"<b>{ticker}</b>\n"
                f"Signal: <code>{signal_dt:%Y-%m-%d %H:%M}</code>\n"
                f"Actual H1 open: <b>{fmt_price(entry)}</b>\n"
                f"Open time: <code>{next_dt:%H:%M}</code>\n"
                f"Execution gap: <b>{fmt_pct(entry_gap)}</b>\n"
                f"Delay detected: <b>{age_minutes:.1f} min</b>\n\n"
                "⛔ Entry window sudah lewat.\n"
                "Jangan mengejar harga."
            )
            status = "MISSED_OPEN"

        send_message(msg)

        state["resolved"][key] = {
            **pending,
            "status": status,
            "next_bar_datetime": next_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "entry_open": entry,
            "entry_gap": entry_gap,
            "intraday_source": source,
            "resolved_at": now_wib().strftime("%Y-%m-%d %H:%M:%S"),
        }

        state["pending"].pop(key, None)
        resolved += 1

    return resolved


def main():
    print("=" * 100)
    print("YOIKI H1 V2.6 — INTRADAY OPEN EXECUTION RESOLVER")
    print("=" * 100)

    print(f"REPO ROOT : {ROOT}")
    print(f"MODEL DIR : {MODEL_DIR}")

    universe = load_universe()
    print(f"Universe: {len(universe)}")

    model, calibrator, feature_cols = load_model()
    recent = fetch_recent(universe)

    if recent.empty:
        raise RuntimeError("No Yahoo data returned. DATA ERROR, not NO SIGNAL.")

    got = set(
        recent["ticker"].dropna().astype(str).str.upper().unique()
    )
    coverage = 100.0 * len(got) / max(len(universe), 1)
    print(f"DATA HEALTH: {len(got)}/{len(universe)} ({coverage:.1f}%)")

    if coverage < 90.0:
        raise RuntimeError(f"DATA HEALTH too low: {coverage:.1f}%")

    history = pd.concat([load_history(), recent], ignore_index=True)
    save_history(history)

    latest = latest_features(history)
    if latest.empty:
        print("NO FEATURES — history belum cukup.")
        return

    missing = [c for c in feature_cols if c not in latest.columns]
    if missing:
        raise RuntimeError("Missing production feature(s): " + ", ".join(missing))

    raw_score = model.predict(latest[feature_cols])
    latest["pred_calibrated"] = np.asarray(
        calibrator.predict(raw_score), dtype=float
    )

    candidates = latest[
        (latest["pred_calibrated"] >= THRESHOLD)
        & (latest["close"] >= MIN_PRICE)
        & (pd.to_numeric(latest["gap_1h"], errors="coerce") <= SIGNAL_GAP_MAX)
    ].copy()

    candidates = candidates.sort_values("pred_calibrated", ascending=False)
    print(f"Candidates: {len(candidates)}")

    state = load_state()

    new_pending = register_new_pending(state, candidates, history)
    resolved_count = resolve_pending(state)

    closed_history = keep_closed(history)
    latest_bar = None
    if not closed_history.empty:
        latest_bar = pd.to_datetime(
            closed_history["datetime"], errors="coerce"
        ).max()

    state["version"] = VERSION
    state["last_run"] = now_wib().strftime("%Y-%m-%d %H:%M:%S")
    state["last_completed_bar"] = (
        latest_bar.strftime("%Y-%m-%d %H:%M:%S")
        if pd.notna(latest_bar)
        else None
    )

    prune_state(state)
    save_state(state)

    print(
        f"[{state['last_run']}] candidates={len(candidates)} "
        f"new_pending={new_pending} resolved={resolved_count} "
        f"pending={len(state['pending'])} resolved_total={len(state['resolved'])}"
    )
    print("GitHub Actions run finished.")


if __name__ == "__main__":
    main()
