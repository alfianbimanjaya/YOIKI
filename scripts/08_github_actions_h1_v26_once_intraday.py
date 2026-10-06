# -*- coding: utf-8 -*-
"""
YOIKI H1 V2.6 — SIMPLE PUBLIC GITHUB ACTIONS RUNNER
PATH-FIXED VERSION

Important:
- Does NOT modify h1_v2_config.py.
- Production model paths are resolved explicitly from repository root.
- Standalone: no dependency on the old realtime Telegram bot.
- Pending signal state survives between GitHub Actions runs.
- H1 is used only for signal generation; next-H1 OPEN is resolved from 1m/5m intraday data.
"""

from __future__ import annotations

import json
import os
import sys
import time
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

# Execution-price resolver:
# - H1 remains the signal timeframe.
# - 1m is the primary source for the next H1 OPEN.
# - 5m is the fallback.
# - This does NOT change the frozen entry rule; it only obtains the
#   actual next-session/open price faster than waiting for the next H1 bar.
INTRADAY_RETRY_SECONDS = 10
INTRADAY_MAX_WAIT_SECONDS = 50
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



def infer_next_h1_clock(history, ticker, signal_dt):
    """
    Infer the next H1 bar clock time from the historical H1 sequence.

    We do NOT use the future/next H1 bar for live execution anymore.
    Historical H1 structure is only used to determine the expected clock
    (e.g. 10:00 -> 11:00, 16:00 -> 09:00 next trading day).

    The actual execution price is always taken from current 1m/5m data.
    """
    ticker = str(ticker).upper()
    df = history[
        history["ticker"].astype(str).str.upper() == ticker
    ].copy()

    if df.empty:
        # Conservative fallback for normal IDX H1 structure.
        if signal_dt.hour >= 16:
            return (9, 0)
        return ((signal_dt.hour + 1) % 24, signal_dt.minute)

    df["datetime"] = pd.to_datetime(
        df["datetime"], errors="coerce"
    )
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
        next_clocks.append(
            (int(nxt.hour), int(nxt.minute))
        )

    if next_clocks:
        counts = {}
        for clock in next_clocks:
            counts[clock] = counts.get(clock, 0) + 1
        return max(
            counts.items(),
            key=lambda x: (x[1], x[0])
        )[0]

    if signal_dt.hour >= 16:
        return (9, 0)

    return ((signal_dt.hour + 1) % 24, signal_dt.minute)


def fetch_intraday_open(
    ticker,
    signal_dt,
    expected_clock,
):
    """
    Get the actual next H1 OPEN from intraday data.

    Priority:
      1m -> 5m

    We look for the first intraday bar strictly after signal_dt whose clock
    matches the expected next H1 start.

    Returns:
      {
        "datetime": pandas.Timestamp,
        "open": float,
        "source": "1m" or "5m"
      }
    """
    now = now_wib()
    deadline = time.time() + INTRADAY_MAX_WAIT_SECONDS

    while True:
        for interval in ("1m", "5m"):
            try:
                symbol = (
                    ticker if ticker.endswith(".JK")
                    else f"{ticker}.JK"
                )

                df = yf.download(
                    symbol,
                    period="1d",
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )

                if df is None or df.empty:
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(-1)

                df = df.dropna(subset=["Open"]).reset_index()

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

                df["local_dt"] = dt
                df["open_num"] = pd.to_numeric(
                    df["Open"],
                    errors="coerce",
                )

                mask = (
                    df["local_dt"].notna()
                    & df["open_num"].notna()
                    & (df["local_dt"] > pd.Timestamp(signal_dt))
                    & (df["local_dt"].dt.hour == expected_clock[0])
                    & (df["local_dt"].dt.minute == expected_clock[1])
                )

                candidates = (
                    df.loc[mask]
                    .sort_values("local_dt")
                )

                if not candidates.empty:
                    r = candidates.iloc[0]
                    return {
                        "datetime": pd.Timestamp(r["local_dt"]),
                        "open": float(r["open_num"]),
                        "source": interval,
                    }

            except Exception as exc:
                print(
                    f"INTRADAY {interval} FETCH ERROR "
                    f"{ticker}: {exc}"
                )

        if time.time() >= deadline:
            break

        time.sleep(INTRADAY_RETRY_SECONDS)

    return None


def register_new_pending(state, candidates, history):
    """
    Register a new signal and record the expected NEXT H1 clock.

    The entry price is intentionally NOT read from H1.
    It will be resolved from 1m/5m when the next H1 open exists.
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

        next_clock = infer_next_h1_clock(
            history,
            ticker,
            signal_dt,
        )

        state["pending"][key] = {
            "ticker": ticker,
            "signal_datetime": signal_dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "signal_close": signal_close,
            "pred_calibrated": score,
            "gap_1h": gap1h,
            "next_open_hour": int(next_clock[0]),
            "next_open_minute": int(next_clock[1]),
            "created_at": now_wib().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }

        print(
            f"PENDING {ticker}: "
            f"signal={signal_dt:%Y-%m-%d %H:%M} "
            f"next_h1_clock={next_clock[0]:02d}:{next_clock[1]:02d}"
        )

        created += 1

    return created


def resolve_pending(state):
    """
    Resolve pending signals using 1m/5m intraday data.

    Important operational rule:
    - We never claim an H1 OPEN simply because the next H1 candle exists.
    - We obtain the actual OPEN price from 1m first, 5m fallback.
    - If the actual OPEN is already more than EXECUTION_ALERT_GRACE_MINUTES
      in the past, we send MISSED OPEN instead of telling the user to chase.
    """
    resolved = 0

    for key, pending in list(state["pending"].items()):
        ticker = pending["ticker"]
        signal_dt = pd.Timestamp(pending["signal_datetime"])
        signal_close = float(pending["signal_close"])
        score = float(pending["pred_calibrated"])

        next_clock = (
            int(pending["next_open_hour"]),
            int(pending["next_open_minute"]),
        )

        intraday = fetch_intraday_open(
            ticker,
            signal_dt,
            next_clock,
        )

        if intraday is None:
            continue

        entry = float(intraday["open"])
        next_dt = pd.Timestamp(intraday["datetime"])
        source = intraday["source"]

        entry_gap = entry / signal_close - 1.0
        age_minutes = (
            pd.Timestamp(now_wib()) - next_dt
        ).total_seconds() / 60.0

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
                f"Portfolio cap: {MAX_POSITIONS} × "
                f"{ALLOC_PCT * 100:.0f}%\n\n"
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
            "next_bar_datetime": next_dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "entry_open": entry,
            "entry_gap": entry_gap,
            "intraday_source": source,
            "resolved_at": now_wib().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
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

    # 1) Register newly detected signal bars.
    #    The expected next H1 clock is stored, but the actual execution price
    #    will be resolved from 1m/5m intraday data.
    new_pending = register_new_pending(
        state,
        candidates,
        history,
    )

    # 2) Resolve pending signals using actual intraday OPEN data.
    resolved_count = resolve_pending(
        state,
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
