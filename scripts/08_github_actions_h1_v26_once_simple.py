# -*- coding: utf-8 -*-
"""YOIKI H1 V2.6 — simple standalone GitHub Actions runner."""
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

from h1_v2_config import (
    CALIBRATOR_FILE, FEATURES_JSON, MODEL_FILE, MODEL_META_JSON,
    MIN_PRICE, TIMEZONE,
)

THRESHOLD = 0.4336
SIGNAL_GAP_MAX = 0.015
ENTRY_GAP_MAX = 0.001
TP_PCT = 0.04
SL_PCT = 0.02
BE_TRIGGER_PCT = 0.025
FETCH_DAYS = 12
MIN_HISTORY_ROWS = 210
KEEP_HISTORY_ROWS = 700
UNIVERSE_FILE = ROOT / "universe_v26.csv"
SEED_HISTORY = ROOT / "v26_initial_history.parquet"
RUNTIME_DIR = ROOT / "cloud_runtime"
HISTORY_FILE = RUNTIME_DIR / "history.parquet"
STATE_FILE = RUNTIME_DIR / "state.json"


def now_wib():
    return datetime.now(ZoneInfo(TIMEZONE)).replace(tzinfo=None)


def fmt_price(x):
    return f"Rp {float(x):,.0f}".replace(",", ".")


def fmt_pct(x):
    return f"{float(x) * 100:+.2f}%"


def send_telegram(text):
    token = os.environ["TELEGRAM_BOT_TOKEN"].strip()
    chat_id = os.environ["TELEGRAM_CHAT_ID"].strip()
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text, "parse_mode": "HTML"}).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=data, method="POST")
    with urllib.request.urlopen(req, timeout=20) as r:
        result = json.loads(r.read().decode())
    if not result.get("ok"):
        raise RuntimeError(result)


def load_universe():
    df = pd.read_csv(UNIVERSE_FILE)
    return (df["ticker"].dropna().astype(str).str.strip().str.upper().drop_duplicates().tolist())


def load_model():
    meta = json.loads(Path(MODEL_META_FILE).read_text(encoding="utf-8"))
    feature_cols = json.loads(Path(FEATURES_JSON).read_text(encoding="utf-8"))
    count = int(meta.get("feature_count", 0))
    if count and count != len(feature_cols):
        raise RuntimeError(f"Feature mismatch metadata={count} json={len(feature_cols)}")
    return lgb.Booster(model_file=str(MODEL_FILE)), joblib.load(CALIBRATOR_FILE), feature_cols


def fetch_recent(tickers):
    now = datetime.now(ZoneInfo(TIMEZONE))
    raw = yf.download(
        tickers=[t if t.endswith(".JK") else f"{t}.JK" for t in tickers],
        start=(now - timedelta(days=FETCH_DAYS)).strftime("%Y-%m-%d"),
        end=(now + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1h", group_by="ticker", auto_adjust=False,
        progress=False, threads=True,
    )
    rows = []
    yft = [t if t.endswith(".JK") else f"{t}.JK" for t in tickers]
    for t in yft:
        try:
            df = raw.copy() if len(yft) == 1 else raw[t].copy() if hasattr(raw, "columns") and t in raw.columns.levels[0] else pd.DataFrame()
            if df.empty:
                continue
            df = df.dropna(subset=["Close"]).reset_index()
            date_col = "Datetime" if "Datetime" in df.columns else "Date"
            dt = pd.to_datetime(df[date_col], errors="coerce")
            if getattr(dt.dt, "tz", None) is not None:
                dt = dt.dt.tz_convert(TIMEZONE).dt.tz_localize(None)
            rows.append(pd.DataFrame({
                "ticker": t.replace(".JK", ""), "datetime": dt,
                "open": df["Open"].to_numpy(), "high": df["High"].to_numpy(),
                "low": df["Low"].to_numpy(), "close": df["Close"].to_numpy(),
                "volume": df["Volume"].to_numpy(),
            }))
        except Exception as exc:
            print(f"FETCH ERROR {t}: {exc}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_history():
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if HISTORY_FILE.exists():
        return pd.read_parquet(HISTORY_FILE)
    if not SEED_HISTORY.exists():
        raise FileNotFoundError(SEED_HISTORY)
    return pd.read_parquet(SEED_HISTORY)


def save_history(df):
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["ticker", "datetime", "close"]).drop_duplicates(["ticker", "datetime"], keep="last").sort_values(["ticker", "datetime"])
    df = df.groupby("ticker", group_keys=False).tail(KEEP_HISTORY_ROWS).reset_index(drop=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(HISTORY_FILE, index=False)


def build_latest(history):
    dt = pd.to_datetime(history["datetime"], errors="coerce")
    closed = history[dt + pd.Timedelta(hours=1) <= pd.Timestamp(now_wib())].copy()
    out = []
    for ticker, g in closed.groupby("ticker", sort=False):
        if len(g) < MIN_HISTORY_ROWS:
            continue
        try:
            out.append(compute_ticker_features(g))
        except Exception as exc:
            print(f"FEATURE ERROR {ticker}: {exc}")
    if not out:
        return pd.DataFrame()
    f = pd.concat(out, ignore_index=True)
    return f.sort_values(["ticker", "datetime"]).groupby("ticker", as_index=False).tail(1).copy()


def next_bar(history, ticker, signal_dt):
    d = history[history["ticker"].astype(str).str.upper() == ticker].copy()
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")
    d = d[d["datetime"] > pd.Timestamp(signal_dt)].sort_values("datetime")
    return None if d.empty else d.iloc[0]


def load_state():
    if not STATE_FILE.exists():
        return {"sent": {}}
    try:
        s = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return s if isinstance(s, dict) and "sent" in s else {"sent": {}}
    except Exception:
        return {"sent": {}}


def save_state(s):
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(s, indent=2), encoding="utf-8")


def main():
    print("=" * 100)
    print("YOIKI H1 V2.6 — SIMPLE PUBLIC GITHUB ACTIONS")
    print("=" * 100)
    universe = load_universe()
    print(f"Universe: {len(universe)}")
    model, calibrator, feature_cols = load_model()
    recent = fetch_recent(universe)
    if recent.empty:
        raise RuntimeError("No Yahoo data returned. DATA ERROR, not NO SIGNAL.")
    got = set(recent["ticker"].astype(str).str.upper().unique())
    coverage = 100 * len(got) / max(len(universe), 1)
    print(f"DATA HEALTH: {len(got)}/{len(universe)} ({coverage:.1f}%)")
    if coverage < 90:
        raise RuntimeError(f"DATA HEALTH too low: {coverage:.1f}%")
    history = pd.concat([load_history(), recent], ignore_index=True)
    save_history(history)
    latest = build_latest(history)
    if latest.empty:
        print("NO FEATURES — history belum cukup.")
        return
    missing = [c for c in feature_cols if c not in latest.columns]
    if missing:
        raise RuntimeError("Missing production feature(s): " + ", ".join(missing))
    latest["pred_calibrated"] = calibrator.predict(model.predict(latest[feature_cols]))
    candidates = latest[(latest["pred_calibrated"] >= THRESHOLD) & (latest["close"] >= MIN_PRICE) & (pd.to_numeric(latest["gap_1h"], errors="coerce") <= SIGNAL_GAP_MAX)].sort_values("pred_calibrated", ascending=False)
    state = load_state()
    new_count = 0
    for _, row in candidates.iterrows():
        ticker = str(row["ticker"]).upper()
        signal_dt = pd.Timestamp(row["datetime"])
        key = f"{ticker}|{signal_dt:%Y-%m-%d %H:%M:%S}"
        if key in state["sent"]:
            continue
        nxt = next_bar(history, ticker, signal_dt)
        if nxt is None:
            continue
        close = float(row["close"]); entry = float(nxt["open"]); entry_gap = entry / close - 1.0; score = float(row["pred_calibrated"])
        if abs(entry_gap) <= ENTRY_GAP_MAX:
            msg = (f"🟢 <b>YOIKI H1 V2.6 — ENTRY READY</b>\n\n<b>{ticker}</b>\nSignal: <code>{signal_dt:%Y-%m-%d %H:%M}</code>\nP: <b>{score:.4f}</b>\nSignal close: {fmt_price(close)}\nNext open: <b>{fmt_price(entry)}</b>\nExecution gap: <b>{fmt_pct(entry_gap)}</b>\n\nTP: {fmt_price(entry * (1 + TP_PCT))} (+4%)\nSL: {fmt_price(entry * (1 - SL_PCT))} (-2%)\nBE: {fmt_price(entry * (1 + BE_TRIGGER_PCT))} (+2.5%)\n⚠️ BOT TIDAK MENEMPATKAN ORDER.")
        else:
            msg = (f"🔴 <b>YOIKI H1 V2.6 — SKIP</b>\n\n<b>{ticker}</b>\nSignal: <code>{signal_dt:%Y-%m-%d %H:%M}</code>\nP: <b>{score:.4f}</b>\nNext open: {fmt_price(entry)}\nExecution gap: <b>{fmt_pct(entry_gap)}</b>\n\n❌ Gap > 0.10% — SKIP.")
        send_message(msg)
        state["sent"][key] = now_wib().strftime("%Y-%m-%d %H:%M:%S")
        new_count += 1
    cutoff = now_wib() - timedelta(days=7)
    for key, stamp in list(state["sent"].items()):
        try:
            if pd.Timestamp(stamp).to_pydatetime() < cutoff:
                del state["sent"][key]
        except Exception:
            pass
    save_state(state)
    print(f"[{now_wib():%Y-%m-%d %H:%M:%S}] candidates={len(candidates)} new_signal={new_count}")
    print("GitHub Actions run finished.")


if __name__ == "__main__":
    main()
