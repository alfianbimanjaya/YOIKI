import numpy as np
import pandas as pd

from h1_v2_config import FEATURES_FILE, PROCESSED_DIR, RAW_FILE


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    # Normalized least-squares slope; uses past/current values only.
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def _calc(values):
        y = np.asarray(values, dtype=float)
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / (denom + 1e-12)
        scale = abs(y_mean) + 1e-9
        return slope / scale

    return series.rolling(window).apply(_calc, raw=True)


def compute_ticker_features(group: pd.DataFrame) -> pd.DataFrame:
    df = group.copy().sort_values("datetime").reset_index(drop=True)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    volume = df["volume"].astype(float)
    prev_close = close.shift(1)

    # -------------------------------------------------------------------------
    # 1. Returns / price action
    # -------------------------------------------------------------------------
    for lag in [1, 2, 3, 5, 8, 13, 21]:
        df[f"return_{lag}h"] = close.pct_change(lag)
    df["gap_1h"] = open_ / (prev_close + 1e-9) - 1

    # -------------------------------------------------------------------------
    # 2. Candle microstructure
    # -------------------------------------------------------------------------
    rng = (high - low).replace(0, np.nan)
    body = (close - open_)
    df["candle_range_pct"] = rng / (close + 1e-9)
    df["body_ratio"] = body.abs() / rng
    df["body_signed_ratio"] = body / rng
    df["upper_shadow_ratio"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / rng
    df["lower_shadow_ratio"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / rng
    df["close_location_value"] = (close - low) / (rng + 1e-9)

    # -------------------------------------------------------------------------
    # 3. Reversal / drawdown structure
    # -------------------------------------------------------------------------
    for w in [20, 50, 100]:
        rolling_high = high.rolling(w).max()
        rolling_low = low.rolling(w).min()
        df[f"dist_from_high_{w}h"] = close / (rolling_high + 1e-9) - 1
        df[f"dist_from_low_{w}h"] = close / (rolling_low + 1e-9) - 1
        df[f"range_position_{w}h"] = (close - rolling_low) / (rolling_high - rolling_low + 1e-9)

    # -------------------------------------------------------------------------
    # 4. Bollinger / Donchian
    # -------------------------------------------------------------------------
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_percent_b"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)
    df["bb_width"] = (bb_upper - bb_lower) / (bb_mid.abs() + 1e-9)

    for w in [20, 50]:
        dc_high = high.rolling(w).max()
        dc_low = low.rolling(w).min()
        df[f"donchian_pos_{w}"] = (close - dc_low) / (dc_high - dc_low + 1e-9)
        df[f"donchian_width_{w}"] = (dc_high - dc_low) / (close + 1e-9)

    # -------------------------------------------------------------------------
    # 5. EMA / trend distance
    # -------------------------------------------------------------------------
    emas = {}
    for w in [5, 10, 20, 50, 100, 200]:
        ema = close.ewm(span=w, adjust=False).mean()
        emas[w] = ema
        df[f"ema_{w}"] = ema
        df[f"dist_ema_{w}"] = close / (ema + 1e-9) - 1

    for fast, slow in [(5, 20), (10, 50), (20, 50), (20, 200), (50, 200)]:
        df[f"ema_spread_{fast}_{slow}"] = emas[fast] / (emas[slow] + 1e-9) - 1

    # -------------------------------------------------------------------------
    # 6. Volatility / ATR regime
    # -------------------------------------------------------------------------
    for p in [7, 14, 28]:
        a = atr(df, p)
        df[f"atr_{p}"] = a
        df[f"natr_{p}"] = a / (close + 1e-9)
    df["volatility_std_20"] = df["return_1h"].rolling(20).std()
    df["volatility_std_5"] = df["return_1h"].rolling(5).std()
    df["volatility_ratio_5_20"] = df["volatility_std_5"] / (df["volatility_std_20"] + 1e-9)

    # -------------------------------------------------------------------------
    # 7. Momentum / oscillators
    # -------------------------------------------------------------------------
    for p in [7, 14, 28]:
        df[f"rsi_{p}"] = rsi(close, p)

    for lag in [3, 6, 12, 24]:
        df[f"roc_{lag}"] = close.pct_change(lag)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = macd - signal
    df["macd_hist_pct_price"] = (macd - signal) / (close + 1e-9)

    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_k = 100 * (close - low14) / (high14 - low14 + 1e-9)
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_k.rolling(3).mean()
    df["williams_r"] = -100 * (high14 - close) / (high14 - low14 + 1e-9)

    # -------------------------------------------------------------------------
    # 8. Volume / participation
    # -------------------------------------------------------------------------
    for p in [5, 20, 50]:
        vol_ma = volume.rolling(p).mean()
        df[f"vol_ratio_{p}"] = volume / (vol_ma + 1e-9)

    df["turnover_est"] = close * volume
    df["log_turnover"] = np.log1p(df["turnover_est"].clip(lower=0))

    obv_step = np.where(close > prev_close, volume, np.where(close < prev_close, -volume, 0.0))
    obv = pd.Series(obv_step, index=df.index).cumsum()
    df["obv"] = obv
    for w in [5, 20, 50]:
        df[f"obv_slope_{w}"] = rolling_slope(obv, w)

    df["bullish_rejection_score"] = df["lower_shadow_ratio"] * df["vol_ratio_20"]
    df["bearish_rejection_score"] = df["upper_shadow_ratio"] * df["vol_ratio_20"]

    # -------------------------------------------------------------------------
    # 9. Trend / momentum slope
    # -------------------------------------------------------------------------
    for w in [5, 10, 20, 50]:
        df[f"price_slope_{w}"] = rolling_slope(close, w)
    df["momentum_accel"] = df["return_3h"] - df["return_8h"]
    df["reversal_score_raw"] = (
        (-df["dist_from_high_20h"]).clip(lower=0)
        * (1 - (df["bb_percent_b"] - 0.5).abs().clip(lower=0, upper=0.5))
        * (1 + df["bullish_rejection_score"].clip(lower=0))
    )

    # -------------------------------------------------------------------------
    # 10. Time-of-day / day-of-week — deterministic, future-known metadata
    # -------------------------------------------------------------------------
    df["hour"] = df["datetime"].dt.hour.astype(float)
    df["day_of_week"] = df["datetime"].dt.dayofweek.astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 5.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 5.0)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)
    df["is_late_session"] = (df["hour"] >= 15).astype(int)

    # Keep stable metadata and numeric features only.
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw data tidak ditemukan: {RAW_FILE}")

    print("=" * 100)
    print("YOIKI H1 V2 — FEATURE FACTORY")
    print(f"Input : {RAW_FILE}")
    print(f"Output: {FEATURES_FILE}")
    print("=" * 100)

    raw = pd.read_parquet(RAW_FILE)
    raw["datetime"] = pd.to_datetime(raw["datetime"], errors="coerce")
    raw = raw.dropna(subset=["ticker", "datetime"]).copy()

    result = []
    for ticker, group in raw.groupby("ticker", sort=False):
        result.append(compute_ticker_features(group))

    features = pd.concat(result, ignore_index=True)
    # EMA-200 is the longest mandatory warm-up feature.
    features = features.dropna(subset=["ema_200"]).reset_index(drop=True)
    features.to_parquet(FEATURES_FILE, index=False)

    feature_cols = [
        c for c in features.columns
        if c not in {"ticker", "datetime", "open", "high", "low", "close", "volume"}
    ]
    print(f"Rows   : {len(features):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Saved  : {FEATURES_FILE}")


if __name__ == "__main__":
    main()
