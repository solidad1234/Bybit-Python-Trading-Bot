"""
Standalone Multi-Asset Backtest for Bybit Futures Bot (futures.py)
==================================================================

Replays the exact multi-asset scanning logic, 4h trend filter, ATR stop/target calculation,
position sizing, partial profit taking, breakeven stop, early scratch exit, and trailing-stop
rules from futures.py against historical Bybit kline data across SOLUSDT, ETHUSDT, AVAXUSDT,
LINKUSDT, and BNBUSDT.

Outputs a rich feature dataset (trade_log.csv) formatted for XGBoost ML model training.

REQUIREMENTS:
    pip install pandas numpy requests ta-lib python-dotenv

USAGE:
    python backtest.py --days 365 --balance 100
    python backtest.py --start-year 2022 --balance 100
"""

import argparse
import sys
import os
import time
import numpy as np
import pandas as pd
import requests
try:
    import talib
except ImportError:
    talib = None
from datetime import datetime, timezone, timedelta
from factors.support_resistance import detect_sr_levels_from_arrays

BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
BYBIT_MKT_URL   = "https://api.bybit.com/v5/market"

# ---- Multi-Asset Universe Configuration (7 Coins) ----
TRADE_SYMBOLS = ["SOLUSDT", "ETHUSDT", "BNBUSDT", "LINKUSDT", "BTCUSDT", "NEARUSDT", "INJUSDT"]
PRIMARY_TF = "15"
HIGHER_TF = "60"

FUTURES_RISK_PER_TRADE = 0.02
MIN_REWARD_RATIO = 2.0       # 2:1 R:R target
SIGNAL_STRENGTH_THRESHOLD = 4
MAX_DAILY_TRADES = 15
MAX_CONSECUTIVE_LOSSES = 3

TAKER_FEE = 0.00055
FUNDING_RATE_PER_8H = 0.0001   # fallback when historical funding unavailable

CONTRACT_SPECS = {
    "SOLUSDT":  {"min_size": 0.1,   "step_size": 0.1,   "decimals": 1},
    "ETHUSDT":  {"min_size": 0.01,  "step_size": 0.01,  "decimals": 2},
    "BNBUSDT":  {"min_size": 0.01,  "step_size": 0.01,  "decimals": 2},
    "LINKUSDT": {"min_size": 0.1,   "step_size": 0.1,   "decimals": 1},
    "BTCUSDT":  {"min_size": 0.001, "step_size": 0.001, "decimals": 3},
    "NEARUSDT": {"min_size": 0.1,   "step_size": 0.1,   "decimals": 1},
    "INJUSDT":  {"min_size": 0.1,   "step_size": 0.1,   "decimals": 1},
}

# ---- Multi-factor weights (mirrors factors/aggregator.py) ----
MF_WEIGHTS          = {"regime": 0.25, "derivatives": 0.22,
                       "technical": 0.20, "support_resistance": 0.15,
                       "sentiment": 0.12, "news": 0.06}
MF_LONG_THRESHOLD   = 0.25   # production threshold
MF_SHORT_THRESHOLD  = 0.15   # production threshold


# ----------------------------------------------------------------------
# Data fetching
# ----------------------------------------------------------------------
def fetch_klines(symbol, interval, start_ms, end_ms):
    """Paginate Bybit's public kline endpoint to build a full historical range."""
    all_rows = []
    cursor_end = end_ms
    print(f"   Fetching {symbol} ({interval}m)...")
    while cursor_end > start_ms:
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "start": start_ms,
            "end": cursor_end,
            "limit": 1000,
        }
        try:
            resp = requests.get(BYBIT_KLINE_URL, params=params, timeout=20)
            data = resp.json()
            if data.get("retCode") != 0:
                print(f"⚠️ Bybit API warning ({symbol}): {data.get('retMsg')}")
                break
            rows = data["result"]["list"]
            if not rows:
                break
            all_rows.extend(rows)
            oldest_ts = int(rows[-1][0])
            if oldest_ts <= start_ms:
                break
            cursor_end = oldest_ts - 1
            time.sleep(0.12)  # rate limit safety
        except Exception as e:
            print(f"⚠️ Fetch exception ({symbol}): {e}")
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_rows,
        columns=["ts", "open", "high", "low", "close", "volume", "turnover"],
    )
    df["ts"] = pd.to_numeric(df["ts"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c])
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["ts"], unit="ms")
    return df


def build_indicators(df):
    """Vectorized calculation of indicators matching calculate_indicators() in futures.py."""
    if df.empty:
        return pd.DataFrame()

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    out = pd.DataFrame(index=df.index)
    out["rsi"] = talib.RSI(close, timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_histogram"] = macd_hist
    out["ema_21"] = talib.EMA(close, timeperiod=21)
    out["ema_50"] = talib.EMA(close, timeperiod=50)
    out["atr"] = talib.ATR(high, low, close, timeperiod=14)
    out["volume_sma"] = talib.SMA(volume, timeperiod=20)
    out["adx"] = talib.ADX(high, low, close, timeperiod=14)
    stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_d
    out["volume_ratio"] = volume / out["volume_sma"].replace(0, np.nan)
    out["close"] = close
    out["high"] = high
    out["low"] = low
    out["time"] = df["time"].values
    out["ts"] = df["ts"].values
    return out


# ----------------------------------------------------------------------
# Historical factor data helpers
# ----------------------------------------------------------------------

def fetch_historical_funding(symbol: str, start_ms: int, end_ms: int) -> dict:
    """Pull full Bybit funding-rate history for a symbol."""
    url = f"{BYBIT_MKT_URL}/funding/history"
    result = {}
    cursor_end = end_ms
    while cursor_end > start_ms:
        try:
            resp = requests.get(url, params={
                "category": "linear", "symbol": symbol,
                "startTime": start_ms, "endTime": cursor_end, "limit": 200,
            }, timeout=15)
            data = resp.json()
            if data.get("retCode") != 0:
                break
            items = data["result"]["list"]
            if not items:
                break
            for item in items:
                ts = int(item["fundingRateTimestamp"])
                result[ts] = float(item["fundingRate"])
            oldest = int(items[-1]["fundingRateTimestamp"])
            if oldest <= start_ms:
                break
            cursor_end = oldest - 1
            time.sleep(0.12)
        except Exception:
            break
    return result


def fetch_historical_fng(days: int) -> dict:
    """Fetch Fear & Greed history from alternative.me."""
    try:
        resp = requests.get("https://api.alternative.me/fng/",
                            params={"limit": days}, timeout=12)
        data = resp.json()
        result = {}
        for entry in data.get("data", []):
            dt = datetime.fromtimestamp(int(entry["timestamp"]), tz=timezone.utc)
            key = dt.strftime("%Y-%m-%d")
            result[key] = int(entry["value"])
        return result
    except Exception as e:
        print(f"⚠️ Could not fetch historical F&G: {e}")
        return {}


def build_regime_scores(btc1h: pd.DataFrame) -> dict:
    """Compute rolling regime score for every 1h bar from BTC data."""
    if btc1h.empty:
        return {}
    closes = btc1h["close"].values
    times  = btc1h["time"].values
    ema30  = pd.Series(closes).ewm(span=30, adjust=False).mean().values
    scores = {}
    for i, (t, c, e30) in enumerate(zip(times, closes, ema30)):
        if c > e30 * 1.03:
            trend = 1.0
        elif c > e30:
            trend = 0.5
        elif c > e30 * 0.95:
            trend = -0.3
        else:
            trend = -1.0
        mom = (c - closes[i - 7]) / closes[i - 7] if i >= 7 else 0.0
        mom = max(-1.0, min(1.0, mom * 10))
        scores[i] = max(-1.0, min(1.0, trend * 0.55 + mom * 0.30))
    return scores


def _fng_to_score(value: int) -> float:
    """Contrarian mapping of F&G value (0-100) -> score (-1 to +1)."""
    if value <= 30:
        return 0.3 + (30 - value) / 42.85
    elif value <= 70:
        return 0.0
    elif value <= 75:
        return -0.2
    else:
        return -0.3 - (value - 75) / 35.71


def compute_multi_factor_details(
    ta_signal: dict,
    bar_time,
    btc1h_idx: int,
    regime_scores: dict,
    funding_map: dict,
    fng_map: dict,
    sr_res: dict = None,
) -> dict:
    """Compute weighted multi-factor scores and return detailed breakdown for logging."""
    bar_dt = pd.Timestamp(bar_time)

    # 1. Technical score
    direction = ta_signal.get("signal")
    strength  = ta_signal.get("strength", 0)
    if direction == "LONG":
        ta_score = min(1.0, (strength - 3) / 4.0)
    elif direction == "SHORT":
        ta_score = -min(1.0, (strength - 5) / 5.0)
    else:
        ta_score = 0.0

    # 2. Regime score
    regime_score = regime_scores.get(btc1h_idx, 0.0)

    # 3. Derivatives
    deriv_score = 0.0
    funding_rate = 0.0
    if funding_map:
        bar_ms = int(bar_dt.timestamp() * 1000)
        past = [ts for ts in funding_map if ts <= bar_ms]
        if past:
            funding_rate = funding_map[max(past)]
            if funding_rate > 0.00015:
                deriv_score = -min(1.0, (funding_rate - 0.00015) / 0.0004)
            elif funding_rate < 0:
                deriv_score = min(1.0, abs(funding_rate) / 0.0003)
            else:
                deriv_score = 0.0

    # 4. Sentiment
    sentiment_score = 0.0
    date_key = bar_dt.strftime("%Y-%m-%d")
    if date_key in fng_map:
        sentiment_score = _fng_to_score(fng_map[date_key])

    # 5. News
    news_score = 0.0

    # 6. S/R score
    sr_score = sr_res.get("score", 0.0) if sr_res else 0.0

    final = (MF_WEIGHTS["technical"]          * ta_score
           + MF_WEIGHTS["regime"]             * regime_score
           + MF_WEIGHTS["derivatives"]        * deriv_score
           + MF_WEIGHTS["support_resistance"] * sr_score
           + MF_WEIGHTS["sentiment"]          * sentiment_score
           + MF_WEIGHTS["news"]               * news_score)
    final_score = max(-1.0, min(1.0, final))

    regime_class = "BULL" if regime_score > 0.3 else ("BEAR" if regime_score < -0.3 else "NEUTRAL")

    return {
        "final_score":       round(final_score, 3),
        "technical_score":   round(ta_score, 3),
        "regime_score":      round(regime_score, 3),
        "derivatives_score": round(deriv_score, 3),
        "sentiment_score":   round(sentiment_score, 3),
        "news_score":        round(news_score, 3),
        "sr_score":          round(sr_score, 3),
        "regime_class":      regime_class,
        "funding_rate":      funding_rate,
    }


# ----------------------------------------------------------------------
# Signal / stop / sizing logic - mirrors futures.py exactly
# ----------------------------------------------------------------------
def calculate_signal(row15, row1h, row4h, volatility, regime_score=0.0):
    """Calculate futures trading signals with hard 4h trend gate and dynamic LONG threshold."""
    trend_bullish = (not pd.isna(row4h["ema_21"])) and row4h["ema_21"] > row4h["ema_50"]
    trend_bearish = (not pd.isna(row4h["ema_21"])) and row4h["ema_21"] < row4h["ema_50"]

    long_conditions = [
        row15["rsi"] < 40,
        row1h["rsi"] < 50,
        row15["macd"] > row15["macd_signal"],
        row15["close"] > row15["ema_21"] * 0.998,
        row15["volume_ratio"] > 1.3,
        volatility > 0.02,
        row15["adx"] > 18,
    ]
    short_conditions = [
        row15["rsi"] > 65,
        row1h["rsi"] > 55,
        row15["macd"] < row15["macd_signal"],
        (row15["macd_histogram"] / row15["close"]) < -0.0005 if row15["close"] > 0 else False,
        row15["close"] < row15["ema_21"],
        row15["volume_ratio"] > 1.4,
        volatility > 0.025,
        row15["stoch_k"] > 80,
        row15["close"] > row1h["ema_50"] * 0.98,
        row15["adx"] > 18,
    ]
    long_score  = sum(bool(c) for c in long_conditions)
    short_score = sum(bool(c) for c in short_conditions)

    min_long_score = 5 if regime_score <= -0.4 else 4

    if long_score >= min_long_score and trend_bullish:
        return {"signal": "LONG",  "strength": long_score,  "leverage": 10.0}
    if short_score >= 6 and trend_bearish:
        return {"signal": "SHORT", "strength": short_score, "leverage": 10.5}
    return {"signal": None, "strength": max(long_score, short_score)}


def calculate_stops(direction, entry_price, atr15, atr1h, strength):
    """Calculate improved ATR stops matching futures.py."""
    primary_atr = max(atr15, atr1h * 0.7)
    base_stop_distance = 1.5 * primary_atr
    strength_multiplier = 1.0 + (strength / 20)
    volatility_factor = min(atr1h / atr15, 1.5) if atr15 > 0 else 1.2
    stop_distance = base_stop_distance * strength_multiplier * volatility_factor

    min_stop_distance = entry_price * 0.008
    if stop_distance < min_stop_distance:
        stop_distance = min_stop_distance

    reward_distance = stop_distance * MIN_REWARD_RATIO

    if direction == "SHORT":
        stop_loss = entry_price + stop_distance
        take_profit = entry_price - reward_distance
    else:
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + reward_distance

    return stop_loss, take_profit, stop_distance


def calculate_position_size(symbol, balance, entry_price, stop_loss, leverage):
    """Calculate safe position size using symbol-specific step and min sizes."""
    max_usable_margin = balance * 0.7
    risk_amount = balance * FUTURES_RISK_PER_TRADE
    stop_distance = abs(entry_price - stop_loss)
    if stop_distance <= 0 or balance < 5:
        return None

    spec = CONTRACT_SPECS.get(symbol, {"min_size": 0.1, "step_size": 0.1, "decimals": 1})
    min_order_size = spec["min_size"]
    step_size = spec["step_size"]
    decimals = spec["decimals"]

    max_position_by_margin = (max_usable_margin * leverage) / entry_price
    max_position_by_risk = risk_amount / stop_distance
    position_size = min(max_position_by_margin, max_position_by_risk)

    if position_size < min_order_size:
        position_size = min_order_size
    else:
        position_size = round(position_size / step_size) * step_size

    required_margin = (position_size * entry_price) / leverage
    if required_margin > max_usable_margin:
        position_size = round((max_usable_margin * leverage) / entry_price / step_size) * step_size
        required_margin = (position_size * entry_price) / leverage
        if position_size < min_order_size:
            return None

    if required_margin < 2:
        return None

    return {
        "position_size": round(position_size, decimals),
        "required_margin": round(required_margin, 2),
    }


# ----------------------------------------------------------------------
# Multi-Asset Backtest Execution Engine
# ----------------------------------------------------------------------
def run_multi_asset_backtest(symbols, start_ms, end_ms, starting_balance, no_factors=False):
    """Executes multi-asset bar-by-bar backtest across all symbols in universe."""

    print("\n" + "=" * 60)
    print("PRE-FETCHING MULTI-ASSET HISTORICAL KLINE DATA")
    print("=" * 60)

    # Fetch 15m, 1h, 4h data for each symbol
    data15, data1h, data4h = {}, {}, {}
    for sym in symbols:
        df15 = fetch_klines(sym, PRIMARY_TF, start_ms, end_ms)
        df1h = fetch_klines(sym, HIGHER_TF, start_ms, end_ms)
        df4h = fetch_klines(sym, "240", start_ms, end_ms)

        data15[sym] = build_indicators(df15)
        data1h[sym] = build_indicators(df1h)
        data4h[sym] = build_indicators(df4h)

    # Fetch BTC data for macro correlation and regime
    dfbtc = fetch_klines("BTCUSDT", HIGHER_TF, start_ms, end_ms)
    btc1h = build_indicators(dfbtc)

    if btc1h.empty or data15[symbols[0]].empty:
        raise RuntimeError("Failed to load historical data for backtesting.")

    # Multi-factor datasets
    regime_scores = {}
    funding_maps = {}
    fng_map = {}

    if not no_factors:
        print("\n" + "=" * 60)
        print("PRE-FETCHING MULTI-FACTOR DATA (REGIME, FUNDING, F&G)")
        print("=" * 60)
        regime_scores = build_regime_scores(btc1h)
        days_count = (end_ms - start_ms) // (24 * 3600 * 1000)
        fng_map = fetch_historical_fng(min(days_count, 365))

        for sym in symbols:
            funding_maps[sym] = fetch_historical_funding(sym, start_ms, end_ms)
            print(f"  Funding history for {sym}: {len(funding_maps[sym])} records ✓")

    # Establish master timeline based on first symbol's 15m candles
    master_df = data15[symbols[0]]
    master_times = master_df["time"].values
    master_ts = master_df["ts"].values

    balance = starting_balance
    equity_curve = []
    trades = []

    position = None
    daily_trades = 0
    consecutive_losses = 0
    last_day = None

    btc_times = btc1h["time"].values
    btc_close = btc1h["close"].values

    warmup = 60
    print("\n" + "=" * 60)
    print("RUNNING MULTI-ASSET SIMULATION LOOP...")
    print("=" * 60)

    for i in range(warmup, len(master_times)):
        t = master_times[i]
        ts_curr = master_ts[i]
        
        # daily reset: trades counter and consecutive loss limit
        day = pd.Timestamp(t).date()
        if day != last_day:
            daily_trades = 0
            consecutive_losses = 0
            last_day = day

        # Align BTC correlation
        b_idx = np.searchsorted(btc_times, t, side="right") - 1
        if b_idx < 4:
            btc_bear = False
        else:
            btc_now = btc_close[b_idx]
            btc_1h_ago = btc_close[b_idx - 1]
            btc_4h_ago = btc_close[b_idx - 4]
            chg_1h = (btc_now - btc_1h_ago) / btc_1h_ago * 100
            chg_4h = (btc_now - btc_4h_ago) / btc_4h_ago * 100
            btc_bear = chg_1h < -2.0 or chg_4h < -5.0

        # ---- Manage open position first ----
        if position is not None:
            sym = position["symbol"]
            ind15_curr = data15[sym]
            
            # Find current bar index for active symbol
            s_idx = np.searchsorted(ind15_curr["ts"].values, ts_curr)
            if s_idx < len(ind15_curr) and ind15_curr["ts"].values[s_idx] == ts_curr:
                row15 = ind15_curr.iloc[s_idx]
                current_price = row15["close"]
                hi, lo = row15["high"], row15["low"]
                direction = position["direction"]

                exit_price = None
                result = None
                bars_held = i - position["entry_master_idx"]

                # Early scratch exit (-0.7% adverse in first 3 bars)
                if not position["stop_moved_to_be"] and bars_held <= 3:
                    adverse_pct = ((position["entry"] - current_price) / position["entry"]
                                   if direction == "LONG" else
                                   (current_price - position["entry"]) / position["entry"])
                    if adverse_pct >= 0.007:
                        exit_price, result = current_price, "SCRATCH"

                if exit_price is None:
                    if direction == "LONG":
                        if lo <= position["stop"]:
                            exit_price, result = position["stop"], "LOSS" if position["stop"] < position["entry"] else "BE/WIN"
                        elif hi >= position["target"]:
                            exit_price, result = position["target"], "WIN"
                    else:  # SHORT
                        if hi >= position["stop"]:
                            exit_price, result = position["stop"], "LOSS" if position["stop"] > position["entry"] else "BE/WIN"
                        elif lo <= position["target"]:
                            exit_price, result = position["target"], "WIN"

                # Check partial profit & trailing stop if still open
                if exit_price is None:
                    pnl_pct = ((current_price - position["entry"]) / position["entry"]
                               if direction == "LONG" else
                               (position["entry"] - current_price) / position["entry"])

                    if pnl_pct >= 0.015 and not position["stop_moved_to_be"]:
                        position["stop"] = position["entry"]
                        position["stop_moved_to_be"] = True

                    atr15 = row15["atr"]
                    if direction == "LONG":
                        position["highest"] = max(position["highest"], current_price)
                        if (position["highest"] - position["entry"]) / position["entry"] > 0.015:
                            proposed = position["highest"] - (1.5 * atr15)
                            if proposed > position["stop"]:
                                position["stop"] = proposed
                    else:
                        position["lowest"] = min(position["lowest"], current_price)
                        if (position["entry"] - position["lowest"]) / position["entry"] > 0.015:
                            proposed = position["lowest"] + (1.5 * atr15)
                            if proposed < position["stop"]:
                                position["stop"] = proposed

                # Execute Exit
                if exit_price is not None:
                    size = position["size"]
                    gross_pnl = ((exit_price - position["entry"]) * size if direction == "LONG"
                                 else (position["entry"] - exit_price) * size)

                    notional_in = position["entry"] * size
                    notional_out = exit_price * size
                    fees = (notional_in + notional_out) * TAKER_FEE

                    funding_periods = max(1, (bars_held * 15) // (8 * 60))
                    funding_cost = notional_in * FUNDING_RATE_PER_8H * funding_periods

                    net_pnl = gross_pnl - fees - funding_cost
                    balance += net_pnl

                    fc = position["factor_context"]
                    trades.append({
                        "symbol":             sym,
                        "entry_time":         position["entry_time"],
                        "exit_time":          t,
                        "direction":          direction,
                        "entry_price":        position["entry"],
                        "exit_price":         exit_price,
                        "size":               size,
                        "gross_pnl":          round(gross_pnl, 4),
                        "fees":               round(fees, 4),
                        "funding_cost":       round(funding_cost, 4),
                        "net_pnl":            round(net_pnl, 4),
                        "result":             result,
                        "balance_after":      round(balance, 2),

                        # Rich multi-factor feature set for XGBoost training
                        "ta_signal_strength": fc.get("ta_signal_strength"),
                        "aggregated_score":   fc.get("aggregated_score"),
                        "volatility":         fc.get("volatility"),
                        "atr_15m":            fc.get("atr_15m"),
                        "technical_score":    fc.get("technical_score"),
                        "regime_score":       fc.get("regime_score"),
                        "derivatives_score":  fc.get("derivatives_score"),
                        "sentiment_score":    fc.get("sentiment_score"),
                        "news_score":         fc.get("news_score"),
                        "regime_class":       fc.get("regime_class"),
                        "funding_rate":       fc.get("funding_rate"),
                        "market_trend_4h":    fc.get("market_trend_4h"),
                    })

                    consecutive_losses = 0 if net_pnl > 0 else consecutive_losses + 1
                    position = None

        # ---- Scan Universe for New Setup (only if flat) ----
        if position is None:
            if daily_trades >= MAX_DAILY_TRADES or consecutive_losses >= MAX_CONSECUTIVE_LOSSES or balance < 5:
                equity_curve.append({"time": t, "balance": balance})
                continue

            best_setup = None
            best_score = -1.0

            for sym in symbols:
                ind15_df = data15[sym]
                ind1h_df = data1h[sym]
                ind4h_df = data4h[sym]

                s_idx15 = np.searchsorted(ind15_df["ts"].values, ts_curr)
                if s_idx15 >= len(ind15_df) or ind15_df["ts"].values[s_idx15] != ts_curr:
                    continue

                row15 = ind15_df.iloc[s_idx15]
                if pd.isna(row15["rsi"]) or pd.isna(row15["atr"]):
                    continue

                idx1h = ind1h_df["ts"].searchsorted(ts_curr, side="right") - 1
                if idx1h < warmup:
                    continue
                row1h = ind1h_df.iloc[idx1h]

                idx4h = ind4h_df["ts"].searchsorted(ts_curr, side="right") - 1
                if idx4h < 10:
                    continue
                row4h = ind4h_df.iloc[idx4h]

                # Regime filter: skip flat market
                if row1h["adx"] < 18:
                    continue

                current_price = row15["close"]
                if idx1h >= 24:
                    price_24h_ago = ind1h_df.iloc[idx1h - 24]["close"]
                    volatility = abs((current_price - price_24h_ago) / price_24h_ago)
                else:
                    volatility = 0.02

                b_idx1h = btc1h["ts"].searchsorted(ts_curr, side="right") - 1
                regime_t = regime_scores.get(b_idx1h, 0.0)

                signal = calculate_signal(row15, row1h, row4h, volatility, regime_score=regime_t)

                if signal["signal"] == "LONG" and btc_bear:
                    continue
                if not signal["signal"] or signal["strength"] < SIGNAL_STRENGTH_THRESHOLD:
                    continue

                # Compute S/R levels
                h1 = ind1h_df["high"].values[:idx1h + 1]
                l1 = ind1h_df["low"].values[:idx1h + 1]
                h4 = ind4h_df["high"].values[:idx4h + 1]
                l4 = ind4h_df["low"].values[:idx4h + 1]
                sr_res = detect_sr_levels_from_arrays(h1, l1, h4, l4, current_price)

                # Multi-Factor Consensus Evaluation
                if no_factors:
                    direction_ok = True
                    mf_details = {
                        "final_score": 0.0, "technical_score": 0.0, "regime_score": 0.0,
                        "derivatives_score": 0.0, "sentiment_score": 0.0, "news_score": 0.0,
                        "sr_score": 0.0, "regime_class": "NEUTRAL", "funding_rate": 0.0
                    }
                else:
                    mf_details = compute_multi_factor_details(
                        signal, t, b_idx1h, regime_scores,
                        funding_maps.get(sym, {}), fng_map,
                        sr_res=sr_res
                    )
                    mf_score = mf_details["final_score"]
                    direction_ok = (
                        (signal["signal"] == "LONG"  and mf_score >=  MF_LONG_THRESHOLD) or
                        (signal["signal"] == "SHORT" and mf_score <= -MF_SHORT_THRESHOLD)
                    )

                if not direction_ok:
                    continue

                score_abs = abs(mf_details["final_score"]) if not no_factors else float(signal["strength"])
                if score_abs > best_score:
                    best_score = score_abs
                    trend_4h = "BULL" if row4h["ema_21"] > row4h["ema_50"] else "BEAR"

                    trade_leverage = signal["leverage"]
                    if sr_res.get("suggested_stop") and sr_res.get("suggested_target") and sr_res.get("scenario") != "MID_RANGE":
                        stop_loss = sr_res["suggested_stop"]
                        take_profit = sr_res["suggested_target"]
                        if sr_res.get("suggested_leverage"):
                            trade_leverage = sr_res["suggested_leverage"]
                    else:
                        stop_loss, take_profit, _ = calculate_stops(
                            signal["signal"], current_price, row15["atr"], row1h["atr"], signal["strength"]
                        )

                    sizing = calculate_position_size(sym, balance, current_price, stop_loss, trade_leverage)

                    if sizing:
                        best_setup = {
                            "symbol": sym,
                            "direction": signal["signal"],
                            "entry": current_price,
                            "stop": stop_loss,
                            "target": take_profit,
                            "size": sizing["position_size"],
                            "entry_time": t,
                            "entry_master_idx": i,
                            "stop_moved_to_be": False,
                            "highest": current_price,
                            "lowest": current_price,
                            "factor_context": {
                                "ta_signal_strength": signal["strength"],
                                "aggregated_score":   mf_details["final_score"],
                                "volatility":         round(volatility, 4),
                                "atr_15m":            round(row15["atr"], 4),
                                "technical_score":    mf_details["technical_score"],
                                "regime_score":       mf_details["regime_score"],
                                "derivatives_score":  mf_details["derivatives_score"],
                                "sentiment_score":    mf_details["sentiment_score"],
                                "news_score":         mf_details["news_score"],
                                "sr_score":           mf_details.get("sr_score", 0.0),
                                "sr_scenario":        sr_res.get("scenario", "MID_RANGE"),
                                "regime_class":       mf_details["regime_class"],
                                "funding_rate":       mf_details["funding_rate"],
                                "market_trend_4h":    trend_4h,
                            }
                        }

            # Execute best setup found across universe
            if best_setup:
                position = best_setup
                daily_trades += 1

        equity_curve.append({"time": t, "balance": balance})

    return pd.DataFrame(trades), pd.DataFrame(equity_curve)


def summarize(trades, equity, starting_balance):
    """Summarize and display backtest results."""
    if trades.empty:
        print("\n❌ No trades were triggered over this backtest period.")
        return

    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] <= 0]
    final_balance = equity["balance"].iloc[-1]
    total_return_pct = (final_balance - starting_balance) / starting_balance * 100

    running_max = equity["balance"].cummax()
    drawdown = (equity["balance"] - running_max) / running_max
    max_dd_pct = drawdown.min() * 100

    gross_win = wins["net_pnl"].sum()
    gross_loss = abs(losses["net_pnl"].sum())
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    print("\n" + "=" * 60)
    print("MULTI-ASSET BACKTEST SUMMARY (2022 - PRESENT)")
    print("=" * 60)
    print(f"Universe Assets:     {', '.join(TRADE_SYMBOLS)}")
    print(f"Total Trades:        {len(trades)}")
    print(f"Win Rate:            {len(wins) / len(trades) * 100:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"Starting Balance:    ${starting_balance:.2f}")
    print(f"Ending Balance:      ${final_balance:.2f}")
    print(f"Total Return:        {total_return_pct:+.1f}%")
    print(f"Max Drawdown:        {max_dd_pct:.1f}%")
    print(f"Profit Factor:       {profit_factor:.2f}")
    print(f"Total Fees Paid:     ${trades['fees'].sum():.2f}")
    print(f"Total Funding Paid:  ${trades['funding_cost'].sum():.2f}")
    print(f"Avg Net PnL/Trade:   ${trades['net_pnl'].mean():.2f}")
    print("=" * 60)

    print("\nBreakdown by Asset:")
    for sym in TRADE_SYMBOLS:
        sym_trades = trades[trades["symbol"] == sym]
        if not sym_trades.empty:
            sym_wins = sym_trades[sym_trades["net_pnl"] > 0]
            sym_wr = len(sym_wins) / len(sym_trades) * 100
            sym_pnl = sym_trades["net_pnl"].sum()
            print(f"  • {sym:<10}: {len(sym_trades):>3} trades | Win Rate: {sym_wr:>5.1f}% | Net PnL: ${sym_pnl:>+7.2f}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Bybit Futures Bot Backtest")
    parser.add_argument("--days",       type=int,   default=365,   help="Days of history to test")
    parser.add_argument("--start-year", type=int,   default=None,  help="Start calendar year (e.g. 2022)")
    parser.add_argument("--balance",    type=float, default=100.0, help="Starting balance in USDT")
    parser.add_argument("--no-factors", action="store_true",    help="Pure TA mode — skip multi-factor gating")
    args = parser.parse_args()

    if args.start_year:
        start_ms = int(datetime(args.start_year, 1, 1).timestamp() * 1000)
        end_ms   = int(time.time() * 1000)
        days_label = f"{args.start_year}-Present"
    else:
        end_ms   = int(time.time() * 1000)
        start_ms = end_ms - args.days * 24 * 60 * 60 * 1000
        days_label = f"{args.days}d"

    print(f"{'='*60}")
    print(f"MULTI-ASSET FUTURES BOT BACKTEST")
    print(f"Universe: {', '.join(TRADE_SYMBOLS)}")
    print(f"Period:   {days_label}")
    print(f"Balance:  ${args.balance}")
    print(f"{'='*60}")

    trades, equity = run_multi_asset_backtest(
        TRADE_SYMBOLS, start_ms, end_ms, args.balance, no_factors=args.no_factors
    )

    tlog = "trade_log.csv"
    ecurv = "equity_curve.csv"
    trades.to_csv(tlog, index=False)
    equity.to_csv(ecurv, index=False)

    summarize(trades, equity, args.balance)
    print(f"\n✅ Clean dataset written to: {tlog}")
    print(f"✅ Equity curve written to:  {ecurv}")


if __name__ == "__main__":
    main()