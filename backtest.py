"""
Backtest for the SOLUSDT Futures Bot (futures.py)
==================================================

Replays the SAME signal logic, stop/target calculation, position sizing,
partial-exit, breakeven, and trailing-stop rules from futures.py against
historical Bybit kline data, so you can see how the strategy would have
performed before risking real capital on it live.

REQUIREMENTS (run this on your server, not in a sandboxed env):
    pip install pandas numpy requests ta-lib python-dotenv

USAGE:
    python backtest.py --days 730 --balance 100

WHAT THIS DOES:
    1. Pulls ~N days of 15m, 1h history for SOLUSDT and BTCUSDT from
       Bybit's public REST API (no API key needed - historical data is public).
    2. Recomputes the exact indicators/conditions from futures.py bar-by-bar.
    3. Simulates entries, ATR stops/targets, partial profit-taking,
       breakeven-stop, trailing stop, daily trade limits, and the
       consecutive-loss halt.
    4. Applies realistic taker fees (both sides) and an approximate funding
       cost for time spent in a position.
    5. Prints a summary and writes trade_log.csv + equity_curve.csv.

IMPORTANT LIMITATIONS (read before trusting the output):
    - Trade management (partials/trailing stop) is evaluated on each 15m
      candle's CLOSE, not on the live bot's 10-second fast loop. This means
      intrabar moves that would have triggered management in real life may
      be missed or delayed here. Stop-loss / take-profit HITS, however, are
      checked against each candle's HIGH/LOW, so those are realistic.
    - Funding rate is approximated as a flat rate you configure (default
      0.01%/8h) rather than pulled from actual historical funding data.
      Actual SOLUSDT funding varies and can meaningfully affect results,
      especially if the strategy tends to hold through funding windows in
      one direction more than the other.
    - Slippage is not modeled - all fills are assumed at the exact signal
      price. Real fills, especially at 10x+ leverage during volatility
      spikes, will be worse than this.
    - This assumes liquidation never happens before the stop is hit. At
      high leverage that assumption can be wrong - check your liquidation
      price against your stop distance separately.
    - Backtest results, even done carefully, do not guarantee future
      performance. Treat this as a way to reject clearly-broken logic and
      get a rough sense of edge and fee-drag, not as a promise.
"""

import argparse
import sys
import os
import time
import numpy as np
import pandas as pd
import requests
import talib
from datetime import datetime, timezone, timedelta

BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
BYBIT_MKT_URL   = "https://api.bybit.com/v5/market"

# ---- Strategy constants (mirrored from futures.py) ----
SYMBOL = "SOLUSDT"
PRIMARY_TF = "15"
HIGHER_TF = "60"

FUTURES_RISK_PER_TRADE = 0.02
MIN_REWARD_RATIO = 2.0       # 3.0 target never reached on 15m; 2.0 is realistically achievable
SIGNAL_STRENGTH_THRESHOLD = 5
MAX_DAILY_TRADES = 15
MAX_CONSECUTIVE_LOSSES = 3

TAKER_FEE = 0.00055
FUNDING_RATE_PER_8H = 0.0001   # fallback when no real funding data available

# ---- Multi-factor weights (mirrors factors/aggregator.py) ----
# Rebalanced: regime ↑ 0.15→0.30 (macro is most reliable crash signal);
# sentiment ↓ 0.20→0.15 (contrarian F&G was fighting regime in rapid downturns).
# Principle: weights changed for sound structural reasons, not to fix a known event.
MF_WEIGHTS          = {"regime": 0.30, "derivatives": 0.25,
                       "technical": 0.20, "sentiment": 0.15, "news": 0.10}
MF_LONG_THRESHOLD   = 0.25   # |score| to allow LONG
MF_SHORT_THRESHOLD  = 0.15   # lower bar for SHORT — sentiment factor is structurally biased long


# ----------------------------------------------------------------------
# Data fetching
# ----------------------------------------------------------------------
def fetch_klines(symbol, interval, start_ms, end_ms):
    """Paginate Bybit's public kline endpoint to build a full historical range."""
    all_rows = []
    cursor_end = end_ms
    while cursor_end > start_ms:
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "start": start_ms,
            "end": cursor_end,
            "limit": 1000,
        }
        resp = requests.get(BYBIT_KLINE_URL, params=params, timeout=20)
        data = resp.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit API error: {data.get('retMsg')}")
        rows = data["result"]["list"]
        if not rows:
            break
        all_rows.extend(rows)
        oldest_ts = int(rows[-1][0])
        if oldest_ts <= start_ms:
            break
        cursor_end = oldest_ts - 1
        time.sleep(0.15)  # be polite to the API

    if not all_rows:
        raise RuntimeError(f"No kline data returned for {symbol} {interval}")

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
    """Vectorized version of calculate_indicators() from futures.py, applied
    across full history so every bar has a causal (backward-looking only) value."""
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
    return out


# ----------------------------------------------------------------------
# Historical factor data helpers
# ----------------------------------------------------------------------

def fetch_historical_funding(symbol: str, start_ms: int, end_ms: int) -> dict:
    """
    Pull full Bybit funding-rate history for a symbol.
    Returns {funding_timestamp_ms: rate} for every 8h settlement.
    """
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
            time.sleep(0.15)
        except Exception:
            break
    return result


def fetch_historical_fng(days: int) -> dict:
    """
    Fetch Fear & Greed history from alternative.me (free, up to 365 days).
    Returns {"YYYY-MM-DD": value} dict.
    """
    try:
        resp = requests.get("https://api.alternative.me/fng/",
                            params={"limit": days}, timeout=12)
        data = resp.json()
        result = {}
        for entry in data.get("data", []):
            # timestamp is Unix seconds
            dt  = datetime.fromtimestamp(int(entry["timestamp"]), tz=timezone.utc)
            key = dt.strftime("%Y-%m-%d")
            result[key] = int(entry["value"])
        return result
    except Exception as e:
        print(f"⚠️  Could not fetch historical F&G: {e}")
        return {}


def build_regime_scores(btc1h: pd.DataFrame) -> dict:
    """
    Compute a rolling regime score for every 1h bar from BTC data.
    Returns {bar_index: score} where score ∈ [-1, +1].
    """
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
    """Contrarian mapping of F&G value (0-100) → score (-1 to +1)."""
    if value <= 20:
        return 0.8 + (20 - value) / 100
    elif value <= 40:
        return 0.2 + (40 - value) / 50
    elif value <= 60:
        return 0.0
    elif value <= 80:
        return -0.2 - (value - 60) / 50
    else:
        return -0.8 - (value - 80) / 100


def compute_multi_factor_score(
    ta_signal: dict,
    bar_time,
    btc1h_idx: int,
    regime_scores: dict,
    funding_map: dict,
    fng_map: dict,
) -> float:
    """
    Compute a weighted multi-factor consensus score for one backtest bar.
    Returns the final score ∈ [-1, +1].
    News factor is treated as 0 (neutral) — no reliable historical data.
    """
    bar_dt = pd.Timestamp(bar_time)

    # 1. Technical score
    direction = ta_signal.get("signal")
    strength  = ta_signal.get("strength", 0)
    if direction == "LONG":
        ta_score = min(1.0, (strength - 4) / 3.0)
    elif direction == "SHORT":
        ta_score = -min(1.0, (strength - 5) / 5.0)
    else:
        ta_score = 0.0

    # 2. Regime score (from pre-computed rolling BTC EMA)
    regime_score = regime_scores.get(btc1h_idx, 0.0)

    # 3. Derivatives — nearest historical funding rate (contrarian)
    deriv_score = 0.0
    if funding_map:
        bar_ms = int(bar_dt.timestamp() * 1000)
        # find closest funding timestamp at or before this bar
        past = [ts for ts in funding_map if ts <= bar_ms]
        if past:
            rate = funding_map[max(past)]
            # high positive → longs crowded → SHORT (negative score)
            if rate >= 0:
                deriv_score = -min(1.0, rate / 0.0005)
            else:
                deriv_score = min(1.0, abs(rate) / 0.0003)

    # 4. Sentiment — daily F&G (contrarian)
    sentiment_score = 0.0
    date_key = bar_dt.strftime("%Y-%m-%d")
    if date_key in fng_map:
        sentiment_score = _fng_to_score(fng_map[date_key])

    # 5. News — neutral in backtest
    news_score = 0.0

    final = (MF_WEIGHTS["technical"]    * ta_score
           + MF_WEIGHTS["regime"]       * regime_score
           + MF_WEIGHTS["derivatives"]  * deriv_score
           + MF_WEIGHTS["sentiment"]    * sentiment_score
           + MF_WEIGHTS["news"]         * news_score)
    return max(-1.0, min(1.0, final))


# ----------------------------------------------------------------------
# Signal / stop / sizing logic - mirrors futures.py exactly
# ----------------------------------------------------------------------
def calculate_signal(row15, row1h, row4h, volatility, regime_score=0.0):
    """
    4h trend is a hard gate, not an additive score condition.
    LONG only allowed when 4h EMA_21 > EMA_50 (bullish structure).
    SHORT only allowed when 4h EMA_21 < EMA_50 (bearish structure).

    Dynamic threshold: when macro regime is strongly bearish (regime_score <= -0.4),
    require 6/7 TA conditions for LONG instead of 5/7 — more evidence needed to fight
    a macro headwind. This is a general principle, not tied to any specific event.
    """
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
        row15["macd_histogram"] < -0.2,
        row15["close"] < row15["ema_21"],
        row15["volume_ratio"] > 1.4,
        volatility > 0.025,
        row15["stoch_k"] > 80,
        row15["close"] > row1h["ema_50"] * 0.98,
        row15["adx"] > 18,
    ]
    long_score  = sum(bool(c) for c in long_conditions)
    short_score = sum(bool(c) for c in short_conditions)

    # Dynamic LONG threshold: more evidence required when macro regime is bearish.
    # This is a continuous, principled rule — not tied to any specific event date.
    min_long_score = 6 if regime_score <= -0.4 else 5

    # Hard gates: trend alignment is non-negotiable.
    if long_score >= min_long_score and trend_bullish:
        return {"signal": "LONG",  "strength": long_score,  "leverage": 10.0}
    if short_score >= 6 and trend_bearish:
        return {"signal": "SHORT", "strength": short_score, "leverage": 10.5}
    return {"signal": None, "strength": max(long_score, short_score)}


def calculate_stops(direction, entry_price, atr15, atr1h, strength):
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


def calculate_position_size(balance, entry_price, stop_loss, leverage):
    max_usable_margin = balance * 0.7
    risk_amount = balance * FUTURES_RISK_PER_TRADE
    stop_distance = abs(entry_price - stop_loss)
    if stop_distance <= 0 or balance < 5:
        return None

    max_position_by_margin = (max_usable_margin * leverage) / entry_price
    max_position_by_risk = risk_amount / stop_distance
    position_size = min(max_position_by_margin, max_position_by_risk)

    min_order_size = 0.1
    step_size = 0.1
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
        "position_size": round(position_size, 1),
        "required_margin": round(required_margin, 2),
    }


# ----------------------------------------------------------------------
# Backtest loop
# ----------------------------------------------------------------------
def run_backtest(ind15, ind1h, ind4h, btc1h, starting_balance, factor_data=None):
    """factor_data keys: regime_scores (dict), funding (dict), fng (dict)"""
    if factor_data is None:
        factor_data = {"regime_scores": {}, "funding": {}, "fng": {}}
    balance = starting_balance
    equity_curve = []
    trades = []

    position = None
    daily_trades = 0
    consecutive_losses = 0
    last_day = None

    btc_times = btc1h["time"].values
    btc_close = btc1h["close"].values

    warmup = 60  # bars needed for indicators to stabilize
    for i in range(warmup, len(ind15)):
        row15 = ind15.iloc[i]
        t = row15["time"]

        if pd.isna(row15["rsi"]) or pd.isna(row15["atr"]):
            continue

        # daily reset: trades counter only
        # consecutive_losses is NOT reset daily — it acts as a session-level brake
        day = pd.Timestamp(t).date()
        if day != last_day:
            daily_trades = 0
            last_day = day

        # align 1h data: last 1h bar closed at or before t
        idx1h = ind1h["time"].searchsorted(t, side="right") - 1
        if idx1h < warmup:
            continue
        row1h = ind1h.iloc[idx1h]

        # align 4h data: last 4h bar closed at or before t
        idx4h = ind4h["time"].searchsorted(t, side="right") - 1
        if idx4h < 10:
            continue
        row4h = ind4h.iloc[idx4h]

        # BTC correlation (mirrors check_btc_correlation: 1h and 4h-ago change)
        b_idx = np.searchsorted(btc_times, t.to_datetime64(), side="right") - 1
        if b_idx < 4:
            btc_bull, btc_bear = True, False
        else:
            btc_now = btc_close[b_idx]
            btc_1h_ago = btc_close[b_idx - 1]
            btc_4h_ago = btc_close[b_idx - 4]
            chg_1h = (btc_now - btc_1h_ago) / btc_1h_ago * 100
            chg_4h = (btc_now - btc_4h_ago) / btc_4h_ago * 100
            btc_bull = chg_1h > -1.0 and chg_4h > -2.0
            btc_bear = chg_1h < -2.0 or chg_4h < -5.0

        current_price = row15["close"]

        # volatility: 24h change on the 1h series (24 bars back), same as futures.py
        if idx1h >= 24:
            price_24h_ago = ind1h.iloc[idx1h - 24]["close"]
            volatility = abs((current_price - price_24h_ago) / price_24h_ago)
        else:
            volatility = 0.02

        # ---- manage open position first ----
        if position is not None:
            direction = position["direction"]
            hi, lo = row15["high"], row15["low"]

            # Step 1: Check SL/TP using the PREVIOUS bar's stop (before any updates).
            # Trailing/breakeven only apply from the NEXT bar onward.
            exit_price = None
            result     = None
            bars_held  = i - position["entry_index"]

            # -- Early scratch: cut losses quickly if trade goes adverse immediately --
            # If position is -0.7% adverse within the first 3 bars AND stop hasn't moved
            # to breakeven, exit at current price. This reduces average loss R-multiple
            # without being specific to any event — it's a universal risk management rule.
            if (not position["stop_moved_to_be"] and bars_held <= 3):
                adverse_pct = ((position["entry"] - current_price) / position["entry"]
                               if direction == "LONG" else
                               (current_price - position["entry"]) / position["entry"])
                if adverse_pct >= 0.007:  # -0.7% threshold
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

            # Step 2: Only update breakeven/trailing if we didn't exit this bar.
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

            if exit_price is not None:
                size = position["size"]
                gross_pnl = ((exit_price - position["entry"]) * size if direction == "LONG"
                             else (position["entry"] - exit_price) * size)

                notional_in = position["entry"] * size
                notional_out = exit_price * size
                fees = (notional_in + notional_out) * TAKER_FEE

                bars_held = i - position["entry_index"]
                funding_periods = max(1, (bars_held * 15) // (8 * 60))
                funding_cost = notional_in * FUNDING_RATE_PER_8H * funding_periods

                net_pnl = gross_pnl - fees - funding_cost
                balance += net_pnl

                trades.append({
                    "entry_time": position["entry_time"], "exit_time": t,
                    "direction": direction, "entry": position["entry"],
                    "exit": exit_price, "size": size, "gross_pnl": gross_pnl,
                    "fees": fees, "funding_cost": funding_cost, "net_pnl": net_pnl,
                    "result": result, "balance_after": balance,
                })

                consecutive_losses = 0 if net_pnl > 0 else consecutive_losses + 1
                position = None

        # ---- look for new entry only if flat ----
        if position is None:
            if row1h["adx"] < 18:  # lowered from 20 — captures more trending bars
                pass  # regime filter: flat market, skip
            elif daily_trades >= MAX_DAILY_TRADES:
                pass
            elif consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                pass
            else:
                # pass regime_score for dynamic LONG threshold
                regime_t = factor_data["regime_scores"].get(idx1h, 0.0)
                signal = calculate_signal(row15, row1h, row4h, volatility, regime_score=regime_t)
                if signal["signal"] == "LONG" and btc_bear:
                    signal["signal"] = None

                if signal["signal"] and signal["strength"] >= SIGNAL_STRENGTH_THRESHOLD and balance >= 5:

                    # ---- Multi-factor consensus gate ----
                    if not factor_data.get("regime_scores"):  # --no-factors mode
                        direction_ok = True
                        mf_score = 0.0
                    else:
                        mf_score = compute_multi_factor_score(
                            signal, t, idx1h,
                            factor_data["regime_scores"],
                            factor_data["funding"],
                            factor_data["fng"],
                        )
                        direction_ok = (
                            (signal["signal"] == "LONG"  and mf_score >=  MF_LONG_THRESHOLD) or
                            (signal["signal"] == "SHORT" and mf_score <= -MF_SHORT_THRESHOLD)
                        )
                    if not direction_ok:
                        equity_curve.append({"time": t, "balance": balance})
                        continue   # skip this bar — multi-factor rejects it
                    # -------------------------------------

                    stop_loss, take_profit, _ = calculate_stops(
                        signal["signal"], current_price, row15["atr"], row1h["atr"], signal["strength"]
                    )
                    sizing = calculate_position_size(balance, current_price, stop_loss, signal["leverage"])
                    if sizing:
                        position = {
                            "direction": signal["signal"], "entry": current_price,
                            "stop": stop_loss, "target": take_profit,
                            "size": sizing["position_size"], "entry_time": t,
                            "entry_index": i, "stop_moved_to_be": False,
                            "highest": current_price, "lowest": current_price,
                            "mf_score": round(mf_score, 3),
                        }
                        daily_trades += 1

        equity_curve.append({"time": t, "balance": balance})

    return pd.DataFrame(trades), pd.DataFrame(equity_curve)


def summarize(trades, equity, starting_balance):
    if trades.empty:
        print("No trades were triggered over this period - the signal conditions "
              "never fired, or the ADX regime filter blocked the whole window. "
              "Try a longer date range.")
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
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Total trades:        {len(trades)}")
    print(f"Win rate:            {len(wins) / len(trades) * 100:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"Starting balance:    ${starting_balance:.2f}")
    print(f"Ending balance:      ${final_balance:.2f}")
    print(f"Total return:        {total_return_pct:+.1f}%")
    print(f"Max drawdown:        {max_dd_pct:.1f}%")
    print(f"Profit factor:       {profit_factor:.2f}")
    print(f"Total fees paid:     ${trades['fees'].sum():.2f}")
    print(f"Total funding paid:  ${trades['funding_cost'].sum():.2f}")
    print(f"Avg net PnL/trade:   ${trades['net_pnl'].mean():.2f}")
    print("=" * 60)
    print("\nSaved: trade_log.csv, equity_curve.csv")


def main():
    parser = argparse.ArgumentParser(description="SOLUSDT Multi-Factor Backtest")
    parser.add_argument("--days",    type=int,   default=365,   help="Days of history to test (ignored if --year set)")
    parser.add_argument("--balance", type=float, default=100.0, help="Starting balance in USDT")
    parser.add_argument("--no-factors", action="store_true",    help="Pure TA mode — skip multi-factor gating")
    parser.add_argument("--year",    type=int,   default=None,  help="Test a full calendar year: 2023 | 2024 | 2025")
    parser.add_argument("--out",     type=str,   default="",    help="Output file prefix (e.g. '2024_')")
    args = parser.parse_args()

    if args.year:
        import calendar
        y = args.year
        start_ms = int(datetime(y, 1, 1).timestamp() * 1000)
        # end is Dec 31 of that year OR now — whichever is earlier
        end_of_year = int(datetime(y, 12, 31, 23, 59, 59).timestamp() * 1000)
        end_ms      = min(end_of_year, int(time.time() * 1000))
        days_label  = f"{y}"
        args.days   = (end_ms - start_ms) // (24 * 3600 * 1000)
    else:
        end_ms     = int(time.time() * 1000)
        start_ms   = end_ms - args.days * 24 * 60 * 60 * 1000
        days_label = f"{args.days}d"

    prefix = args.out or (f"{args.year}_" if args.year else "")

    print(f"{'='*60}")
    print(f"BACKTEST: {SYMBOL}  |  Period: {days_label}  |  Balance: ${args.balance}")
    print(f"{'='*60}")
    print(f"Fetching {args.days} days of {SYMBOL} data ({PRIMARY_TF}m, {HIGHER_TF}m, 4h)...")
    df15  = fetch_klines(SYMBOL, PRIMARY_TF, start_ms, end_ms)
    df1h  = fetch_klines(SYMBOL, HIGHER_TF,  start_ms, end_ms)
    dfbtc = fetch_klines("BTCUSDT", HIGHER_TF, start_ms, end_ms)
    df4h  = fetch_klines(SYMBOL, "240", start_ms, end_ms)

    print(f"Got {len(df15)} x 15m | {len(df1h)} x 1h | {len(df4h)} x 4h bars. Computing indicators...")
    ind15 = build_indicators(df15)
    ind1h = build_indicators(df1h)
    btc1h = build_indicators(dfbtc)
    ind4h = build_indicators(df4h)

    factor_data = {"regime_scores": {}, "funding": {}, "fng": {}}

    if not args.no_factors:
        print("\n── Pre-fetching multi-factor data ──────────────────────")
        factor_data["regime_scores"] = build_regime_scores(btc1h)
        print(f"  Regime scores   : {len(factor_data['regime_scores'])} bars ✓")
        factor_data["funding"] = fetch_historical_funding(SYMBOL, start_ms, end_ms)
        print(f"  Funding history : {len(factor_data['funding'])} records ✓")
        factor_data["fng"] = fetch_historical_fng(min(args.days, 365))
        print(f"  Fear & Greed    : {len(factor_data['fng'])} days ✓")
        print("────────────────────────────────────────────────────────\n")
    else:
        print("⚠️  --no-factors: pure TA mode")

    print("Running backtest...")
    trades, equity = run_backtest(ind15, ind1h, ind4h, btc1h, args.balance, factor_data)

    tlog  = f"{prefix}trade_log.csv"
    ecurv = f"{prefix}equity_curve.csv"
    trades.to_csv(tlog,  index=False)
    equity.to_csv(ecurv, index=False)

    summarize(trades, equity, args.balance)
    if not args.no_factors and "mf_score" in trades.columns and len(trades):
        print(f"Avg MF score (taken trades): {trades['mf_score'].mean():+.3f}")
    print(f"Saved: {tlog}, {ecurv}")


if __name__ == "__main__":
    main()
