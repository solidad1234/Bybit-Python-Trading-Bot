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
import time
import numpy as np
import pandas as pd
import requests
import talib

BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"

# ---- Strategy constants (mirrored from futures.py) ----
SYMBOL = "SOLUSDT"
PRIMARY_TF = "15"
HIGHER_TF = "60"

FUTURES_RISK_PER_TRADE = 0.02
MIN_REWARD_RATIO = 3.0
SIGNAL_STRENGTH_THRESHOLD = 5
MAX_DAILY_TRADES = 15
MAX_CONSECUTIVE_LOSSES = 3

TAKER_FEE = 0.00055          # Bybit USDT perpetual taker fee (approx, per side)
FUNDING_RATE_PER_8H = 0.0001  # 0.01% / 8h flat approximation - adjust if you pull real funding history


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
# Signal / stop / sizing logic - mirrors futures.py exactly
# ----------------------------------------------------------------------
def calculate_signal(row15, row1h, volatility):
    long_conditions = [
        row15["rsi"] < 40,
        row1h["rsi"] < 50,
        row15["macd"] > row15["macd_signal"],
        row15["close"] > row15["ema_21"] * 0.998,
        row15["volume_ratio"] > 1.3,
        volatility > 0.02,
        row15["adx"] > 20,
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
    long_score = sum(bool(c) for c in long_conditions)
    short_score = sum(bool(c) for c in short_conditions)

    if long_score >= 5:
        return {"signal": "LONG", "strength": long_score, "leverage": 10.0}
    if short_score >= 6:
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
def run_backtest(ind15, ind1h, btc1h, starting_balance):
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

        # reset daily counter
        day = pd.Timestamp(t).date()
        if day != last_day:
            daily_trades = 0
            last_day = day

        # align 1h data: last 1h bar closed at or before t
        idx1h = ind1h["time"].searchsorted(t, side="right") - 1
        if idx1h < warmup:
            continue
        row1h = ind1h.iloc[idx1h]

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

            # partial profit-taking / breakeven (evaluated on close, approximating live bot)
            pnl_pct = ((current_price - position["entry"]) / position["entry"]
                       if direction == "LONG" else
                       (position["entry"] - current_price) / position["entry"])

            if pnl_pct >= 0.015 and not position["stop_moved_to_be"]:
                position["stop"] = position["entry"]
                position["stop_moved_to_be"] = True

            # trailing stop (price-based, mirrors implement_trailing_stop)
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

            # check stop / target against this bar's range
            exit_price = None
            result = None
            if direction == "LONG":
                if lo <= position["stop"]:
                    exit_price, result = position["stop"], "LOSS" if position["stop"] < position["entry"] else "BE/WIN"
                elif hi >= position["target"]:
                    exit_price, result = position["target"], "WIN"
            else:
                if hi >= position["stop"]:
                    exit_price, result = position["stop"], "LOSS" if position["stop"] > position["entry"] else "BE/WIN"
                elif lo <= position["target"]:
                    exit_price, result = position["target"], "WIN"

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
            if row1h["adx"] < 20:
                pass  # regime filter: flat market, skip
            elif daily_trades >= MAX_DAILY_TRADES:
                pass
            elif consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                pass
            else:
                signal = calculate_signal(row15, row1h, volatility)
                if signal["signal"] == "LONG" and btc_bear:
                    signal["signal"] = None

                if signal["signal"] and signal["strength"] >= SIGNAL_STRENGTH_THRESHOLD and balance >= 5:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365, help="How many days of history to test")
    parser.add_argument("--balance", type=float, default=100.0, help="Starting balance in USDT")
    args = parser.parse_args()

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - args.days * 24 * 60 * 60 * 1000

    print(f"Fetching {args.days} days of {SYMBOL} data ({PRIMARY_TF}m and {HIGHER_TF}m)...")
    df15 = fetch_klines(SYMBOL, PRIMARY_TF, start_ms, end_ms)
    df1h = fetch_klines(SYMBOL, HIGHER_TF, start_ms, end_ms)
    print("Fetching BTCUSDT 1h for correlation filter...")
    dfbtc = fetch_klines("BTCUSDT", HIGHER_TF, start_ms, end_ms)

    print(f"Got {len(df15)} x 15m bars, {len(df1h)} x 1h bars. Computing indicators...")
    ind15 = build_indicators(df15)
    ind1h = build_indicators(df1h)
    btc1h = build_indicators(dfbtc)

    print("Running backtest...")
    trades, equity = run_backtest(ind15, ind1h, btc1h, args.balance)

    trades.to_csv("trade_log.csv", index=False)
    equity.to_csv("equity_curve.csv", index=False)

    summarize(trades, equity, args.balance)


if __name__ == "__main__":
    main()
