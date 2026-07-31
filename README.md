# Bybit Python Trading Bot

A high-performance algorithmic trading bot built for Bybit, utilizing technical analysis (TA-LIB) for multi-timeframe trend following and momentum strategies.

## Features

- **Technical Market Analysis**: Uses the `TA-LIB` library for precise calculations.
- **Indicators Used**: RSI, MACD, Moving Averages (MA, EMA, SMA), ATR, Bollinger Bands, and Stochastic RSI based on multi-timeframe Kline data (15m, 1h, 4h).
- **BTC Trend Filter**: Monitors Bitcoin's momentum to filter out false signals in altcoin pairs.
- **Advanced Risk Management**: 
  - Uses Average True Range (ATR) to dynamically calculate realistic Stop Loss and Take Profit levels.
  - Implements trailing stops based on real-time price action to lock in profits.
  - Scales out of positions (25% take profits at 2% and 4% gains).
- **Robust State Management**: Uses **SQLite3 (`trading_state.db`)** to persist open position data. If the bot disconnects or restarts, it will seamlessly recover the active trade and continue managing it without data loss.
- **Dual Execution Loop**: 
  - **Fast Loop (Every 10s)**: Rapidly queries current price tickers to manage open positions, trigger trailing stops, and execute partial take-profits instantly.
  - **Slow Loop (Every 5m)**: Runs heavier technical analysis (TA-LIB) calculations to detect new trade entries on candle closes.

## Setup

1. Rename `.env.example` to `.env` and add your Bybit `API_KEY` and `API_SECRET`.
2. Ensure you have the required dependencies installed (including `TA-LIB` C++ binaries and python wrapper, `pybit`, and `numpy`).
3. Run the bot: `python futures.py`
