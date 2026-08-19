"""
MetaTrader 5 (MT5) Forex Trading Bot
====================================
Focus: Multi-Factor Forex Trading with MT5 API Integration,
Forex Google News Sentiment, Support & Resistance Anchoring, Spread Guard,
Session Timing Controls, 4h Trend Filtering, and Dynamic Pip Lot Sizing.

Mirrors the architecture of futures.py, adapted for Forex market mechanics.
"""

import sys
import os
import time
import json
import re
import sqlite3
import numpy as np
import argparse
from datetime import datetime, timezone, timedelta
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Try importing feedparser for News RSS
try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    feedparser = None
    HAS_FEEDPARSER = False

# Try importing MetaTrader 5 Python library
try:
    import MetaTrader5 as mt5
    HAS_MT5_LIB = True
except ImportError:
    mt5 = None
    HAS_MT5_LIB = False

# Try importing TA-Lib
try:
    import talib
    HAS_TALIB = True
except ImportError:
    talib = None
    HAS_TALIB = False

# Try importing Support & Resistance factor from factors module
try:
    from factors.support_resistance import get_sr_score
    HAS_SR_FACTOR = True
except ImportError:
    get_sr_score = None
    HAS_SR_FACTOR = False

# Load environment variables
load_dotenv()
MT5_LOGIN = os.getenv("MT5_LOGIN", "")
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "MetaQuotes-Demo")
MT5_PATH = os.getenv("MT5_PATH", "")

# Forex Configuration
TRADE_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

# Per-symbol Forex specs (pip size, lot step, min lot, contract size)
# quote_is_usd=True  → quote currency is USD (pip value = contract_size * pip_size)
# quote_is_usd=False → quote currency is NOT USD (pip value must be divided by current price)
SYMBOL_SPECS = {
    "EURUSD": {"pip_size": 0.0001, "min_lot": 0.01, "lot_step": 0.01, "contract_size": 100000, "quote_is_usd": True},
    "GBPUSD": {"pip_size": 0.0001, "min_lot": 0.01, "lot_step": 0.01, "contract_size": 100000, "quote_is_usd": True},
    "USDJPY": {"pip_size": 0.01,   "min_lot": 0.01, "lot_step": 0.01, "contract_size": 100000, "quote_is_usd": False},
    "AUDUSD": {"pip_size": 0.0001, "min_lot": 0.01, "lot_step": 0.01, "contract_size": 100000, "quote_is_usd": True},
    "USDCAD": {"pip_size": 0.0001, "min_lot": 0.01, "lot_step": 0.01, "contract_size": 100000, "quote_is_usd": False},
}

# Bot-level constants
BOT_MAGIC_NUMBER   = 998877  # Unique magic number to identify this bot's orders in MT5
MAX_DAILY_LOSS_PCT = 0.05    # Halt trading if session PnL drops more than 5% of start balance

# Forex Google News Query Mapping
FOREX_NEWS_QUERIES = {
    "MACRO":  "US Dollar Fed inflation interest rates forex",
    "EURUSD": "EURUSD Euro US Dollar forex news",
    "GBPUSD": "GBPUSD British Pound Bank of England forex news",
    "USDJPY": "USDJPY Japanese Yen Bank of Japan forex news",
    "AUDUSD": "AUDUSD Australian Dollar RBA forex news",
    "USDCAD": "USDCAD Canadian Dollar Bank of Canada forex news",
}

# Forex Keywords for Sentiment Analysis
POSITIVE_KEYWORDS = {
    "hike", "hawkish", "growth", "surge", "surging", "rally", "rallies",
    "bullish", "strong", "gains", "gain", "outperform", "recovery", "rebound",
    "stimulus", "optimism", "record", "expansion", "profit"
}

NEGATIVE_KEYWORDS = {
    "cut", "dovish", "recession", "drop", "drops", "plunge", "plunges",
    "bearish", "weak", "decline", "declining", "inflation", "crisis",
    "slump", "deficit", "unemployment", "risk", "warning", "downside"
}

primary_timeframe = "15m"   # 15 minutes
higher_timeframe  = "1h"    # 1 hour
macro_timeframe   = "4h"    # 4 hours

# Risk Management Controls
forex_risk_per_trade     = 0.015  # 1.5% account risk per trade
max_spread_pips          = 2.5    # Max allowed broker spread in pips
min_reward_ratio         = 1.5    # 1.5:1 R:R target for Forex intraday
min_volatility_threshold = 0.0010 # Minimum volatility required

# Multi-Factor Consensus Weights
WEIGHTS = {
    "trend_alignment":    0.25,
    "support_resistance": 0.25,
    "technical":          0.20,
    "news":               0.15,
    "spread_volatility":  0.15,
}

# Session Timing Guards (UTC)
london_open_utc    = 7.0    # 07:00 UTC (London session start)
ny_close_utc       = 16.5   # 16:30 UTC (New York overlap end)
rollover_start_utc = 21.9   # 21:54 UTC
rollover_end_utc   = 22.3   # 22:18 UTC

# Global Bot State
bot_state = {
    'last_trade_time': None,
    'position': None,
    'daily_trades': 0,
    'total_trades': 0,
    'winning_trades': 0,
    'consecutive_losses': 0,
    'max_consecutive_losses': 3,
    'session_start': datetime.now(timezone.utc),
    'available_balance': 1000.0,
    'session_start_balance': 0.0,  # Set to 0.0 as sentinel — populated on first balance fetch
    'session_pnl': 0.0,
}


def get_forex_news_score(symbol: str) -> dict:
    """
    Fetch Google News RSS for USD Macro & Pair Specific Forex News
    """
    if not HAS_FEEDPARSER:
        return {"score": 0.0, "confidence": 0.0, "block_long_only": False, "details": {"reason": "feedparser not installed"}}

    try:
        # Macro USD query
        macro_url = f"https://news.google.com/rss/search?q={quote_plus(FOREX_NEWS_QUERIES['MACRO'])}&hl=en-US&gl=US&ceid=US:en"
        macro_feed = feedparser.parse(macro_url)

        # Pair query
        query_str = FOREX_NEWS_QUERIES.get(symbol, f"{symbol} forex news")
        pair_url = f"https://news.google.com/rss/search?q={quote_plus(query_str)}&hl=en-US&gl=US&ceid=US:en"
        pair_feed = feedparser.parse(pair_url)

        entries = (macro_feed.entries[:5] if macro_feed.entries else []) + (pair_feed.entries[:5] if pair_feed.entries else [])
        if not entries:
            return {"score": 0.0, "confidence": 0.2, "block_long_only": False, "details": {"reason": "No news found"}}

        net_score = 0.0
        for entry in entries:
            title = entry.get("title", "").lower()
            words = set(re.findall(r"[a-z]+", title))
            pos_count = len(words & POSITIVE_KEYWORDS)
            neg_count = len(words & NEGATIVE_KEYWORDS)
            article_score = (pos_count - neg_count) * 0.25
            net_score += max(-1.0, min(1.0, article_score))

        final_score = max(-1.0, min(1.0, net_score / len(entries)))
        block_long_only = final_score <= -0.5

        return {
            "score": round(final_score, 3),
            "confidence": 0.70,
            "block_long_only": block_long_only,
            "block_reason": f"Negative Forex News Score ({final_score:.2f})" if block_long_only else "",
            "details": {"articles_count": len(entries), "score": round(final_score, 3)}
        }
    except Exception as e:
        return {"score": 0.0, "confidence": 0.0, "block_long_only": False, "details": {"err": str(e)}}


class MT5ForexBot:
    def __init__(self, dry_run=False):
        print(f"🚀 Initializing MT5 Forex Trading Bot...")
        self.dry_run = dry_run
        self.connected = False
        self.state_file = 'mt5_trading_state.json'
        
        self.init_db()
        self.init_mt5_connection()
        self.load_position_state()
        self.initialize_balance()

    def init_db(self):
        """Initialize SQLite database for state persistence"""
        self.conn = sqlite3.connect('mt5_trading_state.db', timeout=30.0, check_same_thread=False)
        self.cursor = self.conn.cursor()
        try:
            self.cursor.execute('PRAGMA journal_mode=WAL;')
            self.cursor.execute('PRAGMA busy_timeout=30000;')
        except Exception as e:
            print(f"⚠️ Could not set WAL/busy_timeout PRAGMA: {e}")

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS position (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                direction TEXT,
                size REAL,
                entry REAL,
                stop REAL,
                target REAL,
                order_id TEXT,
                timestamp TEXT,
                exit_25_taken INTEGER,
                exit_50_taken INTEGER,
                stop_moved_to_be INTEGER,
                original_stop REAL,
                highest_price REAL,
                lowest_price REAL
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_log (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol             TEXT,
                entry_time         TEXT,
                exit_time          TEXT,
                direction          TEXT,
                entry_price        REAL,
                exit_price         REAL,
                size               REAL,
                pnl                REAL,
                result             TEXT,
                ta_signal_strength REAL,
                spread_pips        REAL,
                volatility         REAL,
                atr_15m            REAL,
                news_score         REAL,
                sr_scenario        TEXT,
                session_window     TEXT
            )
        ''')

        # --- Schema migrations: safely add columns missing from older DB files ---
        _migrations = [
            ("trade_log", "news_score",         "REAL"),
            ("trade_log", "sr_scenario",        "TEXT"),
            ("trade_log", "session_window",      "TEXT"),
            ("trade_log", "ta_signal_strength",  "REAL"),
            ("trade_log", "spread_pips",         "REAL"),
            ("trade_log", "volatility",          "REAL"),
            ("trade_log", "atr_15m",             "REAL"),
        ]
        existing_cols = {}
        for tbl in ("trade_log", "position"):
            self.cursor.execute(f"PRAGMA table_info({tbl})")
            existing_cols[tbl] = {row[1] for row in self.cursor.fetchall()}
        for tbl, col, col_type in _migrations:
            if col not in existing_cols.get(tbl, set()):
                try:
                    self.cursor.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {col_type}")
                    print(f"🔧 DB migration: added column '{col}' to table '{tbl}'")
                except Exception as mig_err:
                    print(f"⚠️ DB migration notice for {tbl}.{col}: {mig_err}")

        self.conn.commit()

    def init_mt5_connection(self):
        """Establish MT5 Terminal Connection"""
        if not HAS_MT5_LIB:
            print("⚠️ MetaTrader5 library not installed (`pip install MetaTrader5`). Running in DRY-RUN mode.")
            self.dry_run = True
            return

        print("🔌 Connecting to MetaTrader 5 Terminal...")
        init_args = {}
        if MT5_PATH and os.path.exists(MT5_PATH):
            init_args["path"] = MT5_PATH

        try:
            if not mt5.initialize(**init_args):
                print(f"❌ mt5.initialize() failed: {mt5.last_error()}. Running in DRY-RUN mode.")
                self.dry_run = True
                return

            if MT5_LOGIN and MT5_PASSWORD:
                login_id = int(MT5_LOGIN)
                authorized = mt5.login(login=login_id, password=MT5_PASSWORD, server=MT5_SERVER)
                if not authorized:
                    print(f"❌ MT5 login failed for user {login_id} on {MT5_SERVER}: {mt5.last_error()}")
                    self.dry_run = True
                    return
                print(f"✅ Logged into MT5 Account #{login_id} on server '{MT5_SERVER}'")
            else:
                acc_info = mt5.account_info()
                if acc_info is not None:
                    print(f"✅ Connected to active MT5 Account #{acc_info.login} ({acc_info.company})")
                else:
                    print("⚠️ Connected to local MT5 terminal (no login credentials specified in .env)")

            self.connected = True

        except Exception as e:
            print(f"❌ Error during MT5 connection initialization: {e}. Switching to DRY-RUN mode.")
            self.dry_run = True

    def initialize_balance(self):
        """Fetch account balance from MT5 or set default"""
        balance = 1000.0
        if self.connected and HAS_MT5_LIB:
            try:
                acc_info = mt5.account_info()
                if acc_info is not None:
                    balance = float(acc_info.margin_free or acc_info.balance)
            except Exception as e:
                print(f"⚠️ Could not fetch MT5 account info: {e}")

        bot_state['available_balance'] = balance
        if bot_state['session_start_balance'] == 0.0:  # Only set once per session (sentinel check)
            bot_state['session_start_balance'] = balance

        print(f"💰 Forex Balance Initialized:")
        print(f"   Available Free Margin: ${balance:.2f}")
        print(f"   Risk Per Trade: {forex_risk_per_trade*100:.1f}% = ${balance * forex_risk_per_trade:.2f}")
        print(f"   Mode: {'LIVE / DEMO MT5 CONNECTED' if self.connected else 'DRY-RUN SIMULATION'}")

    def save_position_state(self):
        """Save active position state to SQLite"""
        pos = bot_state['position']
        if not pos:
            return
        try:
            self.cursor.execute('DELETE FROM position')
            self.cursor.execute('''
                INSERT INTO position (
                    symbol, direction, size, entry, stop, target,
                    order_id, timestamp, exit_25_taken, exit_50_taken,
                    stop_moved_to_be, original_stop, highest_price, lowest_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pos['symbol'], pos['direction'], pos['size'], pos['entry'], pos['stop'],
                pos['target'], str(pos.get('order_id', '')), str(pos['timestamp']),
                int(pos.get('exit_25_taken', False)), int(pos.get('exit_50_taken', False)),
                int(pos.get('stop_moved_to_be', False)), pos.get('original_stop', pos['stop']),
                pos.get('highest_price', pos['entry']), pos.get('lowest_price', pos['entry'])
            ))
            self.conn.commit()
        except Exception as e:
            print(f"⚠️ Error saving position state to SQLite: {e}")

    def load_position_state(self):
        """Load position state from SQLite"""
        try:
            self.cursor.execute('SELECT * FROM position LIMIT 1')
            row = self.cursor.fetchone()
            if row:
                columns = [col[0] for col in self.cursor.description]
                r = dict(zip(columns, row))
                bot_state['position'] = {
                    'symbol': r.get('symbol'),
                    'direction': r.get('direction'),
                    'size': r.get('size'),
                    'entry': r.get('entry'),
                    'stop': r.get('stop'),
                    'target': r.get('target'),
                    'order_id': r.get('order_id'),
                    'timestamp': r.get('timestamp') or datetime.now(timezone.utc).isoformat(),
                    'exit_25_taken': bool(r.get('exit_25_taken', False)),
                    'exit_50_taken': bool(r.get('exit_50_taken', False)),
                    'stop_moved_to_be': bool(r.get('stop_moved_to_be', False)),
                    'original_stop': r.get('original_stop'),
                    'highest_price': r.get('highest_price'),
                    'lowest_price': r.get('lowest_price')
                }
                print(f"🔄 Recovered active MT5 position: {bot_state['position']['direction']} {bot_state['position']['symbol']}")
            else:
                bot_state['position'] = None
        except Exception as e:
            print(f"⚠️ Error reading position state from SQLite: {e}")
            bot_state['position'] = None

    def clear_position_state(self):
        """Clear active position from SQLite"""
        try:
            self.cursor.execute('DELETE FROM position')
            self.conn.commit()
        except Exception as e:
            print(f"⚠️ Error clearing position state: {e}")
        bot_state['position'] = None

    def check_spread_and_session(self, symbol):
        """Check Forex spread limits and active session timing"""
        now_utc = datetime.now(timezone.utc)
        utc_hour_float = now_utc.hour + (now_utc.minute / 60.0)

        # 1. Rollover Check (21:54 - 22:18 UTC)
        if rollover_start_utc <= utc_hour_float <= rollover_end_utc:
            print(f" 🚫 BLOCKED: Daily broker rollover window (21:55 - 22:15 UTC). Spreads spike.")
            return False, "ROLLOVER_WINDOW", 0.0

        # 2. Active Session Window Check (07:00 - 16:30 UTC)
        in_session = (london_open_utc <= utc_hour_float <= ny_close_utc)
        session_label = "LONDON_NY_OVERLAP" if in_session else "OUT_OF_SESSION"

        if not in_session:
            print(f" 🚫 BLOCKED: Outside London/NY session ({utc_hour_float:.2f} UTC). Active window: 07:00–16:30 UTC.")
            return False, "OUT_OF_SESSION", 0.0

        # 3. Real-Time Spread Check
        pip_size = SYMBOL_SPECS.get(symbol, {}).get("pip_size", 0.0001)
        spread_pips = 1.0  # default fallback

        if self.connected and HAS_MT5_LIB:
            info = mt5.symbol_info(symbol)
            if info is not None:
                if not info.visible:
                    mt5.symbol_select(symbol, True)
                spread_pips = (info.ask - info.bid) / pip_size
            else:
                print(f" ⚠️ Warning: Could not fetch symbol info for {symbol}")

        print(f" 🔍 {symbol} Real-time Spread: {spread_pips:.1f} pips (Max: {max_spread_pips} pips)")
        if spread_pips > max_spread_pips:
            print(f" 🚫 BLOCKED: Spread {spread_pips:.1f} pips exceeds threshold {max_spread_pips} pips")
            return False, f"HIGH_SPREAD_{spread_pips:.1f}", spread_pips

        return True, session_label, spread_pips

    def fetch_multi_timeframe_data(self, symbol):
        """Fetch multi-timeframe candles (15m, 1h, 4h)"""
        data = {}
        tf_map = {}
        if HAS_MT5_LIB:
            tf_map = {
                "15m": mt5.TIMEFRAME_M15,
                "1h":  mt5.TIMEFRAME_H1,
                "4h":  mt5.TIMEFRAME_H4
            }

        for tf_name in ["15m", "1h", "4h"]:
            if self.connected and HAS_MT5_LIB:
                try:
                    rates = mt5.copy_rates_from_pos(symbol, tf_map[tf_name], 0, 100)
                    if rates is not None and len(rates) > 0:
                        data[tf_name] = {
                            'close': rates['close'].astype(float),
                            'high':  rates['high'].astype(float),
                            'low':   rates['low'].astype(float),
                            'volume': rates['tick_volume'].astype(float),
                            'timestamp': rates['time']
                        }
                except Exception as e:
                    print(f"❌ Error fetching {tf_name} MT5 rates for {symbol}: {e}")

            # Fallback simulated data generator for dry-run/testing
            if tf_name not in data:
                np.random.seed(int(time.time() * 1000) % 100000)
                base_price = 1.0850 if symbol == "EURUSD" else (1.2650 if symbol == "GBPUSD" else 155.0)
                noise = np.random.normal(0, 0.0005, 100).cumsum()
                closes = base_price + noise
                highs  = closes + np.abs(np.random.normal(0, 0.0002, 100))
                lows   = closes - np.abs(np.random.normal(0, 0.0002, 100))
                vols   = np.random.randint(100, 1000, 100).astype(float)
                data[tf_name] = {
                    'close': closes,
                    'high': highs,
                    'low': lows,
                    'volume': vols,
                    'timestamp': np.arange(100)
                }

        # Format map keys for compatibility with factors/support_resistance.py ("60" for 1h, "240" for 4h)
        data["60"]  = data["1h"]
        data["240"] = data["4h"]

        return data

    def calculate_indicators(self, data):
        """Calculate technical indicators using TA-Lib or fallback numpy implementation"""
        indicators = {}

        for tf_name in ["15m", "1h", "4h"]:
            closes  = data[tf_name]['close']
            highs   = data[tf_name]['high']
            lows    = data[tf_name]['low']
            volumes = data[tf_name]['volume']

            if HAS_TALIB:
                rsi       = talib.RSI(closes, timeperiod=14)[-1]
                macd_line, macd_signal, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
                ema_21    = talib.EMA(closes, timeperiod=21)[-1]
                ema_50    = talib.EMA(closes, timeperiod=50)[-1]
                atr       = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
                adx       = talib.ADX(highs, lows, closes, timeperiod=14)[-1]
                stoch_k, stoch_d = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowd_period=3)
                volume_sma = talib.SMA(volumes, timeperiod=20)[-1]
            else:
                # --- Helper: proper exponential moving average ---
                def _ema(arr, n):
                    k = 2.0 / (n + 1)
                    val = float(np.mean(arr[:n]))
                    for v in arr[n:]:
                        val = float(v) * k + val * (1.0 - k)
                    return val

                # --- Wilder's RSI (exponential smoothing, not simple average) ---
                diff = np.diff(closes)
                gains = np.where(diff > 0, diff, 0.0)
                losses = np.where(diff < 0, -diff, 0.0)
                rsi_period = 14
                avg_gain = float(np.mean(gains[:rsi_period]))
                avg_loss = float(np.mean(losses[:rsi_period]))
                for i in range(rsi_period, len(gains)):
                    avg_gain = (avg_gain * (rsi_period - 1) + gains[i]) / rsi_period
                    avg_loss = (avg_loss * (rsi_period - 1) + losses[i]) / rsi_period
                rs = avg_gain / (avg_loss + 1e-9)
                rsi = 100.0 - (100.0 / (1.0 + rs))

                # --- Proper EMA values ---
                ema_21 = _ema(closes, 21)
                ema_50 = _ema(closes, 50)

                # --- Proper MACD: EMA12 - EMA26, Signal = EMA9 of MACD line ---
                ema_12 = _ema(closes, 12)
                ema_26 = _ema(closes, 26)
                macd_line_val = ema_12 - ema_26
                # Build enough MACD history to compute 9-period signal EMA
                start_idx = max(26, len(closes) - 20)
                macd_vals = [_ema(closes[:i + 1], 12) - _ema(closes[:i + 1], 26)
                             for i in range(start_idx, len(closes))]
                macd_signal_val = _ema(np.array(macd_vals), 9) if len(macd_vals) >= 9 else macd_vals[-1]
                macd_hist_val = macd_line_val - macd_signal_val
                macd_line   = np.array([macd_line_val])
                macd_signal = np.array([macd_signal_val])
                macd_hist   = np.array([macd_hist_val])

                # --- ATR: proper True Range (Wilder's smoothing) ---
                tr_arr = np.maximum(
                    highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]),
                               np.abs(lows[1:]  - closes[:-1]))
                )
                atr_period = 14
                atr = float(np.mean(tr_arr[:atr_period]))
                for v in tr_arr[atr_period:]:
                    atr = (atr * (atr_period - 1) + float(v)) / atr_period

                # --- Stochastic %K / %D from actual high/low range ---
                stoch_period = 14
                recent_high = float(np.max(highs[-stoch_period:]))
                recent_low  = float(np.min(lows[-stoch_period:]))
                stoch_k_val = ((float(closes[-1]) - recent_low) /
                               (recent_high - recent_low + 1e-9)) * 100.0
                stoch_k = np.array([stoch_k_val])
                stoch_d = np.array([stoch_k_val])  # Simplified: no 3-bar smoothing history

                # --- ADX: directional movement index (simplified single-window DX) ---
                dm_plus  = np.where((highs[1:] - highs[:-1]) > (lows[:-1] - lows[1:]),
                                    np.maximum(highs[1:] - highs[:-1], 0.0), 0.0)
                dm_minus = np.where((lows[:-1] - lows[1:]) > (highs[1:] - highs[:-1]),
                                    np.maximum(lows[:-1] - lows[1:], 0.0), 0.0)
                atr_window = np.mean(tr_arr[-atr_period:])
                di_plus  = 100.0 * np.mean(dm_plus[-atr_period:])  / (atr_window + 1e-9)
                di_minus = 100.0 * np.mean(dm_minus[-atr_period:]) / (atr_window + 1e-9)
                dx = 100.0 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-9)
                adx = dx  # Single DX bar as ADX proxy

                volume_sma = np.mean(volumes[-20:])

            indicators[tf_name] = {
                'rsi': float(rsi),
                'macd': float(macd_line[-1]),
                'macd_signal': float(macd_signal[-1]),
                'macd_histogram': float(macd_hist[-1]),
                'ema_21': float(ema_21),
                'ema_50': float(ema_50),
                'atr': float(atr),
                'adx': float(adx),
                'stoch_k': float(stoch_k[-1]),
                'stoch_d': float(stoch_d[-1]),
                'volume_ratio': float(volumes[-1] / volume_sma if volume_sma > 0 else 1.0)
            }

        current_price = float(data["15m"]['close'][-1])
        price_24h_ago = float(data["1h"]['close'][-24]) if len(data["1h"]['close']) >= 24 else float(data["1h"]['close'][0])
        volatility = abs((current_price - price_24h_ago) / price_24h_ago)

        return indicators, current_price, volatility

    def evaluate_multi_factor_consensus(self, symbol, indicators, current_price, volatility, data, spread_pips):
        """
        Multi-Factor Consensus Aggregator (Technical + S/R + Forex News + Spread/Volatility + 4h Trend)
        """
        # 1. 4h Trend Score
        ind4h = indicators.get("4h", {})
        ema21_4h = ind4h.get("ema_21", 0.0)
        ema50_4h = ind4h.get("ema_50", 0.0)
        trend_bullish = ema21_4h > ema50_4h if ema50_4h > 0 else True
        trend_score = 1.0 if trend_bullish else -1.0

        # 2. Technical Signal
        long_conds = [
            indicators["15m"]["rsi"] < 45,
            indicators["1h"]["rsi"] < 52,
            indicators["15m"]["macd"] > indicators["15m"]["macd_signal"],
            current_price > indicators["15m"]["ema_21"] * 0.999,
            volatility > min_volatility_threshold,
            indicators["15m"]["adx"] > 18,
            indicators["15m"]["stoch_k"] < 35,
        ]
        short_conds = [
            indicators["15m"]["rsi"] > 55,
            indicators["1h"]["rsi"] > 48,
            indicators["15m"]["macd"] < indicators["15m"]["macd_signal"],
            current_price < indicators["15m"]["ema_21"] * 1.001,
            volatility > min_volatility_threshold,
            indicators["15m"]["adx"] > 18,
            indicators["15m"]["stoch_k"] > 65,
        ]
        long_ta  = sum(long_conds)
        short_ta = sum(short_conds)
        ta_score = (long_ta - short_ta) / 7.0

        # 3. Support & Resistance Score
        sr_data = {"score": 0.0, "scenario": "MID_RANGE", "suggested_stop": None, "suggested_target": None}
        if HAS_SR_FACTOR and get_sr_score is not None:
            try:
                sr_data = get_sr_score(symbol, current_price, indicators, data)
            except Exception as e:
                print(f" ⚠️ S/R Factor evaluation notice: {e}")
        sr_score = sr_data.get("score", 0.0)

        # 4. Forex News Score
        news_data = get_forex_news_score(symbol)
        news_score = news_data.get("score", 0.0)
        block_long_only = news_data.get("block_long_only", False)

        # 5. Spread / Volatility Quality Score
        spread_score = max(0.0, 1.0 - (spread_pips / max_spread_pips))

        # Composite Multi-Factor Score calculation
        final_score = (
            WEIGHTS["trend_alignment"] * trend_score +
            WEIGHTS["support_resistance"] * sr_score +
            WEIGHTS["technical"] * ta_score +
            WEIGHTS["news"] * news_score +
            WEIGHTS["spread_volatility"] * spread_score
        )

        print(f" 🔬 Multi-Factor Consensus ({symbol}):")
        print(f"    Trend: {trend_score:+.2f} | S/R ({sr_data.get('scenario')}): {sr_score:+.2f} | TA: {ta_score:+.2f} | News: {news_score:+.2f}")
        print(f"    Final Score: {final_score:+.3f}")

        # Signal determination
        signal = None
        if final_score >= 0.25 and trend_bullish and not block_long_only:
            signal = "LONG"
        elif final_score <= -0.25 and not trend_bullish:
            signal = "SHORT"

        if block_long_only and final_score >= 0.25:
            print(f" 🚫 LONG Signal blocked by Forex News Guard: {news_data.get('block_reason')}")

        return {
            "signal": signal,
            "final_score": round(final_score, 3),
            "news_score": news_score,
            "sr_scenario": sr_data.get("scenario", "MID_RANGE"),
            "suggested_stop": sr_data.get("suggested_stop"),
            "suggested_target": sr_data.get("suggested_target"),
            "strength": max(long_ta, short_ta)
        }

    def calculate_forex_position_size(self, symbol, entry_price, stop_loss_price):
        """Calculate Forex position size in MT5 Standard Lots"""
        specs = SYMBOL_SPECS.get(symbol, SYMBOL_SPECS["EURUSD"])
        pip_size      = specs["pip_size"]
        min_lot       = specs["min_lot"]
        lot_step      = specs["lot_step"]
        contract_size = specs["contract_size"]

        balance = bot_state['available_balance']
        risk_amount = balance * forex_risk_per_trade

        stop_distance = abs(entry_price - stop_loss_price)
        if stop_distance <= 0:
            print("❌ Invalid stop loss distance")
            return None

        pip_distance = stop_distance / pip_size

        # Pip value in USD depends on quote currency:
        #   USD-quoted (EURUSD, GBPUSD, AUDUSD): pip_value = contract_size * pip_size  → ~$10/pip/lot
        #   Non-USD quote (USDJPY, USDCAD):      pip_value = (contract_size * pip_size) / entry_price
        if specs.get("quote_is_usd", True):
            pip_value_per_lot = contract_size * pip_size
        else:
            pip_value_per_lot = (contract_size * pip_size) / entry_price

        raw_lots = risk_amount / (pip_distance * pip_value_per_lot)
        lots = round(raw_lots / lot_step) * lot_step
        lots = max(min_lot, round(lots, 2))

        print(f" 📊 Position Size Calculation ({symbol}):")
        print(f"    Available Balance: ${balance:.2f}")
        print(f"    Risk Amount ({forex_risk_per_trade*100:.1f}%): ${risk_amount:.2f}")
        print(f"    Stop Distance: {pip_distance:.1f} pips (${stop_distance:.5f})")
        print(f"    Calculated Lots: {lots:.2f} standard lots")

        return {
            'lots': lots,
            'risk_amount': risk_amount,
            'pip_distance': pip_distance,
            'pip_value': pip_value_per_lot * lots
        }

    def execute_trade(self, symbol, direction, entry_price, indicators, signal_data, spread_pips=0.0, volatility=0.0):
        """Execute Forex Trade (via MT5 or Dry-Run)"""
        pip_size = SYMBOL_SPECS.get(symbol, {}).get("pip_size", 0.0001)
        atr_15m  = indicators["15m"]["atr"]

        # Use S/R anchored stops if available, otherwise calculate ATR stops
        sug_stop   = signal_data.get("suggested_stop")
        sug_target = signal_data.get("suggested_target")

        if sug_stop and sug_target and signal_data.get("sr_scenario") != "MID_RANGE":
            stop_loss   = sug_stop
            take_profit = sug_target
            stop_pips   = abs(entry_price - stop_loss) / pip_size
            reward_pips = abs(take_profit - entry_price) / pip_size
            print(f" 🎯 Using S/R-Anchored Stop ({signal_data.get('sr_scenario')}): SL={stop_loss:.5f}, TP={take_profit:.5f}")
        else:
            stop_pips   = max(15.0, (1.5 * atr_15m) / pip_size)
            reward_pips = stop_pips * min_reward_ratio
            if direction == "LONG":
                stop_loss   = entry_price - (stop_pips * pip_size)
                take_profit = entry_price + (reward_pips * pip_size)
            else:
                stop_loss   = entry_price + (stop_pips * pip_size)
                take_profit = entry_price - (reward_pips * pip_size)

        pos_size = self.calculate_forex_position_size(symbol, entry_price, stop_loss)
        if not pos_size:
            return False

        lots = pos_size['lots']

        # Live MT5 Order Execution
        order_id = f"DRY_RUN_{int(time.time())}"
        if self.connected and HAS_MT5_LIB and not self.dry_run:
            info = mt5.symbol_info(symbol)
            trade_type = mt5.ORDER_TYPE_BUY if direction == "LONG" else mt5.ORDER_TYPE_SELL
            price = info.ask if direction == "LONG" else info.bid

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lots,
                "type": trade_type,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 10,
                "magic": BOT_MAGIC_NUMBER,
                "comment": "MT5 Forex Bot Signal",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            res = mt5.order_send(request)
            if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
                order_id = str(res.order)
                entry_price = float(res.price)  # Use actual MT5 fill price, not candle close
                print(f"✅ MT5 Order Successfully Placed! Order ID #{order_id} | Fill: {entry_price:.5f}")
            else:
                err_msg = mt5.last_error() if res is None else res.comment
                print(f"❌ MT5 Order Execution Failed: {err_msg}")
                return False

        bot_state['position'] = {
            'symbol': symbol,
            'direction': direction,
            'size': lots,
            'entry': entry_price,
            'stop': stop_loss,
            'target': take_profit,
            'original_stop': stop_loss,
            'order_id': order_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'highest_price': entry_price,
            'lowest_price': entry_price,
            'exit_25_taken': False,
            'exit_50_taken': False,
            'stop_moved_to_be': False,
            # Runtime metadata (not persisted to DB — defaults to 0 on crash recovery)
            'spread_pips': spread_pips,
            'volatility': volatility,
            'atr_15m': indicators["15m"]["atr"],
            'news_score': 0.0,  # updated below if available
        }
        self.save_position_state()

        print(f" 🚀 {direction} POSITION OPENED [{symbol}]")
        print(f"    Lots: {lots} | Entry: {entry_price:.5f} | SL: {stop_loss:.5f} ({stop_pips:.1f} pips) | TP: {take_profit:.5f} ({reward_pips:.1f} pips)")
        return True

    def manage_position(self, current_price):
        """Manage active open position (Trailing stop & Break-even checks)"""
        pos = bot_state['position']
        if not pos:
            return

        symbol     = pos['symbol']
        direction  = pos['direction']
        entry      = pos['entry']
        stop       = pos['stop']
        target     = pos['target']
        pip_size   = SYMBOL_SPECS.get(symbol, {}).get("pip_size", 0.0001)

        # Track high / low watermark
        pos['highest_price'] = max(pos.get('highest_price', entry), current_price)
        pos['lowest_price']  = min(pos.get('lowest_price', entry), current_price)

        if direction == "LONG":
            profit_pips = (current_price - entry) / pip_size
            risk_pips   = (entry - pos['original_stop']) / pip_size

            # Move to Break-Even at 1.0 R profit
            if profit_pips >= risk_pips and not pos.get('stop_moved_to_be'):
                pos['stop'] = entry + (2.0 * pip_size)  # Entry + 2 pips
                pos['stop_moved_to_be'] = True
                self.save_position_state()
                print(f" 🛡️ [{symbol}] Stop moved to BREAK-EVEN at {pos['stop']:.5f}")

            # Check SL / TP hits
            if current_price <= stop:
                self.close_position(current_price, "STOP_LOSS")
            elif current_price >= target:
                self.close_position(current_price, "TAKE_PROFIT")

        elif direction == "SHORT":
            profit_pips = (entry - current_price) / pip_size
            risk_pips   = (pos['original_stop'] - entry) / pip_size

            # Move to Break-Even at 1.0 R profit
            if profit_pips >= risk_pips and not pos.get('stop_moved_to_be'):
                pos['stop'] = entry - (2.0 * pip_size)
                pos['stop_moved_to_be'] = True
                self.save_position_state()
                print(f" 🛡️ [{symbol}] Stop moved to BREAK-EVEN at {pos['stop']:.5f}")

            # Check SL / TP hits
            if current_price >= stop:
                self.close_position(current_price, "STOP_LOSS")
            elif current_price <= target:
                self.close_position(current_price, "TAKE_PROFIT")

    def close_position(self, exit_price, reason):
        """Close active position and record trade outcome"""
        pos = bot_state['position']
        if not pos:
            return

        direction  = pos['direction']
        entry      = pos['entry']
        lots       = pos['size']
        symbol     = pos['symbol']
        pip_size   = SYMBOL_SPECS.get(symbol, {}).get("pip_size", 0.0001)

        pips_earned = (exit_price - entry) / pip_size if direction == "LONG" else (entry - exit_price) / pip_size

        # Quote-currency-aware pip value (fixes USDJPY / USDCAD sizing)
        specs = SYMBOL_SPECS.get(symbol, SYMBOL_SPECS["EURUSD"])
        if specs.get("quote_is_usd", True):
            pip_value_per_lot = specs["contract_size"] * pip_size
        else:
            pip_value_per_lot = (specs["contract_size"] * pip_size) / exit_price
        pnl = pips_earned * pip_value_per_lot * lots

        # Update win/loss counters and circuit breaker
        bot_state['total_trades']  += 1
        bot_state['session_pnl']   += pnl
        bot_state['daily_trades']  += 1
        if pnl > 0:
            bot_state['winning_trades']    += 1
            bot_state['consecutive_losses'] = 0
        else:
            bot_state['consecutive_losses'] += 1

        # Log outcome with real metadata from position state
        try:
            self.cursor.execute('''
                INSERT INTO trade_log (
                    symbol, entry_time, exit_time, direction, entry_price, exit_price,
                    size, pnl, result, ta_signal_strength, spread_pips, volatility, atr_15m, news_score, sr_scenario, session_window
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                pos['timestamp'],
                datetime.now(timezone.utc).isoformat(),
                direction, entry, exit_price, lots, pnl, reason,
                float(pos.get('strength', 0.0)),
                float(pos.get('spread_pips', 0.0)),
                float(pos.get('volatility', 0.0)),
                float(pos.get('atr_15m', 0.0)),
                float(pos.get('news_score', 0.0)),
                str(pos.get('sr_scenario', 'UNKNOWN')),
                'LIVE' if self.connected else 'DRY_RUN'
            ))
            self.conn.commit()
        except Exception as e:
            print(f"⚠️ Error logging trade outcome to SQLite: {e}")

        print(f" 🏁 POSITION CLOSED [{symbol}] — Reason: {reason} | Exit: {exit_price:.5f} | PnL: ${pnl:.2f} ({pips_earned:.1f} pips)")
        win_rate = (bot_state['winning_trades'] / bot_state['total_trades'] * 100) if bot_state['total_trades'] > 0 else 0.0
        print(f" 📊 Session Stats — Trades: {bot_state['total_trades']} | Win Rate: {win_rate:.0f}% | Session PnL: ${bot_state['session_pnl']:.2f} | Consec. Losses: {bot_state['consecutive_losses']}")
        self.clear_position_state()

        # --- Daily Loss Killswitch ---
        max_daily_loss = bot_state['session_start_balance'] * MAX_DAILY_LOSS_PCT
        if bot_state['session_pnl'] < -max_daily_loss:
            print(f"🛑 DAILY LOSS LIMIT HIT: Session PnL ${bot_state['session_pnl']:.2f} exceeded -{max_daily_loss:.2f} ({MAX_DAILY_LOSS_PCT*100:.0f}% of start balance). Bot halted for today.")
            sys.exit(0)

        # --- Consecutive-Loss Circuit Breaker ---
        if bot_state['consecutive_losses'] >= bot_state['max_consecutive_losses']:
            print(f"🛑 CIRCUIT BREAKER: {bot_state['consecutive_losses']} consecutive losses hit the limit of {bot_state['max_consecutive_losses']}. Bot halted.")
            sys.exit(0)

    def run_cycle(self):
        """Single polling cycle across supported Forex symbols"""
        print(f"\n🔄 --- Running MT5 Forex Cycle [{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] ---")

        # Handle open position management first
        if bot_state['position']:
            pos = bot_state['position']
            symbol = pos['symbol']

            # Use live bid/ask tick for position management (not stale candle close)
            cur_price = None
            if self.connected and HAS_MT5_LIB:
                tick = mt5.symbol_info_tick(symbol)
                if tick is not None:
                    cur_price = tick.bid if pos['direction'] == 'LONG' else tick.ask

            # Fallback: fetch candle data and use last close
            if cur_price is None:
                data = self.fetch_multi_timeframe_data(symbol)
                cur_price = float(data['15m']['close'][-1]) if data else None

            if cur_price is not None:
                print(f" 📊 Open Position Active: {pos['direction']} {symbol} | Live Price: {cur_price:.5f}")
                self.manage_position(cur_price)
            else:
                print(f" ⚠️ Could not fetch price for open position on {symbol}")
            return

        # Scan symbols for new entry signals
        for symbol in TRADE_SYMBOLS:
            print(f"\n🔍 Evaluating {symbol}...")
            passed_checks, session_label, spread_pips = self.check_spread_and_session(symbol)
            if not passed_checks:
                continue

            data = self.fetch_multi_timeframe_data(symbol)
            if not data:
                print(f" ⚠️ Could not fetch market data for {symbol}")
                continue

            indicators, current_price, volatility = self.calculate_indicators(data)
            print(f"   Current Price: {current_price:.5f} | Volatility: {volatility*100:.3f}% | 15m ATR: {indicators['15m']['atr']:.5f}")

            signal_data = self.evaluate_multi_factor_consensus(symbol, indicators, current_price, volatility, data, spread_pips)
            signal = signal_data.get("signal")

            if signal in ["LONG", "SHORT"]:
                print(f" ⚡ CONSENSUS SIGNAL DETECTED: {signal} on {symbol}")
                success = self.execute_trade(symbol, signal, current_price, indicators, signal_data, spread_pips, volatility)
                if success:
                    break  # Execute one position per cycle max

        print("\n✅ Cycle complete.")


def main():
    parser = argparse.ArgumentParser(description="MetaTrader 5 Forex Trading Bot")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run simulation mode")
    parser.add_argument("--single-cycle", action="store_true", help="Run a single evaluation cycle and exit")
    args = parser.parse_args()

    bot = MT5ForexBot(dry_run=args.dry_run)

    if args.single_cycle:
        bot.run_cycle()
    else:
        print("🔄 Starting MT5 Forex Bot continuous polling loop (15s interval). Press Ctrl+C to exit.")
        try:
            while True:
                bot.run_cycle()
                time.sleep(15)
        except KeyboardInterrupt:
            print("\n🛑 MT5 Forex Bot stopped by user.")


if __name__ == "__main__":
    main()
