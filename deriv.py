"""
Deriv WebSocket API Forex Trading Bot
======================================
Linux-native Forex trading bot using Deriv (Binary.com) v3 WebSocket API.
Identical multi-factor signal logic, session timing, S/R anchoring, spread guards,
risk management, break-even stops, daily loss killswitch, and SQLite state persistence
as mt5.py and oanda.py.

Credentials (.env):
    DERIV_API_TOKEN - Personal API token from Deriv Account Settings -> API Token
    DERIV_APP_ID    - App ID (default: 1089 for general testing or your registered App ID)
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

# Try importing websocket-client
try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    websocket = None
    HAS_WEBSOCKET = False

# Try importing feedparser for News RSS
try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    feedparser = None
    HAS_FEEDPARSER = False

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
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN", "")
DERIV_APP_ID    = os.getenv("DERIV_APP_ID", "1089")

DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

# ── Forex symbols & Deriv symbol mapping ──────────────────────────────────
TRADE_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

DERIV_SYMBOLS = {
    "EURUSD": "frxEURUSD",
    "GBPUSD": "frxGBPUSD",
    "USDJPY": "frxUSDJPY",
    "AUDUSD": "frxAUDUSD",
    "USDCAD": "frxUSDCAD",
}

DERIV_GRANULARITY = {
    "15m": 900,
    "1h":  3600,
    "4h":  14400,
}

# Per-symbol specs
# quote_is_usd=True  -> quote currency is USD (pip value = contract_size * pip_size)
# quote_is_usd=False -> quote currency is NOT USD (pip value = (contract_size * pip_size) / price)
SYMBOL_SPECS = {
    "EURUSD": {"pip_size": 0.0001, "min_lot": 0.01, "lot_step": 0.01, "contract_size": 100000, "quote_is_usd": True},
    "GBPUSD": {"pip_size": 0.0001, "min_lot": 0.01, "lot_step": 0.01, "contract_size": 100000, "quote_is_usd": True},
    "USDJPY": {"pip_size": 0.01,   "min_lot": 0.01, "lot_step": 0.01, "contract_size": 100000, "quote_is_usd": False},
    "AUDUSD": {"pip_size": 0.0001, "min_lot": 0.01, "lot_step": 0.01, "contract_size": 100000, "quote_is_usd": True},
    "USDCAD": {"pip_size": 0.0001, "min_lot": 0.01, "lot_step": 0.01, "contract_size": 100000, "quote_is_usd": False},
}

BOT_MAGIC_NUMBER   = 998877
MAX_DAILY_LOSS_PCT = 0.05

# Forex Google News Query Mapping
FOREX_NEWS_QUERIES = {
    "MACRO":  "US Dollar Fed inflation interest rates forex",
    "EURUSD": "EURUSD Euro US Dollar forex news",
    "GBPUSD": "GBPUSD British Pound Bank of England forex news",
    "USDJPY": "USDJPY Japanese Yen Bank of Japan forex news",
    "AUDUSD": "AUDUSD Australian Dollar RBA forex news",
    "USDCAD": "USDCAD Canadian Dollar Bank of Canada forex news",
}

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

primary_timeframe = "15m"
higher_timeframe  = "1h"
macro_timeframe   = "4h"

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
    'session_start_balance': 0.0,
    'session_pnl': 0.0,
}


def get_forex_news_score(symbol: str) -> dict:
    """Fetch Google News RSS for USD Macro & Pair Specific Forex News"""
    if not HAS_FEEDPARSER:
        return {"score": 0.0, "confidence": 0.0, "block_long_only": False, "details": {"reason": "feedparser not installed"}}

    try:
        macro_url = f"https://news.google.com/rss/search?q={quote_plus(FOREX_NEWS_QUERIES['MACRO'])}&hl=en-US&gl=US&ceid=US:en"
        macro_feed = feedparser.parse(macro_url)

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


class DerivForexBot:
    def __init__(self, dry_run=False):
        print("🚀 Initializing Deriv Forex Trading Bot...")
        self.dry_run = dry_run
        self.connected = False
        self.ws = None
        self.state_file = 'deriv_trading_state.json'
        
        self.init_db()
        self.init_deriv_connection()
        self.load_position_state()
        self.initialize_balance()

    # ── Deriv WebSocket Communication Helpers ─────────────────────────────

    def _send_request(self, payload: dict, timeout=10.0):
        """Send JSON request over WebSocket and return response."""
        if not self.ws:
            return None
        try:
            self.ws.send(json.dumps(payload))
            self.ws.settimeout(timeout)
            response_str = self.ws.recv()
            if response_str:
                return json.loads(response_str)
        except Exception as e:
            print(f"❌ Deriv WS request error ({payload.get('ticks_history') or list(payload.keys())[0]}): {e}")
        return None

    def init_deriv_connection(self):
        """Establish WebSocket connection with Deriv API supporting both PAT (REST OTP) and legacy WS auth."""
        if not HAS_WEBSOCKET:
            print("⚠️ websocket-client library not installed (`pip install websocket-client`). Running in DRY-RUN mode.")
            self.dry_run = True
            return

        if not DERIV_API_TOKEN:
            print("⚠️ DERIV_API_TOKEN not set in .env. Running in DRY-RUN mode.")
            self.dry_run = True
            return

        print(f"🔌 Connecting to Deriv API (App ID: {DERIV_APP_ID})...")

        # ── Method 1: Deriv REST OTP Auth (for PAT pat_... tokens) ──────────
        if DERIV_API_TOKEN.startswith("pat_") or len(DERIV_API_TOKEN) > 40:
            try:
                import requests
                headers = {
                    "Authorization": f"Bearer {DERIV_API_TOKEN}",
                    "Deriv-App-ID": DERIV_APP_ID,
                    "Content-Type": "application/json"
                }
                res = requests.get("https://api.derivws.com/trading/v1/options/accounts", headers=headers, timeout=10)
                if res.status_code == 200:
                    accounts_data = res.json().get("data", [])
                    # Pick demo or real account based on dry_run mode
                    selected_acc = None
                    target_type = "demo" if self.dry_run else "real"
                    for acc in accounts_data:
                        if acc.get("account_type") == target_type and acc.get("status") == "active":
                            selected_acc = acc
                            break
                    if not selected_acc and accounts_data:
                        selected_acc = accounts_data[0]

                    if selected_acc:
                        acc_id = selected_acc.get("account_id")
                        balance = selected_acc.get("balance", "0.00")
                        curr = selected_acc.get("currency", "USD")
                        acc_type = selected_acc.get("account_type", "demo").upper()

                        # Request OTP WebSocket URL
                        otp_res = requests.post(f"https://api.derivws.com/trading/v1/options/accounts/{acc_id}/otp", headers=headers, timeout=10)
                        if otp_res.status_code in [200, 201]:
                            ws_url = otp_res.json().get("data", {}).get("url")
                            if ws_url:
                                self.ws = websocket.create_connection(ws_url, timeout=10.0)
                                print(f"✅ Authorized Deriv PAT [{acc_type}]: Account {acc_id} | Balance: ${balance} {curr}")
                                self.connected = True
                                return
            except Exception as e:
                print(f"⚠️ REST OTP auth failed ({e}). Trying fallback WebSocket auth...")

        # ── Method 2: Legacy Direct WebSocket Auth ─────────────────────────
        try:
            self.ws = websocket.create_connection(DERIV_WS_URL, timeout=10.0)
            auth_res = self._send_request({"authorize": DERIV_API_TOKEN})
            if auth_res and "authorize" in auth_res:
                auth = auth_res["authorize"]
                print(f"✅ Authorized Deriv Account: {auth.get('email')} | Account ID: {auth.get('loginid')} | Currency: {auth.get('currency')}")
                self.connected = True
            else:
                err_msg = auth_res.get("error", {}).get("message", "Authorization failed") if auth_res else "No response"
                print(f"❌ Deriv authorization failed: {err_msg}. Running in DRY-RUN mode.")
                self.dry_run = True
        except Exception as e:
            print(f"❌ Error connecting to Deriv WebSocket: {e}. Switching to DRY-RUN mode.")
            self.dry_run = True

    # ── Database Persistence ──────────────────────────────────────────────

    def init_db(self):
        """Initialize SQLite database for state persistence"""
        self.conn = sqlite3.connect('deriv_trading_state.db', timeout=30.0, check_same_thread=False)
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

        # Schema migrations for existing databases
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
                except Exception as mig_err:
                    pass

        self.conn.commit()

    def initialize_balance(self):
        """Fetch account balance from Deriv API or set default"""
        balance = 1000.0
        if self.connected and not self.dry_run:
            try:
                bal_res = self._send_request({"balance": 1})
                if bal_res and "balance" in bal_res:
                    balance = float(bal_res["balance"].get("balance", 1000.0))
            except Exception as e:
                print(f"⚠️ Could not fetch Deriv account balance: {e}")

        bot_state['available_balance'] = balance
        if bot_state['session_start_balance'] == 0.0:
            bot_state['session_start_balance'] = balance

        print(f"💰 Deriv Balance Initialized:")
        print(f"   Available Balance: ${balance:.2f}")
        print(f"   Risk Per Trade: {forex_risk_per_trade*100:.1f}% = ${balance * forex_risk_per_trade:.2f}")
        print(f"   Mode: {'LIVE DERIV CONNECTED' if self.connected and not self.dry_run else 'DRY-RUN SIMULATION'}")

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
                print(f"🔄 Recovered active Deriv position: {bot_state['position']['direction']} {bot_state['position']['symbol']}")
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

    # ── Market Data & Session Guard ───────────────────────────────────────

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

        # 3. Real-Time Spread Check via Deriv WS proposal/pricing
        pip_size = SYMBOL_SPECS.get(symbol, {}).get("pip_size", 0.0001)
        spread_pips = 1.0  # default fallback

        if self.connected and not self.dry_run:
            deriv_sym = DERIV_SYMBOLS.get(symbol, symbol)
            proposal = self._send_request({
                "proposal": 1,
                "amount": 10,
                "basis": "stake",
                "contract_type": "MULTUP",
                "currency": "USD",
                "symbol": deriv_sym
            })
            if proposal and "proposal" in proposal:
                p = proposal["proposal"]
                spot = float(p.get("spot", 0.0))
                ask  = float(p.get("ask_price", spot))
                bid  = float(p.get("bid_price", spot))
                if ask > 0 and bid > 0:
                    spread_pips = abs(ask - bid) / pip_size

        print(f" 🔍 {symbol} Real-time Spread: {spread_pips:.1f} pips (Max: {max_spread_pips} pips)")
        if spread_pips > max_spread_pips:
            print(f" 🚫 BLOCKED: Spread {spread_pips:.1f} pips exceeds threshold {max_spread_pips} pips")
            return False, f"HIGH_SPREAD_{spread_pips:.1f}", spread_pips

        return True, session_label, spread_pips

    def fetch_multi_timeframe_data(self, symbol):
        """Fetch multi-timeframe candles (15m, 1h, 4h) via Deriv WebSocket `ticks_history`"""
        data = {}
        deriv_sym = DERIV_SYMBOLS.get(symbol, symbol)

        for tf_name in ["15m", "1h", "4h"]:
            fetched = False
            granularity_sec = DERIV_GRANULARITY[tf_name]

            if self.connected:
                try:
                    res = self._send_request({
                        "ticks_history": deriv_sym,
                        "adjust_start_time": 1,
                        "count": 100,
                        "end": "latest",
                        "granularity": granularity_sec,
                        "style": "candles"
                    })
                    if res and "candles" in res:
                        candles = res["candles"]
                        closes  = np.array([float(c["close"]) for c in candles])
                        highs   = np.array([float(c["high"]) for c in candles])
                        lows    = np.array([float(c["low"]) for c in candles])
                        vols    = np.array([float(c.get("volume", 100)) for c in candles])
                        data[tf_name] = {
                            'close': closes,
                            'high': highs,
                            'low': lows,
                            'volume': vols,
                            'timestamp': np.array([int(c["epoch"]) for c in candles])
                        }
                        fetched = True
                except Exception as e:
                    print(f"❌ Error fetching {tf_name} Deriv candles for {symbol}: {e}")

            # Fallback simulated data generator for dry-run/testing
            if not fetched or tf_name not in data:
                np.random.seed(int(time.time() * 1000) % 100000)
                base_prices = {"EURUSD": 1.0850, "GBPUSD": 1.2650, "USDJPY": 155.00, "AUDUSD": 0.6550, "USDCAD": 1.3550}
                base_price = base_prices.get(symbol, 1.0000)
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
                def _ema(arr, n):
                    k = 2.0 / (n + 1)
                    val = float(np.mean(arr[:n]))
                    for v in arr[n:]:
                        val = float(v) * k + val * (1.0 - k)
                    return val

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

                ema_21 = _ema(closes, 21)
                ema_50 = _ema(closes, 50)

                ema_12 = _ema(closes, 12)
                ema_26 = _ema(closes, 26)
                macd_line_val = ema_12 - ema_26
                start_idx = max(26, len(closes) - 20)
                macd_vals = [_ema(closes[:i + 1], 12) - _ema(closes[:i + 1], 26)
                             for i in range(start_idx, len(closes))]
                macd_signal_val = _ema(np.array(macd_vals), 9) if len(macd_vals) >= 9 else macd_vals[-1]
                macd_hist_val = macd_line_val - macd_signal_val
                macd_line   = np.array([macd_line_val])
                macd_signal = np.array([macd_signal_val])
                macd_hist   = np.array([macd_hist_val])

                tr_arr = np.maximum(
                    highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]),
                               np.abs(lows[1:]  - closes[:-1]))
                )
                atr_period = 14
                atr = float(np.mean(tr_arr[:atr_period]))
                for v in tr_arr[atr_period:]:
                    atr = (atr * (atr_period - 1) + float(v)) / atr_period

                stoch_period = 14
                recent_high = float(np.max(highs[-stoch_period:]))
                recent_low  = float(np.min(lows[-stoch_period:]))
                stoch_k_val = ((float(closes[-1]) - recent_low) /
                               (recent_high - recent_low + 1e-9)) * 100.0
                stoch_k = np.array([stoch_k_val])
                stoch_d = np.array([stoch_k_val])

                dm_plus  = np.where((highs[1:] - highs[:-1]) > (lows[:-1] - lows[1:]),
                                    np.maximum(highs[1:] - highs[:-1], 0.0), 0.0)
                dm_minus = np.where((lows[:-1] - lows[1:]) > (highs[1:] - highs[:-1]),
                                    np.maximum(lows[:-1] - lows[1:], 0.0), 0.0)
                atr_window = np.mean(tr_arr[-atr_period:])
                di_plus  = 100.0 * np.mean(dm_plus[-atr_period:])  / (atr_window + 1e-9)
                di_minus = 100.0 * np.mean(dm_minus[-atr_period:]) / (atr_window + 1e-9)
                dx = 100.0 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-9)
                adx = dx

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
        ind4h = indicators.get("4h", {})
        ema21_4h = ind4h.get("ema_21", 0.0)
        ema50_4h = ind4h.get("ema_50", 0.0)
        trend_bullish = ema21_4h > ema50_4h if ema50_4h > 0 else True
        trend_score = 1.0 if trend_bullish else -1.0

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

        sr_data = {"score": 0.0, "scenario": "MID_RANGE", "suggested_stop": None, "suggested_target": None}
        if HAS_SR_FACTOR and get_sr_score is not None:
            try:
                sr_data = get_sr_score(symbol, current_price, indicators, data)
            except Exception as e:
                print(f" ⚠️ S/R Factor evaluation notice: {e}")
        sr_score = sr_data.get("score", 0.0)

        news_data = get_forex_news_score(symbol)
        news_score = news_data.get("score", 0.0)
        block_long_only = news_data.get("block_long_only", False)

        spread_score = max(0.0, 1.0 - (spread_pips / max_spread_pips))

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
        """Calculate position size in Standard Lots"""
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
        """Execute Forex Trade (via Deriv WS API or Dry-Run)"""
        pip_size = SYMBOL_SPECS.get(symbol, {}).get("pip_size", 0.0001)
        atr_15m  = indicators["15m"]["atr"]

        sug_stop   = signal_data.get("suggested_stop")
        sug_target = signal_data.get("suggested_target")

        if sug_stop and sug_target and signal_data.get("sr_scenario") != "MID_RANGE":
            stop_loss   = sug_stop
            take_profit = sug_target
            stop_pips   = abs(entry_price - stop_loss) / pip_size
            reward_pips = abs(take_profit - entry_price) / pip_size
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
        order_id = f"DRY_RUN_{int(time.time())}"

        if self.connected and not self.dry_run:
            deriv_sym = DERIV_SYMBOLS.get(symbol, symbol)
            contract_type = "MULTUP" if direction == "LONG" else "MULTDOWN"
            stake_amount = max(1.00, round(pos_size['risk_amount'], 2))

            # Request proposal from Deriv
            proposal_req = {
                "proposal": 1,
                "amount": stake_amount,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "underlying_symbol": deriv_sym,
                "multiplier": 100,
                "limit_order": {
                    "stop_loss": round(min(stake_amount * 0.9, stake_amount * (stop_pips / 100.0)), 2),
                    "take_profit": round(stake_amount * (reward_pips / 100.0), 2)
                }
            }
            prop_res = self._send_request(proposal_req)
            if prop_res and "proposal" in prop_res:
                proposal_id = prop_res["proposal"]["id"]
                buy_req = {
                    "buy": proposal_id,
                    "price": stake_amount
                }
                buy_res = self._send_request(buy_req)
                if buy_res and "buy" in buy_res:
                    order_id = str(buy_res["buy"].get("contract_id", order_id))
                    entry_price = float(buy_res["buy"].get("buy_price", entry_price))
                    print(f"✅ Deriv Order Placed Successfully! Contract ID #{order_id} | Fill Price: {entry_price:.5f}")
                else:
                    err_info = buy_res.get("error", {}) if buy_res else {}
                    err_code = err_info.get("code", "UNKNOWN")
                    err_msg  = err_info.get("message", "No buy response")
                    print(f"❌ Deriv Buy Execution Failed [{err_code}]: {err_msg}")
                    if "balance" in err_msg.lower() or err_code == "InsufficientBalance":
                        print("💡 REASON: Insufficient funds in account to cover the trade stake.")
                    return False
            else:
                err_info = prop_res.get("error", {}) if prop_res else {}
                err_code = err_info.get("code", "UNKNOWN")
                err_msg  = err_info.get("message", "No proposal response")
                print(f"❌ Deriv Proposal Failed [{err_code}]: {err_msg}")
                if "balance" in err_msg.lower() or err_code == "InsufficientBalance":
                    print(f"💡 REASON: Insufficient account balance (${bot_state['available_balance']:.2f}) for minimum stake (${stake_amount:.2f}).")
                elif "contract" in err_msg.lower() or "market" in err_msg.lower():
                    print("💡 REASON: Market closed or contract parameter out of allowed bounds.")
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
            'spread_pips': spread_pips,
            'volatility': volatility,
            'atr_15m': indicators["15m"]["atr"],
            'news_score': signal_data.get("news_score", 0.0),
            'sr_scenario': signal_data.get("sr_scenario", "MID_RANGE"),
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

        pos['highest_price'] = max(pos.get('highest_price', entry), current_price)
        pos['lowest_price']  = min(pos.get('lowest_price', entry), current_price)

        if direction == "LONG":
            profit_pips = (current_price - entry) / pip_size
            risk_pips   = (entry - pos['original_stop']) / pip_size

            if profit_pips >= risk_pips and not pos.get('stop_moved_to_be'):
                pos['stop'] = entry + (2.0 * pip_size)
                pos['stop_moved_to_be'] = True
                self.save_position_state()
                print(f" 🛡️ [{symbol}] Stop moved to BREAK-EVEN at {pos['stop']:.5f}")

            if current_price <= stop:
                self.close_position(current_price, "STOP_LOSS")
            elif current_price >= target:
                self.close_position(current_price, "TAKE_PROFIT")

        elif direction == "SHORT":
            profit_pips = (entry - current_price) / pip_size
            risk_pips   = (pos['original_stop'] - entry) / pip_size

            if profit_pips >= risk_pips and not pos.get('stop_moved_to_be'):
                pos['stop'] = entry - (2.0 * pip_size)
                pos['stop_moved_to_be'] = True
                self.save_position_state()
                print(f" 🛡️ [{symbol}] Stop moved to BREAK-EVEN at {pos['stop']:.5f}")

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

        specs = SYMBOL_SPECS.get(symbol, SYMBOL_SPECS["EURUSD"])
        if specs.get("quote_is_usd", True):
            pip_value_per_lot = specs["contract_size"] * pip_size
        else:
            pip_value_per_lot = (specs["contract_size"] * pip_size) / exit_price
        pnl = pips_earned * pip_value_per_lot * lots

        bot_state['total_trades']  += 1
        bot_state['session_pnl']   += pnl
        bot_state['daily_trades']  += 1
        if pnl > 0:
            bot_state['winning_trades']    += 1
            bot_state['consecutive_losses'] = 0
        else:
            bot_state['consecutive_losses'] += 1

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
                'LIVE' if self.connected and not self.dry_run else 'DRY_RUN'
            ))
            self.conn.commit()
        except Exception as e:
            print(f"⚠️ Error logging trade outcome to SQLite: {e}")

        print(f" 🏁 POSITION CLOSED [{symbol}] — Reason: {reason} | Exit: {exit_price:.5f} | PnL: ${pnl:.2f} ({pips_earned:.1f} pips)")
        win_rate = (bot_state['winning_trades'] / bot_state['total_trades'] * 100) if bot_state['total_trades'] > 0 else 0.0
        print(f" 📊 Session Stats — Trades: {bot_state['total_trades']} | Win Rate: {win_rate:.0f}% | Session PnL: ${bot_state['session_pnl']:.2f} | Consec. Losses: {bot_state['consecutive_losses']}")
        self.clear_position_state()

        max_daily_loss = bot_state['session_start_balance'] * MAX_DAILY_LOSS_PCT
        if bot_state['session_pnl'] < -max_daily_loss:
            print(f"🛑 DAILY LOSS LIMIT HIT: Session PnL ${bot_state['session_pnl']:.2f} exceeded -{max_daily_loss:.2f} ({MAX_DAILY_LOSS_PCT*100:.0f}% of start balance). Bot halted for today.")
            sys.exit(0)

        if bot_state['consecutive_losses'] >= bot_state['max_consecutive_losses']:
            print(f"🛑 CIRCUIT BREAKER: {bot_state['consecutive_losses']} consecutive losses hit the limit of {bot_state['max_consecutive_losses']}. Bot halted.")
            sys.exit(0)

    def run_cycle(self):
        """Single polling cycle across supported Forex symbols"""
        print(f"\n🔄 --- Running Deriv Forex Cycle [{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] ---")

        if bot_state['position']:
            pos = bot_state['position']
            symbol = pos['symbol']

            cur_price = None
            if self.connected and not self.dry_run:
                deriv_sym = DERIV_SYMBOLS.get(symbol, symbol)
                tick_res = self._send_request({"ticks": deriv_sym})
                if tick_res and "tick" in tick_res:
                    cur_price = float(tick_res["tick"].get("quote", 0.0))

            if cur_price is None:
                data = self.fetch_multi_timeframe_data(symbol)
                cur_price = float(data['15m']['close'][-1]) if data else None

            if cur_price is not None:
                print(f" 📊 Open Position Active: {pos['direction']} {symbol} | Live Price: {cur_price:.5f}")
                self.manage_position(cur_price)
            else:
                print(f" ⚠️ Could not fetch price for open position on {symbol}")
            return

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
                    break

        print("\n✅ Cycle complete.")


def main():
    parser = argparse.ArgumentParser(description="Deriv WebSocket API Forex Trading Bot")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run simulation mode")
    parser.add_argument("--single-cycle", action="store_true", help="Run a single evaluation cycle and exit")
    args = parser.parse_args()

    bot = DerivForexBot(dry_run=args.dry_run)

    if args.single_cycle:
        bot.run_cycle()
    else:
        print("🔄 Starting Deriv Forex Bot continuous polling loop (15s interval). Press Ctrl+C to exit.")
        try:
            while True:
                bot.run_cycle()
                time.sleep(15)
        except KeyboardInterrupt:
            print("\n🛑 Deriv Forex Bot stopped by user.")


if __name__ == "__main__":
    main()
