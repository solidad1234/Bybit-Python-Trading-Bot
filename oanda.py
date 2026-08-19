"""
OANDA v20 REST API Forex Trading Bot
======================================
Linux-native alternative to mt5.py. Identical multi-factor signal logic,
session timing, S/R anchoring, spread guards, risk management, and SQLite
persistence. Only the broker data-feed and order execution layer differs.

Credentials (.env):
    OANDA_API_KEY     - Personal access token from OANDA Hub (hub.oanda.com)
    OANDA_ACCOUNT_ID  - Account ID shown in OANDA Hub (e.g. 001-001-1234567-001)
    OANDA_ENVIRONMENT - "practice" (default, free demo) or "live"
"""

import sys
import os
import time
import json
import re
import sqlite3
import numpy as np
import argparse
import requests
from datetime import datetime, timezone, timedelta
from urllib.parse import quote_plus
from dotenv import load_dotenv

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    feedparser = None
    HAS_FEEDPARSER = False

try:
    import talib
    HAS_TALIB = True
except ImportError:
    talib = None
    HAS_TALIB = False

try:
    from factors.support_resistance import get_sr_score
    HAS_SR_FACTOR = True
except ImportError:
    get_sr_score = None
    HAS_SR_FACTOR = False

load_dotenv()
OANDA_API_KEY     = os.getenv("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID  = os.getenv("OANDA_ACCOUNT_ID", "")
OANDA_ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")

OANDA_BASE_URL = (
    "https://api-fxtrade.oanda.com"
    if OANDA_ENVIRONMENT == "live"
    else "https://api-fxpractice.oanda.com"
)

# ── Forex symbols & OANDA instrument name mapping ──────────────────────────
TRADE_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

OANDA_INSTRUMENTS = {
    "EURUSD": "EUR_USD", "GBPUSD": "GBP_USD", "USDJPY": "USD_JPY",
    "AUDUSD": "AUD_USD", "USDCAD": "USD_CAD",
}

OANDA_GRANULARITY = {"15m": "M15", "1h": "H1", "4h": "H4"}

# quote_is_usd=True  → pip value = contract_size * pip_size (~$10/pip/lot)
# quote_is_usd=False → pip value = (contract_size * pip_size) / price
SYMBOL_SPECS = {
    "EURUSD": {"pip_size": 0.0001, "min_lot": 0.01, "lot_step": 0.01, "contract_size": 100000, "quote_is_usd": True},
    "GBPUSD": {"pip_size": 0.0001, "min_lot": 0.01, "lot_step": 0.01, "contract_size": 100000, "quote_is_usd": True},
    "USDJPY": {"pip_size": 0.01,   "min_lot": 0.01, "lot_step": 0.01, "contract_size": 100000, "quote_is_usd": False},
    "AUDUSD": {"pip_size": 0.0001, "min_lot": 0.01, "lot_step": 0.01, "contract_size": 100000, "quote_is_usd": True},
    "USDCAD": {"pip_size": 0.0001, "min_lot": 0.01, "lot_step": 0.01, "contract_size": 100000, "quote_is_usd": False},
}

BOT_MAGIC_NUMBER   = 998877
MAX_DAILY_LOSS_PCT = 0.05

FOREX_NEWS_QUERIES = {
    "MACRO":  "US Dollar Fed inflation interest rates forex",
    "EURUSD": "EURUSD Euro US Dollar forex news",
    "GBPUSD": "GBPUSD British Pound Bank of England forex news",
    "USDJPY": "USDJPY Japanese Yen Bank of Japan forex news",
    "AUDUSD": "AUDUSD Australian Dollar RBA forex news",
    "USDCAD": "USDCAD Canadian Dollar Bank of Canada forex news",
}
POSITIVE_KEYWORDS = {
    "hike","hawkish","growth","surge","surging","rally","rallies",
    "bullish","strong","gains","gain","outperform","recovery","rebound",
    "stimulus","optimism","record","expansion","profit"
}
NEGATIVE_KEYWORDS = {
    "cut","dovish","recession","drop","drops","plunge","plunges",
    "bearish","weak","decline","declining","inflation","crisis",
    "slump","deficit","unemployment","risk","warning","downside"
}

primary_timeframe = "15m"
higher_timeframe  = "1h"
macro_timeframe   = "4h"

forex_risk_per_trade     = 0.015
max_spread_pips          = 2.5
min_reward_ratio         = 1.5
min_volatility_threshold = 0.0010

WEIGHTS = {
    "trend_alignment": 0.25, "support_resistance": 0.25,
    "technical": 0.20, "news": 0.15, "spread_volatility": 0.15,
}

london_open_utc    = 7.0
ny_close_utc       = 16.5
rollover_start_utc = 21.9
rollover_end_utc   = 22.3

bot_state = {
    'last_trade_time': None, 'position': None,
    'daily_trades': 0, 'total_trades': 0, 'winning_trades': 0,
    'consecutive_losses': 0, 'max_consecutive_losses': 3,
    'session_start': datetime.now(timezone.utc),
    'available_balance': 1000.0, 'session_start_balance': 0.0, 'session_pnl': 0.0,
}


def get_forex_news_score(symbol: str) -> dict:
    """Fetch Google News RSS for Macro USD + pair-specific sentiment."""
    if not HAS_FEEDPARSER:
        return {"score": 0.0, "confidence": 0.0, "block_long_only": False,
                "details": {"reason": "feedparser not installed"}}
    try:
        macro_url = f"https://news.google.com/rss/search?q={quote_plus(FOREX_NEWS_QUERIES['MACRO'])}&hl=en-US&gl=US&ceid=US:en"
        pair_url  = f"https://news.google.com/rss/search?q={quote_plus(FOREX_NEWS_QUERIES.get(symbol, symbol + ' forex'))}&hl=en-US&gl=US&ceid=US:en"
        entries   = (feedparser.parse(macro_url).entries[:5] + feedparser.parse(pair_url).entries[:5])
        if not entries:
            return {"score": 0.0, "confidence": 0.2, "block_long_only": False}
        net = 0.0
        for e in entries:
            words = set(re.findall(r"[a-z]+", e.get("title", "").lower()))
            net  += max(-1.0, min(1.0, (len(words & POSITIVE_KEYWORDS) - len(words & NEGATIVE_KEYWORDS)) * 0.25))
        final = max(-1.0, min(1.0, net / len(entries)))
        return {
            "score": round(final, 3), "confidence": 0.70,
            "block_long_only": final <= -0.5,
            "block_reason": f"Negative news ({final:.2f})" if final <= -0.5 else "",
            "details": {"articles_count": len(entries)},
        }
    except Exception as e:
        return {"score": 0.0, "confidence": 0.0, "block_long_only": False, "details": {"err": str(e)}}

class OandaForexBot:
    def __init__(self, dry_run=False):
        print("🚀 Initializing OANDA Forex Trading Bot...")
        self.dry_run = dry_run
        self.connected = False
        self.headers = {
            "Authorization": f"Bearer {OANDA_API_KEY}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339",
        }
        self.init_db()
        self.init_oanda_connection()
        self.load_position_state()
        self.initialize_balance()

    # ── OANDA REST helpers ─────────────────────────────────────────────────

    def _get(self, path: str, params: dict = None):
        try:
            r = requests.get(f"{OANDA_BASE_URL}/v3/{path}",
                             headers=self.headers, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"❌ OANDA GET {path}: {e}")
            return None

    def _post(self, path: str, payload: dict):
        try:
            r = requests.post(f"{OANDA_BASE_URL}/v3/{path}",
                              headers=self.headers, json=payload, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"❌ OANDA POST {path}: {e}")
            return None

    def _put(self, path: str, payload: dict):
        try:
            r = requests.put(f"{OANDA_BASE_URL}/v3/{path}",
                             headers=self.headers, json=payload, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"❌ OANDA PUT {path}: {e}")
            return None

    # ── Database ───────────────────────────────────────────────────────────

    def init_db(self):
        """Initialize SQLite for state persistence (identical schema to mt5.py)"""
        self.conn = sqlite3.connect('oanda_trading_state.db', timeout=30.0, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('PRAGMA journal_mode=WAL;')
        self.cursor.execute('PRAGMA busy_timeout=30000;')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS position (
                id INTEGER PRIMARY KEY, symbol TEXT, direction TEXT,
                size REAL, entry REAL, stop REAL, target REAL,
                order_id TEXT, timestamp TEXT, exit_25_taken INTEGER,
                exit_50_taken INTEGER, stop_moved_to_be INTEGER,
                original_stop REAL, highest_price REAL, lowest_price REAL
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT, entry_time TEXT, exit_time TEXT, direction TEXT,
                entry_price REAL, exit_price REAL, size REAL, pnl REAL,
                result TEXT, ta_signal_strength REAL, spread_pips REAL,
                volatility REAL, atr_15m REAL, news_score REAL,
                sr_scenario TEXT, session_window TEXT
            )
        ''')
        # Schema migrations for existing DB files
        existing = {}
        for tbl in ("trade_log", "position"):
            self.cursor.execute(f"PRAGMA table_info({tbl})")
            existing[tbl] = {row[1] for row in self.cursor.fetchall()}
        for tbl, col, typ in [
            ("trade_log","news_score","REAL"), ("trade_log","sr_scenario","TEXT"),
            ("trade_log","session_window","TEXT"), ("trade_log","ta_signal_strength","REAL"),
            ("trade_log","spread_pips","REAL"), ("trade_log","volatility","REAL"),
            ("trade_log","atr_15m","REAL"),
        ]:
            if col not in existing.get(tbl, set()):
                try:
                    self.cursor.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {typ}")
                except Exception:
                    pass
        self.conn.commit()

    # ── Connection & balance ───────────────────────────────────────────────

    def init_oanda_connection(self):
        """Verify OANDA API credentials and account access."""
        if not OANDA_API_KEY or not OANDA_ACCOUNT_ID:
            print("⚠️  OANDA_API_KEY or OANDA_ACCOUNT_ID not set in .env. Running in DRY-RUN mode.")
            self.dry_run = True
            return
        data = self._get(f"accounts/{OANDA_ACCOUNT_ID}/summary")
        if data and "account" in data:
            acc = data["account"]
            print(f"✅ Connected to OANDA account #{acc.get('id')} | "
                  f"Currency: {acc.get('currency')} | "
                  f"Env: {OANDA_ENVIRONMENT.upper()}")
            self.connected = True
        else:
            print("❌ Could not reach OANDA API. Running in DRY-RUN mode.")
            self.dry_run = True

    def initialize_balance(self):
        """Fetch free margin from OANDA or use default."""
        balance = 1000.0
        if self.connected:
            data = self._get(f"accounts/{OANDA_ACCOUNT_ID}/summary")
            if data and "account" in data:
                balance = float(data["account"].get("marginAvailable", 1000.0))
        bot_state['available_balance'] = balance
        if bot_state['session_start_balance'] == 0.0:
            bot_state['session_start_balance'] = balance
        print(f"💰 OANDA Balance: ${balance:.2f} | "
              f"Risk/trade: {forex_risk_per_trade*100:.1f}% = ${balance*forex_risk_per_trade:.2f} | "
              f"Mode: {'LIVE' if self.connected and not self.dry_run else 'DRY-RUN'}")

    # ── Position state persistence ─────────────────────────────────────────

    def save_position_state(self):
        pos = bot_state['position']
        if not pos:
            return
        try:
            self.cursor.execute('DELETE FROM position')
            self.cursor.execute('''
                INSERT INTO position (symbol,direction,size,entry,stop,target,order_id,
                    timestamp,exit_25_taken,exit_50_taken,stop_moved_to_be,
                    original_stop,highest_price,lowest_price)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                pos['symbol'], pos['direction'], pos['size'], pos['entry'],
                pos['stop'], pos['target'], str(pos.get('order_id','')),
                str(pos['timestamp']),
                int(pos.get('exit_25_taken', False)), int(pos.get('exit_50_taken', False)),
                int(pos.get('stop_moved_to_be', False)),
                pos.get('original_stop', pos['stop']),
                pos.get('highest_price', pos['entry']),
                pos.get('lowest_price', pos['entry']),
            ))
            self.conn.commit()
        except Exception as e:
            print(f"⚠️ Save position error: {e}")

    def load_position_state(self):
        try:
            self.cursor.execute('SELECT * FROM position LIMIT 1')
            row = self.cursor.fetchone()
            if row:
                cols = [c[0] for c in self.cursor.description]
                r = dict(zip(cols, row))
                bot_state['position'] = {
                    'symbol': r['symbol'], 'direction': r['direction'],
                    'size': r['size'], 'entry': r['entry'],
                    'stop': r['stop'], 'target': r['target'],
                    'order_id': r.get('order_id', ''),
                    'timestamp': r.get('timestamp') or datetime.now(timezone.utc).isoformat(),
                    'exit_25_taken': bool(r.get('exit_25_taken', False)),
                    'exit_50_taken': bool(r.get('exit_50_taken', False)),
                    'stop_moved_to_be': bool(r.get('stop_moved_to_be', False)),
                    'original_stop': r.get('original_stop', r['stop']),
                    'highest_price': r.get('highest_price', r['entry']),
                    'lowest_price': r.get('lowest_price', r['entry']),
                }
                print(f"🔄 Recovered active position: {r['direction']} {r['symbol']}")
            else:
                bot_state['position'] = None
        except Exception as e:
            print(f"⚠️ Load position error: {e}")
            bot_state['position'] = None

    def clear_position_state(self):
        try:
            self.cursor.execute('DELETE FROM position')
            self.conn.commit()
        except Exception as e:
            print(f"⚠️ Clear position error: {e}")
        bot_state['position'] = None

    # ── Spread & session guard ─────────────────────────────────────────────

    def check_spread_and_session(self, symbol):
        now_utc = datetime.now(timezone.utc)
        h = now_utc.hour + now_utc.minute / 60.0
        if rollover_start_utc <= h <= rollover_end_utc:
            print(f" 🚫 BLOCKED: Rollover window ({h:.2f} UTC).")
            return False, "ROLLOVER_WINDOW", 0.0
        if not (london_open_utc <= h <= ny_close_utc):
            print(f" 🚫 BLOCKED: Outside London/NY session ({h:.2f} UTC). Active: 07:00–16:30 UTC.")
            return False, "OUT_OF_SESSION", 0.0

        pip_size    = SYMBOL_SPECS[symbol]["pip_size"]
        spread_pips = 1.0
        if self.connected and not self.dry_run:
            instrument = OANDA_INSTRUMENTS[symbol]
            resp = self._get(f"accounts/{OANDA_ACCOUNT_ID}/pricing",
                             params={"instruments": instrument})
            if resp and resp.get("prices"):
                p   = resp["prices"][0]
                ask = float(p.get("asks", [{"price": "0"}])[0]["price"])
                bid = float(p.get("bids", [{"price": "0"}])[0]["price"])
                spread_pips = (ask - bid) / pip_size

        print(f" 🔍 {symbol} Spread: {spread_pips:.1f} pips (max {max_spread_pips})")
        if spread_pips > max_spread_pips:
            print(f" 🚫 BLOCKED: Spread too wide.")
            return False, f"HIGH_SPREAD_{spread_pips:.1f}", spread_pips
        return True, "LONDON_NY_OVERLAP", spread_pips

    def fetch_multi_timeframe_data(self, symbol):
        data = {}
        instrument = OANDA_INSTRUMENTS.get(symbol, symbol)
        for tf_name, gran in OANDA_GRANULARITY.items():
            fetched = False
            if self.connected and not self.dry_run:
                resp = self._get(f"instruments/{instrument}/candles",
                                 params={"granularity": gran, "count": 100, "price": "M"})
                if resp and "candles" in resp:
                    cs = [c for c in resp["candles"] if c.get("complete", True)]
                    if len(cs) >= 30:
                        data[tf_name] = {
                            "close":  np.array([float(c["mid"]["c"]) for c in cs]),
                            "high":   np.array([float(c["mid"]["h"]) for c in cs]),
                            "low":    np.array([float(c["mid"]["l"]) for c in cs]),
                            "volume": np.array([float(c.get("volume", 100)) for c in cs]),
                        }
                        fetched = True
            if not fetched:
                np.random.seed(int(time.time() * 1000) % 100000)
                base   = 1.0850 if symbol == "EURUSD" else (1.265 if symbol == "GBPUSD" else 155.0)
                closes = base + np.random.normal(0, 0.0005, 100).cumsum()
                data[tf_name] = {
                    "close":  closes,
                    "high":   closes + np.abs(np.random.normal(0, 0.0002, 100)),
                    "low":    closes - np.abs(np.random.normal(0, 0.0002, 100)),
                    "volume": np.random.randint(100, 1000, 100).astype(float),
                }
        data["60"] = data["1h"]; data["240"] = data["4h"]
        return data

    def calculate_indicators(self, data):
        indicators = {}
        for tf in ["15m", "1h", "4h"]:
            closes = data[tf]["close"]; highs = data[tf]["high"]
            lows   = data[tf]["low"];   volumes = data[tf]["volume"]
            if HAS_TALIB:
                rsi     = talib.RSI(closes, 14)[-1]
                ml,ms,mh = talib.MACD(closes, 12, 26, 9)
                ema_21  = talib.EMA(closes, 21)[-1]; ema_50 = talib.EMA(closes, 50)[-1]
                atr     = talib.ATR(highs, lows, closes, 14)[-1]
                adx     = talib.ADX(highs, lows, closes, 14)[-1]
                sk, sd  = talib.STOCH(highs, lows, closes, 14, 3, 3)
                vol_sma = talib.SMA(volumes, 20)[-1]
            else:
                def _ema(a, n):
                    k, v = 2.0/(n+1), float(np.mean(a[:n]))
                    for x in a[n:]: v = float(x)*k + v*(1-k)
                    return v
                P = 14
                d = np.diff(closes)
                g = np.where(d>0,d,0.0); l = np.where(d<0,-d,0.0)
                ag, al = float(np.mean(g[:P])), float(np.mean(l[:P]))
                for i in range(P, len(g)):
                    ag=(ag*(P-1)+g[i])/P; al=(al*(P-1)+l[i])/P
                rsi    = 100.0 - 100.0/(1.0+ag/(al+1e-9))
                ema_21 = _ema(closes,21); ema_50 = _ema(closes,50)
                mv  = [_ema(closes[:i+1],12)-_ema(closes[:i+1],26)
                       for i in range(max(26,len(closes)-20), len(closes))]
                mlv = _ema(closes,12)-_ema(closes,26)
                msv = _ema(np.array(mv),9) if len(mv)>=9 else mv[-1]
                ml  = np.array([mlv]); ms = np.array([msv]); mh = np.array([mlv-msv])
                tr  = np.maximum(highs[1:]-lows[1:],np.maximum(
                      np.abs(highs[1:]-closes[:-1]),np.abs(lows[1:]-closes[:-1])))
                atr = float(np.mean(tr[:P]))
                for v in tr[P:]: atr=(atr*(P-1)+float(v))/P
                rh  = float(np.max(highs[-P:])); rl = float(np.min(lows[-P:]))
                skv = ((float(closes[-1])-rl)/(rh-rl+1e-9))*100
                sk  = np.array([skv]); sd = np.array([skv])
                dmp = np.where((highs[1:]-highs[:-1])>(lows[:-1]-lows[1:]),
                               np.maximum(highs[1:]-highs[:-1],0.0),0.0)
                dmm = np.where((lows[:-1]-lows[1:])>(highs[1:]-highs[:-1]),
                               np.maximum(lows[:-1]-lows[1:],0.0),0.0)
                aw  = np.mean(tr[-P:])
                adx = 100*abs(100*np.mean(dmp[-P:])/(aw+1e-9)-100*np.mean(dmm[-P:])/(aw+1e-9)) / \
                      (100*np.mean(dmp[-P:])/(aw+1e-9)+100*np.mean(dmm[-P:])/(aw+1e-9)+1e-9)
                vol_sma = np.mean(volumes[-20:])
            indicators[tf] = {
                "rsi": float(rsi), "macd": float(ml[-1]), "macd_signal": float(ms[-1]),
                "macd_histogram": float(mh[-1]), "ema_21": float(ema_21), "ema_50": float(ema_50),
                "atr": float(atr), "adx": float(adx),
                "stoch_k": float(sk[-1]), "stoch_d": float(sd[-1]),
                "volume_ratio": float(volumes[-1]/(vol_sma+1e-9)),
            }
        cur = float(data["15m"]["close"][-1])
        ref = float(data["1h"]["close"][-24] if len(data["1h"]["close"])>=24 else data["1h"]["close"][0])
        return indicators, cur, abs((cur-ref)/ref)

    def evaluate_multi_factor_consensus(self, symbol, indicators, current_price, volatility, data, spread_pips):
        """Identical multi-factor logic to mt5.py."""
        ind4h = indicators.get("4h", {})
        trend_bullish = ind4h.get("ema_21", 0) > ind4h.get("ema_50", 0)
        trend_score   = 1.0 if trend_bullish else -1.0

        long_conds  = [indicators["15m"]["rsi"] < 45, indicators["1h"]["rsi"] < 52,
                       indicators["15m"]["macd"] > indicators["15m"]["macd_signal"],
                       current_price > indicators["15m"]["ema_21"] * 0.999,
                       volatility > min_volatility_threshold,
                       indicators["15m"]["adx"] > 18, indicators["15m"]["stoch_k"] < 35]
        short_conds = [indicators["15m"]["rsi"] > 55, indicators["1h"]["rsi"] > 48,
                       indicators["15m"]["macd"] < indicators["15m"]["macd_signal"],
                       current_price < indicators["15m"]["ema_21"] * 1.001,
                       volatility > min_volatility_threshold,
                       indicators["15m"]["adx"] > 18, indicators["15m"]["stoch_k"] > 65]
        long_ta  = sum(long_conds); short_ta = sum(short_conds)
        ta_score = (long_ta - short_ta) / 7.0

        sr_data = {"score": 0.0, "scenario": "MID_RANGE", "suggested_stop": None, "suggested_target": None}
        if HAS_SR_FACTOR and get_sr_score:
            try: sr_data = get_sr_score(symbol, current_price, indicators, data)
            except Exception as e: print(f" ⚠️ S/R: {e}")
        sr_score = sr_data.get("score", 0.0)

        news_data       = get_forex_news_score(symbol)
        news_score      = news_data.get("score", 0.0)
        block_long_only = news_data.get("block_long_only", False)
        spread_score    = max(0.0, 1.0 - (spread_pips / max_spread_pips))

        final_score = (WEIGHTS["trend_alignment"]    * trend_score +
                       WEIGHTS["support_resistance"] * sr_score    +
                       WEIGHTS["technical"]          * ta_score    +
                       WEIGHTS["news"]               * news_score  +
                       WEIGHTS["spread_volatility"]  * spread_score)

        print(f" 🔬 Multi-Factor ({symbol}): Trend={trend_score:+.2f} S/R={sr_score:+.2f} "
              f"TA={ta_score:+.2f} News={news_score:+.2f} → Score={final_score:+.3f}")

        signal = None
        if final_score >= 0.25 and trend_bullish and not block_long_only:
            signal = "LONG"
        elif final_score <= -0.25 and not trend_bullish:
            signal = "SHORT"
        if block_long_only and final_score >= 0.25:
            print(f" 🚫 LONG blocked by news: {news_data.get('block_reason')}")

        return {"signal": signal, "final_score": round(final_score, 3),
                "news_score": news_score, "sr_scenario": sr_data.get("scenario", "MID_RANGE"),
                "suggested_stop": sr_data.get("suggested_stop"),
                "suggested_target": sr_data.get("suggested_target"),
                "strength": max(long_ta, short_ta)}

    def calculate_forex_position_size(self, symbol, entry_price, stop_loss_price):
        specs    = SYMBOL_SPECS.get(symbol, SYMBOL_SPECS["EURUSD"])
        pip_size = specs["pip_size"]; min_lot = specs["min_lot"]
        lot_step = specs["lot_step"]; contract_size = specs["contract_size"]
        balance  = bot_state['available_balance']
        risk_amt = balance * forex_risk_per_trade
        stop_dist = abs(entry_price - stop_loss_price)
        if stop_dist <= 0:
            print("❌ Invalid stop distance"); return None
        pip_dist = stop_dist / pip_size
        pip_value = (contract_size * pip_size if specs.get("quote_is_usd", True)
                     else (contract_size * pip_size) / entry_price)
        raw_lots = risk_amt / (pip_dist * pip_value)
        lots     = max(min_lot, round(round(raw_lots / lot_step) * lot_step, 2))
        print(f" 📊 {symbol}: balance=${balance:.2f} risk=${risk_amt:.2f} "
              f"stop={pip_dist:.1f}pips lots={lots:.2f}")
        return {"lots": lots, "risk_amount": risk_amt, "pip_distance": pip_dist}

    # ── OANDA-specific: execute trade ─────────────────────────────────────

    def execute_trade(self, symbol, direction, entry_price, indicators, signal_data,
                      spread_pips=0.0, volatility=0.0):
        pip_size   = SYMBOL_SPECS[symbol]["pip_size"]
        atr_15m    = indicators["15m"]["atr"]
        sug_stop   = signal_data.get("suggested_stop")
        sug_target = signal_data.get("suggested_target")

        if sug_stop and sug_target and signal_data.get("sr_scenario") != "MID_RANGE":
            stop_loss   = sug_stop; take_profit = sug_target
            stop_pips   = abs(entry_price - stop_loss) / pip_size
            reward_pips = abs(take_profit - entry_price) / pip_size
        else:
            stop_pips   = max(15.0, (1.5 * atr_15m) / pip_size)
            reward_pips = stop_pips * min_reward_ratio
            if direction == "LONG":
                stop_loss   = entry_price - stop_pips * pip_size
                take_profit = entry_price + reward_pips * pip_size
            else:
                stop_loss   = entry_price + stop_pips * pip_size
                take_profit = entry_price - reward_pips * pip_size

        pos_size = self.calculate_forex_position_size(symbol, entry_price, stop_loss)
        if not pos_size: return False
        lots  = pos_size["lots"]
        units = int(lots * 100000)  # OANDA uses units (base currency)
        if direction == "SHORT": units = -units

        order_id    = f"DRY_RUN_{int(time.time())}"
        fill_price  = entry_price

        if self.connected and not self.dry_run:
            instrument = OANDA_INSTRUMENTS[symbol]
            payload = {"order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "stopLossOnFill":   {"price": f"{stop_loss:.5f}",   "timeInForce": "GTC"},
                "takeProfitOnFill": {"price": f"{take_profit:.5f}", "timeInForce": "GTC"},
                "tradeClientExtensions": {"comment": f"OandaBot_{BOT_MAGIC_NUMBER}"},
            }}
            resp = self._post(f"accounts/{OANDA_ACCOUNT_ID}/orders", payload)
            if resp and "orderFillTransaction" in resp:
                fill  = resp["orderFillTransaction"]
                order_id   = fill.get("tradeOpened", {}).get("tradeID", order_id)
                fill_price = float(fill.get("price", entry_price))
                print(f"✅ OANDA Order filled! Trade #{order_id} | Fill: {fill_price:.5f}")
            else:
                print(f"❌ OANDA Order failed: {resp}")
                return False

        bot_state['position'] = {
            'symbol': symbol, 'direction': direction, 'size': lots,
            'entry': fill_price, 'stop': stop_loss, 'target': take_profit,
            'original_stop': stop_loss, 'order_id': order_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'highest_price': fill_price, 'lowest_price': fill_price,
            'exit_25_taken': False, 'exit_50_taken': False, 'stop_moved_to_be': False,
            'spread_pips': spread_pips, 'volatility': volatility,
            'atr_15m': atr_15m, 'news_score': signal_data.get("news_score", 0.0),
            'sr_scenario': signal_data.get("sr_scenario", "MID_RANGE"),
        }
        self.save_position_state()
        print(f" 🚀 {direction} [{symbol}] lots={lots} entry={fill_price:.5f} "
              f"SL={stop_loss:.5f} TP={take_profit:.5f}")
        return True

    # ── Position management ────────────────────────────────────────────────

    def get_live_price(self, symbol, direction):
        """Fetch live bid (for LONG) or ask (for SHORT) from OANDA pricing endpoint."""
        if self.connected and not self.dry_run:
            instrument = OANDA_INSTRUMENTS[symbol]
            resp = self._get(f"accounts/{OANDA_ACCOUNT_ID}/pricing",
                             params={"instruments": instrument})
            if resp and resp.get("prices"):
                p = resp["prices"][0]
                return float(p["bids"][0]["price"] if direction == "LONG"
                             else p["asks"][0]["price"])
        return None

    def update_oanda_stop(self, order_id, new_stop):
        """Modify stop loss on an open OANDA trade (break-even move)."""
        if self.connected and not self.dry_run and not str(order_id).startswith("DRY"):
            payload = {"stopLoss": {"price": f"{new_stop:.5f}", "timeInForce": "GTC"}}
            self._put(f"accounts/{OANDA_ACCOUNT_ID}/trades/{order_id}/orders", payload)

    def manage_position(self, current_price):
        pos = bot_state['position']
        if not pos: return
        symbol    = pos['symbol']; direction = pos['direction']
        entry     = pos['entry']; stop = pos['stop']
        target    = pos['target']; pip_size = SYMBOL_SPECS[symbol]["pip_size"]

        pos['highest_price'] = max(pos.get('highest_price', entry), current_price)
        pos['lowest_price']  = min(pos.get('lowest_price',  entry), current_price)

        if direction == "LONG":
            profit_pips = (current_price - entry) / pip_size
            risk_pips   = (entry - pos['original_stop']) / pip_size
            if profit_pips >= risk_pips and not pos.get('stop_moved_to_be'):
                new_stop = entry + 2.0 * pip_size
                pos['stop'] = new_stop; pos['stop_moved_to_be'] = True
                self.update_oanda_stop(pos.get('order_id'), new_stop)
                self.save_position_state()
                print(f" 🛡️ [{symbol}] Break-even stop → {new_stop:.5f}")
            if current_price <= stop:   self.close_position(current_price, "STOP_LOSS")
            elif current_price >= target: self.close_position(current_price, "TAKE_PROFIT")

        elif direction == "SHORT":
            profit_pips = (entry - current_price) / pip_size
            risk_pips   = (pos['original_stop'] - entry) / pip_size
            if profit_pips >= risk_pips and not pos.get('stop_moved_to_be'):
                new_stop = entry - 2.0 * pip_size
                pos['stop'] = new_stop; pos['stop_moved_to_be'] = True
                self.update_oanda_stop(pos.get('order_id'), new_stop)
                self.save_position_state()
                print(f" 🛡️ [{symbol}] Break-even stop → {new_stop:.5f}")
            if current_price >= stop:   self.close_position(current_price, "STOP_LOSS")
            elif current_price <= target: self.close_position(current_price, "TAKE_PROFIT")

    def close_position(self, exit_price, reason):
        pos = bot_state['position']
        if not pos: return
        symbol    = pos['symbol']; direction = pos['direction']
        entry     = pos['entry']; lots = pos['size']
        pip_size  = SYMBOL_SPECS[symbol]["pip_size"]
        specs     = SYMBOL_SPECS[symbol]

        # Close on OANDA side
        if self.connected and not self.dry_run and not str(pos.get('order_id','')).startswith("DRY"):
            instrument = OANDA_INSTRUMENTS[symbol]
            key = "longUnits" if direction == "LONG" else "shortUnits"
            self._put(f"accounts/{OANDA_ACCOUNT_ID}/positions/{instrument}/close",
                      {key: "ALL"})

        pips = (exit_price - entry)/pip_size if direction == "LONG" else (entry - exit_price)/pip_size
        pip_val = (specs["contract_size"] * pip_size if specs.get("quote_is_usd", True)
                   else (specs["contract_size"] * pip_size) / exit_price)
        pnl  = pips * pip_val * lots

        bot_state['total_trades']   += 1
        bot_state['session_pnl']    += pnl
        bot_state['daily_trades']   += 1
        if pnl > 0:
            bot_state['winning_trades']    += 1
            bot_state['consecutive_losses'] = 0
        else:
            bot_state['consecutive_losses'] += 1

        try:
            self.cursor.execute('''
                INSERT INTO trade_log (symbol,entry_time,exit_time,direction,entry_price,
                exit_price,size,pnl,result,ta_signal_strength,spread_pips,volatility,
                atr_15m,news_score,sr_scenario,session_window) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (symbol, pos['timestamp'], datetime.now(timezone.utc).isoformat(),
                  direction, entry, exit_price, lots, pnl, reason,
                  float(pos.get('strength', 0.0)), float(pos.get('spread_pips', 0.0)),
                  float(pos.get('volatility', 0.0)), float(pos.get('atr_15m', 0.0)),
                  float(pos.get('news_score', 0.0)), str(pos.get('sr_scenario', 'UNKNOWN')),
                  'LIVE' if self.connected else 'DRY_RUN'))
            self.conn.commit()
        except Exception as e:
            print(f"⚠️ Trade log error: {e}")

        wr = (bot_state['winning_trades']/bot_state['total_trades']*100) if bot_state['total_trades'] else 0
        print(f" 🏁 CLOSED [{symbol}] {reason} | Exit={exit_price:.5f} PnL=${pnl:.2f} ({pips:.1f}pips)")
        print(f" 📊 Session: trades={bot_state['total_trades']} WR={wr:.0f}% "
              f"PnL=${bot_state['session_pnl']:.2f} consec_loss={bot_state['consecutive_losses']}")
        self.clear_position_state()

        max_loss = bot_state['session_start_balance'] * MAX_DAILY_LOSS_PCT
        if bot_state['session_pnl'] < -max_loss:
            print(f"🛑 DAILY LOSS LIMIT HIT (${bot_state['session_pnl']:.2f}). Bot halted.")
            sys.exit(0)
        if bot_state['consecutive_losses'] >= bot_state['max_consecutive_losses']:
            print(f"🛑 CIRCUIT BREAKER: {bot_state['consecutive_losses']} consecutive losses. Bot halted.")
            sys.exit(0)

    # ── Main cycle ─────────────────────────────────────────────────────────

    def run_cycle(self):
        print(f"\n🔄 --- OANDA Cycle [{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] ---")

        if bot_state['position']:
            pos    = bot_state['position']
            symbol = pos['symbol']
            cur    = self.get_live_price(symbol, pos['direction'])
            if cur is None:
                data = self.fetch_multi_timeframe_data(symbol)
                cur  = float(data['15m']['close'][-1]) if data else None
            if cur:
                print(f" 📊 Open: {pos['direction']} {symbol} | Price={cur:.5f}")
                self.manage_position(cur)
            else:
                print(f" ⚠️ Cannot fetch price for {symbol}")
            return

        for symbol in TRADE_SYMBOLS:
            print(f"\n🔍 Evaluating {symbol}...")
            ok, label, spread_pips = self.check_spread_and_session(symbol)
            if not ok: continue
            data = self.fetch_multi_timeframe_data(symbol)
            if not data: continue
            indicators, cur_price, volatility = self.calculate_indicators(data)
            print(f"   Price={cur_price:.5f} vol={volatility*100:.3f}% ATR={indicators['15m']['atr']:.5f}")
            sig_data = self.evaluate_multi_factor_consensus(
                symbol, indicators, cur_price, volatility, data, spread_pips)
            signal = sig_data.get("signal")
            if signal in ("LONG", "SHORT"):
                print(f" ⚡ SIGNAL: {signal} on {symbol}")
                if self.execute_trade(symbol, signal, cur_price, indicators,
                                      sig_data, spread_pips, volatility):
                    break
        print("\n✅ Cycle complete.")


def main():
    parser = argparse.ArgumentParser(description="OANDA v20 REST API Forex Trading Bot")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without placing real orders")
    parser.add_argument("--single-cycle", action="store_true", help="Run one cycle then exit")
    args = parser.parse_args()

    bot = OandaForexBot(dry_run=args.dry_run)
    if args.single_cycle:
        bot.run_cycle()
    else:
        print("🔄 OANDA Bot running (15s interval). Ctrl+C to stop.")
        try:
            while True:
                bot.run_cycle()
                time.sleep(15)
        except KeyboardInterrupt:
            print("\n🛑 OANDA Bot stopped by user.")


if __name__ == "__main__":
    main()



