"""
Standalone Futures Trading Bot - Extracted from Hybrid Bot
Focus: Pure futures trading with enhanced risk management
"""

import requests
import hmac
import hashlib
import time
try:
    import talib
except ImportError:
    talib = None
import numpy as np
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
import os
import json
import sqlite3
from factors.aggregator import MultiFactorAggregator
from factors.regime import get_regime_score

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")

# Trading Configuration
TRADE_SYMBOLS = ["SOLUSDT", "ETHUSDT", "AVAXUSDT", "LINKUSDT", "BNBUSDT"]

# Per-symbol Bybit linear contract specs (last verified 2026-08)
# min_qty  : minimum order size in base coin units
# step_size: order size increment
# ticker   : human-readable base coin label for log messages
SYMBOL_CONTRACT_SPECS = {
    "SOLUSDT":  {"min_qty": 0.1,  "step_size": 0.1,  "ticker": "SOL"},
    "ETHUSDT":  {"min_qty": 0.01, "step_size": 0.01, "ticker": "ETH"},
    "AVAXUSDT": {"min_qty": 0.1,  "step_size": 0.1,  "ticker": "AVAX"},
    "LINKUSDT": {"min_qty": 0.1,  "step_size": 0.1,  "ticker": "LINK"},
    "BNBUSDT":  {"min_qty": 0.01, "step_size": 0.01, "ticker": "BNB"},
}
primary_timeframe = "15"   # Primary analysis
higher_timeframe = "60"    # Trend confirmation

# Futures Risk Management
futures_risk_per_trade = 0.02  # 2% risk per futures trade
max_leverage = 20.0            # Conservative max leverage
min_reward_ratio = 2.0         # 2:1 R:R — achievable on 15m; 3:1 target was never reached in backtesting
min_volatility_threshold = 0.02

# Trading Limits
max_daily_trades = 15
min_trade_gap_hours = 2
signal_strength_threshold = 5

# Initialize Bybit session
session = HTTP(
    testnet=False,
    api_key=api_key,
    api_secret=api_secret,
    recv_window=15000,
)

# Futures Trading State
futures_state = {
    'last_trade_time': None,
    'position': None,
    'daily_trades': 0,
    'total_trades': 0,
    'winning_trades': 0,
    'consecutive_losses': 0,
    'max_consecutive_losses': 3,
    'session_start': datetime.now(),
    'available_balance': 0,
    # P&L circuit breaker: halt if session losses exceed 5% of starting balance
    'session_start_balance': 0,
    'session_pnl': 0.0,
}

class FuturesTradingBot:
    
    def __init__(self):
        print(f"🚀 Initializing Futures Trading Bot...")
        self.init_db()
        self.load_position_state()
        self.initialize_balance()
        self.state_file = 'trading_state.json'

    def init_db(self):
        """Initialize SQLite database for state persistence"""
        self.conn = sqlite3.connect('trading_state.db', timeout=30.0, check_same_thread=False)
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
                leverage REAL,
                margin REAL,
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
            CREATE TABLE IF NOT EXISTS ai_training_data (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                timestamp TEXT,
                current_price REAL,
                rsi_15m REAL,
                rsi_1h REAL,
                macd_15m REAL,
                macd_hist_15m REAL,
                adx_15m REAL,
                adx_1h REAL,
                volatility REAL,
                btc_1h_change REAL,
                btc_4h_change REAL,
                volume_ratio_15m REAL
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_trade_outcomes (
                id INTEGER PRIMARY KEY,
                entry_time TEXT,
                exit_time TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                max_favorable_price REAL,
                max_adverse_price REAL
            )
        ''')
        # ── Rich trade log — all factor context saved at entry ──────────
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

                -- Signal quality
                ta_signal_strength REAL,
                aggregated_score   REAL,
                volatility         REAL,
                atr_15m            REAL,

                -- Multi-factor scores (−1 → +1)
                technical_score    REAL,
                regime_score       REAL,
                derivatives_score  REAL,
                sentiment_score    REAL,
                news_score         REAL,
                sr_score           REAL,
                sr_scenario        TEXT,

                -- Regime classification
                regime_class       TEXT,

                -- Derivatives snapshot at entry
                funding_rate       REAL,
                open_interest      REAL,
                long_short_ratio   REAL,

                -- News & trend context
                news_sentiment     TEXT,
                market_trend_4h    TEXT
            )
        ''')
        self.conn.commit()

    def save_position_state(self):
        """Save current position state to SQLite"""
        if not futures_state['position']:
            return
            
        pos = futures_state['position']
        for attempt in range(3):
            try:
                self.cursor.execute('DELETE FROM position')
                self.cursor.execute('''
                    INSERT INTO position (
                        symbol, direction, size, entry, stop, target, leverage, margin,
                        order_id, timestamp, exit_25_taken, exit_50_taken,
                        stop_moved_to_be, original_stop, highest_price, lowest_price
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pos['symbol'], pos['direction'], pos['size'], pos['entry'], pos['stop'],
                    pos['target'], pos['leverage'], pos['margin'], pos['order_id'],
                    str(pos['timestamp']), int(pos.get('exit_25_taken', False)), 
                    int(pos.get('exit_50_taken', False)), int(pos.get('stop_moved_to_be', False)),
                    pos.get('original_stop', pos['stop']), pos.get('highest_price', pos['entry']),
                    pos.get('lowest_price', pos['entry'])
                ))
                self.conn.commit()
                break
            except sqlite3.OperationalError as e:
                if attempt < 2:
                    time.sleep(0.2 * (attempt + 1))
                else:
                    print(f"⚠️ Warning: database is locked when saving position state ({e}). Will retry next cycle.")
            except Exception as e:
                print(f"⚠️ Error saving position state: {e}")
                break
        
    def load_position_state(self):
        """Load active position from SQLite on startup"""
        try:
            self.cursor.execute('SELECT * FROM position LIMIT 1')
            row = self.cursor.fetchone()
        except Exception as e:
            print(f"⚠️ Error reading position state from DB: {e}")
            row = None

        if row:
            columns = [col[0] for col in self.cursor.description]
            row_dict = dict(zip(columns, row))
            recovered_symbol = row_dict.get('symbol')
            if not recovered_symbol:
                print("⚠️ DB recovery: position record has no symbol. Clearing stale state.")
                self.clear_position_state()
                return
            print(f"🔄 Recovering active position from database...")
            futures_state['position'] = {
                'symbol': recovered_symbol,
                'direction': row_dict.get('direction'),
                'size': row_dict.get('size'),
                'entry': row_dict.get('entry'),
                'stop': row_dict.get('stop'),
                'target': row_dict.get('target'),
                'leverage': row_dict.get('leverage'),
                'margin': row_dict.get('margin'),
                'order_id': row_dict.get('order_id'),
                'timestamp': row_dict.get('timestamp') or datetime.now(),
                'exit_25_taken': bool(row_dict.get('exit_25_taken', False)),
                'exit_50_taken': bool(row_dict.get('exit_50_taken', False)),
                'stop_moved_to_be': bool(row_dict.get('stop_moved_to_be', False)),
                'original_stop': row_dict.get('original_stop'),
                'highest_price': row_dict.get('highest_price'),
                'lowest_price': row_dict.get('lowest_price')
            }
            # Verify with exchange (optional but safe)
            try:
                result = session.get_positions(category="linear", symbol=futures_state['position']['symbol'])
                if result.get("retCode") == 0:
                    pos_list = result.get("result", {}).get("list", [])
                    active_size = float(pos_list[0].get("size", "0")) if pos_list else 0
                    if active_size == 0:
                        print("⚠️ DB position found, but Bybit reports no open position. Clearing state.")
                        self.clear_position_state()
                    else:
                        print(f"✅ Bybit confirmed open position of size {active_size}.")
            except Exception as e:
                print(f"⚠️ Could not verify position with Bybit: {e}")
        else:
            futures_state['position'] = None

    def clear_position_state(self):
        """Clear position from SQLite"""
        for attempt in range(3):
            try:
                self.cursor.execute('DELETE FROM position')
                self.conn.commit()
                break
            except sqlite3.OperationalError as e:
                if attempt < 2:
                    time.sleep(0.2 * (attempt + 1))
                else:
                    print(f"⚠️ Warning: database is locked when clearing position state ({e}).")
            except Exception as e:
                print(f"⚠️ Error clearing position state: {e}")
                break
        futures_state['position'] = None
        
    def save_state(self):
        """Save critical state to disk"""
        try:
            state_to_save = {
                'position': self.serialize_position(futures_state['position']),
                'daily_trades': futures_state['daily_trades'],
                'total_trades': futures_state['total_trades'],
                'winning_trades': futures_state['winning_trades'],
                'consecutive_losses': futures_state['consecutive_losses']
            }
            with open(self.state_file, 'w') as f:
                json.dump(state_to_save, f)
        except Exception as e:
            print(f"⚠️ Error saving state: {e}")

    def load_state(self):
        """Load state from disk"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    saved_state = json.load(f)
                    futures_state['position'] = self.deserialize_position(saved_state.get('position'))
                    futures_state['daily_trades'] = saved_state.get('daily_trades', 0)
                    futures_state['total_trades'] = saved_state.get('total_trades', 0)
                    futures_state['winning_trades'] = saved_state.get('winning_trades', 0)
                    futures_state['consecutive_losses'] = saved_state.get('consecutive_losses', 0)
                print("✅ State loaded successfully")
            except Exception as e:
                print(f"⚠️ Error loading state: {e}")

    def serialize_position(self, pos):
        if not pos: return None
        pos_copy = pos.copy()
        if 'timestamp' in pos_copy and isinstance(pos_copy['timestamp'], datetime):
            pos_copy['timestamp'] = pos_copy['timestamp'].isoformat()
        return pos_copy

    def deserialize_position(self, pos):
        if not pos: return None
        if 'timestamp' in pos and isinstance(pos['timestamp'], str):
            pos['timestamp'] = datetime.fromisoformat(pos['timestamp'])
        return pos

    def log_features(self, symbol, current_price, indicators, btc_data, volatility):
        """Log features for AI training (symbol column added for multi-asset ML)"""
        try:
            self.cursor.execute('''
                INSERT INTO ai_training_data (
                    symbol, timestamp, current_price, rsi_15m, rsi_1h, macd_15m,
                    macd_hist_15m, adx_15m, adx_1h, volatility,
                    btc_1h_change, btc_4h_change, volume_ratio_15m
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                datetime.now().isoformat(),
                current_price,
                indicators['15m']['rsi'],
                indicators['1h']['rsi'],
                indicators['15m']['macd'],
                indicators['15m']['macd_histogram'],
                indicators['15m']['adx'],
                indicators['1h']['adx'],
                volatility,
                btc_data['1h_change'],
                btc_data['4h_change'],
                indicators['15m']['volume_ratio']
            ))
            self.conn.commit()
        except Exception as e:
            print(f"⚠️ Error logging features: {e}")

    def log_trade_outcome(self, pnl=None, exit_price=None, result=None,
                          exit_time=None, factor_context=None):
        """
        Log trade outcome to BOTH ai_trade_outcomes (legacy) and the new
        trade_log table which includes all multi-factor context.

        factor_context dict (all keys optional):
            ta_signal_strength, aggregated_score, volatility, atr_15m,
            technical_score, regime_score, derivatives_score, sentiment_score,
            news_score, regime_class,
            funding_rate, open_interest, long_short_ratio,
            news_sentiment, market_trend_4h
        """
        pos = futures_state['position']
        if not pos: return

        fc = factor_context or {}
        try:
            entry_time = pos.get('timestamp')
            if isinstance(entry_time, datetime):
                entry_time = entry_time.isoformat()
            now_str = (exit_time or datetime.now()).isoformat() if not isinstance(
                exit_time, str) else exit_time

            # ── legacy table (unchanged) ───────────────────────────────
            self.cursor.execute('''
                INSERT INTO ai_trade_outcomes (
                    entry_time, exit_time, direction, entry_price,
                    exit_price, pnl, max_favorable_price, max_adverse_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry_time, now_str, pos['direction'], pos['entry'],
                exit_price or pos.get('target'), pnl or 0,
                pos.get('highest_price', pos['entry']),
                pos.get('lowest_price', pos['entry'])
            ))

            # ── rich trade_log table ────────────────────────────────────
            self.cursor.execute('''
                INSERT INTO trade_log (
                    symbol, entry_time, exit_time, direction, entry_price, exit_price,
                    size, pnl, result,
                    ta_signal_strength, aggregated_score, volatility, atr_15m,
                    technical_score, regime_score, derivatives_score,
                    sentiment_score, news_score, sr_score, sr_scenario, regime_class,
                    funding_rate, open_interest, long_short_ratio,
                    news_sentiment, market_trend_4h
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                pos.get('symbol', 'UNKNOWN'),
                entry_time,
                now_str,
                pos['direction'],
                pos['entry'],
                exit_price or pos.get('target'),
                pos.get('size', 0),
                pnl or 0,
                result or 'UNKNOWN',
                fc.get('ta_signal_strength'),
                fc.get('aggregated_score'),
                fc.get('volatility'),
                fc.get('atr_15m'),
                fc.get('technical_score'),
                fc.get('regime_score'),
                fc.get('derivatives_score'),
                fc.get('sentiment_score'),
                fc.get('news_score'),
                fc.get('sr_score'),
                fc.get('sr_scenario'),
                fc.get('regime_class'),
                fc.get('funding_rate'),
                fc.get('open_interest'),
                fc.get('long_short_ratio'),
                fc.get('news_sentiment'),
                fc.get('market_trend_4h'),
            ))

            self.conn.commit()
            print(f"✅ Trade logged — PnL: ${pnl:.2f}  Result: {result}")
        except Exception as e:
            print(f"⚠️ Error logging outcome: {e}")
        
    def initialize_balance(self):
        """Initialize futures trading balance"""
        usdt_balance = self.get_usdt_balance()
        
        # Minimum balance validation
        min_balance_required = 25  # $25 minimum for meaningful futures trading
        
        if usdt_balance < min_balance_required:
            print(f"⚠️ WARNING: USDT balance ${usdt_balance:.2f} is below recommended minimum ${min_balance_required}")
        
        futures_state['available_balance'] = usdt_balance
        # Seed circuit breaker baseline (only set once at startup)
        if futures_state['session_start_balance'] == 0:
            futures_state['session_start_balance'] = usdt_balance
        
        print(f"💰 Futures Balance Initialized:")
        print(f"   Available USDT: ${usdt_balance:.2f}")
        print(f"   Risk Per Trade: {futures_risk_per_trade*100:.1f}% = ${usdt_balance * futures_risk_per_trade:.2f}")
        print(f"   Max Leverage: {max_leverage:.1f}x")
        
        if usdt_balance < 15:
            print(f"⚠️ USDT balance too low for meaningful futures trades (need $15+ USDT margin)")
    
    def get_usdt_balance(self):
        """Get actual USDT balance (needed for futures margin)"""
        try:
            result = session.get_wallet_balance(accountType="UNIFIED")
            if result.get("retCode") == 0:
                account_list = result.get("result", {}).get("list", [])
                if account_list:
                    coins = account_list[0].get("coin", [])
                    for coin in coins:
                        if coin["coin"] == "USDT":
                            balance = float(coin.get("walletBalance", "0"))
                            return balance
            return 0.0
        except Exception as e:
            print(f"❌ Error getting USDT balance: {e}")
            return 0.0
    
    def fetch_multi_timeframe_data(self, symbol):
        """Fetch data from multiple timeframes"""
        data = {}
        
        for tf in [primary_timeframe, higher_timeframe, "240"]:  # 15m, 1h, 4h
            try:
                result = session.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=tf,
                    limit=100
                )
                if result.get("retCode") == 0:
                    candles = result["result"]["list"]
                    candles = list(reversed(candles))
                    
                    data[tf] = {
                        'close': np.array([float(c[4]) for c in candles]),
                        'high': np.array([float(c[2]) for c in candles]),
                        'low': np.array([float(c[3]) for c in candles]),
                        'volume': np.array([float(c[5]) for c in candles]),
                        'timestamp': [int(c[0]) for c in candles]
                    }
            except Exception as e:
                print(f"❌ Error fetching {tf} data: {e}")
                return None
            
            time.sleep(0.1)
        
        return data if len(data) == 3 else None
    
    def calculate_indicators(self, data):
        """Calculate technical indicators using TA-Lib"""
        indicators = {}
        
        for tf, prices in [("15m", data[primary_timeframe]), ("1h", data[higher_timeframe]), ("4h", data["240"])]:
            closes = prices['close']
            highs = prices['high']
            lows = prices['low']
            volumes = prices['volume']
            
            # TA-Lib calculations
            rsi = talib.RSI(closes, timeperiod=14)[-1]
            macd_line, macd_signal, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
            ema_21 = talib.EMA(closes, timeperiod=21)[-1]
            ema_50 = talib.EMA(closes, timeperiod=50)[-1]
            atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
            volume_sma = talib.SMA(volumes, timeperiod=20)[-1]
            bb_upper, bb_middle, bb_lower = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            adx = talib.ADX(highs, lows, closes, timeperiod=14)[-1]
            stoch_k, stoch_d = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowd_period=3)
            
            indicators[tf] = {
                'rsi': rsi,
                'macd': macd_line[-1],
                'macd_signal': macd_signal[-1],
                'macd_histogram': macd_hist[-1],
                'ema_21': ema_21,
                'ema_50': ema_50,
                'atr': atr,
                'volume_sma': volume_sma,
                'current_volume': volumes[-1],
                'bb_upper': bb_upper[-1],
                'bb_lower': bb_lower[-1],
                'bb_middle': bb_middle[-1],
                'adx': adx,
                'stoch_k': stoch_k[-1],
                'stoch_d': stoch_d[-1],
                'volume_ratio': volumes[-1] / volume_sma if volume_sma > 0 else 1
            }
        
        current_price = data[primary_timeframe]['close'][-1]
        
        # Calculate volatility
        if len(data[higher_timeframe]['close']) >= 24:
            price_24h_ago = data[higher_timeframe]['close'][-24]
            volatility = abs((current_price - price_24h_ago) / price_24h_ago)
        else:
            volatility = 0.02
        
        return indicators, current_price, volatility

    def check_btc_correlation(self):
        """Check Bitcoin trend correlation"""
        try:
            result = session.get_kline(
                category="linear",
                symbol="BTCUSDT",
                interval="60",
                limit=10
            )
            if result.get("retCode") == 0:
                candles = result["result"]["list"]
                candles = list(reversed(candles))
                
                btc_closes = [float(c[4]) for c in candles]
                btc_current = btc_closes[-1]
                btc_1h_ago = btc_closes[-2] if len(btc_closes) > 1 else btc_current
                btc_4h_ago = btc_closes[-5] if len(btc_closes) > 4 else btc_current
                
                btc_1h_change = (btc_current - btc_1h_ago) / btc_1h_ago * 100
                btc_4h_change = (btc_current - btc_4h_ago) / btc_4h_ago * 100
                
                return {
                        'bullish': btc_1h_change > -1.0 and btc_4h_change > -2.0,
                        'bearish': btc_1h_change < -2.0 or btc_4h_change < -5.0,
                        '1h_change': btc_1h_change,
                        '4h_change': btc_4h_change
                    }
        except Exception as e:
            print(f"❌ BTC correlation check failed: {e}")
        
        return {'bullish': True, 'bearish': False, '1h_change': 0, '4h_change': 0}
    
    def calculate_futures_signals(self, indicators, current_price, volatility, regime_score=0.0):
        """
        Calculate futures trading signals.

        4h trend alignment is a HARD GATE (non-negotiable):
          - LONG  only when 4h EMA_21 > EMA_50  (4h uptrend confirmed)
          - SHORT only when 4h EMA_21 < EMA_50  (4h downtrend confirmed)
        This prevents buying into sustained downtrends and shorting into uptrends,
        which was the primary cause of losses identified during backtesting.
        """
        ind4h = indicators.get("4h", {})
        ema21_4h = ind4h.get("ema_21", None)
        ema50_4h = ind4h.get("ema_50", None)

        # Hard trend gates — fall back to allowing both if 4h data unavailable
        if ema21_4h is not None and ema50_4h is not None and ema50_4h > 0:
            trend_bullish = ema21_4h > ema50_4h   # 4h uptrend
            trend_bearish = ema21_4h < ema50_4h   # 4h downtrend
        else:
            print("⚠️  4h EMA unavailable — trend gate relaxed for this cycle")
            trend_bullish = True
            trend_bearish = True

        # Futures LONG conditions (7 conditions, need 5+)
        futures_long_conditions = [
            indicators["15m"]["rsi"] < 40,
            indicators["1h"]["rsi"] < 50,
            indicators["15m"]["macd"] > indicators["15m"]["macd_signal"],
            current_price > indicators["15m"]["ema_21"] * 0.998,
            indicators["15m"]["volume_ratio"] > 1.3,
            volatility > 0.02,
            indicators["15m"]["adx"] > 18,
        ]

        # Futures SHORT conditions (10 conditions, need 6+)
        futures_short_conditions = [
            indicators["15m"]["rsi"] > 65,
            indicators["1h"]["rsi"] > 55,
            indicators["15m"]["macd"] < indicators["15m"]["macd_signal"],
            (indicators["15m"]["macd_histogram"] / current_price) < -0.0005 if current_price > 0 else False,
            current_price < indicators["15m"]["ema_21"],
            indicators["15m"]["volume_ratio"] > 1.4,
            volatility > 0.025,
            indicators["15m"]["stoch_k"] > 80,
            current_price > indicators["1h"]["ema_50"] * 0.98,
            indicators["15m"]["adx"] > 18,
        ]

        long_score  = sum(futures_long_conditions)
        short_score = sum(futures_short_conditions)

        # Dynamic LONG threshold: require more evidence during bearish macro regime
        min_long_score = 6 if regime_score <= -0.4 else 5

        # Log 4h trend context
        trend_label = "BULL" if trend_bullish else ("BEAR" if trend_bearish else "FLAT")
        print(f"   4h Trend: {trend_label}  (EMA21={ema21_4h:.2f} {'>' if trend_bullish else '<'} EMA50={ema50_4h:.2f})" if ema21_4h else "   4h Trend: UNKNOWN")
        print(f"   LONG score={long_score}/7 (min {min_long_score})  SHORT score={short_score}/10")

        # Hard gate: TA signal is only valid when 4h trend aligns
        if long_score >= min_long_score and trend_bullish:
            return {"signal": "LONG",  "strength": long_score,  "leverage": 10.0}
        if short_score >= 6 and trend_bearish:
            return {"signal": "SHORT", "strength": short_score, "leverage": 10.5}

        # Log why signal was blocked
        if long_score >= min_long_score and not trend_bullish:
            print(f"   🚫 LONG blocked — 4h trend is bearish (EMA21 < EMA50)")
        if short_score >= 6 and not trend_bearish:
            print(f"   🚫 SHORT blocked — 4h trend is bullish (EMA21 > EMA50)")

        return {"signal": None, "strength": max(long_score, short_score)}
    
    def calculate_futures_position_size(self, entry_price, stop_loss_price,
                                        leverage=10.0, symbol=None):
        """Calculate safe futures position size with symbol-aware lot constraints."""
        if not symbol or symbol not in SYMBOL_CONTRACT_SPECS:
            raise ValueError(f"calculate_futures_position_size requires valid symbol in {list(SYMBOL_CONTRACT_SPECS.keys())}")

        specs = SYMBOL_CONTRACT_SPECS[symbol]
        min_order_size = specs["min_qty"]
        step_size      = specs["step_size"]
        ticker         = specs["ticker"]

        usdt_balance = self.get_usdt_balance()

        print(f"💰 Margin Check:")
        print(f"   Available USDT: ${usdt_balance:.2f}")

        if usdt_balance < 5:
            print(f"❌ Insufficient USDT for futures margin: ${usdt_balance:.2f} (minimum $5)")
            return None

        max_usable_margin = usdt_balance * 0.7
        risk_amount       = usdt_balance * futures_risk_per_trade

        stop_distance = abs(entry_price - stop_loss_price)
        if stop_distance <= 0:
            print("❌ Invalid stop loss distance")
            return None

        max_position_by_margin = (max_usable_margin * leverage) / entry_price
        max_position_by_risk   = risk_amount / stop_distance
        position_size          = min(max_position_by_margin, max_position_by_risk)

        if position_size < min_order_size:
            print(f"⚠️ Calculated position ({position_size:.4f}) below minimum ({min_order_size}) — using minimum")
            position_size = min_order_size
        else:
            position_size = round(position_size / step_size) * step_size
            print(f"📊 Rounded position to step size: {position_size} {ticker}")

        required_margin = (position_size * entry_price) / leverage

        if required_margin > max_usable_margin:
            print(f"⚠️ Required margin ${required_margin:.2f} exceeds available ${max_usable_margin:.2f}")
            position_size   = (max_usable_margin * leverage) / entry_price
            position_size   = round(position_size / step_size) * step_size
            required_margin = (position_size * entry_price) / leverage
            if position_size < min_order_size:
                print(f"❌ Even reduced position ({position_size}) below minimum ({min_order_size})")
                return None

        if required_margin < 2:
            print(f"❌ Margin too small: ${required_margin:.2f} (min: $2)")
            return None

        actual_risk = position_size * stop_distance

        print(f"📊 Position Size Calculation ({symbol}):")
        print(f"   Max Usable (70%): ${max_usable_margin:.2f}")
        print(f"   Risk Amount ({futures_risk_per_trade*100:.0f}%): ${risk_amount:.2f}")
        print(f"   Stop Distance: ${stop_distance:.4f}")
        print(f"   Position Size: {position_size} {ticker}")
        print(f"   Required Margin: ${required_margin:.2f}")
        print(f"   Actual Risk: ${actual_risk:.2f}")

        return {
            'position_size':   position_size,
            'required_margin': round(required_margin, 2),
            'leverage':        leverage,
            'risk_amount':     risk_amount,
            'actual_risk':     actual_risk,
            'ticker':          ticker,
        }
    
    def calculate_improved_futures_stops(self, direction, current_price, indicators, signal_strength, sr_data=None):
        """Calculate wider, more realistic stops for futures trading (with S/R anchor support)."""
        
        atr_15m = indicators["15m"]["atr"]
        atr_1h = indicators["1h"]["atr"]
        
        # Use LARGER ATR for futures to avoid noise
        primary_atr = max(atr_15m, atr_1h * 0.7)
        
        # Priority 1: Use S/R-anchored stops if actionable scenario
        if sr_data and sr_data.get("suggested_stop") and sr_data.get("suggested_target") and sr_data.get("scenario") != "MID_RANGE":
            stop_loss = sr_data["suggested_stop"]
            take_profit = sr_data["suggested_target"]
            stop_distance = abs(current_price - stop_loss)
            reward_amount = abs(take_profit - current_price)
            reward_ratio = reward_amount / stop_distance if stop_distance > 0 else min_reward_ratio
            
            print(f"📊 S/R-Anchored Stop Calculation ({sr_data.get('scenario')}):")
            print(f"   Entry: ${current_price:.4f}")
            print(f"   Stop Loss: ${stop_loss:.4f}")
            print(f"   Take Profit: ${take_profit:.4f}")
            print(f"   Risk: ${stop_distance:.4f}")
            print(f"   Reward: ${reward_amount:.4f}")
            print(f"   Ratio: {reward_ratio:.2f}:1")
            
            return {
                'stop_loss': round(stop_loss, 4),
                'take_profit': round(take_profit, 4),
                'risk_amount': stop_distance,
                'reward_amount': reward_amount,
                'reward_ratio': reward_ratio,
                'method_used': f"sr_anchored_{sr_data.get('scenario')}",
                'primary_atr_used': primary_atr
            }

        print(f"📊 ATR Analysis:")
        print(f"   15m ATR: ${atr_15m:.2f}")
        print(f"   1h ATR: ${atr_1h:.2f}")
        print(f"   Using Primary ATR: ${primary_atr:.2f}")
        
        if direction == "SHORT":
            base_stop_distance = 1.5 * primary_atr
            strength_multiplier = 1.0 + (signal_strength / 20)
            volatility_factor = min(atr_1h / atr_15m, 1.5) if atr_15m > 0 else 1.2
            
            stop_distance = base_stop_distance * strength_multiplier * volatility_factor
            stop_loss = current_price + stop_distance
            
            reward_distance = stop_distance * min_reward_ratio
            take_profit = current_price - reward_distance
            
        else:  # LONG
            base_stop_distance = 1.5 * primary_atr
            strength_multiplier = 1.0 + (signal_strength / 20)
            volatility_factor = min(atr_1h / atr_15m, 1.5) if atr_15m > 0 else 1.2
            
            stop_distance = base_stop_distance * strength_multiplier * volatility_factor
            stop_loss = current_price - stop_distance
            
            reward_distance = stop_distance * min_reward_ratio
            take_profit = current_price + reward_distance
        
        # Ensure minimum distances
        min_stop_distance = current_price * 0.008  # 0.8% minimum
        if stop_distance < min_stop_distance:
            print(f"⚠️ Stop distance too small, increasing to minimum 0.8%")
            stop_distance = min_stop_distance
            
            if direction == "SHORT":
                stop_loss = current_price + stop_distance
                take_profit = current_price - (stop_distance * min_reward_ratio)
            else:
                stop_loss = current_price - stop_distance
                take_profit = current_price + (stop_distance * min_reward_ratio)
        
        return {
            'stop_loss': round(stop_loss, 4),
            'take_profit': round(take_profit, 4),
            'risk_amount': stop_distance,
            'reward_amount': abs(take_profit - current_price),
            'reward_ratio': abs(take_profit - current_price) / stop_distance,
            'method_used': 'improved_wider_stops',
            'primary_atr_used': primary_atr
        }
    
    def place_futures_order(self, signal, current_price, indicators, factor_context=None, symbol_override=None):
        """Place futures order with improved stops. factor_context holds all
        multi-factor scores collected at signal time for logging."""

        if not symbol_override:
            raise ValueError(
                "place_futures_order() called without symbol_override. "
                "Always pass symbol_override explicitly to prevent silent SOLUSDT fallback."
            )
        trade_sym = symbol_override
        ticker    = SYMBOL_CONTRACT_SPECS.get(trade_sym, {}).get("ticker", trade_sym)
        
        sr_data = None
        if factor_context:
            sr_scenario = factor_context.get("sr_scenario", "MID_RANGE")
            if sr_scenario != "MID_RANGE":
                sr_data = {
                    "scenario": sr_scenario,
                    "suggested_stop": factor_context.get("sr_suggested_stop"),
                    "suggested_target": factor_context.get("sr_suggested_target"),
                    "suggested_leverage": factor_context.get("sr_suggested_leverage", 10.0),
                }
                if sr_data.get("suggested_leverage"):
                    signal["leverage"] = sr_data["suggested_leverage"]

        # Calculate improved stops (with S/R anchoring if present)
        stop_data = self.calculate_improved_futures_stops(
            signal['signal'], current_price, indicators, signal['strength'], sr_data=sr_data
        )
        
        stop_loss = stop_data['stop_loss']
        take_profit = stop_data['take_profit']
        
        print(f"\n🎯 Futures Stop Calculation:")
        print(f"   Entry: ${current_price:.2f}")
        print(f"   Stop Loss: ${stop_loss:.2f}")
        print(f"   Take Profit: ${take_profit:.2f}")
        print(f"   Risk: ${stop_data['risk_amount']:.2f}")
        print(f"   Reward: ${stop_data['reward_amount']:.2f}")
        print(f"   Ratio: {stop_data['reward_ratio']:.2f}:1")
        
        # Calculate position size (symbol-aware lot constraints)
        position_data = self.calculate_futures_position_size(
            current_price, stop_loss, signal["leverage"], symbol=trade_sym
        )
        
        if position_data is None:
            print("❌ Cannot calculate valid futures position size")
            return False
        
        try:
            # Set leverage
            print(f"🔧 Setting leverage to {signal['leverage']:.1f}x...")
            try:
                leverage_result = session.set_leverage(
                    category="linear",
                    symbol=trade_sym,
                    buyLeverage=str(int(signal["leverage"])),
                    sellLeverage=str(int(signal["leverage"]))
                )
                
                if leverage_result.get("retCode") == 0:
                    print(f"✅ Leverage set successfully")
                elif leverage_result.get("retCode") == 110043:
                    print(f"ℹ️ Leverage already set (continuing)")
                else:
                    print(f"⚠️ Leverage response: {leverage_result.get('retMsg')} (continuing)")
                    
            except Exception as leverage_error:
                print(f"⚠️ Leverage setting error: {leverage_error} (continuing)")
            
            # Place the order — SL/TP included atomically to eliminate the race
            # window that exists when stops are set in a separate API call after fill.
            side = "Buy" if signal["signal"] == "LONG" else "Sell"
            
            # Format qty and SL/TP according to contract precision per symbol
            step_size     = SYMBOL_CONTRACT_SPECS.get(trade_sym, {}).get("step_size", 0.1)
            qty_precision = 2 if step_size < 0.1 else 1
            price_precision = 4 if current_price < 100 else 2
            
            order_params = {
                "category": "linear",
                "symbol": trade_sym,
                "side": side,
                "orderType": "Market",
                "qty": f"{position_data['position_size']:.{qty_precision}f}",
                "stopLoss": f"{stop_loss:.{price_precision}f}",
                "takeProfit": f"{take_profit:.{price_precision}f}",
                "slTriggerBy": "MarkPrice",
                "tpTriggerBy": "MarkPrice",
            }
            
            print(f"\n🚀 FUTURES {signal['signal']} Order ({trade_sym}):")
            print(f"   💰 Margin: ${position_data['required_margin']:.2f}")
            print(f"   📊 Position: {position_data['position_size']} {ticker}")
            print(f"   ⚡ Leverage: {signal['leverage']:.1f}x")
            print(f"   🎯 Entry: ${current_price:.4f}")
            print(f"   🛡️ Stop: ${stop_loss:.4f} (atomic, mark-price triggered)")
            print(f"   💎 Target: ${take_profit:.4f} (atomic, mark-price triggered)")
            print(f"   💀 Max Risk: ${position_data['actual_risk']:.2f}")
            
            result = session.place_order(**order_params)
            
            if result.get("retCode") == 0:
                print(f"✅ Futures {signal['signal']} order placed with SL/TP set atomically!")
                
                # Store position
                futures_state['position'] = {
                    'symbol': trade_sym,
                    'direction': signal['signal'],
                    'size': position_data['position_size'],
                    'entry': current_price,
                    'stop': stop_loss,
                    'target': take_profit,
                    'leverage': signal['leverage'],
                    'margin': position_data['required_margin'],
                    'order_id': result.get('result', {}).get('orderId'),
                    'timestamp': datetime.now(),
                    'exit_25_taken': False,
                    'exit_50_taken': False,
                    'stop_moved_to_be': False,
                    'original_stop': stop_loss,
                    # Track best price for trailing stop
                    'highest_price': current_price,
                    'lowest_price': current_price,
                    # Factor context logged at entry — persisted on close
                    'factor_context': factor_context or {},
                }
                
                self.save_position_state()
                
                futures_state['daily_trades'] += 1
                futures_state['total_trades'] += 1
                self.save_state()
                
                return True
            else:
                print(f"❌ Futures order failed: {result.get('retMsg')}")
                return False
                
        except Exception as e:
            print(f"❌ Error placing futures order: {e}")
            return False
    
    def sync_position_with_bybit(self):
        """Sync local position state with Bybit API"""
        if not futures_state['position']: return
        
        try:
            result = session.get_positions(
                category="linear",
                symbol=futures_state['position']['symbol']
            )
            if result.get("retCode") == 0:
                positions = result.get("result", {}).get("list", [])
                active_pos = next((p for p in positions if float(p.get("size", 0)) > 0), None)
                
                if not active_pos:
                    print("⚠️ Bybit reports no open position, but local state had one. Resolving...")
                    # Get closed PnL to determine if win or loss
                    pnl_result = session.get_closed_pnl(
                        category="linear",
                        symbol=futures_state['position']['symbol'],
                        limit=1
                    )
                    win = False
                    pnl = 0
                    if pnl_result.get("retCode") == 0:
                        pnl_list = pnl_result.get("result", {}).get("list", [])
                        if pnl_list:
                            pnl = float(pnl_list[0].get("closedPnl", 0))
                            win = pnl > 0
                    
                    print(f"🔄 Sync: Position closed externally. PnL: {pnl:.2f}")
                    fc = futures_state['position'].get('factor_context', {})
                    self.log_trade_outcome(pnl=pnl, result='WIN' if win else 'LOSS', factor_context=fc)
                    self.close_position("WIN" if win else "LOSS", log_already_done=True)
        except Exception as e:
            print(f"❌ Error syncing position: {e}")

    def check_position_exits(self, current_price, indicators):
        """Check and manage position exits"""
        
        self.sync_position_with_bybit()
        
        if not futures_state['position']:
            return
        
        pos = futures_state['position']
        direction = pos['direction']
        
        # Partial exit management
        self.partial_position_management(pos, current_price, indicators)
        
        # Trailing stop
        new_stop = self.implement_trailing_stop(pos, current_price, indicators)
        if new_stop and new_stop != pos['stop']:
            pos['stop'] = new_stop
            try:
                session.set_trading_stop(
                    category="linear",
                    symbol=pos['symbol'],
                    stopLoss=str(round(new_stop, 2))
                )
                self.save_position_state()
            except Exception as e:
                print(f"⚠️ Error setting trailing stop: {e}")
        
        # -- Early Scratch Exit --
        # If position goes adverse by -0.7% within first 45 minutes and stop hasn't
        # been moved to breakeven, cut the loss immediately.
        
        pos_time = pos['timestamp']
        if isinstance(pos_time, str):
            try:
                # Handle ISO format string from DB
                pos_time = datetime.fromisoformat(pos_time)
            except Exception:
                # Fallback if unparseable
                pos_time = datetime.now()
                
        time_held_secs = (datetime.now() - pos_time).total_seconds()
        
        if not pos.get('stop_moved_to_be') and time_held_secs <= 2700:
            if direction == "LONG":
                adverse_pct = (pos['entry'] - current_price) / pos['entry']
            else:
                adverse_pct = (current_price - pos['entry']) / pos['entry']
                
            if adverse_pct >= 0.007:
                print(f"🔪 EARLY SCRATCH: Trade went adverse -0.7% quickly. Cutting losses at ${current_price:.2f}")
                self.close_position("SCRATCH", exit_price=current_price)
                return

        # Check structural exit conditions
        if direction == "LONG":
            if current_price <= pos['stop']:
                print(f"🛑 STOP LOSS hit at ${current_price:.2f}")
                self.close_position("LOSS", exit_price=current_price)
            elif current_price >= pos['target']:
                print(f"🎯 TAKE PROFIT hit at ${current_price:.2f}")
                self.close_position("WIN", exit_price=current_price)
        else:  # SHORT
            if current_price >= pos['stop']:
                print(f"🛑 STOP LOSS hit at ${current_price:.2f}")
                self.close_position("LOSS", exit_price=current_price)
            elif current_price <= pos['target']:
                print(f"🎯 TAKE PROFIT hit at ${current_price:.2f}")
                self.close_position("WIN", exit_price=current_price)
    
    def implement_trailing_stop(self, position, current_price, indicators):
        """Implement trailing stop to lock in profits - price-based trailing"""

        if not position:
            return None

        direction = position['direction']
        entry_price = position['entry']
        current_stop = position['stop']
        atr_15m = indicators["15m"]["atr"]

        if direction == "SHORT":
            # Track the lowest price reached (best for SHORT)
            if 'lowest_price' not in position or current_price < position['lowest_price']:
                position['lowest_price'] = current_price
                print(f"📉 New best SHORT price: ${current_price:.2f}")
                self.save_position_state()

            best_price = position['lowest_price']
            unrealized_pnl_pct = (entry_price - best_price) / entry_price

            if unrealized_pnl_pct > 0.03:  # raised from 1.5% — avoids stop at breakeven converting winners
                # Place stop just above best price with ATR buffer
                proposed_stop = best_price + (1.5 * atr_15m)

                # Only update if better (lower than before)
                if proposed_stop < current_stop:
                    print(f"🔄 Trailing stop (SHORT): ${current_stop:.2f} → ${proposed_stop:.2f}")
                    return proposed_stop

        else:  # LONG
            # Track the highest price reached (best for LONG)
            if 'highest_price' not in position or current_price > position['highest_price']:
                position['highest_price'] = current_price
                print(f"📈 New best LONG price: ${current_price:.2f}")
                self.save_position_state()

            best_price = position['highest_price']
            unrealized_pnl_pct = (best_price - entry_price) / entry_price

            if unrealized_pnl_pct > 0.03:  # raised from 1.5% — avoids stop at breakeven converting winners
                # Place stop just below best price with ATR buffer
                proposed_stop = best_price - (1.5 * atr_15m)

                # Only update if better (higher than before)
                if proposed_stop > current_stop:
                    print(f"🔄 Trailing stop (LONG): ${current_stop:.2f} → ${proposed_stop:.2f}")
                    return proposed_stop

        # No change
        return current_stop

    
    def _place_reduce_only_order(self, direction, qty):
        """Place a reduceOnly market order to partially or fully close a position.
        Returns True on success, False on failure. Safe to call if already closed
        (Bybit will reject cleanly with retCode != 0)."""
        try:
            close_side = "Sell" if direction == "LONG" else "Buy"
            result = session.place_order(
                category="linear",
                symbol=futures_state['position']['symbol'],
                side=close_side,
                orderType="Market",
                qty=str(round(qty, 1)),
                reduceOnly=True,
            )
            if result.get("retCode") == 0:
                return True
            else:
                print(f"⚠️ Reduce-only order rejected: {result.get('retMsg')}")
                return False
        except Exception as e:
            print(f"⚠️ Reduce-only order error: {e}")
            return False

    def partial_position_management(self, position, current_price, indicators):
        """Take partial profits to reduce risk — sends real reduce-only orders."""
        
        if not position:
            return None
        
        direction = position['direction']
        entry_price = position['entry']
        
        # Calculate unrealized P&L
        if direction == "SHORT":
            pnl_pct = (entry_price - current_price) / entry_price
        else:  # LONG
            pnl_pct = (current_price - entry_price) / entry_price
        
        # Symbol-aware lot constraints
        sym    = position.get('symbol', 'SOLUSDT')
        specs  = SYMBOL_CONTRACT_SPECS.get(sym, SYMBOL_CONTRACT_SPECS['SOLUSDT'])
        min_q  = specs['min_qty']
        step_q = specs['step_size']
        ticker = specs['ticker']

        # Take 25% profit at 2% gain (halfway to the 2:1 target)
        if pnl_pct >= 0.02 and not position.get('exit_25_taken'):
            partial_qty = round(position['size'] * 0.25 / step_q) * step_q
            if partial_qty < min_q:
                partial_qty = min_q
            print(f"💰 Placing 25% partial close ({partial_qty} {ticker}) at ${current_price:.4f} (+{pnl_pct*100:.1f}%)")
            if self._place_reduce_only_order(direction, partial_qty):
                position['size'] = round((position['size'] - partial_qty) / step_q) * step_q
                position['exit_25_taken'] = True
                print(f"✅ 25% partial closed — remaining size: {position['size']} {ticker}")
                self.save_position_state()

        # Take another 25% at 3.5% gain (just before the 2:1 TP at ~4%)
        if pnl_pct >= 0.035 and not position.get('exit_50_taken'):
            partial_qty = round(position['size'] * 0.25 / step_q) * step_q
            if partial_qty < min_q:
                partial_qty = min_q
            print(f"💰 Placing second 25% partial ({partial_qty} {ticker}) at ${current_price:.4f} (+{pnl_pct*100:.1f}%)")
            if self._place_reduce_only_order(direction, partial_qty):
                position['size'] = round((position['size'] - partial_qty) / step_q) * step_q
                position['exit_50_taken'] = True
                print(f"✅ 50% total closed — remaining size: {position['size']} {ticker}")
                self.save_position_state()
        
        # Move stop to breakeven after first partial (risk-free remainder)
        if position.get('exit_25_taken') and not position.get('stop_moved_to_be'):
            position['stop'] = entry_price
            position['stop_moved_to_be'] = True
            print(f"🛡️ Stop moved to breakeven: ${entry_price:.2f}")
            try:
                session.set_trading_stop(
                    category="linear",
                    symbol=position['symbol'],
                    stopLoss=str(round(entry_price, 2)),
                )
            except Exception as e:
                print(f"⚠️ Could not update exchange SL to breakeven: {e}")
            self.save_position_state()
    
    def close_position(self, result_type, log_already_done=False, exit_price=None):
        """Close futures position. Sends a reduceOnly market order as a safety net
        (harmless if exchange SL/TP already closed it — Bybit will reject cleanly)."""
        if not futures_state['position']:
            return
        
        pos = futures_state['position']
        print(f"🚀 Closing futures {pos['direction']} position ({result_type})")
        
        # Safety-net close: reduceOnly so it fails gracefully if already closed by exchange SL/TP
        remaining_size = pos.get('size', 0)
        if remaining_size >= 0.1:
            closed = self._place_reduce_only_order(pos['direction'], remaining_size)
            if closed:
                print(f"✅ Market close order sent for {remaining_size:.1f} SOL")
            else:
                print(f"ℹ️ Close order rejected — position likely already closed by exchange SL/TP")
        
        if not log_already_done:
            fc = pos.get('factor_context', {})
            self.log_trade_outcome(
                exit_price=exit_price,
                result=result_type,
                factor_context=fc,
            )
        
        # Update session P&L for circuit breaker
        if exit_price and pos.get('entry'):
            size = pos.get('size', 0)
            if pos['direction'] == 'LONG':
                trade_pnl = (exit_price - pos['entry']) * size
            else:
                trade_pnl = (pos['entry'] - exit_price) * size
            futures_state['session_pnl'] += trade_pnl
            print(f"📊 Session PnL updated: ${futures_state['session_pnl']:.2f}")
        
        # Update statistics
        if result_type == "WIN":
            futures_state['winning_trades'] += 1
            futures_state['consecutive_losses'] = 0
        else:
            futures_state['consecutive_losses'] += 1
        
        self.clear_position_state()
    
    def can_trade(self):
        """Check if trading is allowed"""
        if futures_state['daily_trades'] >= max_daily_trades:
            return False, "Daily limit reached"
        
        if futures_state['consecutive_losses'] >= futures_state['max_consecutive_losses']:
            return False, "Too many consecutive losses"
        
        # Daily P&L circuit breaker: halt if session loss exceeds 5% of starting balance
        start_bal = futures_state.get('session_start_balance', 0)
        if start_bal > 0:
            session_loss_pct = futures_state['session_pnl'] / start_bal
            if session_loss_pct < -0.05:
                return False, f"Daily P&L circuit breaker tripped ({session_loss_pct*100:.1f}% session loss)"
        
        return True, "Can trade"
    
    def run_futures_strategy(self):
        """Main futures strategy execution (Multi-Asset Scanner)"""
        
        print(f"\n{'='*80}")
        print(f"🔄 FUTURES Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        can_trade, trade_reason = self.can_trade()
        usdt_balance = self.get_usdt_balance()
        futures_state['available_balance'] = usdt_balance

        if not can_trade:
            print(f"⚠️ Trading blocked: {trade_reason}")
            return None
        if usdt_balance < 5:
            print(f"⚠️ Insufficient USDT margin: ${usdt_balance:.2f}")
            return None
        if futures_state['position']:
            # We already have an open position. Do not scan new pairs.
            # Active management is handled by the fast loop.
            return None

        best_setup = None
        best_score = -1.0

        print(f"🔍 Scanning Universe: {', '.join(TRADE_SYMBOLS)}")

        # ── Pre-loop: fetch market-wide signals ONCE (avoids 5× redundant API calls) ──
        btc_data = self.check_btc_correlation()
        try:
            regime_info = get_regime_score()
        except Exception:
            regime_info = {"score": 0.0, "confidence": 0.0, "block_trade": False,
                           "regime": "NEUTRAL", "details": {}}
        regime_score_global = regime_info.get("score", 0.0)
        precomputed         = {"regime": regime_info}   # passed to aggregator for every symbol

        aggregator = MultiFactorAggregator()

        for current_sym in TRADE_SYMBOLS:
            print(f"\n📊 Analyzing {current_sym}...")

            data = self.fetch_multi_timeframe_data(current_sym)
            if not data:
                print(f"   ❌ Failed to fetch data")
                continue

            indicators, current_price, volatility = self.calculate_indicators(data)
            print(f"   💲 Price: ${current_price:,.4f}")

            # Regime Filter: Block if flat
            if indicators['1h']['adx'] < 18:
                print(f"   🚫 Flat market (1h ADX: {indicators['1h']['adx']:.1f} < 18)")
                continue

            signal = self.calculate_futures_signals(
                indicators, current_price, volatility, regime_score=regime_score_global
            )

            if signal["signal"] not in ["LONG", "SHORT"] or signal["strength"] < signal_strength_threshold:
                continue

            if signal["signal"] == "LONG" and btc_data['bearish']:
                print("   ❌ BTC bearish — skipping LONG")
                continue

            # Full multi-factor evaluation (regime pre-fetched, sentiment uses 1h cache)
            # Pass indicators+data so S/R factor can detect swing levels
            consensus = aggregator.evaluate(signal, current_sym, current_price,
                                            precomputed=precomputed,
                                            indicators=indicators,
                                            data=data)
            if consensus["block_trade"] or consensus["signal"] is None:
                continue

            score_abs = abs(consensus["final_score"])
            print(f"   ✅ {consensus['signal']} Passed! Score: {consensus['final_score']:+.3f}")

            if score_abs > best_score:
                best_score = score_abs
                
                scores    = consensus.get("factor_scores", {})
                sr_fs     = scores.get("support_resistance", {})
                deriv_det = scores.get("derivatives", {}).get("details", {})
                ind4h_now = indicators.get("4h", {})
                trend_4h  = "BULL" if ind4h_now.get("ema_21", 0) > ind4h_now.get("ema_50", 0) else "BEAR"
                
                context = {
                    "ta_signal_strength":  signal.get("strength"),
                    "aggregated_score":    consensus.get("final_score"),
                    "volatility":          volatility,
                    "atr_15m":             indicators["15m"]["atr"],
                    "technical_score":     scores.get("technical", {}).get("score"),
                    "regime_score":        scores.get("regime",    {}).get("score"),
                    "derivatives_score":   scores.get("derivatives", {}).get("score"),
                    "sentiment_score":     scores.get("sentiment", {}).get("score"),
                    "news_score":          scores.get("news",      {}).get("score"),
                    "sr_score":            sr_fs.get("score"),
                    "sr_scenario":         consensus.get("sr_scenario", "MID_RANGE"),
                    "sr_suggested_stop":   consensus.get("sr_suggested_stop"),
                    "sr_suggested_target": consensus.get("sr_suggested_target"),
                    "sr_suggested_leverage": consensus.get("sr_suggested_leverage", 10.0),
                    "regime_class":        scores.get("regime", {}).get("details", {}).get("regime"),
                    "funding_rate":        deriv_det.get("funding",        {}).get("current_rate_pct"),
                    "open_interest":       deriv_det.get("open_interest",  {}).get("oi_change_pct"),
                    "long_short_ratio":    deriv_det.get("long_short_ratio", {}).get("long_ratio_pct"),
                    "news_sentiment":      scores.get("news", {}).get("details", {}).get("sentiment_label"),
                    "market_trend_4h":     trend_4h,
                }

                best_setup = {
                    "symbol": current_sym,
                    "signal": consensus["signal"],
                    "strength": signal["strength"],
                    "leverage": signal["leverage"],
                    "current_price": current_price,
                    "indicators": indicators,
                    "context": context
                }

        # ── Execute the Best Setup ──────────────────────────────────────────
        if best_setup:
            print(f"\n🏆 WINNING SETUP: {best_setup['signal']} on {best_setup['symbol']} (Score: {best_score:.3f})")
            self.place_futures_order(
                {"signal": best_setup["signal"], "strength": best_setup["strength"], "leverage": best_setup["leverage"]}, 
                best_setup["current_price"], 
                best_setup["indicators"], 
                factor_context=best_setup["context"],
                symbol_override=best_setup["symbol"]
            )
        else:
            print("\n💤 No valid setups found across universe.")
            
        # Display status
        pos = futures_state['position']
        print(f"\n📊 Status:")
        print(f"🚀 Position: {pos['symbol'] + ' ' + pos['direction'] if pos else 'None'}")
        print(f"📈 Daily Trades: {futures_state['daily_trades']}/{max_daily_trades}")
        print(f"💰 Available Balance: ${futures_state['available_balance']:.2f}")
        
        if futures_state['total_trades'] > 0:
            win_rate = futures_state['winning_trades'] / futures_state['total_trades'] * 100
            print(f"🎯 Win Rate: {win_rate:.1f}% ({futures_state['winning_trades']}/{futures_state['total_trades']})")
            
        return indicators
    
    def get_current_price(self, symbol):
        """Fast API call to get latest price for active position management"""
        try:
            result = session.get_tickers(category="linear", symbol=symbol)
            if result.get("retCode") == 0:
                list_data = result.get("result", {}).get("list", [])
                if list_data:
                    return float(list_data[0].get("lastPrice", "0"))
        except Exception as e:
            pass
        return None

    def run_bot(self):
        """Main bot execution loop"""
        
        print("🚀" + "="*80)
        print("🚀 STANDALONE FUTURES TRADING BOT (FAST LOOP ENABLED)")
        print("🚀" + "="*80)
        print(f"💎 Scanning Universe: {', '.join(TRADE_SYMBOLS)}")
        print(f"⚡ Strategy: Pure Futures with BTC Correlation")
        print(f"📊 Risk Per Trade: {futures_risk_per_trade*100:.0f}% of USDT balance")
        print(f"⏰ Analysis Frequency: Every 5 minutes")
        print(f"🚀 Execution Frequency: Every 10 seconds (Price check & trailing stops)")
        print(f"💾 Storage: SQLite3 State Persistence")
        print(f"🎯 Signal Threshold: {signal_strength_threshold}/10")
        print(f"💰 Max Leverage: {max_leverage:.1f}x")
        print(f"🎯 Reward Ratio: {min_reward_ratio:.1f}:1 minimum")
        print("="*80)
        
        cycle_count = 0
        last_analysis_time = 0
        analysis_interval = 300  # 5 minutes
        indicators_cache = None
        # Date-based daily reset — reliable regardless of loop timing or restarts
        last_reset_date = datetime.now().date()
        
        # If we restarted with a position, try to get initial indicators for trailing stop math
        if futures_state['position']:
            print("🔄 Initializing indicators for active position management...")
            data = self.fetch_multi_timeframe_data(futures_state['position']['symbol'])
            if data:
                indicators_cache, _, _ = self.calculate_indicators(data)
        
        while True:
            try:
                current_time = time.time()
                
                # --- FAST LOOP: Active Position Management (Every 10 seconds) ---
                if futures_state['position']:
                    fast_price = self.get_current_price(futures_state['position']['symbol'])
                    if fast_price and indicators_cache:
                        self.check_position_exits(fast_price, indicators_cache)
                
                # --- SLOW LOOP: Market Analysis & Signals (Every 5 minutes) ---
                if current_time - last_analysis_time >= analysis_interval:
                    cycle_count += 1
                    
                    indicators = self.run_futures_strategy()
                    if indicators:
                        indicators_cache = indicators
                    
                    # Date-based daily counter reset (reliable across restarts & slow loops)
                    today = datetime.now().date()
                    if today != last_reset_date:
                        futures_state['daily_trades'] = 0
                        futures_state['session_pnl'] = 0.0
                        futures_state['session_start_balance'] = self.get_usdt_balance()
                        last_reset_date = today
                        print(f"🌅 New trading day ({today}) — counters and circuit breaker reset")
                    
                    # Update balance periodically
                    if cycle_count % 12 == 0:  # Every hour (12 * 5 mins)
                        futures_state['available_balance'] = self.get_usdt_balance()
                    
                    # Session statistics
                    runtime = datetime.now() - futures_state['session_start']
                    print(f"\n📊 Session Stats:")
                    print(f"⏰ Runtime: {runtime}")
                    print(f"🔄 Analysis Cycles: {cycle_count}")
                    print(f"💼 Total Trades: {futures_state['total_trades']}")
                    print(f"💰 Current Balance: ${futures_state['available_balance']:.2f}")
                    
                    next_cycle = datetime.now() + timedelta(minutes=5)
                    print(f"\n💤 Next analysis at {next_cycle.strftime('%H:%M:%S')} (Checking stops every 10s)")
                    print("="*80)
                    
                    last_analysis_time = time.time()
                
                # Short sleep for fast reaction time
                time.sleep(10)
                
            except KeyboardInterrupt:
                print("\n🛑 Futures bot stopped by user")
                print(f"📊 Final Stats: {futures_state['total_trades']} trades, {cycle_count} cycles")
                if hasattr(self, 'conn'):
                    self.conn.close()
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("⏰ Waiting 30 seconds before retry...")
                time.sleep(30)

if __name__ == "__main__":
    bot = FuturesTradingBot()
    bot.run_bot()