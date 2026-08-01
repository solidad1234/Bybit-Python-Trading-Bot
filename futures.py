"""
Standalone Futures Trading Bot - Extracted from Hybrid Bot
Focus: Pure futures trading with enhanced risk management
"""

import requests
import hmac
import hashlib
import time
import talib
import numpy as np
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
import os
import json
import sqlite3

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")

# Trading Configuration
symbol = "SOLUSDT"
primary_timeframe = "15"   # Primary analysis
higher_timeframe = "60"    # Trend confirmation

# Futures Risk Management
futures_risk_per_trade = 0.02  # 2% risk per futures trade
max_leverage = 20.0            # Conservative max leverage
min_reward_ratio = 3.0         # Minimum reward:risk ratio (3:1)
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
    'available_balance': 0
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
        self.conn = sqlite3.connect('trading_state.db', check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS position (
                id INTEGER PRIMARY KEY,
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
        self.conn.commit()

    def save_position_state(self):
        """Save current position state to SQLite"""
        if not futures_state['position']:
            return
            
        pos = futures_state['position']
        self.cursor.execute('DELETE FROM position')
        self.cursor.execute('''
            INSERT INTO position (
                direction, size, entry, stop, target, leverage, margin,
                order_id, timestamp, exit_25_taken, exit_50_taken,
                stop_moved_to_be, original_stop, highest_price, lowest_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pos['direction'], pos['size'], pos['entry'], pos['stop'],
            pos['target'], pos['leverage'], pos['margin'], pos['order_id'],
            str(pos['timestamp']), int(pos.get('exit_25_taken', False)), 
            int(pos.get('exit_50_taken', False)), int(pos.get('stop_moved_to_be', False)),
            pos.get('original_stop', pos['stop']), pos.get('highest_price', pos['entry']),
            pos.get('lowest_price', pos['entry'])
        ))
        self.conn.commit()
        
    def load_position_state(self):
        """Load active position from SQLite on startup"""
        self.cursor.execute('SELECT * FROM position LIMIT 1')
        row = self.cursor.fetchone()
        if row:
            print(f"🔄 Recovering active position from database...")
            futures_state['position'] = {
                'direction': row[1],
                'size': row[2],
                'entry': row[3],
                'stop': row[4],
                'target': row[5],
                'leverage': row[6],
                'margin': row[7],
                'order_id': row[8],
                'timestamp': row[9] if row[9] else datetime.now(),
                'exit_25_taken': bool(row[10]),
                'exit_50_taken': bool(row[11]),
                'stop_moved_to_be': bool(row[12]),
                'original_stop': row[13],
                'highest_price': row[14],
                'lowest_price': row[15]
            }
            # Verify with exchange (optional but safe)
            try:
                result = session.get_positions(category="linear", symbol=symbol)
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
        self.cursor.execute('DELETE FROM position')
        self.conn.commit()
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

    def log_features(self, current_price, indicators, btc_data, volatility):
        """Log features for AI training"""
        try:
            self.cursor.execute('''
                INSERT INTO ai_training_data (
                    timestamp, current_price, rsi_15m, rsi_1h, macd_15m, 
                    macd_hist_15m, adx_15m, adx_1h, volatility, 
                    btc_1h_change, btc_4h_change, volume_ratio_15m
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
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

    def log_trade_outcome(self, pnl=None, exit_price=None):
        """Log outcome for AI training"""
        pos = futures_state['position']
        if not pos: return
        
        try:
            entry_time = pos.get('timestamp')
            if isinstance(entry_time, datetime):
                entry_time = entry_time.isoformat()
                
            self.cursor.execute('''
                INSERT INTO ai_trade_outcomes (
                    entry_time, exit_time, direction, entry_price, 
                    exit_price, pnl, max_favorable_price, max_adverse_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry_time,
                datetime.now().isoformat(),
                pos['direction'],
                pos['entry'],
                exit_price or pos.get('target'),
                pnl or 0,
                pos.get('highest_price', pos['entry']),
                pos.get('lowest_price', pos['entry'])
            ))
            self.conn.commit()
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
    
    def fetch_multi_timeframe_data(self):
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
    
    def calculate_futures_signals(self, indicators, current_price, volatility):
        """Calculate futures trading signals"""
        
        # Futures LONG conditions
        futures_long_conditions = [
            indicators["15m"]["rsi"] < 40,
            indicators["1h"]["rsi"] < 50,
            indicators["15m"]["macd"] > indicators["15m"]["macd_signal"],
            current_price > indicators["15m"]["ema_21"] * 0.998,
            indicators["15m"]["volume_ratio"] > 1.3,
            volatility > 0.02,
            indicators["15m"]["adx"] > 20
        ]
        
        # Futures SHORT conditions
        futures_short_conditions = [
            indicators["15m"]["rsi"] > 65,
            indicators["1h"]["rsi"] > 55,
            indicators["15m"]["macd"] < indicators["15m"]["macd_signal"],
            indicators["15m"]["macd_histogram"] < -0.2,
            current_price < indicators["15m"]["ema_21"],
            indicators["15m"]["volume_ratio"] > 1.4,
            volatility > 0.025,
            indicators["15m"]["stoch_k"] > 80,
            current_price > indicators["1h"]["ema_50"] * 0.98,
            indicators["15m"]["adx"] > 18
        ]
        
        long_score = sum(futures_long_conditions)
        short_score = sum(futures_short_conditions)
        
        if long_score >= 5:
            return {"signal": "LONG", "strength": long_score, "leverage": 10.0}
        elif short_score >= 6:
            return {"signal": "SHORT", "strength": short_score, "leverage": 10.5}
        
        return {"signal": None, "strength": max(long_score, short_score)}
    
    def calculate_futures_position_size(self, entry_price, stop_loss_price, leverage=10.0):
        """Calculate safe futures position size with proper USDT margin validation"""
        
        usdt_balance = self.get_usdt_balance()
        
        print(f"💰 Margin Check:")
        print(f"   Available USDT: ${usdt_balance:.2f}")
        
        # Check if we have sufficient USDT margin
        if usdt_balance < 5:
            print(f"❌ Insufficient USDT for futures margin: ${usdt_balance:.2f} (minimum $5)")
            return None
        
        # Use conservative margin - max 70% of available USDT for safety
        max_usable_margin = usdt_balance * 0.7
        risk_amount = usdt_balance * futures_risk_per_trade  # 5% of USDT balance
        
        # Distance to stop loss
        stop_distance = abs(entry_price - stop_loss_price)
        if stop_distance <= 0:
            print("❌ Invalid stop loss distance")
            return None
        
        # Calculate position size based on available margin
        max_position_by_margin = (max_usable_margin * leverage) / entry_price
        
        # Calculate position size based on risk (DO NOT multiply by leverage)
        max_position_by_risk = risk_amount / stop_distance
        
        # Use the smaller of the two for safety
        position_size = min(max_position_by_margin, max_position_by_risk)
        
        # Apply Bybit futures requirements for SOLUSDT
        min_order_size = 0.1  # Minimum 0.1 SOL for futures
        step_size = 0.1       # Must be in increments of 0.1
        
        if position_size < min_order_size:
            print(f"⚠️ Calculated position ({position_size:.4f}) below minimum ({min_order_size})")
            position_size = min_order_size
        else:
            position_size = round(position_size / step_size) * step_size
            print(f"📊 Rounded position to step size: {position_size:.1f} SOL")
        
        # Calculate required margin
        required_margin = (position_size * entry_price) / leverage
        
        # Final validation
        if required_margin > max_usable_margin:
            print(f"⚠️ Required margin ${required_margin:.2f} exceeds available ${max_usable_margin:.2f}")
            position_size = (max_usable_margin * leverage) / entry_price
            position_size = round(position_size / step_size) * step_size
            required_margin = (position_size * entry_price) / leverage
            
            if position_size < min_order_size:
                print(f"❌ Even reduced position ({position_size:.1f}) below minimum")
                return None
        
        if required_margin < 2:
            print(f"❌ Margin too small: ${required_margin:.2f} (min: $2)")
            return None
        
        actual_risk = position_size * stop_distance
        
        print(f"📊 Position Size Calculation:")
        print(f"   Max Usable (50%): ${max_usable_margin:.2f}")
        print(f"   Risk Amount ({futures_risk_per_trade*100:.0f}%): ${risk_amount:.2f}")
        print(f"   Stop Distance: ${stop_distance:.2f}")
        print(f"   Position Size: {position_size:.1f} SOL")
        print(f"   Required Margin: ${required_margin:.2f}")
        print(f"   Actual Risk: ${actual_risk:.2f}")
        
        return {
            'position_size': round(position_size, 1),
            'required_margin': round(required_margin, 2),
            'leverage': leverage,
            'risk_amount': risk_amount,
            'actual_risk': actual_risk
        }
    
    def calculate_improved_futures_stops(self, direction, current_price, indicators, signal_strength):
        """Calculate wider, more realistic stops for futures trading"""
        
        atr_15m = indicators["15m"]["atr"]
        atr_1h = indicators["1h"]["atr"]
        
        # Use LARGER ATR for futures to avoid noise
        primary_atr = max(atr_15m, atr_1h * 0.7)
        
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
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'risk_amount': stop_distance,
            'reward_amount': abs(take_profit - current_price),
            'reward_ratio': abs(take_profit - current_price) / stop_distance,
            'method_used': 'improved_wider_stops',
            'primary_atr_used': primary_atr
        }
    
    def place_futures_order(self, signal, current_price, indicators):
        """Place futures order with improved stops"""
        
        # Calculate improved stops
        stop_data = self.calculate_improved_futures_stops(
            signal['signal'], current_price, indicators, signal['strength']
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
        
        # Calculate position size
        position_data = self.calculate_futures_position_size(
            current_price, stop_loss, signal["leverage"]
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
                    symbol=symbol,
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
            
            # Place the order
            side = "Buy" if signal["signal"] == "LONG" else "Sell"
            order_params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": "Market",
                "qty": f"{position_data['position_size']:.1f}"
            }
            
            print(f"\n🚀 FUTURES {signal['signal']} Order:")
            print(f"   💰 Margin: ${position_data['required_margin']:.2f}")
            print(f"   📊 Position: {position_data['position_size']:.1f} SOL")
            print(f"   ⚡ Leverage: {signal['leverage']:.1f}x")
            print(f"   🎯 Entry: ${current_price:.2f}")
            print(f"   🛡️ Stop: ${stop_loss:.2f}")
            print(f"   💎 Target: ${take_profit:.2f}")
            print(f"   💀 Max Risk: ${position_data['actual_risk']:.2f}")
            
            result = session.place_order(**order_params)
            
            if result.get("retCode") == 0:
                print(f"✅ Futures {signal['signal']} order placed successfully!")
                
                # Set stops
                try:
                    stop_result = session.set_trading_stop(
                        category="linear",
                        symbol=symbol,
                        stopLoss=str(round(stop_loss, 2)),
                        takeProfit=str(round(take_profit, 2))
                    )
                    if stop_result.get("retCode") == 0:
                        print(f"✅ Stops set successfully")
                    else:
                        print(f"⚠️ Could not set stops: {stop_result.get('retMsg')}")
                except Exception as e:
                    print(f"⚠️ Error setting stops: {e}")
                
                # Store position
                futures_state['position'] = {
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
                    'highest_price': current_price,  # For LONG positions
                    'lowest_price': current_price    # For SHORT positions
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
                symbol=symbol
            )
            if result.get("retCode") == 0:
                positions = result.get("result", {}).get("list", [])
                active_pos = next((p for p in positions if float(p.get("size", 0)) > 0), None)
                
                if not active_pos:
                    print("⚠️ Bybit reports no open position, but local state had one. Resolving...")
                    # Get closed PnL to determine if win or loss
                    pnl_result = session.get_closed_pnl(
                        category="linear",
                        symbol=symbol,
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
                    self.log_trade_outcome(pnl=pnl)
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
                    symbol=symbol,
                    stopLoss=str(round(new_stop, 2))
                )
                self.save_position_state()
            except Exception as e:
                print(f"⚠️ Error setting trailing stop: {e}")
        
        # Check exit conditions
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

            if unrealized_pnl_pct > 0.015:  # at least 1.5% profit
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

            if unrealized_pnl_pct > 0.015:  # at least 1.5% profit
                # Place stop just below best price with ATR buffer
                proposed_stop = best_price - (1.5 * atr_15m)

                # Only update if better (higher than before)
                if proposed_stop > current_stop:
                    print(f"🔄 Trailing stop (LONG): ${current_stop:.2f} → ${proposed_stop:.2f}")
                    return proposed_stop

        # No change
        return current_stop

    
    def partial_position_management(self, position, current_price, indicators):
        """Take partial profits to reduce risk"""
        
        if not position:
            return None
        
        direction = position['direction']
        entry_price = position['entry']
        
        # Calculate unrealized P&L
        if direction == "SHORT":
            pnl_pct = (entry_price - current_price) / entry_price
        else:  # LONG
            pnl_pct = (current_price - entry_price) / entry_price
        
        changed = False
        # Take 25% profit at 1.5% gain
        if pnl_pct >= 0.015 and not position.get('exit_25_taken'):
            position['exit_25_taken'] = True
            print(f"💰 Taking 25% profit at ${current_price:.2f} (+{pnl_pct*100:.1f}%)")
            self.save_position_state()
        
        # Take another 25% at 3% gain
        if pnl_pct >= 0.03 and not position.get('exit_50_taken'):
            position['exit_50_taken'] = True
            print(f"💰 Taking 25% more profit at ${current_price:.2f} (+{pnl_pct*100:.1f}%)")
            self.save_position_state()
        
        # Move stop to breakeven after 25% taken (1.5% gain)
        if position.get('exit_25_taken') and not position.get('stop_moved_to_be'):
            position['stop'] = entry_price
            position['stop_moved_to_be'] = True
            print(f"🛡️ Stop moved to breakeven: ${entry_price:.2f}")
            self.save_position_state()
    
    def close_position(self, result_type, log_already_done=False, exit_price=None):
        """Close futures position"""
        if not futures_state['position']:
            return
        
        pos = futures_state['position']
        print(f"🚀 Closing futures {pos['direction']} position")
        
        if not log_already_done:
            self.log_trade_outcome(exit_price=exit_price)
            
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
        
        return True, "Can trade"
    
    def run_futures_strategy(self):
        """Main futures strategy execution"""
        
        print(f"\n{'='*80}")
        print(f"🔄 FUTURES Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # Fetch market data
        print("📊 Fetching multi-timeframe data...")
        data = self.fetch_multi_timeframe_data()
        if not data:
            print("❌ Failed to fetch market data")
            return None
        
        # Calculate indicators
        print("🧮 Calculating technical indicators...")
        indicators, current_price, volatility = self.calculate_indicators(data)
        
        # Check BTC correlation
        btc_data = self.check_btc_correlation()
        
        # Log features for AI Data Pipeline
        self.log_features(current_price, indicators, btc_data, volatility)
        
        # Check position exits
        self.check_position_exits(current_price, indicators)
        
        # Calculate futures signals
        signal = self.calculate_futures_signals(indicators, current_price, volatility)
        
        # Apply BTC filter
        if signal["signal"] == "LONG" and btc_data['bearish']:
            print("❌ BTC bearish - skipping futures LONG")
            signal["signal"] = None
        
        # Execute trades
        can_trade, trade_reason = self.can_trade()
        usdt_balance = self.get_usdt_balance()
        
        # Regime Filter: Block trades if market is ranging/flat
        if indicators['1h']['adx'] < 20:
            can_trade = False
            trade_reason = f"Market is flat/ranging (1h ADX: {indicators['1h']['adx']:.1f} < 20.0)"
        
        # Display market analysis
        print(f"\n📊 {symbol} Futures Analysis:")
        print(f"💰 Current Price: ${current_price:.4f}")
        print(f"📈 RSI (15m/1h/4h): {indicators['15m']['rsi']:.1f}/{indicators['1h']['rsi']:.1f}/{indicators['4h']['rsi']:.1f}")
        print(f"🌊 24h Volatility: {volatility:.2%}")
        print(f"💪 ADX Strength: {indicators['1h']['adx']:.1f}")
        print(f"₿ BTC: 1h={btc_data['1h_change']:+.1f}%, 4h={btc_data['4h_change']:+.1f}%")
        print(f"🚀 Signal: {signal['signal']} (Strength: {signal['strength']}/10)")
        
        if (signal["signal"] in ["LONG", "SHORT"] and 
            signal["strength"] >= signal_strength_threshold and 
            can_trade and 
            not futures_state['position'] and
            usdt_balance >= 5):
            
            print(f"\n🚀 FUTURES {signal['signal']} SIGNAL DETECTED!")
            self.place_futures_order(signal, current_price, indicators)
        elif signal["signal"] in ["LONG", "SHORT"] and usdt_balance < 5:
            print(f"⚠️ Signal detected but insufficient USDT margin: ${usdt_balance:.2f}")
        elif not can_trade:
            print(f"⚠️ Trading blocked: {trade_reason}")
        
        # Display status
        print(f"\n📊 Status:")
        print(f"🚀 Position: {futures_state['position']['direction'] if futures_state['position'] else 'None'}")
        print(f"📈 Daily Trades: {futures_state['daily_trades']}/{max_daily_trades}")
        print(f"💰 Available Balance: ${futures_state['available_balance']:.2f}")
        
        if futures_state['total_trades'] > 0:
            win_rate = futures_state['winning_trades'] / futures_state['total_trades'] * 100
            print(f"🎯 Win Rate: {win_rate:.1f}% ({futures_state['winning_trades']}/{futures_state['total_trades']})")
            
        return indicators
    
    def get_current_price(self):
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
        print(f"💎 Symbol: {symbol}")
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
        
        # If we restarted with a position, try to get initial indicators for trailing stop math
        if futures_state['position']:
            print("🔄 Initializing indicators for active position management...")
            data = self.fetch_multi_timeframe_data()
            if data:
                indicators_cache, _, _ = self.calculate_indicators(data)
        
        while True:
            try:
                current_time = time.time()
                
                # --- FAST LOOP: Active Position Management (Every 10 seconds) ---
                if futures_state['position']:
                    fast_price = self.get_current_price()
                    if fast_price and indicators_cache:
                        self.check_position_exits(fast_price, indicators_cache)
                
                # --- SLOW LOOP: Market Analysis & Signals (Every 5 minutes) ---
                if current_time - last_analysis_time >= analysis_interval:
                    cycle_count += 1
                    
                    indicators = self.run_futures_strategy()
                    if indicators:
                        indicators_cache = indicators
                    
                    # Reset daily counters at midnight
                    if datetime.now().hour == 0 and datetime.now().minute < 5:
                        futures_state['daily_trades'] = 0
                        print("🌅 Daily trading counter reset")
                    
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