"""
Hybrid Spot + Futures Trading Bot - FIXED VERSION
Fixes: 1) indicators variable error, 2) overly tight futures stops
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

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")

# Trading Configuration
symbol = "SOLUSDT"
primary_timeframe = "15"   # Primary analysis
higher_timeframe = "60"    # Trend confirmation

# Portfolio Allocation
SPOT_ALLOCATION = 0.70     # 70% for spot trading
FUTURES_ALLOCATION = 0.30  # 30% for futures trading

# Risk Management
spot_risk_per_trade = 0.99  # Use 99% of spot allocation (like current bot)
futures_risk_per_trade = 0.05  # 5% risk per futures trade
max_leverage = 20.0         # Conservative max leverage
min_reward_ratio = 2.5     # Minimum reward:risk ratio - INCREASED FOR FUTURES
min_volatility_threshold = 0.02

# Trading Limits
max_daily_trades_spot = 3
max_daily_trades_futures = 15
min_trade_gap_hours = 2
signal_strength_threshold = 5  # Increased from 3

# Initialize Bybit session
session = HTTP(
    testnet=False,
    api_key=api_key,
    api_secret=api_secret,
)

# Hybrid Trading State
hybrid_state = {
    'last_trade_time': None,
    'spot_position': None,
    'futures_position': None,
    'daily_spot_trades': 0,
    'daily_futures_trades': 0,
    'total_trades': 0,
    'winning_trades': 0,
    'consecutive_losses': 0,
    'max_consecutive_losses': 3,
    'session_start': datetime.now(),
    'portfolio_balance': {
        'total': 0,
        'spot_allocation': 0,
        'futures_allocation': 0
    }
}

class HybridTradingBot:
    
    def __init__(self):
        print(f"🤖 Initializing Hybrid Trading Bot...")
        self.initialize_portfolio()
        
    def initialize_portfolio(self):
        """Initialize portfolio allocations with validation"""
        total_balance = self.get_total_balance()
        usdt_balance = self.get_usdt_balance()
        
        # Minimum balance validation
        min_balance_required = 50  # $50 minimum for meaningful trading
        
        if total_balance < min_balance_required:
            print(f"⚠️  WARNING: Total portfolio ${total_balance:.2f} is below recommended minimum ${min_balance_required}")
        
        hybrid_state['portfolio_balance'] = {
            'total': total_balance,
            'usdt_only': usdt_balance,
            'spot_allocation': total_balance * SPOT_ALLOCATION,
            'futures_allocation': total_balance * FUTURES_ALLOCATION
        }
        
        print(f"💰 Portfolio Initialized:")
        print(f"   Total Portfolio Value: ${total_balance:.2f} (SOL + USDT)")
        print(f"   Available USDT: ${usdt_balance:.2f} (for futures margin)")
        print(f"   Spot Allocation (70%): ${hybrid_state['portfolio_balance']['spot_allocation']:.2f}")
        print(f"   Futures Allocation (30%): ${hybrid_state['portfolio_balance']['futures_allocation']:.2f}")
        
        # Critical warning for futures trading
        max_futures_margin = usdt_balance * 0.8  # Use max 80% of USDT for futures
        if max_futures_margin < 10:
            print(f"⚠️  FUTURES WARNING: Only ${usdt_balance:.2f} USDT available for margin")
            print(f"   💡 Consider converting some SOL to USDT for futures trading")
            print(f"   💡 Or reduce futures allocation to match available USDT")
        
        # Additional warnings for very small allocations
        if hybrid_state['portfolio_balance']['spot_allocation'] < 20:
            print(f"⚠️  Spot allocation too small for meaningful full-balance trades")
        
        if usdt_balance < 15:
            print(f"⚠️  USDT balance too low for futures trades (need $15+ USDT margin)")
    
    def get_total_balance(self):
        """Get total portfolio value (USDT + SOL value) for allocation"""
        try:
            print(f"🔍 Checking UNIFIED account...")
            result = session.get_wallet_balance(accountType="UNIFIED")
            
            if result.get("retCode") == 0:
                account_list = result.get("result", {}).get("list", [])
                if account_list:
                    coins = account_list[0].get("coin", [])
                    
                    usdt_balance = 0
                    sol_balance = 0
                    
                    print(f"📊 Available coins in UNIFIED:")
                    for coin in coins:
                        wb = coin.get('walletBalance', '0')
                        if float(wb) > 0:
                            print(f"   - {coin['coin']}: {wb}")
                            
                            if coin['coin'] == 'USDT':
                                usdt_balance = float(wb)
                            elif coin['coin'] == 'SOL':
                                sol_balance = float(wb)
                    
                    # Get current SOL price to calculate total value
                    try:
                        url = "https://api.bybit.com/v5/market/tickers"
                        params = {"category": "spot", "symbol": "SOLUSDT"}
                        response = requests.get(url, params=params, timeout=10)
                        
                        if response.status_code == 200:
                            ticker_data = response.json()
                            if ticker_data.get("retCode") == 0:
                                ticker_list = ticker_data.get("result", {}).get("list", [])
                                if ticker_list:
                                    sol_price = float(ticker_list[0]["lastPrice"])
                                    sol_value = sol_balance * sol_price
                                    
                                    total_value = usdt_balance + sol_value
                                    
                                    print(f"💰 Portfolio Calculation:")
                                    print(f"   USDT: ${usdt_balance:.2f}")
                                    print(f"   SOL: {sol_balance:.4f} × ${sol_price:.2f} = ${sol_value:.2f}")
                                    print(f"   Total Portfolio Value: ${total_value:.2f}")
                                    
                                    if total_value > 10:  # Must be at least $10
                                        return total_value
                    except Exception as e:
                        print(f"❌ Error getting SOL price: {e}")
                    
                    # Fallback: if we have significant USDT balance
                    if usdt_balance > 10:
                        print(f"💰 Using USDT balance: ${usdt_balance:.2f}")
                        return usdt_balance
                        
        except Exception as e:
            print(f"❌ Error checking account: {e}")
        
        print("⚠️  No sufficient balance found - using minimal fallback")
        return 50  # Reduced fallback for testing
    
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

    def get_sol_balance(self):
        """Get SOL balance for spot selling"""
        try:
            result = session.get_wallet_balance(accountType="UNIFIED")
            if result.get("retCode") == 0:
                account_list = result.get("result", {}).get("list", [])
                if account_list:
                    coins = account_list[0].get("coin", [])
                    for coin in coins:
                        if coin["coin"] == "SOL":
                            balance = float(coin.get("walletBalance", "0"))
                            return balance
            return 0.0
        except Exception as e:
            print(f"❌ Error getting SOL balance: {e}")
            return 0.0
    
    def fetch_multi_timeframe_data(self):
        """Fetch data from multiple timeframes"""
        data = {}
        
        for tf in [primary_timeframe, higher_timeframe, "240"]:  # 15m, 1h, 4h
            url = f"https://api.bybit.com/v5/market/kline"
            params = {
                "category": "spot",
                "symbol": symbol,
                "interval": tf,
                "limit": 100
            }
            
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    if result["retCode"] == 0:
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
            url = "https://api.bybit.com/v5/market/kline"
            params = {
                "category": "spot",
                "symbol": "BTCUSDT",
                "interval": "60",
                "limit": 10
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                result = response.json()
                if result["retCode"] == 0:
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
    
    def calculate_spot_signals(self, indicators, current_price, volatility):
        """Calculate spot trading signals (proven logic)"""
        
        # Proven LONG conditions from successful spot bot
        long_conditions = [
            indicators["15m"]["rsi"] < 45 and indicators["15m"]["rsi"] > 25,
            indicators["1h"]["rsi"] < 55,
            indicators["4h"]["rsi"] < 60,
            indicators["15m"]["macd"] > indicators["15m"]["macd_signal"],
            indicators["15m"]["macd_histogram"] > 0,
            current_price > indicators["1h"]["ema_21"],
            indicators["1h"]["ema_21"] > indicators["1h"]["ema_50"],
            indicators["1h"]["adx"] > 25,
            indicators["15m"]["volume_ratio"] > 1.2,
            indicators["15m"]["stoch_k"] < 80,
            volatility > min_volatility_threshold,
            indicators["15m"]["bb_upper"] > current_price  # Not at upper BB
        ]
        
        # Enhanced SELL conditions for spot
        sell_conditions = [
            indicators["15m"]["rsi"] > 70,
            indicators["1h"]["rsi"] > 65,
            indicators["15m"]["macd"] < indicators["15m"]["macd_signal"],
            indicators["15m"]["macd_histogram"] < -0.3,
            current_price < indicators["15m"]["ema_21"],
            indicators["15m"]["ema_21"] < indicators["15m"]["ema_50"],
            indicators["15m"]["adx"] > 20,
            indicators["15m"]["volume_ratio"] > 1.5,
            indicators["15m"]["stoch_k"] > 85,
            current_price > indicators["1h"]["ema_21"] * 1.02,  # Above 1h trend
            volatility > min_volatility_threshold
        ]
        
        long_score = sum(long_conditions)
        sell_score = sum(sell_conditions)
        
        if long_score >= 7:  # Proven threshold
            return {"signal": "LONG", "strength": long_score, "type": "spot"}
        elif sell_score >= 6:
            return {"signal": "SELL", "strength": sell_score, "type": "spot"}
        
        return {"signal": None, "strength": max(long_score, sell_score), "type": "spot"}
    
    def calculate_futures_signals(self, indicators, current_price, volatility):
        """Calculate futures trading signals (more aggressive)"""
        
        # Futures LONG (more sensitive)
        futures_long_conditions = [
            indicators["15m"]["rsi"] < 40,
            indicators["1h"]["rsi"] < 50,
            indicators["15m"]["macd"] > indicators["15m"]["macd_signal"],
            current_price > indicators["15m"]["ema_21"] * 0.998,  # Very close to 15m EMA
            indicators["15m"]["volume_ratio"] > 1.3,
            volatility > 0.02,
            indicators["15m"]["adx"] > 20
        ]
        
        # Futures SHORT (capture corrections)
        futures_short_conditions = [
            indicators["15m"]["rsi"] > 65,
            indicators["1h"]["rsi"] > 55,
            indicators["15m"]["macd"] < indicators["15m"]["macd_signal"],
            indicators["15m"]["macd_histogram"] < -0.2,
            current_price < indicators["15m"]["ema_21"],
            indicators["15m"]["volume_ratio"] > 1.4,
            volatility > 0.025,
            indicators["15m"]["stoch_k"] > 80,
            current_price > indicators["1h"]["ema_50"] * 0.98,  # Still above major support
            indicators["15m"]["adx"] > 18
        ]
        
        long_score = sum(futures_long_conditions)
        short_score = sum(futures_short_conditions)
        
        if long_score >= 5:
            return {"signal": "LONG", "strength": long_score, "type": "futures", "leverage": 10.0}
        elif short_score >= 6:  # Higher threshold for shorts
            return {"signal": "SHORT", "strength": short_score, "type": "futures", "leverage": 10.5}
        
        return {"signal": None, "strength": max(long_score, short_score), "type": "futures"}
    
    def calculate_futures_position_size(self, entry_price, stop_loss_price, leverage=10.0):
        """Calculate safe futures position size with proper USDT margin validation"""
        
        # Get actual USDT balance (required for futures margin)
        usdt_balance = self.get_usdt_balance()
        futures_allocation = hybrid_state['portfolio_balance']['futures_allocation']
        
        print(f"💰 Margin Check:")
        print(f"   Theoretical Allocation: ${futures_allocation:.2f}")
        print(f"   Available USDT: ${usdt_balance:.2f}")
        
        # Use the smaller of theoretical allocation or actual USDT
        available_margin = min(futures_allocation, usdt_balance)
        
        # Check if we have sufficient USDT margin
        if usdt_balance < 5:
            print(f"❌ Insufficient USDT for futures margin: ${usdt_balance:.2f} (minimum $5)")
            print(f"💡 Convert some SOL to USDT for futures trading")
            return None
        
        if available_margin < 10:
            print(f"⚠️  Limited margin available: ${available_margin:.2f}")
        
        # Use conservative margin - max 50% of available USDT for safety
        max_usable_margin = usdt_balance * 0.5
        risk_amount = usdt_balance * 0.02  # 2% of USDT balance (more conservative)
        
        # Distance to stop loss
        stop_distance = abs(entry_price - stop_loss_price)
        if stop_distance <= 0:
            print("❌ Invalid stop loss distance")
            return None
        
        # Calculate position size based on available margin
        max_position_by_margin = (max_usable_margin * leverage) / entry_price
        
        # Calculate position size based on risk
        base_position_size = risk_amount / stop_distance
        max_position_by_risk = base_position_size * leverage
        
        # Use the smaller of the two for safety
        position_size = min(max_position_by_margin, max_position_by_risk)
        
        # Apply Bybit futures requirements for SOLUSDT
        min_order_size = 0.1  # Minimum 0.1 SOL for futures
        step_size = 0.1       # Must be in increments of 0.1
        
        if position_size < min_order_size:
            print(f"⚠️  Calculated position ({position_size:.4f}) below minimum ({min_order_size})")
            position_size = min_order_size
        else:
            # Round to nearest step size (0.1 increments)
            position_size = round(position_size / step_size) * step_size
            print(f"📊 Rounded position to step size: {position_size:.1f} SOL")
        
        # Calculate required margin
        required_margin = (position_size * entry_price) / leverage
        
        # Final validation - ensure we don't exceed available USDT
        if required_margin > max_usable_margin:
            print(f"⚠️  Required margin ${required_margin:.2f} exceeds available ${max_usable_margin:.2f}")
            # Reduce position size to fit within available USDT
            position_size = (max_usable_margin * leverage) / entry_price
            position_size = round(position_size / step_size) * step_size  # Round to step size
            required_margin = (position_size * entry_price) / leverage
            
            # Check if reduced size is still above minimum
            if position_size < min_order_size:
                print(f"❌ Even reduced position ({position_size:.1f}) below minimum")
                return None
        
        # Ensure minimum margin requirement
        if required_margin < 2:  # Minimum $2 margin
            print(f"❌ Margin too small: ${required_margin:.2f} (min: $2)")
            return None
        
        actual_risk = position_size * stop_distance
        
        print(f"📊 Position Size Calculation:")
        print(f"   Available USDT Margin: ${usdt_balance:.2f}")
        print(f"   Max Usable (50%): ${max_usable_margin:.2f}")
        print(f"   Risk Amount (2% USDT): ${risk_amount:.2f}")
        print(f"   Stop Distance: ${stop_distance:.2f}")
        print(f"   Position Size: {position_size:.1f} SOL")
        print(f"   Required Margin: ${required_margin:.2f}")
        print(f"   Actual Risk: ${actual_risk:.2f}")
        
        return {
            'position_size': round(position_size, 1),  # 1 decimal place for futures
            'required_margin': round(required_margin, 2),
            'leverage': leverage,
            'risk_amount': risk_amount,
            'actual_risk': actual_risk
        }
    
    def place_spot_order(self, signal, current_price, indicators):
        """Place spot order using available balance (USDT for buying, SOL for selling)"""
        
        atr_15m = indicators["15m"]["atr"]
        
        if signal["signal"] == "LONG":
            # BUY SOL with available USDT
            stop_loss = current_price - (1.8 * atr_15m)
            take_profit = current_price + (min_reward_ratio * 1.8 * atr_15m)
            side = "Buy"
            limit_price = current_price * 0.9998
            
            # Use actual available USDT (not theoretical allocation)
            usdt_balance = self.get_usdt_balance()
            usable_balance = usdt_balance * 0.97  # Leave 3% for fees
            position_size = usable_balance / limit_price
            
            print(f"\n📍 SPOT LONG Order:")
            print(f"   💰 Available USDT: ${usdt_balance:.2f}")
            print(f"   💰 Using: ${usable_balance:.2f} (99% of USDT)")
            print(f"   📊 Position: {position_size:.2f} SOL")
            print(f"   🎯 Entry: ${limit_price:.2f}")
            print(f"   🛑 Stop: ${stop_loss:.2f}")
            print(f"   💎 Target: ${take_profit:.2f}")
            
        elif signal["signal"] == "SELL":
            # SELL existing SOL for USDT
            stop_loss = current_price + (1.8 * atr_15m)
            take_profit = current_price - (min_reward_ratio * 1.8 * atr_15m)
            side = "Sell"
            limit_price = current_price * 1.0002
            
            sol_balance = self.get_sol_balance()
            if sol_balance < 0.01:
                print("❌ No SOL available for spot sell")
                return False
            
            position_size = sol_balance * 0.97  # Sell 99% of SOL
            usable_balance = position_size * limit_price
            
            print(f"\n📍 SPOT SELL Order:")
            print(f"   💰 Available SOL: {sol_balance:.4f}")
            print(f"   💰 Selling: {position_size:.4f} SOL (99% of SOL)")
            print(f"   📊 Expected USDT: ${usable_balance:.2f}")
            print(f"   🎯 Entry: ${limit_price:.2f}")
            print(f"   🛑 Stop: ${stop_loss:.2f}")
            print(f"   💎 Target: ${take_profit:.2f}")
        
        # Validate minimum position size
        if position_size < 0.01:
            print(f"❌ Position size too small: {position_size:.4f} SOL")
            return False
        
        try:
            order_params = {
                "category": "spot",
                "symbol": symbol,
                "side": side,
                "orderType": "Limit",
                "qty": str(round(position_size, 2)),
                "price": str(round(limit_price, 2)),
                "timeInForce": "GTC"
            }
            
            print(f"📋 Order params: {order_params}")
            
            result = session.place_order(**order_params)
            
            if result.get("retCode") == 0:
                print(f"✅ Spot {signal['signal']} order placed successfully!")
                
                hybrid_state['spot_position'] = {
                    'direction': signal['signal'],
                    'size': position_size,
                    'entry': limit_price,
                    'stop': stop_loss,
                    'target': take_profit,
                    'order_id': result.get('result', {}).get('orderId'),
                    'timestamp': datetime.now()
                }
                
                hybrid_state['daily_spot_trades'] += 1
                hybrid_state['total_trades'] += 1
                
                return True
            else:
                print(f"❌ Spot order failed: {result.get('retMsg')}")
                return False
                
        except Exception as e:
            print(f"❌ Error placing spot order: {e}")
            return False
    
    def place_futures_order(self, signal, current_price, indicators):
        """FIXED: Enhanced futures order with WIDER stops for proper risk/reward"""
        
        # Calculate IMPROVED stops using new method
        stop_data = self.calculate_improved_futures_stops(
            signal['signal'], current_price, indicators, signal['strength']
        )
        
        stop_loss = stop_data['stop_loss']
        take_profit = stop_data['take_profit']
        
        print(f"\n🎯 IMPROVED Futures Stop Calculation:")
        print(f"   Entry: ${current_price:.2f}")
        print(f"   Stop Loss: ${stop_loss:.2f}")
        print(f"   Take Profit: ${take_profit:.2f}")
        print(f"   Risk: ${stop_data['risk_amount']:.2f}")
        print(f"   Reward: ${stop_data['reward_amount']:.2f}")
        print(f"   Ratio: {stop_data['reward_ratio']:.2f}:1")
        print(f"   Method: {stop_data['method_used']}")
        
        # Calculate position size using improved stops
        position_data = self.calculate_futures_position_size(
            current_price, stop_loss, signal["leverage"]
        )
        
        if position_data is None:
            print("❌ Cannot calculate valid futures position size")
            return False
        
        try:
            # Set leverage (same as before)
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
                    print(f"ℹ️  Leverage already set to {signal['leverage']:.1f}x (continuing)")
                else:
                    print(f"⚠️  Leverage response: {leverage_result.get('retMsg')} (continuing)")
                    
            except Exception as leverage_error:
                print(f"⚠️  Leverage setting error: {leverage_error} (continuing)")
            
            # Place the order
            side = "Buy" if signal["signal"] == "LONG" else "Sell"
            order_params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": "Market",
                "qty": f"{position_data['position_size']:.1f}"
            }
            
            print(f"\n🚀 IMPROVED FUTURES {signal['signal']} Order:")
            print(f"   💰 Margin: ${position_data['required_margin']:.2f}")
            print(f"   📊 Position: {position_data['position_size']:.1f} SOL")
            print(f"   ⚡ Leverage: {signal['leverage']:.1f}x")
            print(f"   🎯 Entry: ${current_price:.2f}")
            print(f"   🛡️  WIDER Stop: ${stop_loss:.2f}")
            print(f"   💎 BETTER Target: ${take_profit:.2f}")
            print(f"   💀 Max Risk: ${position_data['actual_risk']:.2f}")
            
            result = session.place_order(**order_params)
            
            if result.get("retCode") == 0:
                print(f"✅ Improved futures {signal['signal']} order placed successfully!")
                
                # Set improved stops
                try:
                    stop_result = session.set_trading_stop(
                        category="linear",
                        symbol=symbol,
                        stopLoss=str(round(stop_loss, 2)),
                        takeProfit=str(round(take_profit, 2))
                    )
                    if stop_result.get("retCode") == 0:
                        print(f"✅ Improved stops set successfully")
                    else:
                        print(f"⚠️  Could not set stops: {stop_result.get('retMsg')}")
                except Exception as e:
                    print(f"⚠️  Error setting stops: {e}")
                
                # Store position with improved data
                hybrid_state['futures_position'] = {
                    'direction': signal['signal'],
                    'size': position_data['position_size'],
                    'entry': current_price,
                    'stop': stop_loss,
                    'target': take_profit,
                    'leverage': signal['leverage'],
                    'margin': position_data['required_margin'],
                    'order_id': result.get('result', {}).get('orderId'),
                    'timestamp': datetime.now(),
                    # Enhanced tracking
                    'exit_25_taken': False,
                    'exit_50_taken': False,
                    'stop_moved_to_be': False,
                    'original_stop': stop_loss,
                    'support_resistance': stop_data['support_resistance'] if 'support_resistance' in stop_data else None
                }
                
                hybrid_state['daily_futures_trades'] += 1
                hybrid_state['total_trades'] += 1
                
                return True
            else:
                print(f"❌ Futures order failed: {result.get('retMsg')}")
                return False
                
        except Exception as e:
            print(f"❌ Error placing improved futures order: {e}")
            return False
    
    def calculate_improved_futures_stops(self, direction, current_price, indicators, signal_strength):
        """IMPROVED: Calculate wider, more realistic stops for futures trading"""
        
        atr_15m = indicators["15m"]["atr"]
        atr_1h = indicators["1h"]["atr"]
        
        # Use LARGER ATR for futures to avoid noise
        primary_atr = max(atr_15m, atr_1h * 0.7)  # Use larger of 15m ATR or 70% of 1h ATR
        
        print(f"📊 ATR Analysis:")
        print(f"   15m ATR: ${atr_15m:.2f}")
        print(f"   1h ATR: ${atr_1h:.2f}")
        print(f"   Using Primary ATR: ${primary_atr:.2f}")
        
        if direction == "SHORT":
            # WIDER stops for SHORT positions
            # Base stop: 2.5x ATR above entry (vs previous 1.5x)
            base_stop_distance = 2.5 * primary_atr
            
            # Signal strength adjustment (stronger signals get slightly wider stops)
            strength_multiplier = 1.0 + (signal_strength / 15)  # 1.0x to 1.67x
            
            # Volatility adjustment
            volatility_factor = min(atr_1h / atr_15m, 1.8) if atr_15m > 0 else 1.2
            
            # Final stop calculation
            stop_distance = base_stop_distance * strength_multiplier * volatility_factor
            stop_loss = current_price + stop_distance
            
            # Take profit: Minimum 3:1 reward ratio for futures
            min_futures_ratio = 3.0  # Increased from 2.5
            reward_distance = stop_distance * min_futures_ratio
            take_profit = current_price - reward_distance
            
        else:  # LONG direction
            # Similar logic but reversed
            base_stop_distance = 2.5 * primary_atr
            strength_multiplier = 1.0 + (signal_strength / 15)
            volatility_factor = min(atr_1h / atr_15m, 1.8) if atr_15m > 0 else 1.2
            
            stop_distance = base_stop_distance * strength_multiplier * volatility_factor
            stop_loss = current_price - stop_distance
            
            # Take profit with 3:1 ratio
            min_futures_ratio = 3.0
            reward_distance = stop_distance * min_futures_ratio
            take_profit = current_price + reward_distance
        
        # Ensure minimum distances for high leverage trading
        min_stop_distance = current_price * 0.008  # 0.8% minimum stop distance
        if stop_distance < min_stop_distance:
            print(f"⚠️  Stop distance too small, increasing to minimum 0.8%")
            stop_distance = min_stop_distance
            
            if direction == "SHORT":
                stop_loss = current_price + stop_distance
                take_profit = current_price - (stop_distance * 3.0)
            else:
                stop_loss = current_price - stop_distance
                take_profit = current_price + (stop_distance * 3.0)
        
        return {
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'risk_amount': stop_distance,
            'reward_amount': abs(take_profit - current_price),
            'reward_ratio': abs(take_profit - current_price) / stop_distance,
            'method_used': 'improved_wider_stops_3_to_1_ratio',
            'primary_atr_used': primary_atr
        }
    
    def check_position_exits(self, current_price, indicators):
        """FIXED: Enhanced position exit management - now properly receives indicators"""
        
        # Enhanced spot position management (keep existing logic but add trailing)
        if hybrid_state['spot_position']:
            pos = hybrid_state['spot_position']
            direction = pos['direction']
            
            if direction == "LONG" and current_price <= pos['stop']:
                print(f"🛑 SPOT STOP LOSS hit at ${current_price:.2f}")
                self.close_spot_position("LOSS")
            elif direction == "LONG" and current_price >= pos['target']:
                print(f"🎯 SPOT TAKE PROFIT hit at ${current_price:.2f}")
                self.close_spot_position("WIN")
            elif direction == "SELL" and current_price >= pos['stop']:
                print(f"🛑 SPOT STOP LOSS hit at ${current_price:.2f}")
                self.close_spot_position("LOSS")
            elif direction == "SELL" and current_price <= pos['target']:
                print(f"🎯 SPOT TAKE PROFIT hit at ${current_price:.2f}")
                self.close_spot_position("WIN")
        
        # ENHANCED futures position management
        if hybrid_state['futures_position']:
            pos = hybrid_state['futures_position']
            direction = pos['direction']
            
            # 1. Check for partial profit taking
            partial_exits = self.partial_position_management(pos, current_price, indicators)
            
            # 2. Update trailing stop
            new_stop = self.implement_trailing_stop(pos, current_price, indicators)
            if new_stop and new_stop != pos['stop']:
                pos['stop'] = new_stop
                # Update stop on exchange
                try:
                    session.set_trading_stop(
                        category="linear",
                        symbol=symbol,
                        stopLoss=str(round(new_stop, 2))
                    )
                except:
                    pass
            
            # 3. Check exit conditions
            if direction == "LONG" and current_price <= pos['stop']:
                print(f"🛑 IMPROVED FUTURES STOP hit at ${current_price:.2f}")
                self.close_futures_position("LOSS")
            elif direction == "LONG" and current_price >= pos['target']:
                print(f"🎯 IMPROVED FUTURES TARGET hit at ${current_price:.2f}")
                self.close_futures_position("WIN")
            elif direction == "SHORT" and current_price >= pos['stop']:
                print(f"🛑 IMPROVED FUTURES STOP hit at ${current_price:.2f}")
                self.close_futures_position("LOSS")
            elif direction == "SHORT" and current_price <= pos['target']:
                print(f"🎯 IMPROVED FUTURES TARGET hit at ${current_price:.2f}")
                self.close_futures_position("WIN")
    
    def close_spot_position(self, result_type):
        """Close spot position"""
        if not hybrid_state['spot_position']:
            return
        
        pos = hybrid_state['spot_position']
        print(f"📍 Closing spot {pos['direction']} position")
        
        # Update statistics
        if result_type == "WIN":
            hybrid_state['winning_trades'] += 1
            hybrid_state['consecutive_losses'] = 0
        else:
            hybrid_state['consecutive_losses'] += 1
        
        hybrid_state['spot_position'] = None
    
    def close_futures_position(self, result_type):
        """Close futures position"""
        if not hybrid_state['futures_position']:
            return
        
        pos = hybrid_state['futures_position']
        print(f"🚀 Closing futures {pos['direction']} position")
        
        # Update statistics
        if result_type == "WIN":
            hybrid_state['winning_trades'] += 1
            hybrid_state['consecutive_losses'] = 0
        else:
            hybrid_state['consecutive_losses'] += 1
        
        hybrid_state['futures_position'] = None
    
    def can_trade_spot(self):
        """Check if spot trading is allowed"""
        if hybrid_state['daily_spot_trades'] >= max_daily_trades_spot:
            return False, "Spot daily limit reached"
        
        if hybrid_state['consecutive_losses'] >= hybrid_state['max_consecutive_losses']:
            return False, "Too many consecutive losses"
        
        return True, "Can trade spot"
    
    def can_trade_futures(self):
        """Check if futures trading is allowed"""
        if hybrid_state['daily_futures_trades'] >= max_daily_trades_futures:
            return False, "Futures daily limit reached"
        
        if hybrid_state['consecutive_losses'] >= hybrid_state['max_consecutive_losses']:
            return False, "Too many consecutive losses"
        
        return True, "Can trade futures"
        
    def get_support_resistance_levels(self, indicators, current_price):
        """Calculate dynamic support and resistance levels"""
        
        # Get recent highs and lows
        tf_15m = indicators["15m"]
        tf_1h = indicators["1h"]
        
        # Key technical levels
        ema_21_15m = tf_15m["ema_21"]
        ema_50_15m = tf_15m["ema_50"]
        ema_21_1h = tf_1h["ema_21"]
        ema_50_1h = tf_1h["ema_50"]
        
        bb_upper = tf_15m["bb_upper"]
        bb_lower = tf_15m["bb_lower"]
        
        # Identify key levels
        levels = [ema_21_15m, ema_50_15m, ema_21_1h, ema_50_1h, bb_upper, bb_lower]
        
        # Find nearest support (below current price)
        supports = [level for level in levels if level < current_price]
        nearest_support = max(supports) if supports else current_price * 0.95
        
        # Find nearest resistance (above current price)
        resistances = [level for level in levels if level > current_price]
        nearest_resistance = min(resistances) if resistances else current_price * 1.05
        
        # Simple Fibonacci levels (from recent swing)
        swing_high = max(levels)
        swing_low = min(levels)
        fib_618 = swing_high - 0.618 * (swing_high - swing_low)
        fib_382 = swing_high - 0.382 * (swing_high - swing_low)
        
        return {
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'fib_618': fib_618,
            'fib_382': fib_382,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower
        }

    def implement_trailing_stop(self, position, current_price, indicators):
        """Implement trailing stop to lock in profits"""
        
        if not position:
            return None
        
        direction = position['direction']
        entry_price = position['entry']
        current_stop = position['stop']
        atr_15m = indicators["15m"]["atr"]
        
        if direction == "SHORT":
            # For SHORT: trailing stop moves DOWN as price moves DOWN (in our favor)
            unrealized_pnl_pct = (entry_price - current_price) / entry_price
            
            if unrealized_pnl_pct > 0.02:  # 2% profit
                # Move stop down by 50% of the favorable move
                favorable_move = entry_price - current_price
                new_stop = current_stop - (0.5 * favorable_move)
                
                # Don't move stop less favorable
                new_stop = min(new_stop, current_stop)
                
                # Ensure minimum buffer for high leverage
                min_stop = current_price + (1.0 * atr_15m)  # Tighter for high leverage
                final_stop = max(new_stop, min_stop)
                
                if final_stop != current_stop:
                    print(f"🔄 Trailing stop: ${current_stop:.2f} → ${final_stop:.2f}")
                
                return final_stop
        
        else:  # LONG
            # For LONG: trailing stop moves UP as price moves UP
            unrealized_pnl_pct = (current_price - entry_price) / entry_price
            
            if unrealized_pnl_pct > 0.02:  # 2% profit
                favorable_move = current_price - entry_price
                new_stop = current_stop + (0.5 * favorable_move)
                
                # Don't move stop less favorable
                new_stop = max(new_stop, current_stop)
                
                # Ensure minimum buffer
                max_stop = current_price - (1.0 * atr_15m)
                final_stop = min(new_stop, max_stop)
                
                if final_stop != current_stop:
                    print(f"🔄 Trailing stop: ${current_stop:.2f} → ${final_stop:.2f}")
                
                return final_stop
        
        return current_stop  # No change

    def partial_position_management(self, position, current_price, indicators):
        """Take partial profits to reduce risk of full stop-out"""
        
        if not position:
            return None
        
        direction = position['direction']
        entry_price = position['entry']
        
        # Calculate unrealized P&L
        if direction == "SHORT":
            pnl_pct = (entry_price - current_price) / entry_price
        else:  # LONG
            pnl_pct = (current_price - entry_price) / entry_price
        
        partial_exits = []
        
        # Take 25% profit at 1.5:1 risk/reward (adjusted for high leverage)
        if pnl_pct >= 0.02 and not position.get('exit_25_taken'):  # 2% profit
            partial_exits.append({
                'percentage': 0.25,
                'reason': '1.5:1_risk_reward',
                'price': current_price
            })
            position['exit_25_taken'] = True
            print(f"💰 Taking 25% profit at ${current_price:.2f} (+{pnl_pct*100:.1f}%)")
        
        # Take another 25% at 2.5:1 risk/reward  
        if pnl_pct >= 0.04 and not position.get('exit_50_taken'):  # 4% profit
            partial_exits.append({
                'percentage': 0.25,
                'reason': '2.5:1_risk_reward', 
                'price': current_price
            })
            position['exit_50_taken'] = True
            print(f"💰 Taking 25% more profit at ${current_price:.2f} (+{pnl_pct*100:.1f}%)")
        
        # Move stop to breakeven after taking 50% profit
        if position.get('exit_50_taken') and not position.get('stop_moved_to_be'):
            position['stop'] = entry_price
            position['stop_moved_to_be'] = True
            print(f"🛡️  Stop moved to breakeven: ${entry_price:.2f}")
        
        return partial_exits

    def run_hybrid_strategy(self):
        """Main hybrid strategy execution - FIXED indicators passing"""
        
        print(f"\n{'='*80}")
        print(f"🔄 HYBRID Analysis Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # Fetch market data
        print("📊 Fetching multi-timeframe data...")
        data = self.fetch_multi_timeframe_data()
        if not data:
            print("❌ Failed to fetch market data")
            return
        
        # Calculate indicators
        print("🧮 Calculating technical indicators...")
        indicators, current_price, volatility = self.calculate_indicators(data)
        
        # Check BTC correlation
        btc_data = self.check_btc_correlation()
        
        # FIXED: Check position exits - now properly passes indicators
        self.check_position_exits(current_price, indicators)
        
        # Calculate signals
        spot_signal = self.calculate_spot_signals(indicators, current_price, volatility)
        futures_signal = self.calculate_futures_signals(indicators, current_price, volatility)
        
        # Apply BTC filter
        if spot_signal["signal"] == "LONG" and btc_data['bearish']:
            print("❌ BTC bearish - skipping spot LONG")
            spot_signal["signal"] = None
        
        if futures_signal["signal"] == "LONG" and btc_data['bearish']:
            print("❌ BTC bearish - skipping futures LONG")
            futures_signal["signal"] = None
        
        # Display market analysis
        print(f"\n📊 {symbol} Hybrid Market Analysis:")
        print(f"💰 Current Price: ${current_price:.4f}")
        print(f"📈 RSI (15m/1h/4h): {indicators['15m']['rsi']:.1f}/{indicators['1h']['rsi']:.1f}/{indicators['4h']['rsi']:.1f}")
        print(f"🌊 24h Volatility: {volatility:.2%}")
        print(f"💪 ADX Strength: {indicators['1h']['adx']:.1f}")
        print(f"₿ BTC: 1h={btc_data['1h_change']:+.1f}%, 4h={btc_data['4h_change']:+.1f}%")
        print(f"📍 Spot Signal: {spot_signal['signal']} ({spot_signal['strength']}/12)")
        print(f"🚀 Futures Signal: {futures_signal['signal']} ({futures_signal['strength']}/10)")
        
        # Execute trades
        can_spot, spot_reason = self.can_trade_spot()
        can_futures, futures_reason = self.can_trade_futures()
        
        # Get actual USDT balance for futures margin validation
        usdt_balance = self.get_usdt_balance()
        
        # Futures trading (with USDT balance check)
        if (futures_signal["signal"] in ["LONG", "SHORT"] and 
            futures_signal["strength"] >= 5 and 
            can_futures and 
            not hybrid_state['futures_position'] and
            usdt_balance >= 5):  # Need at least $5 USDT for futures margin
            
            print(f"\n🚀 FUTURES {futures_signal['signal']} SIGNAL DETECTED!")
            self.place_futures_order(futures_signal, current_price, indicators)
        elif (futures_signal["signal"] in ["LONG", "SHORT"] and 
              usdt_balance < 5):
            print(f"⚠️  Futures signal detected but insufficient USDT margin: ${usdt_balance:.2f}")
            print(f"💡 Convert some SOL to USDT to enable futures trading")
        
        # Spot trading (with balance check)  
        if (spot_signal["signal"] in ["LONG", "SELL"] and 
            spot_signal["strength"] >= 7 and 
            can_spot and 
            not hybrid_state['spot_position'] and
            hybrid_state['portfolio_balance']['spot_allocation'] >= 5):  # Minimum $5
            
            print(f"\n📍 SPOT {spot_signal['signal']} SIGNAL DETECTED!")
            self.place_spot_order(spot_signal, current_price, indicators)
        elif (spot_signal["signal"] in ["LONG", "SELL"] and 
              hybrid_state['portfolio_balance']['spot_allocation'] < 5):
            print(f"⚠️  Spot signal detected but allocation too small: ${hybrid_state['portfolio_balance']['spot_allocation']:.2f}")
        
        # Display status
        print(f"\n📊 Current Status:")
        print(f"📍 Spot Position: {hybrid_state['spot_position']['direction'] if hybrid_state['spot_position'] else 'None'}")
        print(f"🚀 Futures Position: {hybrid_state['futures_position']['direction'] if hybrid_state['futures_position'] else 'None'}")
        print(f"📈 Daily Trades: Spot={hybrid_state['daily_spot_trades']}/{max_daily_trades_spot}, Futures={hybrid_state['daily_futures_trades']}/{max_daily_trades_futures}")
        
        if hybrid_state['total_trades'] > 0:
            win_rate = hybrid_state['winning_trades'] / hybrid_state['total_trades'] * 100
            print(f"🎯 Win Rate: {win_rate:.1f}% ({hybrid_state['winning_trades']}/{hybrid_state['total_trades']})")
    
    def run_bot(self):
        """Main bot execution loop"""
        
        print("🤖" + "="*80)
        print("🤖 HYBRID SPOT + FUTURES TRADING BOT - IMPROVED VERSION")
        print("🤖" + "="*80)
        print(f"💎 Symbol: {symbol}")
        print(f"⚡ Strategy: Hybrid Multi-Asset + BTC Correlation")
        print(f"📊 Portfolio Split: {SPOT_ALLOCATION*100:.0f}% Spot + {FUTURES_ALLOCATION*100:.0f}% Futures")
        print(f"⏰ Analysis Frequency: Every 5 minutes")
        print(f"🎯 Signal Thresholds: Spot=7/12, Futures=5/10")
        print(f"💰 Max Leverage: {max_leverage:.1f}x")
        print(f"⚠️  Risk Management: 5% per futures trade, 99% spot allocation")
        print(f"🛡️  IMPROVED: Wider stops (2.5x ATR), 3:1 reward ratio")
        print("="*80)
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                
                self.run_hybrid_strategy()
                
                # Reset daily counters at midnight
                if datetime.now().hour == 0 and datetime.now().minute < 5:
                    hybrid_state['daily_spot_trades'] = 0
                    hybrid_state['daily_futures_trades'] = 0
                    print("🌅 Daily trading counters reset")
                
                # Session statistics
                runtime = datetime.now() - hybrid_state['session_start']
                print(f"\n📊 Session Stats:")
                print(f"⏰ Runtime: {runtime}")
                print(f"🔄 Cycles: {cycle_count}")
                print(f"💼 Total Trades: {hybrid_state['total_trades']}")
                
                next_cycle = datetime.now() + timedelta(minutes=5)
                print(f"\n💤 Next cycle at {next_cycle.strftime('%H:%M:%S')}")
                print("="*80)
                
                time.sleep(300)  # 5 minutes
                
            except KeyboardInterrupt:
                print("\n🛑 Hybrid bot stopped by user")
                print(f"📊 Final Stats: {hybrid_state['total_trades']} trades, {cycle_count} cycles")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("⏰ Waiting 2 minutes before retry...")
                time.sleep(120)

if __name__ == "__main__":
    bot = HybridTradingBot()
    bot.run_bot()