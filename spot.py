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

load_dotenv()
api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")


# # API keys
# api_key = input("Enter you api key: ")
# api_secret = input("Enter your api secret: ")

symbol = "SOLUSDT"
primary_timeframe = "15"   # Primary analysis
higher_timeframe = "60"    # Trend confirmation
risk_per_trade = 0.015     # 1.5% risk per trade (conservative)

# Trading settings for profitability
min_reward_ratio = 2.5     # Minimum 2.5:1 reward:risk ratio
min_trade_gap_hours = 4    # Minimum 4 hours between trades
min_volatility_threshold = 0.02  # Minimum 2% volatility to trade
signal_strength_threshold = 3    # Need at least 3 confirmations

session = HTTP(
    testnet=False,
    api_key=api_key,
    api_secret=api_secret,
)

# Track trading state
trading_state = {
    'last_trade_time': None,
    'current_position': None,
    'daily_trades': 0,
    'daily_pnl': 0,
    'max_daily_trades': 3,    # Limit to 3 quality trades per day
    'consecutive_losses': 0,
    'max_consecutive_losses': 2,
    'total_trades': 0,
    'winning_trades': 0
}

def get_sol_balance():
    """Get SOL balance for selling"""
    account_types = ["UNIFIED", "SPOT"]
    
    for account_type in account_types:
        try:
            result = session.get_wallet_balance(accountType=account_type)
            if result.get("retCode") == 0:
                account_list = result.get("result", {}).get("list", [])
                if account_list:
                    coins = account_list[0].get("coin", [])
                    for coin in coins:
                        if coin["coin"] == "SOL":
                            balance_str = coin.get("availableToWithdraw", "0")
                            wallet_balance = coin.get("walletBalance", "0")
                            
                            # Try both balance fields
                            for bal_str in [balance_str, wallet_balance]:
                                if bal_str and bal_str.strip() and bal_str != '0':
                                    try:
                                        balance = float(bal_str)
                                        if balance > 0:
                                            print(f"💰 Available SOL: {balance:.4f}")
                                            return balance
                                    except ValueError:
                                        continue
        except Exception as e:
            print(f"❌ Error checking SOL balance: {e}")
    
    return 0.0  # No SOL available

def get_account_balance():
    """Get USDT balance for position sizing with improved error handling"""
    account_types = ["UNIFIED", "SPOT"]  # Try different account types
    
    for account_type in account_types:
        try:
            print(f"🔍 Checking {account_type} account...")
            result = session.get_wallet_balance(accountType=account_type)
            print(f"🔍 {account_type} API response code: {result.get('retCode')}")
            
            if result.get("retCode") == 0:
                account_list = result.get("result", {}).get("list", [])
                if not account_list:
                    print(f"⚠️  No accounts found in {account_type}")
                    continue
                    
                coins = account_list[0].get("coin", [])
                print(f"🔍 Found {len(coins)} coins in {account_type} wallet")
                
                for coin in coins:
                    if coin["coin"] == "USDT":
                        balance_str = coin.get("availableToWithdraw", "0")
                        wallet_balance = coin.get("walletBalance", "0")
                        
                        print(f"🔍 USDT availableToWithdraw: '{balance_str}'")
                        print(f"🔍 USDT walletBalance: '{wallet_balance}'")
                        
                        # Try both balance fields
                        for bal_str in [balance_str, wallet_balance]:
                            if bal_str and bal_str.strip() and bal_str != '0':
                                try:
                                    balance = float(bal_str)
                                    if balance > 0:
                                        print(f"💰 {account_type} Balance: ${balance:.2f} USDT")
                                        return balance
                                except ValueError:
                                    continue
                
                print(f"⚠️  No valid USDT balance in {account_type}, checking all coins:")
                for coin in coins:
                    print(f"   - {coin['coin']}: available={coin.get('availableToWithdraw', 'N/A')}, wallet={coin.get('walletBalance', 'N/A')}")
                    
            else:
                print(f"❌ {account_type} API error: {result.get('retMsg', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error checking {account_type} balance: {e}")
    
    print("⚠️  Using fallback balance for testing")
    return 100  # Small fallback for safe testing

def fetch_multi_timeframe_data():
    """Fetch data from multiple timeframes for better analysis"""
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
                    # Reverse to get chronological order (oldest first)
                    candles = list(reversed(candles))
                    
                    data[tf] = {
                        'close': np.array([float(c[4]) for c in candles]),
                        'high': np.array([float(c[2]) for c in candles]),
                        'low': np.array([float(c[3]) for c in candles]),
                        'volume': np.array([float(c[5]) for c in candles]),
                        'timestamp': [int(c[0]) for c in candles]
                    }
                    print(f"✅ Fetched {tf} data: {len(candles)} candles")
        except Exception as e:
            print(f"❌ Error fetching {tf} data: {e}")
            return None
        
        time.sleep(0.1)  # Rate limiting
    
    return data if len(data) == 3 else None

def calculate_market_strength(data):
    """Calculate overall market strength and volatility using TA-Lib"""
    tf_15m = data[primary_timeframe]
    tf_1h = data[higher_timeframe] 
    tf_4h = data["240"]
    
    # Calculate indicators for each timeframe
    indicators = {}
    
    for tf, prices in [("15m", tf_15m), ("1h", tf_1h), ("4h", tf_4h)]:
        closes = prices['close']
        highs = prices['high']
        lows = prices['low']
        volumes = prices['volume']
        
        # TA-Lib calculations (much more accurate)
        rsi = talib.RSI(closes, timeperiod=14)[-1]
        macd_line, macd_signal, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
        ema_21 = talib.EMA(closes, timeperiod=21)[-1]
        ema_50 = talib.EMA(closes, timeperiod=50)[-1]
        atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
        volume_sma = talib.SMA(volumes, timeperiod=20)[-1]
        bb_upper, bb_middle, bb_lower = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        adx = talib.ADX(highs, lows, closes, timeperiod=14)[-1]
        
        # Additional indicators for better signals
        stoch_k, stoch_d = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowd_period=3)
        williams_r = talib.WILLR(highs, lows, closes, timeperiod=14)[-1]
        
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
            'williams_r': williams_r
        }
    
    current_price = tf_15m['close'][-1]
    
    # Calculate volatility (24-hour price change)
    if len(tf_1h['close']) >= 24:
        price_24h_ago = tf_1h['close'][-24]
        price_change_24h = (current_price - price_24h_ago) / price_24h_ago
        volatility = abs(price_change_24h)
    else:
        volatility = 0.02  # Default 2%
    
    return indicators, current_price, volatility

def calculate_signal_strength(indicators, current_price, volatility):
    """Calculate signal strength (0-10) for quality filtering"""
    
    # Get indicator values
    rsi_15m = indicators["15m"]["rsi"]
    rsi_1h = indicators["1h"]["rsi"] 
    rsi_4h = indicators["4h"]["rsi"]
    
    macd_15m = indicators["15m"]["macd"]
    macd_signal_15m = indicators["15m"]["macd_signal"]
    macd_hist_15m = indicators["15m"]["macd_histogram"]
    
    ema_21_1h = indicators["1h"]["ema_21"]
    ema_50_1h = indicators["1h"]["ema_50"]
    ema_21_4h = indicators["4h"]["ema_21"]
    
    adx_1h = indicators["1h"]["adx"]
    volume_ratio = indicators["15m"]["current_volume"] / indicators["15m"]["volume_sma"]
    
    stoch_k_15m = indicators["15m"]["stoch_k"]
    bb_position = (current_price - indicators["15m"]["bb_lower"]) / (indicators["15m"]["bb_upper"] - indicators["15m"]["bb_lower"])
    
    # Enhanced LONG signal conditions
    long_conditions = [
        rsi_15m < 45 and rsi_15m > 25,           # Oversold but not extreme
        rsi_1h < 55,                             # 1h trend not overbought
        rsi_4h < 60,                             # 4h trend bullish room
        macd_15m > macd_signal_15m,              # MACD bullish crossover
        macd_hist_15m > 0,                       # MACD histogram positive
        current_price > ema_21_1h,               # Above 1h trend
        ema_21_1h > ema_50_1h,                   # Strong 1h uptrend
        ema_21_4h > indicators["4h"]["ema_50"],  # 4h uptrend confirmation
        adx_1h > 25,                             # Strong trend
        volume_ratio > 1.2,                      # Above average volume
        stoch_k_15m < 80,                        # Stochastic not overbought
        bb_position < 0.8,                       # Not near upper BB
        volatility > min_volatility_threshold     # Sufficient volatility
    ]
    
    # Enhanced SELL signal conditions (for spot trading)
    sell_conditions = [
        rsi_15m > 55 and rsi_15m < 75,           # Overbought but not extreme
        rsi_1h > 45,                             # 1h trend not oversold
        rsi_4h > 40,                             # 4h trend bearish room
        macd_15m < macd_signal_15m,              # MACD bearish crossover
        macd_hist_15m < 0,                       # MACD histogram negative
        current_price < ema_21_1h,               # Below 1h trend
        ema_21_1h < ema_50_1h,                   # Strong 1h downtrend
        ema_21_4h < indicators["4h"]["ema_50"],  # 4h downtrend confirmation
        adx_1h > 25,                             # Strong trend
        volume_ratio > 1.2,                      # Above average volume
        stoch_k_15m > 20,                        # Stochastic not oversold
        bb_position > 0.2,                       # Not near lower BB
        volatility > min_volatility_threshold     # Sufficient volatility
    ]
    
    long_score = sum(long_conditions)
    sell_score = sum(sell_conditions)
    
    # Return the stronger signal
    if long_score >= signal_strength_threshold:
        return long_score, "LONG"
    elif sell_score >= signal_strength_threshold:
        return sell_score, "SELL"  # Changed from SHORT to SELL for spot trading
    else:
        return max(long_score, sell_score), None

def can_trade():
    """Check if we can trade based on time and daily limits"""
    now = datetime.now()
    
    # Check daily trade limit
    if trading_state['daily_trades'] >= trading_state['max_daily_trades']:
        return False, "Daily trade limit reached"
    
    # Check time gap between trades
    if trading_state['last_trade_time']:
        time_diff = now - trading_state['last_trade_time']
        if time_diff < timedelta(hours=min_trade_gap_hours):
            remaining = min_trade_gap_hours - time_diff.total_seconds() / 3600
            return False, f"Wait {remaining:.1f} hours before next trade"
    
    # Check consecutive losses (cooling off period)
    if trading_state['consecutive_losses'] >= trading_state['max_consecutive_losses']:
        return False, "Too many consecutive losses - cooling off"
    
    return True, "Can trade"

def place_optimized_order(direction, current_price, indicators, account_balance):
    """Place order using full account balance or sell existing SOL"""
    
    atr_15m = indicators["15m"]["atr"]
    
    if direction == "LONG":
        # BUY SOL with USDT
        stop_loss = current_price - (1.8 * atr_15m)
        take_profit = current_price + (min_reward_ratio * 1.8 * atr_15m)
        side = "Buy"
        limit_price = current_price * 0.9998  # 0.02% below market
        
        # Use full USDT balance
        usable_balance = account_balance * 0.999  # Leave 0.1% for fees
        position_size = usable_balance / limit_price  # Convert USDT to SOL quantity
        
        print(f"\n🎯 Placing {direction} order using FULL USDT BALANCE...")
        print(f"💰 Available USDT: ${account_balance:.2f}")
        print(f"💰 Position Size: {position_size:.2f} SOL")
        
    elif direction == "SELL":
        # SELL existing SOL for USDT
        stop_loss = current_price + (1.8 * atr_15m)  # Stop loss above current price
        take_profit = current_price - (min_reward_ratio * 1.8 * atr_15m)  # Take profit below
        side = "Sell"
        limit_price = current_price * 1.0002  # 0.02% above market
        
        # Check if we have SOL to sell
        sol_balance = get_sol_balance()
        if sol_balance < 0.01:
            print("❌ No SOL available to sell")
            return False
            
        position_size = sol_balance * 0.999  # Sell 99.9% of SOL (leave tiny amount for fees)
        
        print(f"\n🎯 Placing {direction} order using FULL SOL BALANCE...")
        print(f"💰 Available SOL: {sol_balance:.4f}")
        print(f"💰 Position Size: {position_size:.2f} SOL")
    
    # Ensure minimum order size
    min_position = 0.01
    if position_size < min_position:
        print(f"❌ Position size too small: {position_size:.4f} SOL")
        return False
    
    try:
        # Place limit order to save on fees
        # Fix decimal precision for Bybit SOLUSDT spot
        order_params = {
            "category": "spot", 
            "symbol": symbol,
            "side": side,
            "orderType": "Limit",
            "qty": str(round(position_size, 4)),      # SOL quantity: 4 decimals max
            "price": str(round(limit_price, 2)),      # USDT price: 2 decimals
            "timeInForce": "GTC"
        }
        
        print(f"\n🎯 Placing {direction} order using FULL BALANCE...")
        print(f"💰 Available Balance: ${account_balance:.2f} USDT")
        print(f"💰 Position Size: {position_size:.2f} SOL (FULL BALANCE)")
        print(f"🎯 Limit Price: ${limit_price:.2f}")
        print(f"🛑 Stop Loss: ${stop_loss:.2f}")
        print(f"💎 Take Profit: ${take_profit:.2f}")
        print(f"💵 Total Investment: ${usable_balance:.2f} USDT")
        print(f"📋 Order params: {order_params}")
        
        result = session.place_order(**order_params)
        
        if result.get("retCode") == 0:
            print(f"✅ {direction} LIMIT order placed successfully!")
            
            if direction == "LONG":
                usable_balance = account_balance * 0.999
                potential_profit = (take_profit - limit_price) * position_size
                print(f"🎉 FULL USDT BALANCE DEPLOYED: ${usable_balance:.2f}")
                print(f"💰 SOL Acquired: {position_size:.2f} SOL")
            else:  # SELL
                expected_usdt = position_size * limit_price
                potential_profit = (limit_price - take_profit) * position_size
                print(f"🎉 FULL SOL BALANCE DEPLOYED: {position_size:.2f} SOL")
                print(f"💰 Expected USDT: ${expected_usdt:.2f}")
            
            print(f"🎯 Entry Price: ${limit_price:.2f}")
            print(f"🛑 Stop Loss: ${stop_loss:.2f}")
            print(f"💎 Take Profit: ${take_profit:.2f}")
            print(f"📊 Risk/Reward: 1:{min_reward_ratio}")
            print(f"🎯 Potential Profit: ${potential_profit:.2f} USDT")
            
            # Update trading state
            trading_state['last_trade_time'] = datetime.now()
            trading_state['daily_trades'] += 1
            trading_state['total_trades'] += 1
            trading_state['current_position'] = {
                'direction': direction,
                'size': position_size,
                'entry': limit_price,
                'stop': stop_loss,
                'target': take_profit,
                'order_id': result.get('result', {}).get('orderId'),
                'investment': usable_balance if direction == "LONG" else expected_usdt
            }
            
            return True
        else:
            print(f"❌ Order failed: {result.get('retMsg')}")
            return False
            
    except Exception as e:
        print(f"❌ Error placing order: {e}")
        return False

def check_position_exit(current_price):
    """Check if current position should be exited"""
    if not trading_state['current_position']:
        return
    
    pos = trading_state['current_position']
    direction = pos['direction']
    entry_price = pos['entry']
    position_size = pos['size']
    investment = pos.get('investment', entry_price * position_size)
    
    # Calculate current P&L in both percentage and USDT
    if direction == "LONG":
        # Long: bought SOL at entry_price, current value at current_price
        pnl_pct = (current_price - entry_price) / entry_price * 100
        pnl_usdt = (current_price - entry_price) * position_size
        current_value = current_price * position_size
    else:  # SELL
        # Sell: sold SOL at entry_price, would need to buy back at current_price
        pnl_pct = (entry_price - current_price) / entry_price * 100  # Profit if price dropped
        pnl_usdt = (entry_price - current_price) * position_size
        current_value = current_price * position_size  # Cost to buy back
    
    print(f"📊 Current Position: {direction}")
    print(f"💰 P&L: {pnl_pct:+.2f}% | ${pnl_usdt:+.2f} USDT")
    if direction == "LONG":
        print(f"💵 Investment: ${investment:.2f} → Current Value: ${current_value:.2f}")
    else:
        print(f"💵 Sold for: ${investment:.2f} → Cost to buy back: ${current_value:.2f}")
    
    # Check stop loss
    if direction == "LONG" and current_price <= pos['stop']:
        print(f"🛑 STOP LOSS triggered at ${current_price:.2f}")
        close_position("LOSS")
        
    elif direction == "SELL" and current_price >= pos['stop']:
        print(f"🛑 STOP LOSS triggered at ${current_price:.2f} (price went UP)")
        close_position("LOSS")
    
    # Check take profit
    elif direction == "LONG" and current_price >= pos['target']:
        print(f"🎯 TAKE PROFIT hit at ${current_price:.2f}")
        close_position("WIN")
        
    elif direction == "SELL" and current_price <= pos['target']:
        print(f"🎯 TAKE PROFIT hit at ${current_price:.2f} (price went DOWN)")
        close_position("WIN")

def close_position(result_type):
    """Close current position and update statistics"""
    if not trading_state['current_position']:
        return
        
    pos = trading_state['current_position']
    direction = pos['direction']
    position_size = pos['size']
    
    # Determine the closing side
    if direction == "LONG":
        side = "Sell"  # Close long position by selling SOL
    else:  # direction == "SELL" 
        side = "Buy"   # Close sell position by buying back SOL
    
    try:
        # Close position with market order
        order_result = session.place_order(
            category="spot",
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=str(round(position_size, 2)),  # Use 2 decimal precision
            timeInForce="GTC"
        )
        
        if order_result.get("retCode") == 0:
            print("✅ Position closed successfully")
            
            # Calculate final P&L if we can get current price
            try:
                response = requests.get(f"https://api.bybit.com/v5/market/kline", params={
                    "category": "spot", "symbol": symbol, "interval": "1", "limit": 1
                })
                if response.status_code == 200:
                    data = response.json()
                    if data["retCode"] == 0:
                        current_price = float(data["result"]["list"][0][4])
                        entry_price = pos['entry']
                        investment = pos.get('investment', entry_price * position_size)
                        
                        if direction == "LONG":
                            # Long: bought SOL, now selling
                            final_value = current_price * position_size
                            pnl_usdt = final_value - investment
                        else:  # SELL
                            # Sell: sold SOL, now buying back
                            cost_to_buyback = current_price * position_size
                            pnl_usdt = investment - cost_to_buyback
                        
                        print(f"💰 Final P&L: ${pnl_usdt:+.2f} USDT")
                        print(f"📊 Investment: ${investment:.2f} → Final Value: ${final_value if direction == 'LONG' else cost_to_buyback:.2f}")
            except:
                pass  # If P&L calculation fails, just continue
            
            # Update statistics
            if result_type == "WIN":
                trading_state['winning_trades'] += 1
                trading_state['consecutive_losses'] = 0
                print("🎉 Trade Result: WIN")
            else:
                trading_state['consecutive_losses'] += 1
                print("😔 Trade Result: LOSS")
            
            # Calculate win rate
            if trading_state['total_trades'] > 0:
                win_rate = trading_state['winning_trades'] / trading_state['total_trades'] * 100
                print(f"📈 Overall Win Rate: {win_rate:.1f}%")
            
            trading_state['current_position'] = None
            
        else:
            print(f"❌ Failed to close position: {order_result.get('retMsg')}")
            
    except Exception as e:
        print(f"❌ Error closing position: {e}")

def profit_optimized_strategy():
    """Main trading strategy optimized for profitability"""
    
    print(f"\n{'='*60}")
    print(f"🔄 Analysis Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Check if we can trade
    can_trade_now, reason = can_trade()
    if not can_trade_now:
        print(f"⏸️  Trading Status: {reason}")
        
        # Still check position exit even if we can't trade
        if trading_state['current_position']:
            data = fetch_multi_timeframe_data()
            if data:
                current_price = data[primary_timeframe]['close'][-1]
                check_position_exit(current_price)
        return
    
    # Get multi-timeframe data
    print("📊 Fetching market data...")
    data = fetch_multi_timeframe_data()
    if not data:
        print("❌ Failed to fetch market data - skipping cycle")
        return
    
    # Calculate market strength
    print("🧮 Calculating technical indicators...")
    indicators, current_price, volatility = calculate_market_strength(data)
    
    # Check exit conditions for existing position first
    check_position_exit(current_price)
    
    # Skip new trades if already in position
    if trading_state['current_position']:
        print(f"📍 Currently in {trading_state['current_position']['direction']} position")
        return
    
    # Calculate signal strength
    signal_strength, direction = calculate_signal_strength(indicators, current_price, volatility)
    
    # Display comprehensive market analysis
    print(f"\n📊 {symbol} Market Analysis:")
    print(f"💰 Current Price: ${current_price:.4f}")
    print(f"📈 RSI (15m/1h/4h): {indicators['15m']['rsi']:.1f}/{indicators['1h']['rsi']:.1f}/{indicators['4h']['rsi']:.1f}")
    print(f"🌊 24h Volatility: {volatility:.2%}")
    print(f"📊 MACD (15m): {indicators['15m']['macd']:.6f} | Signal: {indicators['15m']['macd_signal']:.6f}")
    print(f"📏 EMA21 vs EMA50 (1h): ${indicators['1h']['ema_21']:.2f} vs ${indicators['1h']['ema_50']:.2f}")
    print(f"💪 ADX Trend Strength: {indicators['1h']['adx']:.1f}")
    print(f"📈 Signal Strength: {signal_strength}/13")
    print(f"📅 Daily Trades Used: {trading_state['daily_trades']}/{trading_state['max_daily_trades']}")
    
    # Only trade on strong signals
    if signal_strength >= signal_strength_threshold and direction:
        account_balance = get_account_balance()
        print(f"\n🚀 HIGH QUALITY {direction} SIGNAL DETECTED!")
        print(f"⭐ Signal Strength: {signal_strength}/13 confirmations")
        place_optimized_order(direction, current_price, indicators, account_balance)
    else:
        print(f"\n⏳ Signal strength: {signal_strength}/13 - Need {signal_strength_threshold}+ for trade")
        print("🎯 Waiting for higher quality setup...")

def run_optimized_bot():
    """Run the profit-optimized trading bot"""
    print("🏆" + "="*60)
    print("🏆 FULL-BALANCE SOLANA TRADING BOT")
    print("🏆" + "="*60)
    print(f"💎 Trading Pair: {symbol}")
    print(f"⚡ Strategy: Quality Signals + Full Balance")
    print(f"💰 Max Daily Trades: {trading_state['max_daily_trades']}")
    print(f"⏰ Min Gap Between Trades: {min_trade_gap_hours} hours")
    print(f"📊 Min Signal Strength: {signal_strength_threshold}/13")
    print(f"🎯 Min Risk/Reward Ratio: {min_reward_ratio}:1")
    print(f"💵 Position Sizing: FULL BALANCE (All-In)")
    print(f"📈 Analysis Frequency: Every 20 minutes")
    print("="*60)
    
    cycle_count = 0
    start_time = datetime.now()
    
    while True:
        try:
            cycle_count += 1
            
            profit_optimized_strategy()
            
            # Reset daily counters at midnight
            current_hour = datetime.now().hour
            if current_hour == 0 and datetime.now().minute < 20:
                trading_state['daily_trades'] = 0
                trading_state['daily_pnl'] = 0
                print("🌅 Daily trading counters reset")
            
            # Display session statistics
            runtime = datetime.now() - start_time
            print(f"\n📊 Session Stats:")
            print(f"⏰ Runtime: {runtime}")
            print(f"🔄 Cycles Completed: {cycle_count}")
            print(f"💼 Total Trades: {trading_state['total_trades']}")
            if trading_state['total_trades'] > 0:
                win_rate = trading_state['winning_trades'] / trading_state['total_trades'] * 100
                print(f"🎯 Win Rate: {win_rate:.1f}%")
            
            print(f"\n💤 Sleeping for 20 minutes... Next cycle at {(datetime.now() + timedelta(minutes=20)).strftime('%H:%M:%S')}")
            print("="*60)
            
            time.sleep(1200)  # 20 minutes
            
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
            print(f"📊 Final Session Stats:")
            print(f"🔄 Total Cycles: {cycle_count}")
            print(f"💼 Total Trades: {trading_state['total_trades']}")
            if trading_state['total_trades'] > 0:
                win_rate = trading_state['winning_trades'] / trading_state['total_trades'] * 100
                print(f"🎯 Final Win Rate: {win_rate:.1f}%")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("⏰ Waiting 5 minutes before retry...")
            time.sleep(300)

if __name__ == "__main__":
    run_optimized_bot()