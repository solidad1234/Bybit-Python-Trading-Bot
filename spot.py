import requests
import hmac
import hashlib
import time
import talib
import numpy as np
from datetime import datetime
from pybit.unified_trading import HTTP

# API keys
api_key = ""
api_secret = ""

symbol = "SOLUSDT"
timeframe = "15"  # 15-minute candles 
rsi_period = 14
qty = 0.045


session = HTTP(
    testnet=False,  # False for the live environment
    api_key=api_key,
    api_secret=api_secret,
)
# Fetch recent candle data
def fetch_candle_data(symbol, interval):
    url = f"https://api.bybit.com/v5/market/kline"
    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": interval,
        "limit": 50  # Fetch fewer candles for efficiency
    }
    response = requests.get(url, params=params)

    # Retry for rate limiting
    if response.status_code == 429:
        retry_after = int(response.headers.get('Retry-After', 10))
        print(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
        time.sleep(retry_after)
        return fetch_candle_data(symbol, interval)

    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        return None

    data = response.json()
    if data["retCode"] != 0:
        print("Error:", data["retMsg"])
        return None

    close_prices = [float(candle[4]) for candle in data["result"]["list"]]
    return close_prices

# Calculate RSI
def calculate_rsi(closing_prices):
    return talib.RSI(np.array(closing_prices), timeperiod=rsi_period)

# Place  order 
def place_order(side, qty, price=None, order_type="MARKET", time_in_force="GTC"):
    order_params = {
        "category": "spot",
        "symbol": symbol,
        "side": side.capitalize(),  # e.g., "Buy" or "Sell"
        "orderType": order_type.capitalize(),  # e.g., "Market" or "Limit"
        "qty": str(qty),
        "timeInForce": time_in_force,
        "marketUnit": "basecoin"
    }

    # Add price for limit orders
    if order_type.upper() == "LIMIT" and price:
        order_params["price"] = str(price)

    print("Order Parameters:", order_params)
    print(f"Placing {order_type} order for {qty} {symbol}")

    try:
        result = session.place_order(**order_params)
    except Exception as e:
        print("Error placing order:", e)
        return None

    if result.get("retCode") == 0:
        print(f"{side.upper()} order placed successfully!")
    else:
        print(f"Failed to place {side} order:", result.get("retMsg", "Unknown error"))


# Monitor RSI and trade
def trade_with_rsi():
    closing_prices = fetch_candle_data(symbol, timeframe)
    if closing_prices is None or len(closing_prices) < rsi_period:
        print("Not enough data to calculate RSI.")
        return

    rsi = calculate_rsi(closing_prices)
    latest_rsi = rsi[-1]
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"Current RSI: {latest_rsi:.2f} at : {current_time}") 

    if latest_rsi < 45:
        place_order("buy", qty)
    elif latest_rsi > 65:
        place_order("sell", qty)

#combining all RSI, MA, EMA and MACD
def combined_trade_signal():
    closing_prices = fetch_candle_data(symbol, timeframe)
    
    # Calculate the indicators using TA-Lib
    rsi = talib.RSI(closing_prices, timeperiod=14)
    macd, signal, hist = talib.MACD(closing_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    ema = talib.EMA(closing_prices, timeperiod=14)
    sma = talib.SMA(closing_prices, timeperiod=14)
    
    # Get the most recent values for each indicator
    latest_rsi = rsi[-1]
    latest_macd = macd[-1]
    latest_signal = signal[-1]
    latest_ema = ema[-1]
    latest_sma = sma[-1]
    
    print(f"Current RSI: {latest_rsi:.2f}")
    print(f"Current MACD: {latest_macd:.2f} | Signal: {latest_signal:.2f}")
    print(f"Current EMA: {latest_ema:.2f} | SMA: {latest_sma:.2f}")
    
    if latest_rsi < 30 and latest_macd > latest_signal and closing_prices[-1] > latest_ema:
        print("Buy Signal")
        place_order("buy", qty) 

    elif latest_rsi > 70 and latest_macd < latest_signal and closing_prices[-1] < latest_ema:
        print("Sell Signal")
        place_order("sell", qty) 

    elif closing_prices[-1] > latest_ema and closing_prices[-1] > latest_sma:
        print("Trend Confirmation (Buy)")
        place_order("buy", qty)  

    elif closing_prices[-1] < latest_ema and closing_prices[-1] < latest_sma:
        print("Trend Confirmation (Sell)")
        place_order("sell", qty)  

    else:
        print("No Clear Signal") 


while True:
    print(f"Trading {symbol}")
    # combined_trade_signal()
    trade_with_rsi()
    time.sleep(900)  # Check every 15 minutes
