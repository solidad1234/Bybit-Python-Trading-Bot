import requests
import time
import talib
import numpy as np
from pybit.unified_trading import HTTP

# Bybit API keys
API_KEY = input("Enter your api key: ")
API_SECRET = input("Enter your api secret: ")

# Initialize session
session = HTTP(
    testnet=True,  # Set to False for live trading
    api_key=API_KEY,
    api_secret=API_SECRET,
)

# Trading parameters
SYMBOL = "SOLUSDT"
TIMEFRAME = "15"  # 15-minute candles
RSI_PERIOD = 14
EMA_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
QTY = 0.00011  # Quantity to trade
LEVERAGE = "25"  # Leverage multiplier

# Set leverage for the symbol
session.set_leverage(category="linear", symbol=SYMBOL, buyLeverage=LEVERAGE, sellLeverage=LEVERAGE)

# Fetch candle data
def fetch_candles(symbol, interval, limit=200):
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data["retCode"] == 0:
            return [float(candle[4]) for candle in data["result"]["list"]]  # Close prices
    print("Error fetching candles:", response.json())
    return None

# Fetch order book data
def fetch_order_book(symbol):
    url = f"https://api.bybit.com/v5/market/orderbook"
    params = {"category": "linear", "symbol": symbol}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data["retCode"] == 0:
            return data["result"]
    print("Error fetching order book:", response.json())
    return None

# Calculate indicators
def calculate_indicators(closing_prices):
    rsi = talib.RSI(np.array(closing_prices), timeperiod=RSI_PERIOD)
    ema = talib.EMA(np.array(closing_prices), timeperiod=EMA_PERIOD)
    macd, macd_signal, macd_hist = talib.MACD(
        np.array(closing_prices),
        fastperiod=MACD_FAST,
        slowperiod=MACD_SLOW,
        signalperiod=MACD_SIGNAL,
    )
    return rsi, ema, macd, macd_signal

# Place order with stop-loss and take-profit
def place_order_with_risk_management(side, qty, stop_loss, take_profit):
    order_params = {
        "category": "linear",
        "symbol": SYMBOL,
        "side": side.capitalize(),
        "orderType": "Market",
        "qty": str(qty),
        "timeInForce": "GoodTillCancel",
        "marketUnit": "basecoin"
    }
    print(f"Placing {side} order for {qty} {SYMBOL}...")
    response = session.place_order(**order_params)
    
    if response.get("retCode") == 0:
        print(f"{side.upper()} order placed successfully!")
        position_side = "Buy" if side.lower() == "buy" else "Sell"

        # Set stop-loss and take-profit
        session.set_trading_stop(
            category="linear",
            symbol=SYMBOL,
            side=position_side,
            stopLoss=str(stop_loss),
            takeProfit=str(take_profit),
        )
        print(f"Stop-loss set at {stop_loss}, take-profit set at {take_profit}.")
    else:
        print("Order failed:", response.get("retMsg", "Unknown error"))

# Analyze market and trade
def analyze_and_trade():
    closing_prices = fetch_candles(SYMBOL, TIMEFRAME)
    if not closing_prices or len(closing_prices) < RSI_PERIOD:
        print("Not enough data to analyze.")
        return
    
    rsi, ema, macd, macd_signal = calculate_indicators(closing_prices)
    latest_price = closing_prices[-1]
    latest_rsi = rsi[-1]
    latest_ema = ema[-1]
    latest_macd = macd[-1]
    latest_signal = macd_signal[-1]

    # Fetch order book for additional confirmation
    order_book = fetch_order_book(SYMBOL)
    if order_book:
        best_bid = float(order_book["b"][0][0])  # Highest bid price
        best_ask = float(order_book["a"][0][0])  # Lowest ask price
        print(f"Order Book - Best Bid: {best_bid}, Best Ask: {best_ask}")

    print(f"Latest Price: {latest_price}, RSI: {latest_rsi:.2f}, EMA: {latest_ema:.2f}, MACD: {latest_macd:.2f}, Signal: {latest_signal:.2f}")
    
    # Trading logic
    if latest_rsi < 30 and latest_macd > latest_signal and latest_price > latest_ema:
        print("Buy signal detected.")
        stop_loss = latest_price * 0.98  # Example: 2% below current price
        take_profit = latest_price * 1.05  # Example: 5% above current price
        place_order_with_risk_management("buy", QTY, stop_loss, take_profit)

    elif latest_rsi > 70 and latest_macd < latest_signal and latest_price < latest_ema:
        print("Sell signal detected.")
        stop_loss = latest_price * 1.02  # Example: 2% above current price
        take_profit = latest_price * 0.95  # Example: 5% below current price
        place_order_with_risk_management("sell", QTY, stop_loss, take_profit)

    else:
        print("No clear trading signal.\n")

# Main loop
while True:
    print(f"Analyzing market for {SYMBOL}...")
    analyze_and_trade()
    time.sleep(900)  # Run every 15 minutes
