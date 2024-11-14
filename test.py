from pybit.unified_trading import HTTP

# Initialize the Bybit session
api_key = ""
api_secret = ""
session = HTTP(
    testnet=True,  # Set to False for live trading
    api_key=api_key,
    api_secret=api_secret,
)

def get_symbol_info(symbol):
    try:
        # Fetch symbol information from public endpoint
        response = session.query_symbol(symbol=symbol)
        print("Symbol Info:", response)
        return response
    except Exception as e:
        print(f"Error fetching symbol info: {e}")
        return None

# Fetch info for BTCUSDT symbol
get_symbol_info("BTCUSDT")
