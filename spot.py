"""
Standalone Spot Trading Bot — V2 Engine  (spot.py)
=================================================
Multi-Asset Spot Execution Engine powered by the 6-Factor Consensus Model
(Regime, Derivatives, TA, Support & Resistance, Sentiment, and Google News RSS).

Key Features:
  • Multi-Asset Universe Scanning: SOLUSDT, ETHUSDT, AVAXUSDT, LINKUSDT, BNBUSDT.
  • 6-Factor MultiFactorAggregator: Evaluates consensus before spot accumulation or exit.
  • Support & Resistance Anchoring: Buys at structural support, takes profit at resistance.
  • Symbol-Aware Contract Specs: Enforces min_qty, step_size, and decimal precision per asset.
  • SQLite State Persistence: Tracks spot_position & logs factor context to spot_trade_log.
  • Zero Leverage: No liquidation risk. Capital preservation focused.
"""

import requests
import hmac
import hashlib
import time
import numpy as np
import sqlite3
import json
import os
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

try:
    import talib
except ImportError:
    talib = None

from factors.aggregator import MultiFactorAggregator
from factors.regime import get_regime_score
from factors.derivatives import get_derivatives_score
from factors.news import get_news_score
from factors.sentiment import get_sentiment_score

load_dotenv()
api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")

# Trading Configuration
TRADE_SYMBOLS = ["SOLUSDT", "ETHUSDT", "AVAXUSDT", "LINKUSDT", "BNBUSDT"]

# Per-symbol Bybit Spot Contract Specs
SPOT_CONTRACT_SPECS = {
    "SOLUSDT":  {"min_qty": 0.1,  "step_size": 0.1,  "ticker": "SOL",  "base": "SOL",  "qty_precision": 1, "price_precision": 2},
    "ETHUSDT":  {"min_qty": 0.01, "step_size": 0.01, "ticker": "ETH",  "base": "ETH",  "qty_precision": 2, "price_precision": 2},
    "AVAXUSDT": {"min_qty": 0.1,  "step_size": 0.1,  "ticker": "AVAX", "base": "AVAX", "qty_precision": 1, "price_precision": 2},
    "LINKUSDT": {"min_qty": 0.1,  "step_size": 0.1,  "ticker": "LINK", "base": "LINK", "qty_precision": 1, "price_precision": 3},
    "BNBUSDT":  {"min_qty": 0.01, "step_size": 0.01, "ticker": "BNB",  "base": "BNB",  "qty_precision": 2, "price_precision": 2},
}

primary_timeframe = "15"   # 15m
higher_timeframe  = "60"    # 1h

# Spot Risk & Strategy Limits
min_reward_ratio = 2.0
max_daily_spot_trades = 5
min_trade_gap_hours = 1

session = HTTP(
    testnet=False,
    api_key=api_key,
    api_secret=api_secret,
    recv_window=15000,
)

# Global State Dictionary
spot_state = {
    'last_trade_time': None,
    'position': None,
    'daily_trades': 0,
    'total_trades': 0,
    'winning_trades': 0,
    'consecutive_losses': 0,
    'max_consecutive_losses': 3,
    'session_start': datetime.now(),
}


class SpotTradingBot:
    """V2 Multi-Asset Spot Trading Engine"""

    def __init__(self, allocation_pct: float = 1.0):
        """
        Parameters
        ----------
        allocation_pct : float
            Fraction of total USDT balance to use for spot trading (1.0 for standalone, 0.70 for hybrid).
        """
        self.allocation_pct = max(0.1, min(1.0, allocation_pct))
        self.aggregator = MultiFactorAggregator()
        print(f"🚀 Initializing Spot Trading Bot (V2 Engine)...")
        print(f"💰 Allocation: {self.allocation_pct * 100:.0f}% of USDT Balance")
        self.init_db()
        self.load_position_state()

    def init_db(self):
        """Initialize SQLite database for spot position and trade log persistence."""
        self.conn = sqlite3.connect('trading_state.db', timeout=30.0, check_same_thread=False)
        self.cursor = self.conn.cursor()
        try:
            self.cursor.execute('PRAGMA journal_mode=WAL;')
            self.cursor.execute('PRAGMA busy_timeout=30000;')
        except Exception as e:
            print(f"⚠️ Could not set WAL/busy_timeout PRAGMA: {e}")

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS spot_position (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                direction TEXT,
                size REAL,
                entry REAL,
                stop REAL,
                target REAL,
                investment REAL,
                order_id TEXT,
                timestamp TEXT,
                factor_context TEXT
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS spot_trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                pnl_usdt REAL,
                pnl_pct REAL,
                hold_hours REAL,
                exit_reason TEXT,
                factor_scores TEXT,
                sr_score REAL,
                sr_scenario TEXT
            )
        ''')
        self.conn.commit()

    def save_position_state(self):
        """Persist open spot position state to SQLite."""
        try:
            self.cursor.execute('DELETE FROM spot_position')
            pos = spot_state['position']
            if pos:
                fc_str = json.dumps(pos.get('factor_context', {}))
                self.cursor.execute('''
                    INSERT INTO spot_position (
                        id, symbol, direction, size, entry, stop, target,
                        investment, order_id, timestamp, factor_context
                    ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pos['symbol'], pos['direction'], pos['size'], pos['entry'],
                    pos['stop'], pos['target'], pos.get('investment', 0.0),
                    pos.get('order_id', ''), str(pos['timestamp']), fc_str
                ))
            self.conn.commit()
        except Exception as e:
            print(f"❌ Error saving spot position state: {e}")

    def load_position_state(self):
        """Load open spot position from SQLite."""
        try:
            self.cursor.execute('SELECT symbol, direction, size, entry, stop, target, investment, order_id, timestamp, factor_context FROM spot_position WHERE id=1')
            row = self.cursor.fetchone()
            if row:
                fc = json.loads(row[9]) if row[9] else {}
                spot_state['position'] = {
                    'symbol': row[0],
                    'direction': row[1],
                    'size': row[2],
                    'entry': row[3],
                    'stop': row[4],
                    'target': row[5],
                    'investment': row[6],
                    'order_id': row[7],
                    'timestamp': datetime.fromisoformat(row[8]),
                    'factor_context': fc,
                }
                print(f"📦 Restored open spot position: {row[1]} {row[2]} {row[0]} @ ${row[3]:.4f}")
            else:
                spot_state['position'] = None
        except Exception as e:
            print(f"⚠️ Error loading spot position state: {e}")
            spot_state['position'] = None

    def log_trade_outcome(self, exit_price: float, exit_reason: str):
        """Log closed spot trade to SQLite database."""
        pos = spot_state['position']
        if not pos:
            return
        try:
            entry_price = pos['entry']
            size = pos['size']
            direction = pos['direction']
            investment = pos.get('investment', entry_price * size)

            if direction == "LONG":
                final_value = exit_price * size
                pnl_usdt = final_value - investment
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_usdt = 0.0
                pnl_pct = 0.0

            hold_hours = (datetime.now() - pos['timestamp']).total_seconds() / 3600.0
            fc = pos.get('factor_context', {})

            sr_scen = fc.get('sr_scenario', 'MID_RANGE')
            sr_sc   = fc.get('sr_score', 0.0)

            self.cursor.execute('''
                INSERT INTO spot_trade_log (
                    timestamp, symbol, direction, entry_price, exit_price,
                    pnl_usdt, pnl_pct, hold_hours, exit_reason, factor_scores,
                    sr_score, sr_scenario
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(), pos['symbol'], direction,
                entry_price, exit_price, pnl_usdt, pnl_pct, round(hold_hours, 2),
                exit_reason, json.dumps(fc), sr_sc, sr_scen
            ))
            self.conn.commit()

            if pnl_usdt >= 0:
                spot_state['winning_trades'] += 1
                spot_state['consecutive_losses'] = 0
                print(f"🎉 Spot Trade WIN: +${pnl_usdt:.2f} ({pnl_pct:+.2f}%) [{exit_reason}]")
            else:
                spot_state['consecutive_losses'] += 1
                print(f"😔 Spot Trade LOSS: -${abs(pnl_usdt):.2f} ({pnl_pct:+.2f}%) [{exit_reason}]")

            spot_state['position'] = None
            self.save_position_state()
        except Exception as e:
            print(f"❌ Error logging spot trade outcome: {e}")

    def get_usdt_balance(self) -> float:
        """Get available USDT balance for spot allocation."""
        for acct in ["UNIFIED", "SPOT"]:
            try:
                res = session.get_wallet_balance(accountType=acct)
                if res.get("retCode") == 0:
                    acct_list = res.get("result", {}).get("list", [])
                    if acct_list:
                        coins = acct_list[0].get("coin", [])
                        for coin in coins:
                            if coin["coin"] == "USDT":
                                bal = float(coin.get("availableToWithdraw", coin.get("walletBalance", 0)))
                                if bal > 0:
                                    return bal
            except Exception:
                continue
        return 0.0

    def get_coin_balance(self, coin_ticker: str) -> float:
        """Get available coin balance (e.g. SOL, ETH, AVAX, LINK, BNB)."""
        for acct in ["UNIFIED", "SPOT"]:
            try:
                res = session.get_wallet_balance(accountType=acct)
                if res.get("retCode") == 0:
                    acct_list = res.get("result", {}).get("list", [])
                    if acct_list:
                        coins = acct_list[0].get("coin", [])
                        for coin in coins:
                            if coin["coin"].upper() == coin_ticker.upper():
                                return float(coin.get("availableToWithdraw", coin.get("walletBalance", 0)))
            except Exception:
                continue
        return 0.0

    def fetch_multi_timeframe_data(self, symbol: str) -> dict:
        """Fetch spot OHLCV klines for 15m, 1h, and 4h timeframes."""
        data = {}
        for tf in [primary_timeframe, higher_timeframe, "240"]:
            url = "https://api.bybit.com/v5/market/kline"
            params = {"category": "spot", "symbol": symbol, "interval": tf, "limit": 100}
            try:
                resp = requests.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    res = resp.json()
                    if res.get("retCode") == 0:
                        candles = list(reversed(res["result"]["list"]))
                        data[tf] = {
                            'close': np.array([float(c[4]) for c in candles]),
                            'high':  np.array([float(c[2]) for c in candles]),
                            'low':   np.array([float(c[3]) for c in candles]),
                            'volume':np.array([float(c[5]) for c in candles]),
                            'timestamp': [int(c[0]) for c in candles]
                        }
            except Exception as e:
                print(f"❌ Error fetching spot {tf} data for {symbol}: {e}")
                return None
            time.sleep(0.05)
        return data if len(data) == 3 else None

    def calculate_indicators(self, data: dict) -> tuple:
        """Calculate technical indicators across timeframes."""
        tf_15m = data[primary_timeframe]
        tf_1h  = data[higher_timeframe]
        tf_4h  = data["240"]

        indicators = {}
        for tf, prices in [("15m", tf_15m), ("1h", tf_1h), ("4h", tf_4h)]:
            closes  = prices['close']
            highs   = prices['high']
            lows    = prices['low']
            volumes = prices['volume']

            if talib is not None:
                rsi         = float(talib.RSI(closes, timeperiod=14)[-1])
                macd, macd_sig, macd_h = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
                ema_21      = float(talib.EMA(closes, timeperiod=21)[-1])
                ema_50      = float(talib.EMA(closes, timeperiod=50)[-1])
                atr         = float(talib.ATR(highs, lows, closes, timeperiod=14)[-1])
                vol_sma     = float(talib.SMA(volumes, timeperiod=20)[-1])
                bb_u, bb_m, bb_l = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
                adx         = float(talib.ADX(highs, lows, closes, timeperiod=14)[-1])
                stoch_k, _  = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowd_period=3)
            else:
                # Basic fallback
                rsi, macd, macd_sig, macd_h = 50.0, 0.0, 0.0, 0.0
                ema_21, ema_50 = float(closes[-1]), float(closes[-1])
                atr = float(closes[-1]) * 0.01
                vol_sma = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else 1.0
                bb_u, bb_m, bb_l = float(closes[-1]*1.02), float(closes[-1]), float(closes[-1]*0.98)
                adx, stoch_k = 20.0, 50.0

            indicators[tf] = {
                'rsi': rsi, 'macd': float(macd[-1] if isinstance(macd, np.ndarray) else macd),
                'macd_signal': float(macd_sig[-1] if isinstance(macd_sig, np.ndarray) else macd_sig),
                'macd_histogram': float(macd_h[-1] if isinstance(macd_h, np.ndarray) else macd_h),
                'ema_21': ema_21, 'ema_50': ema_50, 'atr': atr,
                'volume_sma': vol_sma, 'current_volume': float(volumes[-1]),
                'bb_upper': float(bb_u[-1] if isinstance(bb_u, np.ndarray) else bb_u),
                'bb_lower': float(bb_l[-1] if isinstance(bb_l, np.ndarray) else bb_l),
                'adx': adx, 'stoch_k': float(stoch_k[-1] if isinstance(stoch_k, np.ndarray) else stoch_k),
                'volume_ratio': float(volumes[-1] / vol_sma) if vol_sma > 0 else 1.0
            }

        current_price = float(tf_15m['close'][-1])
        volatility = abs(float((current_price - tf_1h['close'][-24]) / tf_1h['close'][-24])) if len(tf_1h['close']) >= 24 else 0.02

        return indicators, current_price, volatility

    def check_btc_correlation(self) -> dict:
        """Check Bitcoin trend correlation for spot safety."""
        try:
            url = "https://api.bybit.com/v5/market/kline"
            params = {"category": "spot", "symbol": "BTCUSDT", "interval": "60", "limit": 10}
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                res = resp.json()
                if res.get("retCode") == 0:
                    candles = list(reversed(res["result"]["list"]))
                    closes = [float(c[4]) for c in candles]
                    cur = closes[-1]
                    c_1h = closes[-2] if len(closes) > 1 else cur
                    c_4h = closes[-5] if len(closes) > 4 else cur
                    chg_1h = (cur - c_1h) / c_1h * 100
                    chg_4h = (cur - c_4h) / c_4h * 100
                    return {
                        'bullish': chg_1h > -1.0 and chg_4h > -2.0,
                        'bearish': chg_1h < -2.0 or chg_4h < -5.0,
                        '1h_change': chg_1h, '4h_change': chg_4h
                    }
        except Exception as e:
            print(f"❌ Spot BTC correlation check failed: {e}")
        return {'bullish': True, 'bearish': False, '1h_change': 0, '4h_change': 0}

    def calculate_spot_signals(self, indicators: dict, current_price: float, volatility: float) -> dict:
        """Calculate technical TA score for spot."""
        long_conds = [
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
            volatility > 0.015,
        ]
        long_score = sum(long_conds)
        if long_score >= 5:
            return {"signal": "LONG", "strength": long_score}
        return {"signal": None, "strength": long_score}

    def place_spot_buy_order(self, symbol: str, current_price: float, factor_context: dict) -> bool:
        """Place Spot BUY order using available USDT balance."""
        usdt_total = self.get_usdt_balance()
        if usdt_total < 5:
            print(f"❌ Insufficient USDT for Spot BUY: ${usdt_total:.2f}")
            return False

        usable_usdt = usdt_total * self.allocation_pct * 0.99  # 1% for fees & slippage
        specs = SPOT_CONTRACT_SPECS.get(symbol, SPOT_CONTRACT_SPECS["SOLUSDT"])

        qty = usable_usdt / current_price
        min_qty = specs["min_qty"]
        step = specs["step_size"]
        ticker = specs["ticker"]

        if qty < min_qty:
            print(f"❌ Spot BUY size ({qty:.4f} {ticker}) below min_qty ({min_qty})")
            return False

        qty = round(qty / step) * step
        qty_prec = specs["qty_precision"]
        price_prec = specs["price_precision"]

        # Stops/Targets: Anchored to S/R if available, else ATR
        sr_stop = factor_context.get("sr_stop")
        sr_target = factor_context.get("sr_target")
        atr_15m = factor_context.get("indicators", {}).get("15m", {}).get("atr", current_price * 0.01)

        stop_loss = sr_stop if sr_stop else current_price - (1.8 * atr_15m)
        take_profit = sr_target if sr_target else current_price + (2.5 * 1.8 * atr_15m)

        try:
            order_params = {
                "category": "spot",
                "symbol": symbol,
                "side": "Buy",
                "orderType": "Market",
                "qty": f"{qty:.{qty_prec}f}",
            }
            print(f"\n🚀 SPOT BUY Order ({symbol}):")
            print(f"   💰 Capital: ${usable_usdt:.2f} USDT")
            print(f"   📊 Position: {qty} {ticker}")
            print(f"   🎯 Entry: ${current_price:.{price_prec}f}")
            print(f"   🛡️ Stop: ${stop_loss:.{price_prec}f}")
            print(f"   💎 Target: ${take_profit:.{price_prec}f}")

            res = session.place_order(**order_params)
            if res.get("retCode") == 0:
                print(f"✅ Spot BUY order placed successfully!")
                spot_state['position'] = {
                    'symbol': symbol,
                    'direction': "LONG",
                    'size': qty,
                    'entry': current_price,
                    'stop': stop_loss,
                    'target': take_profit,
                    'investment': usable_usdt,
                    'order_id': res.get('result', {}).get('orderId'),
                    'timestamp': datetime.now(),
                    'factor_context': factor_context,
                }
                self.save_position_state()
                spot_state['daily_trades'] += 1
                spot_state['total_trades'] += 1
                spot_state['last_trade_time'] = datetime.now()
                return True
            else:
                print(f"❌ Spot BUY order failed: {res.get('retMsg')}")
                return False
        except Exception as e:
            print(f"❌ Error executing spot BUY order: {e}")
            return False

    def place_spot_sell_order(self, symbol: str, current_price: float, exit_reason: str) -> bool:
        """Place Spot SELL order to exit coin position back into USDT."""
        pos = spot_state['position']
        if not pos:
            return False

        specs = SPOT_CONTRACT_SPECS.get(symbol, SPOT_CONTRACT_SPECS["SOLUSDT"])
        ticker = specs["ticker"]
        qty = pos['size']
        qty_prec = specs["qty_precision"]

        try:
            order_params = {
                "category": "spot",
                "symbol": symbol,
                "side": "Sell",
                "orderType": "Market",
                "qty": f"{qty:.{qty_prec}f}",
            }
            print(f"\n🎯 SPOT SELL Order ({symbol}) [{exit_reason}]:")
            print(f"   📊 Quantity: {qty} {ticker}")
            print(f"   🎯 Price: ${current_price:.4f}")

            res = session.place_order(**order_params)
            if res.get("retCode") == 0:
                print(f"✅ Spot SELL order executed successfully!")
                self.log_trade_outcome(exit_price=current_price, exit_reason=exit_reason)
                return True
            else:
                print(f"❌ Spot SELL order failed: {res.get('retMsg')}")
                return False
        except Exception as e:
            print(f"❌ Error executing spot SELL order: {e}")
            return False

    def check_position_exit(self, symbol: str, current_price: float, factor_context: dict = None):
        """Check stop loss, take profit, or factor reversal for open spot position."""
        pos = spot_state['position']
        if not pos or pos['symbol'] != symbol:
            return

        direction   = pos['direction']
        entry_price = pos['entry']
        stop_loss   = pos['stop']
        take_profit = pos['target']

        pnl_pct  = (current_price - entry_price) / entry_price * 100
        pnl_usdt = (current_price - entry_price) * pos['size']

        print(f"📊 Spot Position ({symbol}): {direction} | P&L: {pnl_pct:+.2f}% (${pnl_usdt:+.2f} USDT)")

        # 1. Stop loss hit
        if current_price <= stop_loss:
            print(f"🛑 STOP LOSS triggered for Spot {symbol} at ${current_price:.4f}")
            self.place_spot_sell_order(symbol, current_price, exit_reason="STOP_LOSS")
            return

        # 2. Take profit hit
        if current_price >= take_profit:
            print(f"🎯 TAKE PROFIT hit for Spot {symbol} at ${current_price:.4f}")
            self.place_spot_sell_order(symbol, current_price, exit_reason="TAKE_PROFIT")
            return

        # 3. Factor consensus reversal signal
        if factor_context:
            final_score = factor_context.get("final_score", 0.0)
            if final_score <= -0.35:
                print(f"⚠️ Bearish Consensus Shift (score={final_score:+.3f}) — Exiting Spot position to USDT")
                self.place_spot_sell_order(symbol, current_price, exit_reason="CONSENSUS_REVERSAL")
                return

    def run_cycle(self):
        """Execute one complete Spot analysis cycle across the universe."""
        print(f"\n================================================================================")
        print(f"🔄 SPOT Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"================================================================================")

        # Check existing position exit if open
        if spot_state['position']:
            sym = spot_state['position']['symbol']
            data = self.fetch_multi_timeframe_data(sym)
            if data:
                cur_price = data[primary_timeframe]['close'][-1]
                self.check_position_exit(sym, cur_price)

        # Skip new scans if already holding a spot position
        if spot_state['position']:
            print(f"📍 Spot position active: holding {spot_state['position']['size']} {spot_state['position']['symbol']}")
            return

        btc_corr = self.check_btc_correlation()

        for sym in TRADE_SYMBOLS:
            print(f"\n🔍 Scanning Spot Pair: {sym}")
            data = self.fetch_multi_timeframe_data(sym)
            if not data:
                continue

            indicators, cur_price, volatility = self.calculate_indicators(data)
            ta_sig = self.calculate_spot_signals(indicators, cur_price, volatility)

            regime_sig = get_regime_score()
            deriv_sig  = get_derivatives_score(sym)
            news_sig   = get_news_score(sym)
            sent_sig   = get_sentiment_score()

            precomputed = {"regime": regime_sig, "sentiment": sent_sig}
            consensus  = self.aggregator.evaluate(
                ta_signal=ta_sig, symbol=sym, current_price=cur_price,
                precomputed=precomputed, indicators=indicators, data=data
            )

            final_score = consensus["final_score"]
            news_block  = news_sig.get("block_long_only", False) or news_sig.get("block_trade", False)

            min_buy_score = 0.50 if regime_sig.get("score", 0.0) <= -0.4 else 0.35

            print(f"   Consensus Score: {final_score:+.3f} (min {min_buy_score}) | News Block: {news_block} | BTC Bullish: {btc_corr['bullish']}")

            if final_score >= min_buy_score and btc_corr['bullish'] and not news_block:
                print(f"🚀 HIGH CONVICTION SPOT BUY SIGNAL DETECTED FOR {sym}!")
                factor_ctx = {
                    'indicators': indicators,
                    'final_score': final_score,
                    'sr_scenario': consensus.get('sr_scenario', 'MID_RANGE'),
                    'sr_score': consensus.get('sr_score', 0.0),
                    'sr_stop': consensus.get('suggested_stop'),
                    'sr_target': consensus.get('suggested_target'),
                }
                success = self.place_spot_buy_order(sym, cur_price, factor_ctx)
                if success:
                    break  # Maximum 1 spot position at a time


def run_standalone_spot():
    bot = SpotTradingBot(allocation_pct=1.0)
    print("🚀 Starting Standalone Spot Bot Loop...")
    while True:
        try:
            bot.run_cycle()
            time.sleep(300)  # Scan every 5 minutes
        except KeyboardInterrupt:
            print("\n🛑 Spot Bot stopped by user.")
            break
        except Exception as e:
            print(f"❌ Error in Spot loop: {e}")
            time.sleep(60)


if __name__ == "__main__":
    run_standalone_spot()