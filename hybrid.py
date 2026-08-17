"""
Hybrid Spot + Futures Trading Bot — V2 Orchestrator  (hybrid.py)
================================================================
Combines Spot Accumulation (70% allocation) and Leveraged Futures Trading (30% allocation)
by orchestrating the modular V2 engines from spot.py and futures.py.

Key Features:
  • Capital Partitioning: Allocates 70% of portfolio value to Spot, 30% to Futures margin.
  • Dual-Engine Execution: Delegates spot signals to SpotTradingBot and leveraged futures signals to FuturesTradingBot.
  • Multi-Asset Universe Scanning: Operates across SOLUSDT, ETHUSDT, AVAXUSDT, LINKUSDT, BNBUSDT.
  • Dual Execution Loop:
      - Fast Loop (Every 10s): Position monitoring, trailing stops, atomic exits.
      - Slow Loop (Every 5m): Full multi-asset scanning using 6-Factor consensus & S/R level anchors.
"""

import time
from datetime import datetime
from spot import SpotTradingBot
from futures import FuturesTradingBot, futures_state

# Portfolio Allocations
SPOT_ALLOCATION    = 0.70    # 70% of total capital allocated to spot
FUTURES_ALLOCATION = 0.30    # 30% of total capital allocated to futures margin


class HybridTradingBot:
    """Orchestrator combining Spot and Futures V2 trading engines."""

    def __init__(self):
        print(f"🤖 Initializing Hybrid Trading Bot (V2 Engine)...")
        print(f"================================================================================")
        print(f"💰 Allocation Strategy:")
        print(f"   • Spot Component:    {SPOT_ALLOCATION * 100:.0f}% Capital Allocation (Zero Leverage)")
        print(f"   • Futures Component: {FUTURES_ALLOCATION * 100:.0f}% Capital Allocation (Leveraged Futures)")
        print(f"================================================================================")

        self.spot_bot    = SpotTradingBot(allocation_pct=SPOT_ALLOCATION)
        self.futures_bot = FuturesTradingBot()

    def run_fast_loop(self):
        """
        Fast Loop (Every 10s):
        Rapidly check active positions, trailing stops, and atomic exits.
        """
        try:
            # Sync & monitor futures open position if active
            if futures_state.get('position'):
                self.futures_bot.sync_position_with_bybit()
                self.futures_bot.manage_futures_position()

            # Check spot open position if active
            from spot import spot_state
            if spot_state.get('position'):
                sym = spot_state['position']['symbol']
                data = self.spot_bot.fetch_multi_timeframe_data(sym)
                if data:
                    cur_price = data["15"]['close'][-1]
                    self.spot_bot.check_position_exit(sym, cur_price)

        except Exception as e:
            print(f"⚠️ Error in Hybrid Fast Loop: {e}")

    def run_slow_loop(self):
        """
        Slow Loop (Every 5m):
        Execute multi-asset scanning for Spot and Futures engines.
        """
        print(f"\n🚀 ================================================================================")
        print(f"🔄 HYBRID MULTI-ASSET SCAN CYCLE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🚀 ================================================================================")

        # 1. Run Spot Analysis & Execution Cycle
        try:
            print("\n📍 Running Spot Engine Analysis...")
            self.spot_bot.run_cycle()
        except Exception as e:
            print(f"❌ Spot Engine cycle error: {e}")

        # 2. Run Futures Analysis & Execution Cycle
        try:
            print("\n🚀 Running Futures Engine Analysis...")
            self.futures_bot.run_analysis_cycle()
        except Exception as e:
            print(f"❌ Futures Engine cycle error: {e}")


def run_hybrid_bot():
    bot = HybridTradingBot()
    print("\n🚀 Starting Hybrid Dual-Loop Execution...")
    print("   • Fast Loop:  Every 10 seconds (Price check & trailing exits)")
    print("   • Slow Loop:  Every 5 minutes (Multi-asset scan & 6-factor consensus)\n")

    last_slow_run = 0
    SLOW_INTERVAL = 300   # 5 minutes in seconds
    FAST_INTERVAL = 10    # 10 seconds

    while True:
        try:
            now = time.time()

            # Run slow scan cycle every 5 minutes
            if now - last_slow_run >= SLOW_INTERVAL:
                bot.run_slow_loop()
                last_slow_run = time.time()

            # Run fast position management loop every 10 seconds
            bot.run_fast_loop()
            time.sleep(FAST_INTERVAL)

        except KeyboardInterrupt:
            print("\n🛑 Hybrid Trading Bot stopped by user.")
            break
        except Exception as e:
            print(f"❌ Unexpected error in Hybrid main loop: {e}")
            time.sleep(10)


if __name__ == "__main__":
    run_hybrid_bot()