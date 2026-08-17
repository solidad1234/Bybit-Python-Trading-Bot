# Bybit Python Trading Bot

> [!WARNING]
> ### ⚠️ Disclaimer
> Cryptocurrency trading involves substantial risk of loss and is not suitable for every investor.  
> This trading bot is provided **strictly for educational and research purposes**.  
> - There are **no guarantees of profit**, and you can lose some or all of your collateral.  
> - Always **Do Your Own Research (DYOR)** and test extensively in testnet/sandbox environments before trading live funds.  
> - The author(s) accept **no responsibility or liability** for financial losses incurred through the use of this software.

---

A modular, multi-factor algorithmic trading system built for **Bybit USDT Perpetual Futures & Spot markets**. Powered by **Support & Resistance structural anchors**, multi-timeframe Technical Analysis, Macro Regime detection, Derivatives flow, Sentiment caching, and Google News RSS filtering.

---

## 🌟 Key Features & Architecture (V2 Engine)

### 1. 🛡️ BTC Correlation & Macro News Dual Guard
* **BTC Price Correlation**: Continuously monitors Bitcoin's 1h and 4h price momentum. Automatically blocks LONG setups across all altcoin pairs if BTC exhibits bearish momentum ($1\text{h change} < -2.0\%$ or $4\text{h change} < -5.0\%$).
* **BTC Macro News Filter**: Evaluates macro market news alongside coin-specific headlines. Macro BTC news contributes a 40% weight to every asset's news score. If macro BTC news is severely negative, all trades across the universe are halted.

### 2. 🎯 Support & Resistance (S/R) Execution Engine (`factors/support_resistance.py`)
* **Multi-Timeframe Level Detection**: Combines 1h swing levels (~4 days), 4h structural walls (~17 days), and auto-scaled psychological round-number grids ($5 for SOL/AVAX, $50 for BNB, $100 for ETH).
* **5 Actionable Scenarios**:
  1. **`AT_SUPPORT`**: Price at key support $\rightarrow$ **BOUNCE LONG** setup (Stop: below support wall).
  2. **`AT_RESISTANCE`**: Price at key resistance $\rightarrow$ **REJECTION SHORT** setup (Stop: above resistance wall).
  3. **`BREAKOUT_ABOVE`**: Price closes $\ge 1.0\%$ above resistance with volume $> 1.5\times$ 20-period SMA $\rightarrow$ **BREAKOUT LONG ⚡** (Leverage boosted to **12×**, Stop anchored at broken resistance).
  4. **`BREAKDOWN_BELOW`**: Price closes $\ge 1.0\%$ below support with volume $> 1.5\times$ 20-period SMA $\rightarrow$ **BREAKDOWN SHORT ⚡** (Leverage boosted to **12×**, Stop anchored at broken support).
  5. **`MID_RANGE`**: Price in mid-range $\rightarrow$ Neutral score (-0.1), defaults to ATR-based stops.
* **1% Volume-Confirmed Breakout Boost**: Prevents low-volume fake-outs by enforcing a 15m candle close $\ge 1.0\%$ beyond the level **AND** volume $> 1.5\times$ SMA.

### 3. 🔬 6-Factor Multi-Factor Consensus Engine (`factors/aggregator.py`)
Rebalanced consensus engine evaluates 6 distinct factors before approving any trade:

| Factor | Weight | Key Metrics & Signals |
| :--- | :---: | :--- |
| **`regime`** | **25%** | Macro BTC market regime (Bull / Bear / Neutral) & 4h EMA trend gate |
| **`derivatives`** | **22%** | Bybit Funding Rates, Open Interest (OI) velocity, & Long/Short ratio |
| **`technical`** | **20%** | Multi-timeframe (15m, 1h, 4h) RSI, MACD, ADX, Stochastic & Volume TA |
| **`support_resistance`** | **15%** | S/R proximity, level touch counts, and volume-confirmed breakouts |
| **`sentiment`** | **12%** | 1-hour TTL cached Crypto Fear & Greed contrarian sentiment |
| **`news`** | **6%** | Asymmetric news scoring (Bad coin news blocks LONGs, allows SHORTs) |

### 4. 🌐 Dynamic Multi-Asset Universe Scanning
* Dynamically scans high-liquidity assets: `SOLUSDT`, `ETHUSDT`, `AVAXUSDT`, `LINKUSDT`, `BNBUSDT`.
* **Symbol-Aware Contract Precision**: Enforces exact Bybit lot sizes (`min_qty` & `step_size`) and price precisions per asset.

### 5. 🛡️ Advanced Risk Management & Atomic Orders
* **Atomic Execution**: Market orders are submitted to Bybit with Stop Loss (SL) and Take Profit (TP) parameters set **atomically** in a single API call, eliminating execution race conditions.
* **P&L Session Circuit Breaker**: Automatically halts trading if session losses exceed **5% of initial balance**.
* **Consecutive Loss Cooldown**: Pauses trading after **3 consecutive losses**.
* **Early Scratch Exit**: Immediately exits adverse positions (-0.7%) within the first 45 minutes to protect capital.

### 6. 💾 State Persistence & AI/ML Logging (`trading_state.db`)
* SQLite WAL-mode database persists open position state across restarts.
* Logs rich multi-factor context (`trade_log` table) for every entry/exit, creating a structured dataset for machine learning model training (XGBoost).

---

## 🤖 Available Trading Bots

1. **Futures Trading Bot (`futures.py` - Recommended)**
   - Leveraged USDT perpetual futures execution with multi-asset scanning, S/R anchors, 12× breakout leverage, and 6-factor consensus.
2. **Hybrid Trading Bot (`hybrid.py`)**
   - Allocates 70% portfolio to spot accumulation and 30% to futures hedging/momentum.
3. **Spot Trading Bot (`spot.py`)**
   - Conservative spot accumulation for trending markets.

---

## ⚙️ Setup & Deployment

### 1. Installation & Environment

```bash
# Clone repository
git clone https://github.com/solidad1234/Bybit-Python-Trading-Bot.git
cd bybit-trading-bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables Configuration

Create a `.env` file in the project root:

```env
API_KEY=your_bybit_api_key
API_SECRET=your_bybit_api_secret
CRYPTOPANIC_TOKEN=optional_cryptopanic_token
```

### 3. Execution Commands

```bash
# Run Futures Trading Bot (Multi-Asset V2 Engine)
python3 futures.py

# Run Hybrid Trading Bot
python3 hybrid.py

# Run Backtest Engine (Multi-Asset S/R Historical Verification)
python3 backtest.py
```

### 4. Deploying on VPS (using `screen`)

```bash
screen -S futures-bot
source venv/bin/activate
python3 futures.py
```
*Detach screen*: `Ctrl + A`, then `D`  
*Reattach later*: `screen -r futures-bot`

---

## 🤝 Contributions & Contact

Pull requests and issues are welcome!

**Author**: Solidad Kimeu  
📧 [solidadkimeu@gmail.com](mailto:solidadkimeu@gmail.com)
