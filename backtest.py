"""
Deriv Strategy Backtester
==========================
Backtests the multi-factor consensus logic from deriv.py on real Deriv candle data.

Usage:
    python backtest.py                           # 90 days, all pairs, 10% risk, 2.5 RR
    python backtest.py --symbol EURUSD           # single pair
    python backtest.py --days 180                # last 180 days
    python backtest.py --balance 50              # starting balance
    python backtest.py --risk 0.10 --rr 2.5     # tune parameters (match deriv.py)
    python backtest.py --sweep                   # test all parameter combos, find best

Requires:
    pip install websocket-client python-dotenv numpy pandas
"""

import os, json, time, argparse, sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

try:
    import websocket
    HAS_WS = True
except ImportError:
    HAS_WS = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

load_dotenv()
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN", "")
DERIV_APP_ID    = os.getenv("DERIV_APP_ID", "1089")

# ── Symbol config (mirrors deriv.py) ─────────────────────────────────────
TRADE_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
DERIV_SYMBOLS = {
    "EURUSD": "frxEURUSD", "GBPUSD": "frxGBPUSD", "USDJPY": "frxUSDJPY",
    "AUDUSD": "frxAUDUSD", "USDCAD": "frxUSDCAD",
}
SYMBOL_SPECS = {
    "EURUSD": {"pip_size": 0.0001, "contract_size": 100000, "quote_is_usd": True},
    "GBPUSD": {"pip_size": 0.0001, "contract_size": 100000, "quote_is_usd": True},
    "USDJPY": {"pip_size": 0.01,   "contract_size": 100000, "quote_is_usd": False},
    "AUDUSD": {"pip_size": 0.0001, "contract_size": 100000, "quote_is_usd": True},
    "USDCAD": {"pip_size": 0.0001, "contract_size": 100000, "quote_is_usd": False},
}
WEIGHTS = {
    "trend_alignment": 0.25, "support_resistance": 0.25,
    "technical": 0.20, "news": 0.15, "spread_volatility": 0.15
}

# ── Strategy parameters (set via CLI, match deriv.py) ────────────────────
# Change these in deriv.py, then pass same values here via --risk / --rr
forex_risk_per_trade     = 0.10   # fraction of balance per trade
min_reward_ratio         = 2.5    # take-profit = stop * this ratio
min_signal_threshold     = 0.25   # minimum consensus score to enter
min_volatility_threshold = 0.0010
london_open_utc          = 7.0
ny_close_utc             = 16.5
MAX_DAILY_LOSS_PCT       = 0.05


# ── WebSocket helpers ─────────────────────────────────────────────────────

def get_ws():
    if not HAS_WS or not DERIV_API_TOKEN or not HAS_REQUESTS:
        return None
    try:
        hdrs = {"Authorization": f"Bearer {DERIV_API_TOKEN}",
                "Deriv-App-ID": DERIV_APP_ID, "Content-Type": "application/json"}
        res = requests.get("https://api.derivws.com/trading/v1/options/accounts", headers=hdrs, timeout=10)
        if res.status_code != 200:
            print(f"❌ Auth failed: {res.text}"); return None
        accounts = res.json().get("data", [])
        acc_id = accounts[0]["account_id"] if accounts else None
        if not acc_id:
            print("❌ No account found"); return None
        otp = requests.post(f"https://api.derivws.com/trading/v1/options/accounts/{acc_id}/otp", headers=hdrs, timeout=10)
        ws_url = otp.json().get("data", {}).get("url")
        if not ws_url:
            print(f"❌ OTP failed"); return None
        ws = websocket.create_connection(ws_url, timeout=15)
        print(f"✅ Connected to Deriv (Account: {acc_id})")
        return ws
    except Exception as e:
        print(f"❌ WS error: {e}"); return None


def ws_req(ws, payload, timeout=15):
    try:
        ws.send(json.dumps(payload))
        ws.settimeout(timeout)
        return json.loads(ws.recv())
    except Exception as e:
        print(f"  ⚠️ WS error: {e}"); return None


def fetch_candles(ws, symbol, granularity, count):
    res = ws_req(ws, {
        "ticks_history": DERIV_SYMBOLS.get(symbol, symbol),
        "adjust_start_time": 1, "count": count,
        "end": "latest", "granularity": granularity, "style": "candles"
    })
    if not res or "candles" not in res:
        return None
    df = pd.DataFrame([{"epoch": c["epoch"], "open": float(c["open"]), "high": float(c["high"]),
                         "low": float(c["low"]), "close": float(c["close"])} for c in res["candles"]])
    return df.set_index("epoch").sort_index()


# ── Indicator calculation (mirrors deriv.py) ──────────────────────────────

def _ema(arr, n):
    k = 2.0 / (n + 1); val = float(np.mean(arr[:n]))
    for v in arr[n:]: val = float(v) * k + val * (1 - k)
    return val


def calc_ind(closes, highs, lows):
    diff = np.diff(closes)
    gains = np.where(diff > 0, diff, 0.0); loss = np.where(diff < 0, -diff, 0.0)
    rp = 14; ag = float(np.mean(gains[:rp])); al = float(np.mean(loss[:rp]))
    for i in range(rp, len(gains)):
        ag = (ag*(rp-1)+gains[i])/rp; al = (al*(rp-1)+loss[i])/rp
    rsi = 100 - (100 / (1 + ag/(al+1e-9)))
    ema21 = _ema(closes, 21); ema50 = _ema(closes, 50)
    macd  = _ema(closes, 12) - _ema(closes, 26)
    start = max(26, len(closes)-20)
    mv = [_ema(closes[:i+1],12)-_ema(closes[:i+1],26) for i in range(start, len(closes))]
    macd_sig = _ema(np.array(mv), 9) if len(mv) >= 9 else mv[-1]
    tr = np.maximum(highs[1:]-lows[1:], np.maximum(np.abs(highs[1:]-closes[:-1]), np.abs(lows[1:]-closes[:-1])))
    ap = 14; atr = float(np.mean(tr[:ap]))
    for v in tr[ap:]: atr = (atr*(ap-1)+float(v))/ap
    sp = 14; rh = float(np.max(highs[-sp:])); rl = float(np.min(lows[-sp:]))
    stoch_k = ((float(closes[-1])-rl)/(rh-rl+1e-9))*100
    dm_p = np.where((highs[1:]-highs[:-1])>(lows[:-1]-lows[1:]), np.maximum(highs[1:]-highs[:-1],0),0)
    dm_m = np.where((lows[:-1]-lows[1:])>(highs[1:]-highs[:-1]), np.maximum(lows[:-1]-lows[1:],0),0)
    aw = np.mean(tr[-ap:]); dip=100*np.mean(dm_p[-ap:])/(aw+1e-9); dim=100*np.mean(dm_m[-ap:])/(aw+1e-9)
    adx = 100*abs(dip-dim)/(dip+dim+1e-9)
    return {"rsi":rsi,"macd":macd,"macd_signal":macd_sig,"ema_21":ema21,"ema_50":ema50,
            "atr":atr,"adx":adx,"stoch_k":stoch_k}


def get_signal(i15, i1h, i4h, price, vol):
    trend_bull = i4h["ema_21"] > i4h["ema_50"] if i4h["ema_50"] > 0 else True
    trend = 1.0 if trend_bull else -1.0
    lc = [i15["rsi"]<45, i1h["rsi"]<52, i15["macd"]>i15["macd_signal"],
          price>i15["ema_21"]*0.999, vol>min_volatility_threshold, i15["adx"]>18, i15["stoch_k"]<35]
    sc = [i15["rsi"]>55, i1h["rsi"]>48, i15["macd"]<i15["macd_signal"],
          price<i15["ema_21"]*1.001, vol>min_volatility_threshold, i15["adx"]>18, i15["stoch_k"]>65]
    ta = (sum(lc)-sum(sc))/7.0
    score = WEIGHTS["trend_alignment"]*trend + WEIGHTS["technical"]*ta + WEIGHTS["spread_volatility"]*0.6
    if score >= min_signal_threshold and trend_bull:  return "LONG",  score, i15["atr"]
    if score <= -min_signal_threshold and not trend_bull: return "SHORT", score, i15["atr"]
    return None, score, i15["atr"]


# ── PnL calculation ───────────────────────────────────────────────────────

def calc_pnl(symbol, direction, entry, exit_p, lots):
    s = SYMBOL_SPECS[symbol]; pip = s["pip_size"]
    pips = (exit_p-entry)/pip if direction=="LONG" else (entry-exit_p)/pip
    pvl  = s["contract_size"]*pip if s["quote_is_usd"] else (s["contract_size"]*pip)/exit_p
    return pips * pvl * lots


# ── Core backtester ───────────────────────────────────────────────────────

def backtest_symbol(ws_unused, symbol, df15, df1h, df4h, start_balance):
    specs  = SYMBOL_SPECS[symbol]; pip = specs["pip_size"]
    LOOK   = 100
    trades = []; balance = start_balance; pos = None
    daily_pnl = 0.0; last_day = None

    for i in range(LOOK, len(df15)):
        bar   = df15.iloc[i]; epoch = df15.index[i]
        dt    = datetime.fromtimestamp(epoch, tz=timezone.utc)
        hf    = dt.hour + dt.minute/60.0
        today = dt.date()

        # Reset daily P&L tracker
        if last_day != today:
            daily_pnl = 0.0; last_day = today

        # Daily loss killswitch
        if daily_pnl < -(start_balance * MAX_DAILY_LOSS_PCT):
            pos = None; continue

        # ── Manage open position ──────────────────────────────────────────
        if pos:
            hi, lo = float(bar["high"]), float(bar["low"])
            entry  = pos["entry"]; orig = pos["orig_stop"]

            if pos["dir"] == "LONG":
                if not pos["be"] and (hi-entry)/pip >= (entry-orig)/pip:
                    pos["stop"] = entry + 2*pip; pos["be"] = True
                if lo <= pos["stop"]:
                    ep = pos["stop"]; pnl = calc_pnl(symbol, "LONG", entry, ep, pos["lots"])
                    balance += pnl; daily_pnl += pnl
                    trades.append({"symbol":symbol,"direction":"LONG","entry":entry,"exit":ep,
                                   "pnl":round(pnl,4),"result":"WIN" if pnl>0 else "LOSS",
                                   "pips":round((ep-entry)/pip,1),"date":dt.strftime("%Y-%m-%d %H:%M")})
                    pos = None; continue
                if hi >= pos["target"]:
                    ep = pos["target"]; pnl = calc_pnl(symbol, "LONG", entry, ep, pos["lots"])
                    balance += pnl; daily_pnl += pnl
                    trades.append({"symbol":symbol,"direction":"LONG","entry":entry,"exit":ep,
                                   "pnl":round(pnl,4),"result":"WIN" if pnl>0 else "LOSS",
                                   "pips":round((ep-entry)/pip,1),"date":dt.strftime("%Y-%m-%d %H:%M")})
                    pos = None; continue

            elif pos["dir"] == "SHORT":
                if not pos["be"] and (entry-lo)/pip >= (orig-entry)/pip:
                    pos["stop"] = entry - 2*pip; pos["be"] = True
                if hi >= pos["stop"]:
                    ep = pos["stop"]; pnl = calc_pnl(symbol, "SHORT", entry, ep, pos["lots"])
                    balance += pnl; daily_pnl += pnl
                    trades.append({"symbol":symbol,"direction":"SHORT","entry":entry,"exit":ep,
                                   "pnl":round(pnl,4),"result":"WIN" if pnl>0 else "LOSS",
                                   "pips":round((entry-ep)/pip,1),"date":dt.strftime("%Y-%m-%d %H:%M")})
                    pos = None; continue
                if lo <= pos["target"]:
                    ep = pos["target"]; pnl = calc_pnl(symbol, "SHORT", entry, ep, pos["lots"])
                    balance += pnl; daily_pnl += pnl
                    trades.append({"symbol":symbol,"direction":"SHORT","entry":entry,"exit":ep,
                                   "pnl":round(pnl,4),"result":"WIN" if pnl>0 else "LOSS",
                                   "pips":round((entry-ep)/pip,1),"date":dt.strftime("%Y-%m-%d %H:%M")})
                    pos = None; continue
            continue

        # ── Session guard ─────────────────────────────────────────────────
        if not (london_open_utc <= hf <= ny_close_utc):
            continue

        # ── Indicators ────────────────────────────────────────────────────
        c15 = df15["close"].values[i-LOOK:i+1]
        h15 = df15["high"].values[i-LOOK:i+1]
        l15 = df15["low"].values[i-LOOK:i+1]
        price = float(c15[-1])

        j1 = df1h.index.searchsorted(epoch, side="right") - 1
        j4 = df4h.index.searchsorted(epoch, side="right") - 1
        if j1 < LOOK or j4 < LOOK: continue

        c1h = df1h["close"].values[max(0,j1-LOOK):j1+1]
        h1h = df1h["high"].values[max(0,j1-LOOK):j1+1]
        l1h = df1h["low"].values[max(0,j1-LOOK):j1+1]
        c4h = df4h["close"].values[max(0,j4-LOOK):j4+1]
        h4h = df4h["high"].values[max(0,j4-LOOK):j4+1]
        l4h = df4h["low"].values[max(0,j4-LOOK):j4+1]
        if len(c15)<55 or len(c1h)<55 or len(c4h)<55: continue

        i15 = calc_ind(c15, h15, l15)
        i1h = calc_ind(c1h, h1h, l1h)
        i4h = calc_ind(c4h, h4h, l4h)

        vol = abs((price - (float(c1h[-24]) if len(c1h)>=24 else float(c1h[0]))) / price)
        sig, score, atr = get_signal(i15, i1h, i4h, price, vol)
        if not sig: continue

        # ── Sizing ────────────────────────────────────────────────────────
        stop_pips   = max(15.0, 1.5*atr/pip)
        reward_pips = stop_pips * min_reward_ratio
        if specs["quote_is_usd"]:
            pvl = specs["contract_size"] * pip
        else:
            pvl = (specs["contract_size"] * pip) / price
        lots = max(0.01, round(round((balance*forex_risk_per_trade)/(stop_pips*pvl)/0.01)*0.01, 2))
        sl = price-(stop_pips*pip) if sig=="LONG" else price+(stop_pips*pip)
        tp = price+(reward_pips*pip) if sig=="LONG" else price-(reward_pips*pip)

        pos = {"dir":sig,"entry":price,"stop":sl,"target":tp,"orig_stop":sl,"be":False,"lots":lots}

    return trades, balance


# ── Summary ───────────────────────────────────────────────────────────────

def print_summary(trades, start_bal, final_bal, label=""):
    df = pd.DataFrame(trades)
    if df.empty:
        print("  ⚠️  No trades generated."); return None

    wins   = (df["result"]=="WIN").sum(); losses = (df["result"]=="LOSS").sum(); total = len(df)
    wr     = wins/total*100 if total else 0
    net    = df["pnl"].sum()
    aw     = df[df["result"]=="WIN"]["pnl"].mean() if wins else 0
    al     = df[df["result"]=="LOSS"]["pnl"].mean() if losses else 0
    pf     = abs(df[df["pnl"]>0]["pnl"].sum()/df[df["pnl"]<0]["pnl"].sum()) if losses else 99.0
    bals   = [start_bal]; [bals.append(bals[-1]+p) for p in df["pnl"]]
    peaks  = pd.Series(bals).cummax()
    maxdd  = ((peaks-pd.Series(bals))/peaks*100).max()

    title = f"  📊  BACKTEST RESULTS{' — '+label if label else ''}"
    print("\n" + "═"*58); print(title); print("═"*58)
    print(f"  Start Balance : ${start_bal:.2f}")
    print(f"  Final Balance : ${final_bal:.2f}  ({(final_bal/start_bal-1)*100:+.1f}%)")
    print(f"  Net P&L       : ${net:.2f}")
    print(f"  Total Trades  : {total}  ({wins}W / {losses}L)")
    print(f"  Win Rate      : {wr:.1f}%")
    print(f"  Avg Win       : ${aw:.2f}  |  Avg Loss: ${al:.2f}")
    print(f"  Profit Factor : {pf:.2f}x   ← need >1.0 to be profitable")
    print(f"  Max Drawdown  : {maxdd:.1f}%")
    print("═"*58)
    by_sym = df.groupby("symbol").agg(trades=("pnl","count"), pnl=("pnl","sum"),
                                       wr=("result", lambda x: (x=="WIN").mean()*100)).round(2)
    print("\n  Per-Symbol Breakdown:")
    print(by_sym.to_string()); print()
    return {"pf": pf, "wr": round(wr,1), "net": round(net,2), "maxdd": round(maxdd,1), "trades": total}


# ── Parameter sweep ───────────────────────────────────────────────────────

def run_sweep(cached, start_bal):
    global forex_risk_per_trade, min_reward_ratio, min_signal_threshold
    combos = [
        (0.05, 1.5, 0.25), (0.05, 2.0, 0.25), (0.05, 2.5, 0.30),
        (0.10, 1.5, 0.25), (0.10, 2.0, 0.25), (0.10, 2.5, 0.30), (0.10, 3.0, 0.35),
        (0.15, 2.0, 0.30), (0.15, 2.5, 0.30), (0.15, 3.0, 0.35),
        (0.20, 2.0, 0.35), (0.20, 2.5, 0.35),
    ]
    print("\n🔁 Parameter Sweep")
    print(f"  {'Risk%':>5} {'RR':>5} {'Thr':>5} | {'Trades':>6} {'WR%':>6} {'PF':>6} {'NetPnL':>9} {'MaxDD':>7}")
    print("  " + "-"*62)
    results = []
    for risk, rr, thr in combos:
        forex_risk_per_trade = risk; min_reward_ratio = rr; min_signal_threshold = thr
        all_t = []; bal = start_bal
        for sym, dfs in cached.items():
            if dfs is None: continue
            t, bal = backtest_symbol(None, sym, dfs["15m"], dfs["1h"], dfs["4h"], bal)
            all_t.extend(t)
        if not all_t:
            print(f"  {risk*100:>4.0f}%  {rr:>5.1f}  {thr:>5.2f} |   no trades"); continue
        df = pd.DataFrame(all_t)
        wins = (df["result"]=="WIN").sum(); loss = (df["result"]=="LOSS").sum(); tot = len(df)
        wr   = wins/tot*100 if tot else 0
        pf   = abs(df[df["pnl"]>0]["pnl"].sum()/df[df["pnl"]<0]["pnl"].sum()) if loss else 99.0
        net  = df["pnl"].sum()
        bals = [start_bal]; [bals.append(bals[-1]+p) for p in df["pnl"]]
        peaks = pd.Series(bals).cummax(); maxdd = ((peaks-pd.Series(bals))/peaks*100).max()
        flag = " ✅" if pf > 1.0 else ""
        print(f"  {risk*100:>4.0f}%  {rr:>5.1f}  {thr:>5.2f} | {tot:>6} {wr:>6.1f} {pf:>6.2f} {net:>9.2f} {maxdd:>7.1f}{flag}")
        results.append({"risk":risk,"rr":rr,"thr":thr,"pf":pf,"wr":round(wr,1),"net":round(net,2),"maxdd":round(maxdd,1)})

    profitable = [r for r in results if r["pf"] > 1.0 and r["pf"] < 90]
    if profitable:
        best = max(profitable, key=lambda x: x["pf"])
        print(f"\n  🏆 Best: risk={best['risk']*100:.0f}% | RR={best['rr']} | threshold={best['thr']}")
        print(f"     PF={best['pf']}x | WR={best['wr']}% | Net=${best['net']} | MaxDD={best['maxdd']}%")
        print(f"\n  ➡  Set in deriv.py:")
        print(f"     forex_risk_per_trade = {best['risk']}")
        print(f"     min_reward_ratio     = {best['rr']}")
    else:
        print("\n  ⚠️  No profitable parameter combo found in this period.")
        print("     Consider: longer history, different session, or strategy refinement.")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    global forex_risk_per_trade, min_reward_ratio, min_signal_threshold
    p = argparse.ArgumentParser(description="Deriv Strategy Backtester")
    p.add_argument("--symbol",    default=None)
    p.add_argument("--days",      type=int,   default=90)
    p.add_argument("--balance",   type=float, default=50.0)
    p.add_argument("--risk",      type=float, default=0.10, help="Stake per trade as fraction (0.10=10%%)")
    p.add_argument("--rr",        type=float, default=2.5,  help="Reward:risk ratio (e.g. 2.5)")
    p.add_argument("--threshold", type=float, default=0.25, help="Min consensus score (e.g. 0.25)")
    p.add_argument("--sweep",     action="store_true", help="Test many parameter combos")
    p.add_argument("--csv",       default="backtest_results.csv")
    args = p.parse_args()

    forex_risk_per_trade = args.risk
    min_reward_ratio     = args.rr
    min_signal_threshold = args.threshold

    symbols = [args.symbol] if args.symbol else TRADE_SYMBOLS
    count   = min(5000, args.days * 96)

    print(f"\n🔬 Deriv Strategy Backtester")
    print(f"   Symbols   : {', '.join(symbols)}")
    print(f"   History   : {args.days} days  |  Balance: ${args.balance:.2f}")
    print(f"   Risk/Trade: {args.risk*100:.0f}%  |  RR: {args.rr}:1  |  Threshold: {args.threshold}")

    ws = get_ws()
    if not ws:
        print("\n❌ Cannot connect. Check DERIV_API_TOKEN and DERIV_APP_ID in .env"); sys.exit(1)

    print("\n⬇️  Downloading candle data...")
    cached = {}
    for sym in symbols:
        d15 = fetch_candles(ws, sym, 900,   count);    time.sleep(0.4)
        d1h = fetch_candles(ws, sym, 3600,  min(1500, args.days*24)); time.sleep(0.4)
        d4h = fetch_candles(ws, sym, 14400, min(500,  args.days*6));  time.sleep(0.4)
        if d15 is not None and d1h is not None and d4h is not None:
            cached[sym] = {"15m": d15, "1h": d1h, "4h": d4h}
            print(f"   ✅ {sym}: {len(d15)} × 15m bars")
        else:
            cached[sym] = None
            print(f"   ⚠️  {sym}: data unavailable")
    ws.close()

    if args.sweep:
        run_sweep(cached, args.balance)
        return

    all_trades = []; balance = args.balance
    for sym, dfs in cached.items():
        if dfs is None: continue
        t, balance = backtest_symbol(None, sym, dfs["15m"], dfs["1h"], dfs["4h"], balance)
        all_trades.extend(t)

    print_summary(all_trades, args.balance, balance)
    if all_trades:
        pd.DataFrame(all_trades).to_csv(args.csv, index=False)
        print(f"  📁 Trade log: {args.csv}")


if __name__ == "__main__":
    main()
