"""
Derivatives Factor  (factors/derivatives.py)
=============================================
Analyzes perpetual futures market structure to detect leverage imbalances
and likely squeeze setups.  Uses Bybit public API only (no key required).

Sub-signals:
  funding_rate   – Persistently high positive = longs crowded → SHORT bias
  open_interest  – OI direction vs price direction reveals conviction
  ls_ratio       – Retail long/short ratio as a contrarian signal

Score : -1.0 (longs crowded / SHORT setup) … +1.0 (shorts crowded / LONG setup)
"""

import requests
import numpy as np

BYBIT_URL = "https://api.bybit.com/v5/market"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_derivatives_score(symbol: str) -> dict:
    """Combine funding, OI trend, and L/S ratio into one derivatives score."""
    funding = _funding(symbol)
    oi      = _open_interest(symbol)
    ls      = _long_short_ratio(symbol)

    components = {}
    score = 0.0
    weight_total = 0.0

    if funding is not None:
        score        += funding["score"] * 0.45
        weight_total += 0.45
        components["funding"] = funding

    if oi is not None:
        score        += oi["score"] * 0.35
        weight_total += 0.35
        components["open_interest"] = oi

    if ls is not None:
        score        += ls["score"] * 0.20
        weight_total += 0.20
        components["long_short_ratio"] = ls

    if weight_total == 0:
        return _neutral("All Bybit derivatives endpoints failed")

    score = max(-1.0, min(1.0, score))
    confidence = round(weight_total, 2)   # scales with how many endpoints answered

    return {
        "score":       round(score, 3),
        "confidence":  confidence,
        "block_trade": False,
        "details":     components,
    }


# ---------------------------------------------------------------------------
# Sub-signals
# ---------------------------------------------------------------------------

def _funding(symbol: str):
    """
    Current + recent-average funding rate.
    High positive funding → longs overcrowded → SHORT bias (negative score).
    High negative funding → shorts overcrowded → LONG bias (positive score).
    """
    try:
        # Current rate from ticker
        r = requests.get(f"{BYBIT_URL}/tickers",
                         params={"category": "linear", "symbol": symbol}, timeout=8)
        d = r.json()
        if d.get("retCode") != 0:
            return None
        current_rate = float(d["result"]["list"][0].get("fundingRate", 0))

        # 8-period history (~2.67 days at 8-hour intervals)
        h = requests.get(f"{BYBIT_URL}/funding/history",
                         params={"category": "linear", "symbol": symbol, "limit": 8}, timeout=8)
        hd = h.json()
        avg_rate = current_rate
        if hd.get("retCode") == 0:
            rates = [float(x["fundingRate"]) for x in hd["result"]["list"]]
            avg_rate = float(np.mean(rates)) if rates else current_rate

        # Map to score — contrarian interpretation
        # Standard neutral range: 0.0% to +0.015% per 8h (0.00015) — normal bull market baseline
        # High positive funding: > 0.015%/8h → longs overcrowded → negative score
        # Negative funding: < 0.0% → shorts overcrowded → positive score
        if avg_rate > 0.00015:
            score = -min(1.0, (avg_rate - 0.00015) / 0.0004)   # >0.055%/8h = max negative (-1.0)
        elif avg_rate < 0:
            score = min(1.0, abs(avg_rate) / 0.0003)            # -0.03%/8h = max positive (+1.0)
        else:
            score = 0.0                                         # 0 to 0.015% = neutral baseline

        return {
            "score":              round(score, 3),
            "current_rate_pct":   round(current_rate * 100, 4),
            "avg_8period_pct":    round(avg_rate * 100, 4),
        }
    except Exception:
        return None


def _open_interest(symbol: str):
    """
    OI trend over ~16 hours (4 × 4h bars).
    Rising OI + rising price  → longs building → LONG bias
    Rising OI + falling price → shorts building → SHORT bias
    Falling OI                → deleveraging → neutral
    """
    try:
        r = requests.get(f"{BYBIT_URL}/open-interest",
                         params={"category": "linear", "symbol": symbol,
                                 "intervalTime": "4h", "limit": 8}, timeout=8)
        d = r.json()
        if d.get("retCode") != 0:
            return None

        items = d["result"]["list"]
        if len(items) < 4:
            return None

        oi_now  = float(items[0]["openInterest"])
        oi_then = float(items[3]["openInterest"])
        oi_chg  = (oi_now - oi_then) / oi_then if oi_then > 0 else 0.0

        if abs(oi_chg) < 0.02:  # < 2% OI change → noise
            return {"score": 0.0, "oi_change_pct": round(oi_chg * 100, 2)}

        # Get 24h price change from ticker
        t = requests.get(f"{BYBIT_URL}/tickers",
                         params={"category": "linear", "symbol": symbol}, timeout=8)
        td = t.json()
        price_chg = 0.0
        if td.get("retCode") == 0:
            price_chg = float(td["result"]["list"][0].get("price24hPcnt", 0))

        if oi_chg > 0 and price_chg > 0:
            score = min(1.0, oi_chg * 5)     # rising OI + rising price = LONG
        elif oi_chg > 0 and price_chg < 0:
            score = -min(1.0, oi_chg * 5)    # rising OI + falling price = SHORT
        else:
            score = 0.0  # deleveraging = neutral

        return {
            "score":          round(score, 3),
            "oi_change_pct":  round(oi_chg * 100, 2),
            "price_24h_pct":  round(price_chg * 100, 2),
        }
    except Exception:
        return None


def _long_short_ratio(symbol: str):
    """
    Retail trader long/short ratio — contrarian signal.
    >70% long  → crowded long → SHORT bias
    <30% long  → crowded short → LONG bias
    """
    try:
        r = requests.get(f"{BYBIT_URL}/account-ratio",
                         params={"category": "linear", "symbol": symbol,
                                 "period": "1h", "limit": 4}, timeout=8)
        d = r.json()
        if d.get("retCode") != 0:
            return None

        items = d["result"]["list"]
        if not items:
            return None

        long_ratio = float(items[0].get("buyRatio", 0.5))

        # Contrarian: 70% long → -0.8 score, 30% long → +0.8 score
        deviation = long_ratio - 0.5          # +0.2 means 70% long
        score     = max(-1.0, min(1.0, -deviation * 4))

        return {
            "score":           round(score, 3),
            "long_ratio_pct":  round(long_ratio * 100, 1),
            "short_ratio_pct": round((1 - long_ratio) * 100, 1),
        }
    except Exception:
        return None


def _neutral(reason=""):
    return {"score": 0.0, "confidence": 0.0, "block_trade": False,
            "details": {"reason": reason}}
