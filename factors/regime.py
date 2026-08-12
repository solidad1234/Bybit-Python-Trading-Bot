"""
Market Regime Factor  (factors/regime.py)
==========================================
Assesses macro market conditions using only Bybit's free public API.

Signals derived from BTC daily klines:
  - BTC 30-day EMA position  (trend context)
  - 7-day momentum           (short-term direction)
  - ATR volatility ratio     (regime quality: trending vs whipsawing)

Score : -1.0 (risk-off / downtrend) … +1.0 (risk-on / uptrend)
"""

import requests
import numpy as np

try:
    import talib
    _TALIB = True
except ImportError:
    _TALIB = False

BYBIT_URL = "https://api.bybit.com/v5/market"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_regime_score() -> dict:
    """Return a macro regime assessment based on BTC daily data."""
    try:
        candles = _fetch_btc_daily(90)
        if candles is None or len(candles) < 50:
            return _neutral("Insufficient BTC data")

        closes = np.array([float(c[4]) for c in candles])
        highs  = np.array([float(c[2]) for c in candles])
        lows   = np.array([float(c[3]) for c in candles])

        current = closes[-1]

        # 1. 30-day EMA position
        ema30 = _ema(closes, 30)
        if ema30 > 0:
            if current > ema30 * 1.03:
                trend_score = 1.0
            elif current > ema30 * 1.00:
                trend_score = 0.5
            elif current > ema30 * 0.95:
                trend_score = -0.3
            else:
                trend_score = -1.0
        else:
            trend_score = 0.0

        # 2. 7-day momentum
        mom7 = (current - closes[-8]) / closes[-8] if len(closes) >= 8 else 0.0
        mom_score = max(-1.0, min(1.0, mom7 * 10))  # ±10% → ±1.0

        # 3. ATR volatility penalty (excessive chop → reduce confidence)
        atr_vals = _atr(highs, lows, closes, 14)
        current_atr = atr_vals[-1] if len(atr_vals) else 0
        avg_atr     = float(np.nanmean(atr_vals[-60:])) if len(atr_vals) else 1
        atr_ratio   = current_atr / avg_atr if avg_atr > 0 else 1.0
        vol_penalty = -0.3 if atr_ratio > 2.0 else (-0.1 if atr_ratio > 1.5 else 0.0)

        final = max(-1.0, min(1.0, trend_score * 0.55 + mom_score * 0.30 + vol_penalty * 0.15))
        regime = "RISK_ON" if final >= 0.25 else ("RISK_OFF" if final <= -0.25 else "NEUTRAL")

        return {
            "score":       round(final, 3),
            "confidence":  0.80,
            "regime":      regime,
            "block_trade": False,
            "details": {
                "btc_vs_ema30_pct": round((current / ema30 - 1) * 100, 2) if ema30 else None,
                "7d_momentum_pct":  round(mom7 * 100, 2),
                "atr_ratio":        round(atr_ratio, 2),
            },
        }
    except Exception as e:
        return _neutral(f"Exception: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_btc_daily(limit: int):
    try:
        resp = requests.get(
            f"{BYBIT_URL}/kline",
            params={"category": "linear", "symbol": "BTCUSDT", "interval": "D", "limit": limit},
            timeout=10,
        )
        data = resp.json()
        if data.get("retCode") != 0:
            return None
        return list(reversed(data["result"]["list"]))
    except Exception:
        return None


def _ema(series: np.ndarray, period: int) -> float:
    if _TALIB:
        result = talib.EMA(series.astype(float), timeperiod=period)
        return float(result[-1]) if not np.isnan(result[-1]) else 0.0
    # Fallback: simple EMA via pandas-style calculation
    k = 2.0 / (period + 1)
    ema = float(series[0])
    for v in series[1:]:
        ema = v * k + ema * (1 - k)
    return ema


def _atr(highs, lows, closes, period):
    if _TALIB:
        return talib.ATR(highs.astype(float), lows.astype(float), closes.astype(float), timeperiod=period)
    # Fallback: simple TR-based ATR
    trs = [highs[i] - lows[i] for i in range(len(closes))]
    return np.array(trs)


def _neutral(reason=""):
    return {"score": 0.0, "confidence": 0.0, "regime": "NEUTRAL", "block_trade": False,
            "details": {"reason": reason}}
