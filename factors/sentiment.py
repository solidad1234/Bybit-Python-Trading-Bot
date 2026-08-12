"""
Sentiment Factor  (factors/sentiment.py)
=========================================
Fear & Greed Index as a contrarian sentiment signal.
API: https://api.alternative.me/fng/  (completely free, no API key needed)

Contrarian interpretation:
  Extreme Fear  (0-20)  → LONG opportunity  → score: +0.8 to +1.0
  Fear          (21-40) → Mild LONG bias    → score: +0.2 to +0.6
  Neutral       (41-60) → No signal         → score:  0.0
  Greed         (61-80) → Mild SHORT bias   → score: -0.2 to -0.6
  Extreme Greed (81-100)→ Strong SHORT bias → score: -0.8 to -1.0

Trend modifier: if F&G shifted >15pts in last 7 days, add momentum bias.
Confidence is highest at extremes (where the contrarian edge is proven).

Score: -1.0 (extreme greed → SHORT) … +1.0 (extreme fear → LONG)
"""

import requests

FNG_URL = "https://api.alternative.me/fng/"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_sentiment_score() -> dict:
    """Fetch Fear & Greed Index and return a contrarian directional score."""
    try:
        resp = requests.get(FNG_URL, params={"limit": 7}, timeout=8)
        data = resp.json()

        if "data" not in data or not data["data"]:
            return _neutral("No data returned by alternative.me")

        entries   = data["data"]
        current   = entries[0]
        fng_value = int(current["value"])
        fng_label = current["value_classification"]

        # 7-day average to detect trend direction
        values  = [int(e["value"]) for e in entries]
        avg_7d  = sum(values) / len(values)
        trend   = fng_value - avg_7d   # positive → becoming greedier

        # Base contrarian score
        score = _fng_to_score(fng_value)

        # Trend modifier: rapid shift amplifies signal
        if trend > 15:    # rapidly becoming greedy → SHORT lean
            score -= 0.15
        elif trend < -15: # rapidly becoming fearful → LONG lean
            score += 0.15

        score = max(-1.0, min(1.0, score))

        # Confidence: extremes are more actionable
        if fng_value <= 15 or fng_value >= 85:
            confidence = 0.85
        elif fng_value <= 30 or fng_value >= 70:
            confidence = 0.65
        else:
            confidence = 0.40   # neutral zone — low predictive value

        return {
            "score":       round(score, 3),
            "confidence":  confidence,
            "block_trade": False,
            "details": {
                "fng_value":  fng_value,
                "fng_label":  fng_label,
                "7d_avg":     round(avg_7d, 1),
                "7d_trend":   round(trend, 1),
            },
        }
    except Exception as e:
        return _neutral(f"Exception: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fng_to_score(value: int) -> float:
    """Map F&G index (0-100) to contrarian score (-1.0 to +1.0)."""
    if value <= 20:
        return 0.8 + (20 - value) / 100       # 0.80 … 1.00
    elif value <= 40:
        return 0.2 + (40 - value) / 50        # 0.20 … 0.60
    elif value <= 60:
        return 0.0
    elif value <= 80:
        return -0.2 - (value - 60) / 50       # -0.20 … -0.60
    else:
        return -0.8 - (value - 80) / 100      # -0.80 … -1.00


def _neutral(reason=""):
    return {"score": 0.0, "confidence": 0.0, "block_trade": False,
            "details": {"reason": reason}}


# ---------------------------------------------------------------------------
# Backtest helper — fetch historical F&G (alternative.me supports up to 365d)
# ---------------------------------------------------------------------------

def fetch_historical_fng(days: int = 365) -> dict:
    """
    Returns a dict of {date_str: fng_value} for backtesting.
    date_str format: "YYYY-MM-DD"
    """
    try:
        resp = requests.get(FNG_URL, params={"limit": days, "date_format": "us"}, timeout=15)
        data = resp.json()
        result = {}
        for entry in data.get("data", []):
            # date comes as MM-DD-YYYY in us format
            parts = entry["timestamp"].split("-") if "-" in entry.get("timestamp", "") else []
            if len(parts) == 3:
                date_key = f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
            else:
                # fallback: use Unix timestamp
                from datetime import datetime, timezone
                dt = datetime.fromtimestamp(int(entry["timestamp"]), tz=timezone.utc)
                date_key = dt.strftime("%Y-%m-%d")
            result[date_key] = int(entry["value"])
        return result
    except Exception as e:
        print(f"⚠️ Could not fetch historical F&G: {e}")
        return {}
