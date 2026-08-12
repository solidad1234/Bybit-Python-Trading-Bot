"""
News Factor  (factors/news.py)
================================
Monitors significant crypto news via CryptoPanic's public API (no auth token
required for the free tier, though adding one increases the rate limit).

This factor acts primarily as a TRADE BLOCKER rather than a directional signal:
  - Breaking negative news for SOL/BTC  → suppress LONG entries
  - Breaking positive news               → suppress SHORT entries
  - Major news in last 2 hours           → block_trade = True (hard veto)

When not blocking, the net sentiment of recent headlines provides a small
directional score boost.

Score: -1.0 (strong negative news) … +1.0 (strong positive news)
CryptoPanic API docs: https://cryptopanic.com/developers/api/
"""

import requests
from datetime import datetime, timezone, timedelta

CRYPTOPANIC_URL  = "https://cryptopanic.com/api/v1/posts/"
CRYPTOPANIC_TOKEN = ""          # Optional: paste your free-tier token here
LOOKBACK_HOURS    = 4           # Only consider news from last N hours
BLOCK_THRESHOLD   = 0.6         # |score| above this → hard block_trade = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_news_score(currencies: str = "SOL,BTC") -> dict:
    """
    Fetch recent important news and return a sentiment score + optional veto.
    """
    try:
        params = {
            "currencies": currencies,
            "kind":       "news",
            "filter":     "important",
            "public":     "true",
        }
        if CRYPTOPANIC_TOKEN:
            params["auth_token"] = CRYPTOPANIC_TOKEN

        resp = requests.get(CRYPTOPANIC_URL, params=params, timeout=10)

        if resp.status_code == 429:
            return _neutral("CryptoPanic rate-limited — treating as neutral")
        if resp.status_code != 200:
            return _neutral(f"HTTP {resp.status_code}")

        results = resp.json().get("results", [])
        if not results:
            return _neutral("No important news found")

        # Filter to LOOKBACK_HOURS window
        cutoff  = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
        recent  = _filter_recent(results, cutoff)

        if not recent:
            return _neutral(f"No news in last {LOOKBACK_HOURS}h")

        # Score each article
        score, headline_log = _score_articles(recent)
        score = max(-1.0, min(1.0, score))

        # Hard block when news is decisively one-sided
        block = abs(score) >= BLOCK_THRESHOLD
        block_reason = ""
        if block:
            direction = "negative" if score < 0 else "positive"
            block_reason = f"Strong {direction} news detected (score={score:.2f})"

        # Confidence scales with number of recent articles
        confidence = min(0.90, 0.30 + len(recent) * 0.10)

        return {
            "score":        round(score, 3),
            "confidence":   round(confidence, 2),
            "block_trade":  block,
            "block_reason": block_reason,
            "details": {
                "articles_found":    len(recent),
                "lookback_hours":    LOOKBACK_HOURS,
                "top_headlines":     headline_log[:3],
            },
        }
    except Exception as e:
        return _neutral(f"Exception: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_recent(results: list, cutoff: datetime) -> list:
    recent = []
    for item in results:
        try:
            pub_str = item.get("published_at", "")
            pub     = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            if pub >= cutoff:
                recent.append(item)
        except Exception:
            continue
    return recent


def _score_articles(articles: list):
    """
    CryptoPanic includes vote counts (positive / negative / important).
    Use them to derive a net sentiment score per article.
    """
    net = 0.0
    log = []

    for article in articles:
        votes    = article.get("votes", {})
        pos      = int(votes.get("positive", 0))
        neg      = int(votes.get("negative", 0))
        total    = pos + neg
        if total == 0:
            # No votes yet — treat as mildly negative (unknown risk)
            article_score = -0.1
        else:
            article_score = (pos - neg) / total   # -1 to +1

        # Weight more recent articles higher (simple: equal weight here)
        net += article_score
        log.append({
            "title":  article.get("title", "")[:80],
            "score":  round(article_score, 2),
            "votes":  {"pos": pos, "neg": neg},
        })

    # Normalise by article count
    final = net / len(articles) if articles else 0.0
    return final, log


def _neutral(reason=""):
    return {
        "score":        0.0,
        "confidence":   0.0,
        "block_trade":  False,
        "block_reason": "",
        "details":      {"reason": reason},
    }
