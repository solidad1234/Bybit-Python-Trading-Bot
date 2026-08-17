"""
News Factor  (factors/news.py)
================================
Two-layer news scoring for multi-asset trading using Google News RSS.
Completely FREE — no API key, no rate limits under normal usage.

  Layer 1 — BTC Macro (40% weight):
    BTC / broad-crypto headlines (ETF approvals, exchange hacks, regulatory bans).
    Affects ALL traded coins.

  Layer 2 — Coin-Specific (60% weight):
    Headlines targeting the specific coin (SOL, ETH, AVAX, LINK, BNB).
    Affects ONLY that coin's trade decision.

Composite:  score = btc_macro_score × 0.40 + coin_score × 0.60

Asymmetric Block Logic (preserves SHORT opportunity on bad coin news):
  Coin score  ≤ -BLOCK_THRESHOLD                → block LONG only (SHORT allowed)
  BTC macro   ≤ -BLOCK_THRESHOLD AND coin < 0   → hard-block BOTH directions
  BTC macro   ≤ -BLOCK_THRESHOLD AND coin ≥ 0   → block LONG only
  Coin score  ≥ +BLOCK_THRESHOLD                → score boost, no block

Sentiment scoring method:
  Since we have no vote-counts (unlike CryptoPanic), we score via keyword
  matching on the article title. Each hit adds ±KEYWORD_WEIGHT to the
  article score, capped at [-1, +1]. The per-article scores are averaged
  and then the lookback window is applied.
"""

import feedparser
import re
from datetime import datetime, timezone, timedelta
from urllib.parse import quote_plus

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GOOGLE_NEWS_RSS  = "https://news.google.com/rss/search"
LOOKBACK_HOURS   = 6          # wider window since RSS is slower than live API
BLOCK_THRESHOLD  = 0.55       # |score| above this triggers a block flag
KEYWORD_WEIGHT   = 0.25       # score contribution per keyword hit
MAX_ARTICLES     = 15         # cap to avoid overwhelming keyword scorer

MACRO_WEIGHT = 0.40
COIN_WEIGHT  = 0.60

# Map Bybit linear symbol → search term(s) for Google News
SYMBOL_TO_QUERY = {
    "SOLUSDT":  "Solana SOL crypto",
    "ETHUSDT":  "Ethereum ETH crypto",
    "AVAXUSDT": "Avalanche AVAX crypto",
    "LINKUSDT": "Chainlink LINK crypto",
    "BNBUSDT":  "BNB Binance crypto",
}

MACRO_QUERY = "Bitcoin BTC crypto market"

# Keyword dictionaries — tuned for crypto news
POSITIVE_KEYWORDS = {
    # Institutional / adoption
    "etf", "approval", "approved", "launch", "launches", "listed", "listing",
    "adoption", "institutional", "mainstream", "legal", "regulated",
    "partnership", "integrate", "integrated", "expansion",
    # Price action language
    "surge", "surges", "surging", "rally", "rallies", "bullish", "bull",
    "record", "high", "gains", "gain", "pumps", "pump", "breakout",
    "recover", "recovery", "rebounds", "rebound", "uptrend",
    # Ecosystem / tech
    "upgrade", "upgraded", "update", "milestone", "breakthrough",
    "staking", "yield", "reward", "growth", "positive",
}

NEGATIVE_KEYWORDS = {
    # Security events
    "hack", "hacked", "exploit", "exploited", "vulnerability", "breach",
    "stolen", "theft", "rug", "scam", "fraud", "phishing",
    # Price action language
    "crash", "crashes", "crashing", "dump", "dumps", "dumping", "bear",
    "bearish", "correction", "selloff", "sell-off", "liquidation",
    "drop", "drops", "fell", "plunge", "plunges", "decline", "declining",
    "downtrend", "fear",
    # Regulatory / legal
    "ban", "banned", "restriction", "restrict", "sec", "lawsuit", "sued",
    "fine", "penalty", "warning", "crackdown", "probe", "investigation",
    "illegal", "shutdown", "collapse",
    # Technical failures
    "outage", "down", "offline", "bug", "issue", "delay", "congestion",
    "halt", "halted", "suspend", "suspended",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_news_score(symbol: str = "SOLUSDT") -> dict:
    """
    Fetch Google News RSS headlines and return a two-layer composite score.

    Parameters
    ----------
    symbol : str  Bybit linear symbol, e.g. "ETHUSDT"

    Returns
    -------
    dict with keys:
        score           float   -1.0 … +1.0  (composite)
        confidence      float   0.0 … 0.85
        block_trade     bool    True = hard veto on BOTH directions
        block_long_only bool    True = LONG blocked, SHORT allowed
        block_reason    str
        details         dict
    """
    coin_query = SYMBOL_TO_QUERY.get(symbol.upper(),
                                     symbol.replace("USDT", "") + " crypto")
    coin_code  = symbol.replace("USDT", "")

    # Layer 1: BTC macro
    macro_result = _fetch_and_score(MACRO_QUERY,  label="BTC-macro")

    # Layer 2: Coin-specific
    if coin_code.upper() == "BTC":
        coin_result = macro_result
        mw, cw = 1.0, 0.0
    else:
        coin_result = _fetch_and_score(coin_query, label=coin_code)
        mw, cw = MACRO_WEIGHT, COIN_WEIGHT

    macro_score = macro_result["score"]
    coin_score  = coin_result["score"]
    composite   = max(-1.0, min(1.0, macro_score * mw + coin_score * cw))

    # --- Asymmetric block logic ---
    block_trade     = False
    block_long_only = False
    block_reason    = ""

    if coin_score <= -BLOCK_THRESHOLD:
        block_long_only = True
        block_reason = (
            f"Negative {coin_code} news (score={coin_score:.2f}) "
            f"— LONG blocked, SHORT allowed to capitalise"
        )
    elif macro_score <= -BLOCK_THRESHOLD and coin_score < 0:
        block_trade  = True
        block_reason = (
            f"BTC macro crash (score={macro_score:.2f}) + "
            f"{coin_code} negative (score={coin_score:.2f}) — all trades blocked"
        )
    elif macro_score <= -BLOCK_THRESHOLD and coin_score >= 0:
        block_long_only = True
        block_reason = (
            f"BTC macro crash (score={macro_score:.2f}) with neutral/positive "
            f"{coin_code} news — LONG blocked, SHORT allowed"
        )

    macro_count    = macro_result["details"].get("articles_found", 0)
    coin_count     = coin_result["details"].get("articles_found", 0)
    total_articles = macro_count + coin_count
    confidence     = min(0.85, 0.25 + total_articles * 0.04)

    return {
        "score":           round(composite, 3),
        "confidence":      round(confidence, 2),
        "block_trade":     block_trade,
        "block_long_only": block_long_only,
        "block_reason":    block_reason,
        "details": {
            "macro_score":     round(macro_score, 3),
            "coin_score":      round(coin_score, 3),
            "coin_code":       coin_code,
            "macro_articles":  macro_count,
            "coin_articles":   coin_count,
            "lookback_hours":  LOOKBACK_HOURS,
            "macro_headlines": macro_result["details"].get("top_headlines", [])[:2],
            "coin_headlines":  coin_result["details"].get("top_headlines", [])[:2],
            "sentiment_label": _label(composite),
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_and_score(query: str, label: str = "") -> dict:
    """Fetch Google News RSS for `query`, score by keyword analysis."""
    try:
        url  = f"{GOOGLE_NEWS_RSS}?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)

        if feed.bozo and not feed.entries:
            return _raw_neutral(f"RSS parse error for '{label}'")

        cutoff  = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
        recent  = _filter_recent(feed.entries, cutoff)

        if not recent:
            return _raw_neutral(f"No news in last {LOOKBACK_HOURS}h for '{label}'")

        recent = recent[:MAX_ARTICLES]
        score, headlines = _score_by_keywords(recent)
        score = max(-1.0, min(1.0, score))

        return {
            "score": score,
            "details": {
                "articles_found": len(recent),
                "top_headlines":  headlines[:3],
            },
        }
    except Exception as e:
        return _raw_neutral(f"Exception for '{label}': {e}")


def _filter_recent(entries: list, cutoff: datetime) -> list:
    recent = []
    for entry in entries:
        try:
            # feedparser normalises published_parsed to a time.struct_time in UTC
            pub_struct = entry.get("published_parsed")
            if pub_struct:
                pub = datetime(*pub_struct[:6], tzinfo=timezone.utc)
            else:
                # Fallback: parse published string if struct not available
                pub_str = entry.get("published", "")
                pub     = datetime.strptime(pub_str, "%a, %d %b %Y %H:%M:%S %Z").replace(
                    tzinfo=timezone.utc
                )
            if pub >= cutoff:
                recent.append(entry)
        except Exception:
            recent.append(entry)   # include if date parse fails — better than missing news
    return recent


def _score_by_keywords(entries: list):
    """
    Score articles by positive/negative keyword hits in title.
    Returns (average_score, headline_log).
    """
    net = 0.0
    log = []

    for entry in entries:
        title  = entry.get("title", "")
        words  = set(re.findall(r"[a-z]+", title.lower()))

        pos_hits = words & POSITIVE_KEYWORDS
        neg_hits = words & NEGATIVE_KEYWORDS

        article_score = (len(pos_hits) - len(neg_hits)) * KEYWORD_WEIGHT
        article_score = max(-1.0, min(1.0, article_score))

        net += article_score
        log.append({
            "title":   title[:80],
            "score":   round(article_score, 2),
            "pos":     list(pos_hits)[:3],
            "neg":     list(neg_hits)[:3],
        })

    final = net / len(entries) if entries else 0.0
    return final, log


def _label(score: float) -> str:
    if score >=  0.5: return "Strongly Bullish"
    if score >=  0.2: return "Bullish"
    if score >= -0.2: return "Neutral"
    if score >= -0.5: return "Bearish"
    return "Strongly Bearish"


def _raw_neutral(reason: str = "") -> dict:
    return {
        "score": 0.0,
        "details": {"articles_found": 0, "top_headlines": [], "reason": reason},
    }
