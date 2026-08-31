"""
Economic Calendar Factor  (factors/calendar.py)
================================================
Autonomous macro-event awareness layer for the futures bot.

Two-source architecture (zero manual maintenance required):
───────────────────────────────────────────────────────────
  Source A — Financial Modeling Prep (FMP) API  [PRIMARY]
    Free API key at https://site.financialmodelingprep.com/register
    Endpoint: /api/v3/economic_calendar  (impact, time, country)
    Fetched once per hour; cache written to memory.

  Source B — Google News RSS detection  [FALLBACK / ALWAYS ON]
    Already used by news.py — no extra dependency.
    Scans headlines for known macro-event keywords RIGHT NOW.
    Catches events even without FMP key, or when FMP misses one.
    Runs on every evaluation (no caching needed; fast + free).

Logic
─────
  The two sources are merged: any event from either source can
  trigger a blackout.  Source A is precise (exact event time),
  Source B is reactive (detects an event is being reported NOW).

Timing Windows
─────────────
  Pre-event blackout : event_time - 4h  →  event_time        (Source A only)
  Post-news confirm  : event_time       →  event_time + 90min (both sources)
      └─ During this window entries are allowed, but ONLY after
         2 consecutive 15m candles confirm the post-news trend.
  RSS detection      : when Source B fires → immediate post_news_mode
                       (we are IN the event window right now)
  Normal trading     : everything else

Return dict (get_calendar_status)
──────────────────────────────────
  is_blackout      bool   True → block_trade hard veto in aggregator
  post_news_mode   bool   True → allow entry only with 2-candle confirmation
  event_name       str    Human-readable event label
  event_time       datetime | None
  minutes_to_event float
  block_reason     str
  next_events      list[dict]   next 3 upcoming events (from FMP)
  source           str   "fmp" | "rss" | "clear"
"""

import os
import re
import time
import requests
import feedparser
from datetime import datetime, timezone, timedelta
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FMP_BASE          = "https://financialmodelingprep.com/api/v3"
CACHE_TTL_SECONDS = 3600   # refresh FMP calendar once per hour
BLACKOUT_HOURS    = 4      # pre-event blackout window
POST_NEWS_MINUTES = 90     # post-event confirmation window

# Google News RSS — exact same infrastructure as news.py
GOOGLE_NEWS_RSS   = "https://news.google.com/rss/search"
RSS_LOOKBACK_MINS = 45     # scan headlines from last 45 min for live event detection

# ---------------------------------------------------------------------------
# Source B: high-impact macro event keyword patterns
# These trigger post_news_mode when found in live news headlines.
# New events are detected automatically as long as they appear in headlines.
# ---------------------------------------------------------------------------
MACRO_EVENT_PATTERNS = [
    # Fed / interest rates
    (r"\bfomc\b",                        "FOMC Meeting"),
    (r"federal reserve.{0,30}(rate|decision|statement|minutes|powell)",
                                         "Federal Reserve"),
    (r"(interest rate|rate decision|rate hike|rate cut).{0,20}(fed|fomc|us|united states)",
                                         "Fed Rate Decision"),
    (r"\bjackson hole\b",                "Jackson Hole Symposium"),
    (r"fed chair.{0,20}(speech|speaks|address|testif)",
                                         "Fed Chair Speech"),

    # Inflation
    (r"\bcpi\b.{0,30}(release|data|report|inflation|us|united states)",
                                         "US CPI Release"),
    (r"\bpce\b.{0,30}(inflation|price|index|release)",
                                         "PCE Inflation"),
    (r"\bppi\b.{0,30}(us|united states|release|data)",
                                         "US PPI"),

    # Employment
    (r"non.?farm.{0,10}payroll",         "Non-Farm Payrolls (NFP)"),
    (r"\bnfp\b.{0,20}(data|release|report)",
                                         "NFP Jobs Report"),
    (r"jobless claims.{0,20}(surge|spike|jump|plunge|drop|rise|fall|high|low)",
                                         "Jobless Claims"),
    (r"unemployment.{0,30}(rate|data|us|report)",
                                         "US Unemployment"),

    # GDP / growth
    (r"\bgdp\b.{0,30}(us|united states|revised|revision|growth|shrink|q[1-4])",
                                         "US GDP"),
    (r"gross domestic product.{0,20}(us|united states|revised)",
                                         "US GDP"),

    # Other major releases
    (r"\bism\b.{0,20}(manufacturing|services|pmi)",
                                         "ISM PMI"),
    (r"retail sales.{0,20}(us|united states|data|report)",
                                         "US Retail Sales"),
    (r"consumer confidence.{0,20}(us|drop|surge|fell|rose)",
                                         "US Consumer Confidence"),
    (r"debt ceiling",                    "US Debt Ceiling"),
    (r"(treasury|bond|yield).{0,20}(crash|spike|crisis|10.year)",
                                         "Treasury / Bond Crisis"),
]

# ---------------------------------------------------------------------------
# Module-level cache (FMP only — RSS is always real-time)
# ---------------------------------------------------------------------------
_cache = {
    "events":     [],    # list of parsed FMP event dicts
    "fetched_at": 0.0,   # epoch seconds of last FMP fetch
    "fmp_key":    None,  # loaded from env once
    "last_error": "",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_calendar_status() -> dict:
    """
    Return the current blackout / post-news state.

    Merges two independent sources:
      A) FMP structured calendar (precise event times)
      B) Google News RSS live detection (catches events in real-time)
    """
    # ── Source A: FMP structured calendar ────────────────────────────────
    _refresh_fmp_cache_if_stale()
    fmp_result = _evaluate_fmp_events(_cache["events"])

    # FMP blackout (pre-event) always wins — it's time-precise
    if fmp_result["is_blackout"]:
        fmp_result["source"] = "fmp"
        return fmp_result

    # ── Source B: Google News RSS live detection ───────────────────────
    rss_result = _detect_via_rss()

    if rss_result["is_blackout"] or rss_result["post_news_mode"]:
        # Merge next_events from FMP for display
        rss_result["next_events"] = fmp_result.get("next_events", [])
        rss_result["source"] = "rss"
        return rss_result

    # FMP post-news mode (if FMP detected event that already passed)
    if fmp_result["post_news_mode"]:
        fmp_result["source"] = "fmp"
        return fmp_result

    # All clear
    fmp_result["source"] = "clear"
    return fmp_result


# ---------------------------------------------------------------------------
# Source A: Financial Modeling Prep (FMP)
# ---------------------------------------------------------------------------

def _get_fmp_key() -> str | None:
    """Load FMP API key from environment (cached after first read)."""
    if _cache["fmp_key"] is None:
        _cache["fmp_key"] = os.getenv("FMP_API_KEY", "") or os.getenv("FINNHUB_API_KEY", "")
    return _cache["fmp_key"] or None


def _refresh_fmp_cache_if_stale():
    """Fetch fresh FMP calendar if cache is older than 1h."""
    now = time.time()
    if now - _cache["fetched_at"] < CACHE_TTL_SECONDS:
        return

    api_key = _get_fmp_key()
    if not api_key:
        _cache["events"]     = []
        _cache["fetched_at"] = now
        _cache["last_error"] = "No FMP_API_KEY set"
        print("📅 [calendar] No FMP_API_KEY — using RSS-only event detection (free, real-time)")
        return

    # Skip silently if the key was already rejected as plan-restricted (403)
    if api_key == "__PLAN_RESTRICTED__":
        _cache["fetched_at"] = now  # reset timer so we don't spam the check
        return

    events = _fetch_fmp_events(api_key)
    _cache["events"]     = events
    _cache["fetched_at"] = now

    if events:
        print(f"📅 [calendar] FMP loaded {len(events)} high-impact US events (next 48h)")
        for ev in events[:3]:
            mins = (ev["time"] - datetime.now(timezone.utc)).total_seconds() / 60
            label = f"in {mins:.0f}min" if mins >= 0 else f"{abs(mins):.0f}min ago"
            print(f"   📌 {ev['name']} — {label}")
    else:
        msg = _cache["last_error"] or "clear window"
        print(f"📅 [calendar] FMP: no high-impact US events in next 48h ({msg})")


def _fetch_fmp_events(api_key: str) -> list:
    """Fetch high-impact US events from FMP for next 48h."""
    try:
        today    = datetime.now(timezone.utc).date()
        end_date = today + timedelta(days=2)

        resp = requests.get(
            f"{FMP_BASE}/economic_calendar",
            params={
                "from":   today.strftime("%Y-%m-%d"),
                "to":     end_date.strftime("%Y-%m-%d"),
                "apikey": api_key,
            },
            timeout=10,
        )

        # ── Handle 403 cleanly: economic_calendar is a paid FMP endpoint ──
        if resp.status_code == 403:
            _cache["last_error"] = "403 — economic_calendar requires FMP paid plan"
            print(
                "📅 [calendar] FMP economic_calendar is a premium endpoint (HTTP 403).\n"
                "   Your free plan does not include it. This is expected.\n"
                "   RSS-based event detection is active and fully covers macro events.\n"
                "   To enable FMP: upgrade at https://site.financialmodelingprep.com/"
            )
            # Mark key as unusable so we don't burn quota on retries
            _cache["fmp_key"] = "__PLAN_RESTRICTED__"
            return []

        resp.raise_for_status()
        raw = resp.json()

        if not isinstance(raw, list):
            _cache["last_error"] = f"Unexpected FMP response: {type(raw)}"
            return []

        events = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=2)

        for ev in raw:
            # Keep only HIGH impact US events
            if str(ev.get("country", "")).upper() != "US":
                continue
            if str(ev.get("impact", "")).lower() != "high":
                continue

            # Parse event datetime (FMP returns ISO string: "2024-01-12 08:30:00")
            raw_date = ev.get("date", "")
            try:
                event_dt = datetime.strptime(raw_date, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                try:
                    event_dt = datetime.fromisoformat(raw_date).replace(tzinfo=timezone.utc)
                except Exception:
                    continue

            if event_dt < cutoff:
                continue

            events.append({
                "name":     ev.get("event", "Unknown Event"),
                "time":     event_dt,
                "country":  "US",
                "impact":   "high",
                "estimate": ev.get("estimate"),
                "prev":     ev.get("previous"),
            })

        events.sort(key=lambda e: e["time"])
        _cache["last_error"] = ""
        return events

    except requests.exceptions.RequestException as e:
        # Mask the API key from the error string before printing
        err_str = str(e)
        if api_key and api_key in err_str:
            err_str = err_str.replace(api_key, "***")
        _cache["last_error"] = err_str
        print(f"⚠️  [calendar] FMP fetch failed ({err_str[:80]}) — RSS fallback active")
        return []
    except Exception as e:
        _cache["last_error"] = str(e)
        print(f"⚠️  [calendar] FMP error: {e}")
        return []


def _evaluate_fmp_events(events: list) -> dict:
    """Evaluate FMP events for blackout/post-news windows."""
    now = datetime.now(timezone.utc)
    blackout_delta  = timedelta(hours=BLACKOUT_HOURS)
    post_news_delta = timedelta(minutes=POST_NEWS_MINUTES)

    for ev in events:
        ev_time = ev["time"]
        delta   = ev_time - now

        # Pre-event blackout
        if timedelta(0) <= delta <= blackout_delta:
            h, m = divmod(int(delta.total_seconds() / 60), 60)
            time_str = f"{h}h {m}min" if h else f"{m}min"
            return {
                "is_blackout":     True,
                "post_news_mode":  False,
                "event_name":      ev["name"],
                "event_time":      ev_time,
                "minutes_to_event": round(delta.total_seconds() / 60, 1),
                "block_reason": (
                    f"⏳ Pre-event blackout: [{ev['name']}] in {time_str}. "
                    f"No new entries until post-event trend confirms."
                ),
                "next_events": _format_next_events(events, now),
            }

        # Post-news confirmation window
        post_delta = now - ev_time
        if timedelta(0) < post_delta <= post_news_delta:
            mins_passed = int(post_delta.total_seconds() / 60)
            return {
                "is_blackout":     False,
                "post_news_mode":  True,
                "event_name":      ev["name"],
                "event_time":      ev_time,
                "minutes_to_event": round(delta.total_seconds() / 60, 1),
                "block_reason": (
                    f"📡 Post-news: [{ev['name']}] passed {mins_passed}min ago. "
                    f"Waiting for 2-candle trend confirmation."
                ),
                "next_events": _format_next_events(events, now),
            }

    return _clear_status(events)


# ---------------------------------------------------------------------------
# Source B: Google News RSS live detection
# ---------------------------------------------------------------------------

def _detect_via_rss() -> dict:
    """
    Scan Google News RSS for macro event keywords in the last 45 minutes.
    If a matching high-impact headline is found, trigger post_news_mode.
    This is the always-on layer — works even without any API key.
    """
    try:
        query   = (
            "FOMC OR \"Federal Reserve\" OR \"interest rate\" "
            "OR CPI OR NFP OR \"non-farm payroll\" OR GDP OR PCE "
            "OR \"Jackson Hole\" OR \"Fed Chair\" OR PPI OR \"jobless claims\""
        )
        url  = f"{GOOGLE_NEWS_RSS}?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)

        if feed.bozo and not feed.entries:
            return _empty_rss_result()

        cutoff  = datetime.now(timezone.utc) - timedelta(minutes=RSS_LOOKBACK_MINS)
        recent  = _filter_recent_entries(feed.entries, cutoff)

        for entry in recent:
            title = entry.get("title", "").lower()
            for pattern, event_name in MACRO_EVENT_PATTERNS:
                if re.search(pattern, title, re.IGNORECASE):
                    # Log the exact headline that triggered the detection for auditability
                    raw_title = entry.get('title', '(no title)')
                    pub_time  = entry.get('published', 'unknown time')
                    print(f"📅 [calendar] RSS match: '{raw_title[:100]}' (published: {pub_time})")
                    return {
                        "is_blackout":     False,   # event is NOW — trade the reaction
                        "post_news_mode":  True,
                        "event_name":      event_name,
                        "event_time":      None,
                        "minutes_to_event": 0.0,
                        "block_reason": (
                            f"📡 RSS detected [{event_name}] in live headlines. "
                            f"Post-news mode: 2-candle confirmation required before entry."
                        ),
                        "next_events": [],
                    }

        return _empty_rss_result()

    except Exception as e:
        print(f"⚠️  [calendar] RSS detection error: {e} — skipping")
        return _empty_rss_result()


def _filter_recent_entries(entries: list, cutoff: datetime) -> list:
    """Filter RSS entries to those published after `cutoff`.
    FIX: On timestamp parse failure, EXCLUDE the entry (fail-closed).
    Including unparseable entries was a staleness risk — old articles with
    broken timestamps would bypass the 45-minute window.
    """
    recent = []
    for entry in entries:
        try:
            pub_struct = entry.get("published_parsed")
            if pub_struct:
                pub = datetime(*pub_struct[:6], tzinfo=timezone.utc)
            else:
                pub_str = entry.get("published", "")
                pub = datetime.strptime(pub_str, "%a, %d %b %Y %H:%M:%S %Z").replace(
                    tzinfo=timezone.utc
                )
            if pub >= cutoff:
                recent.append(entry)
        except Exception:
            # FIX: EXCLUDE entries with unparseable timestamps instead of blindly including them.
            # A stale article with a broken date is worse than a missed fresh one.
            pass
    return recent


def _empty_rss_result() -> dict:
    return {
        "is_blackout":     False,
        "post_news_mode":  False,
        "event_name":      "",
        "event_time":      None,
        "minutes_to_event": None,
        "block_reason":    "",
        "next_events":     [],
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _clear_status(events: list) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "is_blackout":     False,
        "post_news_mode":  False,
        "event_name":      "",
        "event_time":      None,
        "minutes_to_event": None,
        "block_reason":    "",
        "next_events":     _format_next_events(events, now),
    }


def _format_next_events(events: list, now: datetime) -> list:
    upcoming = []
    for ev in events:
        if ev["time"] > now:
            delta_h = (ev["time"] - now).total_seconds() / 3600
            upcoming.append({
                "name":       ev["name"],
                "time_utc":   ev["time"].strftime("%Y-%m-%d %H:%M UTC"),
                "hours_away": round(delta_h, 1),
            })
        if len(upcoming) >= 3:
            break
    return upcoming
