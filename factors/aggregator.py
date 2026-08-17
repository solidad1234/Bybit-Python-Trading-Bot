"""
Multi-Factor Signal Aggregator  (factors/aggregator.py)
========================================================
Combines TA + Regime + Derivatives + Sentiment + News into a single
weighted consensus score.

Weights (must sum to 1.0):
  regime       0.30  — macro BTC trend context (most reliable crash indicator)
  derivatives  0.25  — funding + OI + L/S ratio (real money on the line)
  technical    0.20  — RSI/MACD/ADX scoring from existing system
  sentiment    0.15  — contrarian F&G index (1-hour TTL cache)
  news         0.10  — two-layer macro+coin blocker/nudge

Key behaviours:
  - block_trade      = True  → veto both directions
  - block_long_only  = True  → LONG blocked, SHORT allowed (news bad-coin logic)
  - Regime bearish ≤ -0.4   → LONG threshold raised from 0.25 to 0.40

Usage in futures.py:
    from factors.aggregator import MultiFactorAggregator
    agg = MultiFactorAggregator()
    # precomputed contains pre-fetched regime (and optionally sentiment)
    # to avoid redundant API calls across the symbol scanner loop.
    consensus = agg.evaluate(ta_signal, symbol, current_price,
                             precomputed={"regime": regime_info})
"""

import time

from factors.regime      import get_regime_score
from factors.derivatives import get_derivatives_score
from factors.sentiment   import get_sentiment_score
from factors.news        import get_news_score

WEIGHTS = {
    "regime":      0.30,
    "derivatives": 0.25,
    "technical":   0.20,
    "sentiment":   0.15,
    "news":        0.10,
}

LONG_ENTRY_THRESHOLD          = 0.25
SHORT_ENTRY_THRESHOLD         = 0.15
LONG_THRESHOLD_BEARISH_REGIME = 0.40   # elevated when regime_score <= -0.4

# ---------------------------------------------------------------------------
# 1-hour sentiment TTL cache (F&G updates once per day — no need to fetch 5x)
# ---------------------------------------------------------------------------
_sentiment_cache = {"result": None, "fetched_at": 0.0}
_SENTIMENT_TTL   = 3600  # seconds


class MultiFactorAggregator:
    """
    Collect all factor scores and emit a final consensus signal.
    Stateless: safe to instantiate once and reuse across cycles.
    """

    def evaluate(
        self,
        ta_signal:     dict,
        symbol:        str   = "SOLUSDT",
        current_price: float = 0.0,
        precomputed:   dict  = None,
    ) -> dict:
        """
        Parameters
        ----------
        ta_signal    : dict  Output of calculate_futures_signals()
                             Must have keys: 'signal' (str|None), 'strength' (int)
        symbol       : str   Bybit linear symbol, e.g. "ETHUSDT"
        current_price: float Current mark price
        precomputed  : dict  Pre-fetched factor results to avoid redundant API
                             calls when scanning multiple symbols in a loop.
                             Supported keys: "regime", "sentiment"

        Returns
        -------
        dict with:
            signal         : "LONG" | "SHORT" | None
            final_score    : float (-1.0 to +1.0)
            block_trade    : bool   (hard veto — both directions)
            block_long_only: bool   (LONG blocked, SHORT allowed)
            block_reason   : str
            factor_scores  : dict   (per-factor detail for logging)
            elapsed_s      : float
        """
        print("\n🔬 Running multi-factor evaluation...")
        t0 = time.time()
        precomputed = precomputed or {}

        factor_scores = {}

        # 1. Technical Analysis (mapped from existing TA signal)
        factor_scores["technical"] = self._ta_to_score(ta_signal)

        # 2. Regime — use precomputed if caller hoisted it outside the loop
        if "regime" in precomputed:
            factor_scores["regime"] = precomputed["regime"]
        else:
            try:
                factor_scores["regime"] = get_regime_score()
            except Exception as e:
                factor_scores["regime"] = {
                    "score": 0.0, "confidence": 0.0,
                    "block_trade": False, "details": {"err": str(e)},
                }

        # 3. Derivatives — always symbol-specific
        try:
            factor_scores["derivatives"] = get_derivatives_score(symbol)
        except Exception as e:
            factor_scores["derivatives"] = {
                "score": 0.0, "confidence": 0.0,
                "block_trade": False, "details": {"err": str(e)},
            }

        # 4. Sentiment — use precomputed or 1-hour TTL cache
        if "sentiment" in precomputed:
            factor_scores["sentiment"] = precomputed["sentiment"]
        else:
            factor_scores["sentiment"] = _get_cached_sentiment()

        # 5. News — two-layer BTC macro + coin-specific composite
        try:
            factor_scores["news"] = get_news_score(symbol)
        except Exception as e:
            factor_scores["news"] = {
                "score": 0.0, "confidence": 0.0,
                "block_trade": False, "block_long_only": False,
                "block_reason": "", "details": {"err": str(e)},
            }

        # --- Hard vetoes (block_trade = both directions) ---
        block_trade     = False
        block_long_only = False
        block_reason    = ""

        for name, fs in factor_scores.items():
            if fs.get("block_trade", False):
                block_trade  = True
                block_reason = f"{name}: {fs.get('block_reason', 'hard veto')}"
                break

        # News: block_long_only does NOT block shorts
        if not block_trade:
            news_fs = factor_scores.get("news", {})
            if news_fs.get("block_long_only", False):
                block_long_only = True
                block_reason    = news_fs.get("block_reason", "news: LONG blocked")

        # --- Weighted composite score ---
        final_score = 0.0
        for name, fs in factor_scores.items():
            w          = WEIGHTS.get(name, 0.0)
            score      = fs.get("score", 0.0)
            confidence = fs.get("confidence", 1.0)
            final_score += w * score * confidence

        final_score = max(-1.0, min(1.0, final_score))

        # --- Determine consensus signal ---
        ta_direction = ta_signal.get("signal")   # "LONG" | "SHORT" | None

        if block_trade:
            consensus_signal = None
        else:
            regime_s       = factor_scores.get("regime", {}).get("score", 0.0)
            long_threshold = (LONG_THRESHOLD_BEARISH_REGIME
                              if regime_s <= -0.4 else LONG_ENTRY_THRESHOLD)

            if ta_direction == "LONG" and final_score >= long_threshold and not block_long_only:
                consensus_signal = "LONG"
                if long_threshold > LONG_ENTRY_THRESHOLD:
                    print(f"   ⚠️  Bearish regime: elevated LONG threshold {long_threshold:.2f} applied")
            elif ta_direction == "SHORT" and final_score <= -SHORT_ENTRY_THRESHOLD:
                consensus_signal = "SHORT"
            else:
                consensus_signal = None

            if block_long_only and ta_direction == "LONG":
                consensus_signal = None
                print(f"   🚫 LONG blocked by news: {block_reason}")

        elapsed = time.time() - t0
        self._print_summary(
            factor_scores, final_score, consensus_signal,
            block_trade, block_long_only, block_reason, elapsed,
        )

        return {
            "signal":          consensus_signal,
            "final_score":     round(final_score, 3),
            "block_trade":     block_trade,
            "block_long_only": block_long_only,
            "block_reason":    block_reason,
            "factor_scores":   factor_scores,
            "elapsed_s":       round(elapsed, 1),
        }

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _ta_to_score(ta_signal: dict) -> dict:
        """Map existing TA signal strength to standard [-1, +1] factor format."""
        direction = ta_signal.get("signal")
        strength  = ta_signal.get("strength", 0)

        if direction == "LONG":
            score = min(1.0, (strength - 4) / 3.0)   # 5=0.33, 6=0.67, 7=1.0
        elif direction == "SHORT":
            score = -min(1.0, (strength - 5) / 5.0)  # 6=-0.2, 8=-0.6, 10=-1.0
        else:
            score = 0.0

        return {
            "score":       round(score, 3),
            "confidence":  0.85,
            "block_trade": False,
            "details": {"ta_direction": direction, "ta_strength": strength},
        }

    @staticmethod
    def _print_summary(factor_scores, final_score, signal,
                       blocked, block_long_only, block_reason, elapsed):
        bar = "=" * 55
        print(f"\n{bar}")
        print("  MULTI-FACTOR CONSENSUS")
        print(bar)
        for name, fs in factor_scores.items():
            s        = fs.get("score", 0.0)
            c        = fs.get("confidence", 0.0)
            w        = WEIGHTS.get(name, 0.0)
            arrow    = "▲" if s > 0 else ("▼" if s < 0 else "─")
            flag     = "  🚫LONG-ONLY" if (name == "news" and fs.get("block_long_only")) else ""
            print(f"  {name:<12} score={s:+.3f}  conf={c:.2f}  wt={w:.2f}  {arrow}{flag}")
        print(f"  {'─'*51}")
        print(f"  Final score : {final_score:+.3f}")
        if blocked:
            print(f"  🚫 HARD BLOCK: {block_reason}")
        elif block_long_only:
            print(f"  ⚠️  LONG BLOCKED: {block_reason}")
        else:
            print(f"  ✅ Signal    : {signal or 'NONE (below threshold)'}")
        print(f"  ⏱  Elapsed   : {elapsed:.1f}s")
        print(bar)


# ---------------------------------------------------------------------------
# Sentiment cache
# ---------------------------------------------------------------------------

def _get_cached_sentiment() -> dict:
    """Return cached F&G result if < 1h old, else fetch fresh."""
    now = time.time()
    if (_sentiment_cache["result"] is not None
            and (now - _sentiment_cache["fetched_at"]) < _SENTIMENT_TTL):
        return _sentiment_cache["result"]
    try:
        result = get_sentiment_score()
    except Exception as e:
        result = {
            "score": 0.0, "confidence": 0.0,
            "block_trade": False, "details": {"err": str(e)},
        }
    _sentiment_cache["result"]     = result
    _sentiment_cache["fetched_at"] = now
    return result
