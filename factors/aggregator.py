"""
Multi-Factor Signal Aggregator  (factors/aggregator.py)
========================================================
Combines TA + Regime + Derivatives + Sentiment + News into a single
weighted consensus score using Option-A rule-based aggregation.

Weights (must sum to 1.0):
  derivatives  0.30  — real money on the line; most reliable in crypto
  technical    0.25  — existing RSI/MACD/ADX scoring system
  sentiment    0.20  — contrarian F&G index
  regime       0.15  — macro BTC trend context
  news         0.10  — mostly a blocker, small directional nudge

Entry threshold:
  LONG  requires final_score >= +ENTRY_THRESHOLD (default 0.25)
  SHORT requires final_score <= -ENTRY_THRESHOLD (default 0.25)
  Any factor with block_trade=True vetoes the trade entirely.

Usage in futures.py:
    from factors.aggregator import MultiFactorAggregator
    agg = MultiFactorAggregator()
    consensus = agg.evaluate(ta_signal, symbol, current_price)
    # consensus keys: signal, final_score, block_trade, block_reason, factor_scores
"""

import time
from factors.regime      import get_regime_score
from factors.derivatives import get_derivatives_score
from factors.sentiment   import get_sentiment_score
from factors.news        import get_news_score

# Rebalanced weights: regime raised 0.15→0.30 (macro is the most reliable
# crash indicator); sentiment reduced 0.20→0.15 (contrarian F&G was
# fighting regime signal during rapid macro downturns).
# Rule: no weight changed to 'fix' a known event — only principled rebalancing.
WEIGHTS = {
    "regime":      0.30,   # ↑ macro direction dominates
    "derivatives": 0.25,   # funding + OI + L/S ratio — real money signal
    "technical":   0.20,   # RSI/MACD/ADX from existing system
    "sentiment":   0.15,   # contrarian F&G (bias reduced)
    "news":        0.10,   # mostly a blocker
}

# Asymmetric thresholds: LONG needs stronger confirmation than SHORT because
# the sentiment factor (F&G contrarian = long bias when fearful) structurally
# pushes scores toward positive, making it harder to reach -THRESHOLD for shorts.
LONG_ENTRY_THRESHOLD  = 0.25
SHORT_ENTRY_THRESHOLD = 0.15
# When macro regime is strongly bearish, LONG threshold escalates:
LONG_THRESHOLD_BEARISH_REGIME = 0.40  # requires much stronger multi-factor confirmation


class MultiFactorAggregator:
    """
    Collect all factor scores and emit a final consensus signal.
    Thread-safe: can be instantiated once and reused across cycles.
    """

    def evaluate(self, ta_signal: dict, symbol: str = "SOLUSDT",
                 current_price: float = 0.0) -> dict:
        """
        Parameters
        ----------
        ta_signal : dict
            Output of calculate_futures_signals() — must have keys
            'signal' (str|None), 'strength' (int).
        symbol : str
        current_price : float

        Returns
        -------
        dict with:
            signal       : "LONG" | "SHORT" | None
            final_score  : float (-1.0 to +1.0)
            block_trade  : bool
            block_reason : str
            factor_scores: dict   (per-factor detail for logging)
        """
        print("\n🔬 Running multi-factor evaluation...")
        t0 = time.time()

        # --- Collect all factor scores ---
        factor_scores = {}

        # 1. Technical Analysis (mapped from existing signal)
        factor_scores["technical"] = self._ta_to_score(ta_signal)

        # 2. Regime
        try:
            factor_scores["regime"] = get_regime_score()
        except Exception as e:
            factor_scores["regime"] = {"score": 0.0, "confidence": 0.0,
                                        "block_trade": False, "details": {"err": str(e)}}

        # 3. Derivatives
        try:
            factor_scores["derivatives"] = get_derivatives_score(symbol)
        except Exception as e:
            factor_scores["derivatives"] = {"score": 0.0, "confidence": 0.0,
                                             "block_trade": False, "details": {"err": str(e)}}

        # 4. Sentiment
        try:
            factor_scores["sentiment"] = get_sentiment_score()
        except Exception as e:
            factor_scores["sentiment"] = {"score": 0.0, "confidence": 0.0,
                                           "block_trade": False, "details": {"err": str(e)}}

        # 5. News
        try:
            currencies = "SOL,BTC" if "SOL" in symbol else "BTC"
            factor_scores["news"] = get_news_score(currencies)
        except Exception as e:
            factor_scores["news"] = {"score": 0.0, "confidence": 0.0,
                                      "block_trade": False, "details": {"err": str(e)}}

        # --- Check for hard vetoes ---
        block_trade  = False
        block_reason = ""
        for name, fs in factor_scores.items():
            if fs.get("block_trade", False):
                block_trade  = True
                block_reason = f"{name}: {fs.get('block_reason', 'hard veto')}"
                break

        # --- Compute weighted final score ---
        final_score = 0.0
        for name, fs in factor_scores.items():
            w          = WEIGHTS.get(name, 0.0)
            score      = fs.get("score", 0.0)
            confidence = fs.get("confidence", 1.0)   # scale weight by confidence
            final_score += w * score * confidence

        final_score = max(-1.0, min(1.0, final_score))

        # --- Determine consensus signal ---
        ta_direction = ta_signal.get("signal")   # "LONG" | "SHORT" | None

        if block_trade:
            consensus_signal = None
        else:
            # Dynamic LONG threshold: when regime is strongly bearish, require
            # much higher multi-factor confirmation to enter LONG.
            # This is a general principle — not hardcoded to any event.
            regime_s = factor_scores.get("regime", {}).get("score", 0.0)
            long_threshold = (LONG_THRESHOLD_BEARISH_REGIME
                              if regime_s <= -0.4 else LONG_ENTRY_THRESHOLD)

            if ta_direction == "LONG" and final_score >= long_threshold:
                consensus_signal = "LONG"
                if long_threshold > LONG_ENTRY_THRESHOLD:
                    print(f"   ⚠️  Bearish regime: elevated LONG threshold {long_threshold:.2f} applied")
            elif ta_direction == "SHORT" and final_score <= -SHORT_ENTRY_THRESHOLD:
                consensus_signal = "SHORT"
            else:
                # TA fired but multi-factor consensus is too weak / opposing
                consensus_signal = None

        elapsed = time.time() - t0

        # --- Pretty-print summary ---
        self._print_summary(factor_scores, final_score, consensus_signal,
                            block_trade, block_reason, elapsed)

        return {
            "signal":        consensus_signal,
            "final_score":   round(final_score, 3),
            "block_trade":   block_trade,
            "block_reason":  block_reason,
            "factor_scores": factor_scores,
            "elapsed_s":     round(elapsed, 1),
        }

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _ta_to_score(ta_signal: dict) -> dict:
        """
        Convert the existing TA signal dict into the standard factor score format.
        Maps strength (5-7 for LONG, 6-10 for SHORT) to [-1.0, +1.0].
        """
        direction = ta_signal.get("signal")
        strength  = ta_signal.get("strength", 0)

        if direction == "LONG":
            # strength 5 = 0.50, 6 = 0.75, 7 = 1.00
            score = min(1.0, (strength - 4) / 3.0)
        elif direction == "SHORT":
            # strength 6 = -0.50, 8 = -0.83, 10 = -1.00
            score = -min(1.0, (strength - 5) / 5.0)
        else:
            score = 0.0

        return {
            "score":       round(score, 3),
            "confidence":  0.85,   # TA is well-tested on this pair
            "block_trade": False,
            "details": {
                "ta_direction": direction,
                "ta_strength":  strength,
            },
        }

    @staticmethod
    def _print_summary(factor_scores, final_score, signal,
                       blocked, block_reason, elapsed):
        bar = "=" * 55
        print(f"\n{bar}")
        print("  MULTI-FACTOR CONSENSUS")
        print(bar)
        for name, fs in factor_scores.items():
            s   = fs.get("score", 0.0)
            c   = fs.get("confidence", 0.0)
            w   = WEIGHTS.get(name, 0.0)
            bar_char = "▲" if s > 0 else ("▼" if s < 0 else "─")
            print(f"  {name:<12} score={s:+.3f}  conf={c:.2f}  wt={w:.2f}  {bar_char}")
        print(f"  {'─'*51}")
        print(f"  Final score : {final_score:+.3f}")
        if blocked:
            print(f"  🚫 BLOCKED   : {block_reason}")
        else:
            print(f"  ✅ Signal    : {signal or 'NONE (below threshold)'}")
        print(f"  ⏱  Elapsed   : {elapsed:.1f}s")
        print(bar)
