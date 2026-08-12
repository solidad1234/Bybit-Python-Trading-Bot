"""
Multi-Factor Signal Analysis Package
=====================================
Each module exposes a single `get_*_score()` function that returns:
    {
        "score":       float,  # -1.0 (bearish) to +1.0 (bullish)
        "confidence":  float,  # 0.0 to 1.0 (how much to trust this signal)
        "block_trade": bool,   # True = hard veto regardless of other scores
        "details":     dict,   # raw data for logging/debugging
    }

Import the aggregator to combine all factors:
    from factors.aggregator import MultiFactorAggregator
"""
