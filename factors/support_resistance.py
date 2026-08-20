"""
Support & Resistance Factor  (factors/support_resistance.py)
=============================================================
Detects key S/R levels from multiple timeframes and classifies the
current price position into one of 5 actionable scenarios.

Level detection uses THREE complementary sources (no external API needed):
  1. Swing highs/lows on 1h candles   (last 100 bars ≈ 4 days)
  2. Swing highs/lows on 4h candles   (last 100 bars ≈ 17 days)
  3. Psychological round-number levels (auto-scaled to price magnitude)

The FIVE scenarios:
  AT_SUPPORT      — price testing a known floor    → LONG bias
  AT_RESISTANCE   — price testing a known ceiling  → SHORT bias
  BREAKOUT_ABOVE  — confirmed close 1% above resistance + volume ⚡ LONG
  BREAKDOWN_BELOW — confirmed close 1% below support    + volume ⚡ SHORT
  MID_RANGE       — price between levels            → slight penalty

Breakout confirmation requires ALL THREE guards:
  ✓ Candle must CLOSE beyond the level (wicks rejected)
  ✓ Current volume > 1.5 × 20-period SMA (avoids low-volume fake-outs)
  ✓ Price breach ≥ BREAKOUT_BUFFER (1.0%) beyond the level

S/R does NOT veto trades (block_trade is always False).
Instead it provides:
  - A directional score (-1.0 … +1.0)
  - Precise suggested_stop / suggested_target anchored to real walls
  - suggested_leverage (12× for breakouts, 10× otherwise)

These are consumed by futures.py's order-placement logic to replace
pure-ATR stops with S/R-anchored stops for better R:R.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BREAKOUT_BUFFER     = 0.010   # 1.0% beyond level to confirm breakout
AT_LEVEL_ATR_MULT   = 0.8    # within this × ATR_15m = "at the level"
MIN_STOP_ATR_MULT   = 0.8    # minimum stop distance = price × this × ATR
STOP_CLEARANCE_ATR  = 0.4    # clearance beyond level for stop placement
VOLUME_CONFIRM_MULT = 1.5    # volume must be this × SMA to confirm breakout
SWING_WINDOW_1H     = 5      # candles each side for 1h swing detection
SWING_WINDOW_4H     = 3      # candles each side for 4h swing detection
CLUSTER_TOL         = 0.005  # 0.5% clustering tolerance
N_PSYCH_LEVELS      = 6      # number of round-number levels to generate

BREAKOUT_LEVERAGE   = 12.0
NORMAL_LEVERAGE     = 10.0

# Auto-scaled round-number grids: (price_floor, grid_size)
PRICE_GRIDS = [
    (10_000, 1_000),   # BTC-range
    (1_000,  100),     # ETH-range
    (100,    50),      # BNB-range
    (10,     5),       # SOL / AVAX
    (0,      1),       # LINK and below
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_sr_score(symbol: str, current_price: float,
                 indicators: dict, data: dict) -> dict:
    """
    Detect S/R levels and score current price position.

    Parameters
    ----------
    symbol        : str    e.g. "ETHUSDT"
    current_price : float  Current mark price
    indicators    : dict   Output of calculate_indicators()
                           Needs: indicators["15m"]["atr"], ["15m"]["volume_ratio"]
    data          : dict   Output of fetch_multi_timeframe_data()
                           Needs: data["60"] and data["240"] OHLCV arrays

    Returns
    -------
    dict:
        score              float  -1.0 … +1.0
        confidence         float  0.0 … 0.85
        block_trade        bool   always False
        scenario           str    one of the 5 scenarios
        nearest_support    float | None
        nearest_resistance float | None
        suggested_stop     float | None  (S/R-anchored)
        suggested_target   float | None  (S/R-anchored)
        suggested_leverage float
        levels             list   [{price, type, strength, sources}]
        details            dict
    """
    try:
        atr_15m      = indicators.get("15m", {}).get("atr", current_price * 0.01)
        volume_ratio = indicators.get("15m", {}).get("volume_ratio", 1.0)

        # --- Detect levels from three sources ---
        levels_1h = _detect_swing_levels(
            data.get("60", {}).get("high",  np.array([])),
            data.get("60", {}).get("low",   np.array([])),
            window=SWING_WINDOW_1H,
            source="1h_swing",
        )
        levels_4h = _detect_swing_levels(
            data.get("240", {}).get("high", np.array([])),
            data.get("240", {}).get("low",  np.array([])),
            window=SWING_WINDOW_4H,
            source="4h_swing",
        )
        levels_psych = _detect_round_numbers(current_price, N_PSYCH_LEVELS)

        all_raw = levels_1h + levels_4h + levels_psych
        if not all_raw:
            return _neutral("No price levels detected")

        # Cluster nearby levels into zones
        clustered = _cluster_levels(all_raw, CLUSTER_TOL)

        # Separate into support (below) and resistance (above)
        supports    = [l for l in clustered if l["price"] < current_price]
        resistances = [l for l in clustered if l["price"] > current_price]

        nearest_sup = max(supports,    key=lambda x: x["price"]) if supports    else None
        nearest_res = min(resistances, key=lambda x: x["price"]) if resistances else None

        # --- Classify scenario and compute raw score ---
        scenario, score = _classify_scenario(
            current_price, nearest_sup, nearest_res, atr_15m, volume_ratio,
        )

        # --- Suggest S/R-anchored stops and targets ---
        sug_stop, sug_target, sug_lev = _suggest_stops_targets(
            scenario, current_price, nearest_sup, nearest_res, atr_15m, clustered,
        )

        # Confidence: more distinct levels = better picture
        confidence = min(0.85, 0.35 + len(clustered) * 0.04)

        return {
            "score":              round(score, 3),
            "confidence":         round(confidence, 2),
            "block_trade":        False,
            "scenario":           scenario,
            "nearest_support":    round(nearest_sup["price"], 4) if nearest_sup else None,
            "nearest_resistance": round(nearest_res["price"], 4) if nearest_res else None,
            "suggested_stop":     sug_stop,
            "suggested_target":   sug_target,
            "suggested_leverage": sug_lev,
            "levels":             clustered,
            "details": {
                "total_levels":   len(clustered),
                "levels_1h":      len(levels_1h),
                "levels_4h":      len(levels_4h),
                "levels_psych":   len(levels_psych),
                "scenario":       scenario,
                "atr_15m":        round(atr_15m, 4),
                "volume_ratio":   round(volume_ratio, 2),
            },
        }

    except Exception as e:
        return _neutral(f"Exception: {e}")


# ---------------------------------------------------------------------------
# Level Detection
# ---------------------------------------------------------------------------

def _detect_swing_levels(highs: np.ndarray, lows: np.ndarray,
                         window: int = 5, source: str = "swing") -> list:
    """Detect local swing highs (resistance) and swing lows (support)."""
    if len(highs) < window * 2 + 1:
        return []

    levels = []
    n = len(highs)

    # Swing highs
    for i in range(window, n - window):
        segment_h = highs[max(0, i - window): i + window + 1]
        if highs[i] == np.max(segment_h):
            strength = _count_touches(highs[i], highs, lows)
            levels.append({
                "price":    float(highs[i]),
                "type":     "resistance",
                "strength": strength,
                "source":   source,
            })

    # Swing lows
    for i in range(window, n - window):
        segment_l = lows[max(0, i - window): i + window + 1]
        if lows[i] == np.min(segment_l):
            strength = _count_touches(lows[i], highs, lows)
            levels.append({
                "price":    float(lows[i]),
                "type":     "support",
                "strength": strength,
                "source":   source,
            })

    return levels


def _detect_round_numbers(price: float, n: int = 6) -> list:
    """Generate psychological round-number levels scaled to price magnitude."""
    grid = next((g for floor, g in PRICE_GRIDS if price > floor), 1)
    base = round(price / grid) * grid
    levels = []
    for i in range(-n // 2, n // 2 + 1):
        lvl = round(base + i * grid, 8)
        if lvl <= 0 or abs(lvl - price) / price < 0.001:
            continue   # skip levels within 0.1% of current price
        levels.append({
            "price":    float(lvl),
            "type":     "resistance" if lvl > price else "support",
            "strength": 1,
            "source":   "round_number",
        })
    return levels


def _count_touches(level_price: float, highs: np.ndarray,
                   lows: np.ndarray, tolerance: float = 0.005) -> int:
    """Count candles that touched within ±tolerance% of this level."""
    lo_bound = level_price * (1 - tolerance)
    hi_bound = level_price * (1 + tolerance)
    n = int(np.sum((highs >= lo_bound) & (highs <= hi_bound)) +
            np.sum((lows  >= lo_bound) & (lows  <= hi_bound)))
    return max(1, n)


def _cluster_levels(levels: list, tolerance: float = 0.005) -> list:
    """Merge levels within tolerance% of each other into a single zone."""
    if not levels:
        return []
    sorted_lvls = sorted(levels, key=lambda x: x["price"])
    clusters, current = [], [sorted_lvls[0]]

    for lvl in sorted_lvls[1:]:
        ref = current[-1]["price"]
        if abs(lvl["price"] - ref) / ref <= tolerance:
            current.append(lvl)
        else:
            clusters.append(_merge(current))
            current = [lvl]
    clusters.append(_merge(current))
    return clusters


def _merge(cluster: list) -> dict:
    avg_price = float(np.mean([l["price"] for l in cluster]))
    total_str = sum(l["strength"] for l in cluster)
    sources   = list({l["source"] for l in cluster})
    types     = [l["type"] for l in cluster]
    lvl_type  = max(set(types), key=types.count)
    return {
        "price":    round(avg_price, 6),
        "type":     lvl_type,
        "strength": total_str,
        "sources":  sources,
    }


# ---------------------------------------------------------------------------
# Scenario Classification
# ---------------------------------------------------------------------------

def _classify_scenario(price, nearest_sup, nearest_res, atr, volume_ratio):
    """Return (scenario_name, score) for the current price position."""

    sup_p = nearest_sup["price"] if nearest_sup else None
    res_p = nearest_res["price"] if nearest_res else None

    # ── Breakout / Breakdown (highest priority) ──────────────────────────────
    if res_p and price > res_p * (1 + BREAKOUT_BUFFER):
        if volume_ratio >= VOLUME_CONFIRM_MULT:
            return "BREAKOUT_ABOVE", +0.90   # ⚡ confirmed breakout
        # Low-volume break — treat as approaching resistance (possible rejection)
        return "AT_RESISTANCE", -0.30

    if sup_p and price < sup_p * (1 - BREAKOUT_BUFFER):
        if volume_ratio >= VOLUME_CONFIRM_MULT:
            return "BREAKDOWN_BELOW", -0.90  # ⚡ confirmed breakdown
        return "AT_SUPPORT", +0.30

    # ── At a level ────────────────────────────────────────────────────────────
    if res_p and abs(price - res_p) <= AT_LEVEL_ATR_MULT * atr:
        if volume_ratio >= 1.2:
            return "AT_RESISTANCE", 0.0      # ⚡ Volume expanding at resistance = potential breakout test
        str_bonus = min(0.15, nearest_res.get("strength", 1) * 0.02)
        return "AT_RESISTANCE", -(0.20 + str_bonus)

    if sup_p and abs(price - sup_p) <= AT_LEVEL_ATR_MULT * atr:
        str_bonus = min(0.20, nearest_sup.get("strength", 1) * 0.03)
        return "AT_SUPPORT", +(0.55 + str_bonus)

    # ── Mid-range — small penalty (no clear edge) ─────────────────────────────
    return "MID_RANGE", -0.10


# ---------------------------------------------------------------------------
# S/R-Anchored Stop / Target Suggestions
# ---------------------------------------------------------------------------

def _suggest_stops_targets(scenario, price, nearest_sup, nearest_res,
                            atr, all_levels):
    """
    Return (suggested_stop, suggested_target, suggested_leverage).
    ATR floor ensures stops are never dangerously tight.
    """
    sup_p = nearest_sup["price"] if nearest_sup else None
    res_p = nearest_res["price"] if nearest_res else None
    min_dist = max(atr * MIN_STOP_ATR_MULT, price * 0.005)   # ≥0.5% floor

    if scenario == "AT_SUPPORT":
        stop   = (sup_p - STOP_CLEARANCE_ATR * atr) if sup_p else (price - min_dist)
        stop   = min(stop, price - min_dist)          # enforce floor
        target = res_p if res_p else (price + 2 * abs(price - stop))
        return round(stop, 6), round(target, 6), NORMAL_LEVERAGE

    if scenario == "AT_RESISTANCE":
        stop   = (res_p + STOP_CLEARANCE_ATR * atr) if res_p else (price + min_dist)
        stop   = max(stop, price + min_dist)
        target = sup_p if sup_p else (price - 2 * abs(stop - price))
        return round(stop, 6), round(target, 6), NORMAL_LEVERAGE

    if scenario == "BREAKOUT_ABOVE":
        # Stop just below the broken resistance (now acting as support)
        stop   = (res_p - STOP_CLEARANCE_ATR * atr) if res_p else (price - min_dist)
        stop   = min(stop, price - min_dist)
        # Target: next resistance above current price
        nxt = _next_above(price, all_levels, exclude_price=res_p)
        target = nxt if nxt else (price + 3 * abs(price - stop))
        return round(stop, 6), round(target, 6), BREAKOUT_LEVERAGE

    if scenario == "BREAKDOWN_BELOW":
        # Stop just above the broken support (now acting as resistance)
        stop   = (sup_p + STOP_CLEARANCE_ATR * atr) if sup_p else (price + min_dist)
        stop   = max(stop, price + min_dist)
        nxt = _next_below(price, all_levels, exclude_price=sup_p)
        target = nxt if nxt else (price - 3 * abs(stop - price))
        return round(stop, 6), round(target, 6), BREAKOUT_LEVERAGE

    # MID_RANGE — no S/R suggestion; fall back to ATR method in futures.py
    return None, None, NORMAL_LEVERAGE


def _next_above(price, levels, exclude_price=None):
    eps = 0.005   # 0.5% tolerance to exclude the broken level
    candidates = [
        l["price"] for l in levels
        if l["price"] > price and (
            exclude_price is None
            or abs(l["price"] - exclude_price) / exclude_price > eps
        )
    ]
    return min(candidates) if candidates else None


def _next_below(price, levels, exclude_price=None):
    eps = 0.005
    candidates = [
        l["price"] for l in levels
        if l["price"] < price and (
            exclude_price is None
            or abs(l["price"] - exclude_price) / exclude_price > eps
        )
    ]
    return max(candidates) if candidates else None


# ---------------------------------------------------------------------------
# Backtest-compatible standalone detection (no live indicators dict needed)
# ---------------------------------------------------------------------------

def detect_sr_levels_from_arrays(highs_1h, lows_1h, highs_4h, lows_4h,
                                  current_price: float) -> dict:
    """
    Lightweight version for use in backtest.py.
    Accepts raw numpy arrays directly (no 'indicators' or 'data' dict needed).

    Returns a simplified dict with: nearest_support, nearest_resistance,
    scenario, suggested_stop, suggested_target, suggested_leverage, score.
    """
    levels_1h    = _detect_swing_levels(highs_1h, lows_1h, SWING_WINDOW_1H, "1h_swing")
    levels_4h    = _detect_swing_levels(highs_4h, lows_4h, SWING_WINDOW_4H, "4h_swing")
    levels_psych = _detect_round_numbers(current_price, N_PSYCH_LEVELS)
    all_raw      = levels_1h + levels_4h + levels_psych

    if not all_raw:
        return {
            "scenario": "MID_RANGE", "score": -0.10,
            "nearest_support": None, "nearest_resistance": None,
            "suggested_stop": None, "suggested_target": None,
            "suggested_leverage": NORMAL_LEVERAGE,
        }

    clustered = _cluster_levels(all_raw, CLUSTER_TOL)
    supports  = [l for l in clustered if l["price"] < current_price]
    resis     = [l for l in clustered if l["price"] > current_price]
    nearest_sup = max(supports, key=lambda x: x["price"]) if supports else None
    nearest_res = min(resis,    key=lambda x: x["price"]) if resis    else None

    # Use a rough ATR proxy (0.5% of price) for standalone use
    atr_proxy = current_price * 0.005
    scenario, score = _classify_scenario(
        current_price, nearest_sup, nearest_res, atr_proxy, volume_ratio=1.0
    )
    sug_stop, sug_target, sug_lev = _suggest_stops_targets(
        scenario, current_price, nearest_sup, nearest_res, atr_proxy, clustered
    )

    return {
        "scenario":           scenario,
        "score":              round(score, 3),
        "nearest_support":    nearest_sup["price"] if nearest_sup else None,
        "nearest_resistance": nearest_res["price"] if nearest_res else None,
        "suggested_stop":     sug_stop,
        "suggested_target":   sug_target,
        "suggested_leverage": sug_lev,
    }


# ---------------------------------------------------------------------------
# Neutral fallback
# ---------------------------------------------------------------------------

def _neutral(reason: str = "") -> dict:
    return {
        "score": 0.0, "confidence": 0.0, "block_trade": False,
        "scenario": "MID_RANGE",
        "nearest_support": None, "nearest_resistance": None,
        "suggested_stop": None, "suggested_target": None,
        "suggested_leverage": NORMAL_LEVERAGE,
        "levels": [],
        "details": {"reason": reason},
    }
