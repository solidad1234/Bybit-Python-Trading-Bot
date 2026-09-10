"""
Read-only Bybit new-listing watcher.

This tool never places orders. It discovers pre-launch linear contracts and newly
seen spot symbols, then records launch-time price, volume, spread, and order-book
reaction data for later analysis.

Examples:
    python3 new_listing.py
    python3 new_listing.py --interval 15 --state listing_state.json --log listing_reactions.jsonl
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

API_URL = "https://api.bybit.com/v5/market"
DEFAULT_INTERVAL = 30
DEFAULT_LOOKAHEAD_MINUTES = 180
DEFAULT_REACTION_MINUTES = 5
DEFAULT_SAMPLE_INTERVAL = 30


class ListingWatcher:
    def __init__(self, state_path: Path, log_path: Path, lookahead_minutes: int,
                 reaction_minutes: int, sample_interval: int):
        self.state_path = state_path
        self.log_path = log_path
        self.lookahead_minutes = lookahead_minutes
        self.reaction_minutes = reaction_minutes
        self.sample_interval = sample_interval
        self.session = requests.Session()
        self.state = self._load_state()
        self.tracked = {}

    def _load_state(self):
        if not self.state_path.exists():
            return {"spot_symbols": [], "linear_symbols": [], "events": {}}
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError):
            print(f"Warning: could not read {self.state_path}; starting a fresh baseline")
            return {"spot_symbols": [], "linear_symbols": [], "events": {}}

    def _save_state(self):
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True))
        temporary.replace(self.state_path)

    def _get(self, path, params):
        response = self.session.get(f"{API_URL}/{path}", params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
        if payload.get("retCode") != 0:
            raise RuntimeError(f"Bybit {path}: {payload.get('retMsg', payload)}")
        return payload.get("result", {})

    def _instrument_pages(self, category, status=None):
        cursor = None
        while True:
            params = {"category": category, "limit": 1000}
            if status:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor
            result = self._get("instruments-info", params)
            yield from result.get("list", [])
            cursor = result.get("nextPageCursor")
            if not cursor:
                break

    def discover(self):
        now_ms = int(time.time() * 1000)
        lookahead_ms = self.lookahead_minutes * 60 * 1000
        # Fetch all linear instruments so contracts that appear directly as
        # Trading are not missed after leaving PreLaunch.
        linear = list(self._instrument_pages("linear"))
        spot = list(self._instrument_pages("spot"))

        known_linear = set(self.state.get("linear_symbols", []))
        known_spot = set(self.state.get("spot_symbols", []))
        current_linear = {item["symbol"] for item in linear}
        current_spot = {item["symbol"] for item in spot}

        # Older state files only contained PreLaunch contracts. Migrate once by
        # establishing a complete linear baseline instead of emitting hundreds
        # of false new-listing events.
        baseline_initialized = self.state.get("linear_baseline_initialized", False)
        new_linear = []
        new_spot = []
        if baseline_initialized:
            new_linear = [item for item in linear if item["symbol"] not in known_linear]
        if known_spot:
            new_spot = [item for item in spot
                        if item["symbol"] not in known_spot and item.get("status") == "Trading"]

        upcoming = []
        for item in linear:
            launch_ms = int(item.get("launchTime") or 0)
            if (item.get("status") == "PreLaunch" and launch_ms
                    and now_ms <= launch_ms <= now_ms + lookahead_ms):
                upcoming.append(item)

        self.state["linear_symbols"] = sorted(current_linear)
        self.state["spot_symbols"] = sorted(current_spot)
        self.state["linear_baseline_initialized"] = True
        self._save_state()
        return new_linear, new_spot, upcoming

    def _ticker(self, category, symbol):
        result = self._get("tickers", {"category": category, "symbol": symbol})
        items = result.get("list", [])
        return items[0] if items else {}

    def _orderbook(self, category, symbol):
        result = self._get("orderbook", {"category": category, "symbol": symbol, "limit": 25})
        bids = result.get("b", [])
        asks = result.get("a", [])
        if not bids or not asks:
            return {"spread_bps": None, "bid_depth": 0.0, "ask_depth": 0.0}
        bid_price = float(bids[0][0])
        ask_price = float(asks[0][0])
        mid = (bid_price + ask_price) / 2
        return {
            "spread_bps": round((ask_price - bid_price) / mid * 10000, 2) if mid else None,
            "bid_depth": round(sum(float(row[1]) * float(row[0]) for row in bids), 4),
            "ask_depth": round(sum(float(row[1]) * float(row[0]) for row in asks), 4),
        }

    def record_reaction(self, category, symbol, launch_time=None, metadata=None):
        try:
            ticker = self._ticker(category, symbol)
            book = self._orderbook(category, symbol)
        except Exception as exc:
            print(f"  reaction unavailable for {symbol}: {exc}")
            return

        event = self.state.setdefault("events", {}).setdefault(symbol, {
            "category": category,
            "first_seen_utc": datetime.now(timezone.utc).isoformat(),
            "launch_time": launch_time,
            "samples": [],
        })
        if metadata:
            event.update(metadata)
        last_price = float(ticker.get("lastPrice") or 0)
        sample = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "last_price": last_price,
            "mark_price": ticker.get("markPrice"),
            "price_24h_pct": ticker.get("price24hPcnt"),
            "volume_24h": ticker.get("volume24h"),
            "turnover_24h": ticker.get("turnover24h"),
            "open_interest": ticker.get("openInterest"),
            "funding_rate": ticker.get("fundingRate"),
            "bid_price": ticker.get("bid1Price"),
            "ask_price": ticker.get("ask1Price"),
            **book,
        }
        event["last_sample_epoch"] = time.time()
        event["samples"].append(sample)
        self._append_log({"symbol": symbol, **event, "latest": sample})
        print(
            f"  {symbol}: price={last_price:g} spread={book['spread_bps']}bps "
            f"volume24h={ticker.get('volume24h', '?')} "
            f"OI={ticker.get('openInterest', '?')}"
        )

    def _sample_active_events(self):
        """Collect repeated samples until each event's observation window ends."""
        now = time.time()
        for symbol, event in list(self.state.get("events", {}).items()):
            if now >= event.get("monitoring_until_epoch", 0):
                continue
            if now - event.get("last_sample_epoch", 0) < self.sample_interval:
                continue
            self.record_reaction(
                event["category"], symbol, event.get("launch_time"),
                metadata={"status": event.get("status"),
                          "auction_phase": event.get("auction_phase"),
                          "tradable": event.get("tradable", False)},
            )
        self._save_state()

    def _append_log(self, record):
        with self.log_path.open("a") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    def run_once(self):
        self._sample_active_events()
        new_linear, new_spot, upcoming = self.discover()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"\n[{now}] prelaunch={len(upcoming)} new_linear={len(new_linear)} new_spot={len(new_spot)}")

        for item in upcoming:
            launch_ms = int(item.get("launchTime") or 0)
            launch = datetime.fromtimestamp(launch_ms / 1000, timezone.utc).isoformat()
            info = item.get("preListingInfo") or {}
            phase = info.get("curAuctionPhase", "unknown")
            print(f"  UPCOMING {item['symbol']}: launch={launch} phase={phase} "
                  f"max_leverage={item.get('leverageFilter', {}).get('maxLeverage')} "
                  f"max_market_qty={item.get('lotSizeFilter', {}).get('maxMktOrderQty')}")

        for item in new_linear:
            print(f"  NEW LINEAR: {item['symbol']} status={item.get('status')} "
                  f"launchTime={item.get('launchTime')}")
            self._start_observation("linear", item)
        for item in new_spot:
            print(f"  NEW SPOT: {item['symbol']}")
            self._start_observation("spot", item)

    def _start_observation(self, category, item):
        """Create metadata and collect the first sample for a new symbol."""
        info = item.get("preListingInfo") or {}
        phase = info.get("curAuctionPhase")
        status = item.get("status", "Unknown")
        tradable = status == "Trading" and (
            not item.get("isPreListing", False) or phase == "ContinuousTrading"
        )
        metadata = {
            "product_type": ("spot" if category == "spot"
                             else self._product_type(item)),
            "full_name": item.get("fullName"),
            "base_coin": item.get("baseCoin"),
            "quote_coin": item.get("quoteCoin"),
            "market_region": item.get("marketRegion"),
            "underlying_ticker": item.get("underlyingTicker"),
            "contract_type": item.get("contractType"),
            "status": status,
            "is_pre_listing": item.get("isPreListing"),
            "auction_phase": phase,
            "tradable": tradable,
            "max_leverage": (item.get("leverageFilter") or {}).get("maxLeverage"),
            "min_order_qty": (item.get("lotSizeFilter") or {}).get("minOrderQty"),
            "qty_step": (item.get("lotSizeFilter") or {}).get("qtyStep"),
            "max_market_qty": (item.get("lotSizeFilter") or {}).get("maxMktOrderQty"),
            "min_notional_value": (item.get("lotSizeFilter") or {}).get("minNotionalValue"),
            "tick_size": (item.get("priceFilter") or {}).get("tickSize"),
            "monitoring_started_utc": datetime.now(timezone.utc).isoformat(),
            "monitoring_until_epoch": time.time() + self.reaction_minutes * 60,
            "monitoring_minutes": self.reaction_minutes,
        }
        self.record_reaction(category, item["symbol"], item.get("launchTime"), metadata)
        self._save_state()

    @staticmethod
    def _product_type(item):
        """Separate crypto listings from stock/other linear products."""
        base_coin = (item.get("baseCoin") or "").upper()
        region = (item.get("marketRegion") or "").upper()
        underlying = item.get("underlyingTicker")
        if base_coin.endswith("STOCK") or (region and underlying):
            return "stock_linear"
        return "crypto_linear"


def parse_args():
    parser = argparse.ArgumentParser(description="Read-only Bybit new-listing watcher")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        help="Seconds between discovery scans")
    parser.add_argument("--lookahead", type=int, default=DEFAULT_LOOKAHEAD_MINUTES,
                        help="Minutes ahead for upcoming linear launches")
    parser.add_argument("--reaction-minutes", type=int, default=DEFAULT_REACTION_MINUTES,
                        help="Minutes to sample each newly detected symbol")
    parser.add_argument("--sample-interval", type=int, default=DEFAULT_SAMPLE_INTERVAL,
                        help="Seconds between reaction samples")
    parser.add_argument("--state", default="new_listing_state.json")
    parser.add_argument("--log", default="new_listing_reactions.jsonl")
    parser.add_argument("--once", action="store_true", help="Run one scan and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    watcher = ListingWatcher(Path(args.state), Path(args.log), args.lookahead,
                             args.reaction_minutes, args.sample_interval)
    print("Bybit new-listing watcher: READ ONLY; no orders are submitted")
    while True:
        try:
            watcher.run_once()
            if args.once:
                return
            time.sleep(max(5, args.interval))
        except KeyboardInterrupt:
            print("\nWatcher stopped")
            return
        except Exception as exc:
            print(f"Watcher cycle failed: {exc}")
            if args.once:
                raise
            time.sleep(max(15, args.interval))


if __name__ == "__main__":
    main()
