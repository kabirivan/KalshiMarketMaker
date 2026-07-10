from datetime import datetime, timezone
from typing import Dict, List, Tuple


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def seconds_until_close(market: Dict, now_ts: float) -> float:
    close_time = market.get("close_time")
    if not close_time:
        return float("inf")
    try:
        # Kalshi returns ISO-8601 like "2026-07-20T00:00:00Z"
        close_str = close_time.replace("Z", "+00:00")
        close_dt = datetime.fromisoformat(close_str)
    except (ValueError, TypeError, AttributeError):
        return float("inf")
    return close_dt.timestamp() - now_ts


def compute_spread_cents(market: Dict) -> float:
    # Kalshi API now returns *_dollars fields; older field names are kept as fallback.
    bid_dollars = market.get("yes_bid_dollars")
    ask_dollars = market.get("yes_ask_dollars")
    if bid_dollars is not None and ask_dollars is not None:
        return (safe_float(ask_dollars, 0.0) - safe_float(bid_dollars, 0.0)) * 100.0

    yes_bid = safe_float(market.get("yes_bid"), -1)
    yes_ask = safe_float(market.get("yes_ask"), -1)
    if yes_bid < 0 or yes_ask < 0:
        return -1
    return yes_ask - yes_bid


def is_supported_binary_market(market: Dict) -> bool:
    market_type = str(market.get("market_type", "binary")).lower()
    ticker = str(market.get("ticker", ""))

    # Reject non-binary markets (scalar, etc.)
    if market_type != "binary":
        return False

    # Reject MVE (multivariate event) markets.
    # The API's mve_filter=exclude SHOULD filter these, but KXMVE* markets
    # have been observed slipping through. Multiple layers of defense:
    #
    # 1. Hard ticker-prefix gate: all MVE market tickers start with "KXMVE"
    if ticker.upper().startswith("KXMVE"):
        return False

    # 2. API response fields: mve_collection_ticker / mve_selected_legs
    #    (x-omitempty in the spec means absent when empty, present when set)
    if market.get("mve_collection_ticker"):
        return False
    mve_legs = market.get("mve_selected_legs")
    if mve_legs is not None and len(mve_legs) > 0:
        return False

    # 3. MVE combos use strike_type="functional" — reject those too
    strike_type = str(market.get("strike_type", "")).lower()
    if strike_type == "functional":
        return False

    return True


def select_top_markets(markets: List[Dict], selector_cfg: Dict) -> List[Tuple[str, float, float, float]]:
    min_volume_24h = safe_float(selector_cfg.get("min_volume_24h", 100))
    min_spread_cents = safe_float(selector_cfg.get("min_spread_cents", 1))
    top_n = int(selector_cfg.get("top_n", 8))
    volume_weight = safe_float(selector_cfg.get("volume_weight", 0.5))
    spread_weight = safe_float(selector_cfg.get("spread_weight", 0.5))
    min_time_to_close_seconds = safe_float(selector_cfg.get("min_time_to_close_seconds", 0))
    now_ts = datetime.now(timezone.utc).timestamp()

    def collect_candidates(ignore_thresholds: bool) -> List[Dict]:
        collected = []
        for market in markets:
            if not is_supported_binary_market(market):
                continue

            ticker = market.get("ticker")
            if not ticker:
                continue

            # Skip markets that close too soon: no time to unwind, high adverse-selection risk,
            # and the worker is likely to hit HTTP 409 market_closed within a tick or two.
            if seconds_until_close(market, now_ts) < min_time_to_close_seconds:
                continue

            volume_24h = safe_float(
                market.get("volume_24h_fp")
                if market.get("volume_24h_fp") is not None
                else market.get("volume_24h", market.get("volume_fp", market.get("volume", 0)))
            )
            spread_cents = compute_spread_cents(market)
            if spread_cents < 0:
                continue

            if not ignore_thresholds and (volume_24h < min_volume_24h or spread_cents < min_spread_cents):
                continue

            collected.append(
                {
                    "ticker": ticker,
                    "volume_24h": volume_24h,
                    "spread_cents": spread_cents,
                }
            )
        return collected

    backfill_below_top_n = bool(selector_cfg.get("backfill_below_top_n", False))

    def normalize(value: float, low: float, high: float) -> float:
        if high == low:
            return 1.0
        return (value - low) / (high - low)

    def rank_pool(pool: List[Dict]) -> List[Tuple[str, float, float, float]]:
        if not pool:
            return []
        volumes = [market["volume_24h"] for market in pool]
        spreads = [market["spread_cents"] for market in pool]
        min_volume, max_volume = min(volumes), max(volumes)
        min_spread, max_spread = min(spreads), max(spreads)
        ranked_local = []
        for market in pool:
            volume_norm = normalize(market["volume_24h"], min_volume, max_volume)
            spread_norm = 1.0 - normalize(market["spread_cents"], min_spread, max_spread)
            score = volume_weight * volume_norm + spread_weight * spread_norm
            ranked_local.append((market["ticker"], score, market["volume_24h"], market["spread_cents"]))
        ranked_local.sort(key=lambda row: row[1], reverse=True)
        return ranked_local

    strict_pool = collect_candidates(ignore_thresholds=False)
    ranked_strict = rank_pool(strict_pool)

    if len(ranked_strict) >= top_n:
        return ranked_strict[:top_n]

    if not ranked_strict:
        # No strict candidates: fall back to relaxed pool entirely (backward compat).
        return rank_pool(collect_candidates(ignore_thresholds=True))[:top_n]

    if not backfill_below_top_n:
        # Backfill disabled: return whatever the strict pool gave us, even if < top_n.
        return ranked_strict

    # Backfill: strict pool has 1..top_n-1 candidates. Rank the relaxed pool separately
    # (so strict markets always outrank relaxed ones) and append until we hit top_n.
    strict_tickers = {row[0] for row in ranked_strict}
    relaxed_only = [
        market for market in collect_candidates(ignore_thresholds=True)
        if market["ticker"] not in strict_tickers
    ]
    ranked_relaxed = rank_pool(relaxed_only)
    slots_needed = top_n - len(ranked_strict)
    return ranked_strict + ranked_relaxed[:slots_needed]
