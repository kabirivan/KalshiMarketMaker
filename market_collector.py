"""Continuously snapshot Kalshi open markets to CSV for later series-viability analysis.

Runs every `interval_seconds`, fetches all open markets (up to `max_markets`),
extracts market-making relevant fields, and appends to `output_csv`.

Usage:
    python market_collector.py --interval 60 --output market_snapshots.csv
"""
import argparse
import csv
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from kalshi_market_maker.core.kalshi_api import KalshiTradingAPI

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("collector")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


def series_prefix(ticker: str) -> str:
    m = re.match(r"^(KX[A-Z0-9]+?)(?:-|\d{2}[A-Z]{3})", ticker)
    return m.group(1) if m else ticker.split("-")[0]


FIELDS = [
    "ts",
    "ticker",
    "series",
    "status",
    "yes_bid",
    "yes_ask",
    "yes_bid_size",
    "yes_ask_size",
    "spread_cents",
    "mid",
    "last_price",
    "volume_24h",
    "volume_total",
    "open_interest",
    "close_time",
]


def _fdollars(v):
    """Kalshi V2 returns prices as strings like '0.9900'. Return float or None."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return f if f > 0 else None  # treat 0.0000 as unset (Kalshi encodes 'no book' as 0)
    except (TypeError, ValueError):
        return None


def _ffp(v):
    """Kalshi V2 returns sizes/volume as strings like '350.00'. Return float or None."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def snapshot(api: KalshiTradingAPI, writer: csv.DictWriter, ts: int) -> int:
    markets = api.list_all_open_markets(
        mve_filter="exclude",
        page_limit=1000,
        max_pages=10,
        max_markets=10000,
    )
    n = 0
    for m in markets:
        yb = _fdollars(m.get("yes_bid_dollars"))
        ya = _fdollars(m.get("yes_ask_dollars"))
        yb_sz = _ffp(m.get("yes_bid_size_fp"))
        ya_sz = _ffp(m.get("yes_ask_size_fp"))
        spread_c = None
        mid = None
        if yb is not None and ya is not None and ya >= yb:
            spread_c = round((ya - yb) * 100, 2)
            mid = (yb + ya) / 2.0
        writer.writerow({
            "ts": ts,
            "ticker": m.get("ticker", ""),
            "series": series_prefix(m.get("ticker", "")),
            "status": m.get("status", ""),
            "yes_bid": yb,
            "yes_ask": ya,
            "yes_bid_size": yb_sz,
            "yes_ask_size": ya_sz,
            "spread_cents": spread_c,
            "mid": mid,
            "last_price": _fdollars(m.get("last_price_dollars")),
            "volume_24h": _ffp(m.get("volume_24h_fp")),
            "volume_total": _ffp(m.get("volume_fp")),
            "open_interest": _ffp(m.get("open_interest_fp")),
            "close_time": m.get("close_time"),
        })
        n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60, help="Seconds between snapshots")
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "market_snapshots.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    api = KalshiTradingAPI(
        api_key_id=os.environ["KALSHI_API_KEY_ID"],
        private_key_path=os.environ["KALSHI_PRIVATE_KEY_PATH"],
        market_ticker="",
        base_url=os.environ["KALSHI_BASE_URL"],
        logger=logger,
    )

    out_path = Path(args.output)
    file_exists = out_path.exists() and out_path.stat().st_size > 0
    f = open(out_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if not file_exists:
        writer.writeheader()
        f.flush()

    stopping = {"flag": False}

    def _handle_stop(signum, frame):
        stopping["flag"] = True
        logger.info(f"Received signal {signum}; stopping after current tick.")

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    logger.info(f"Collector started. Writing to {out_path}. Interval={args.interval}s.")
    while not stopping["flag"]:
        t0 = time.time()
        ts = int(t0)
        try:
            n = snapshot(api, writer, ts)
            f.flush()
            logger.info(f"Snapshot ts={ts} rows={n} elapsed={time.time()-t0:.1f}s")
        except Exception as exc:
            logger.error(f"Snapshot failed: {exc}", exc_info=False)
        # sleep in short slices so signals interrupt quickly
        remaining = args.interval - (time.time() - t0)
        while remaining > 0 and not stopping["flag"]:
            time.sleep(min(1.0, remaining))
            remaining = args.interval - (time.time() - t0)

    f.close()
    logger.info("Collector stopped.")


if __name__ == "__main__":
    main()
