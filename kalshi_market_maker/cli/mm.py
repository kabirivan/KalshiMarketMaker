import argparse
import logging
import uuid

from dotenv import load_dotenv

from ..config import get_dynamic_config, load_config
from ..core.kalshi_api import KalshiTradingAPI
from ..runtime.dynamic import run_dynamic_strategy


def _install_dry_run_guards() -> None:
    """Patch KalshiTradingAPI so mutating calls log intent but never hit the API.

    This lets the bot read real market data (mid/bid/ask/volume) from prod while
    keeping capital exposure at zero. Only place_order_for_ticker and cancel_order
    are patched; all read methods (list_markets, get_market, list_all_positions,
    list_resting_orders, etc.) still hit the real API.
    """
    log = logging.getLogger("DRY-RUN")

    def _fake_place_order(self, ticker, action, side, price, quantity, expiration_ts=None):
        fake_id = f"dryrun-{uuid.uuid4()}"
        log.warning(
            f"[DRY-RUN] place_order ticker={ticker} action={action} side={side} "
            f"price={price:.4f} qty={quantity} exp={expiration_ts} -> fake_id={fake_id}"
        )
        return fake_id

    def _fake_cancel_order(self, order_id):
        log.warning(f"[DRY-RUN] cancel_order order_id={order_id}")
        return True

    KalshiTradingAPI.place_order_for_ticker = _fake_place_order
    KalshiTradingAPI.cancel_order = _fake_cancel_order


def main():
    parser = argparse.ArgumentParser(description="Kalshi Dynamic Market Maker")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read real market data but never place or cancel orders (capital-safe).",
    )
    args = parser.parse_args()

    load_dotenv()

    if args.dry_run:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("DRY-RUN").warning(
            "======== DRY-RUN MODE ENABLED — NO ORDERS WILL BE PLACED OR CANCELLED ========"
        )
        _install_dry_run_guards()

    raw_config = load_config(args.config)
    dynamic_config = get_dynamic_config(raw_config)
    run_dynamic_strategy(dynamic_config)


if __name__ == "__main__":
    main()
