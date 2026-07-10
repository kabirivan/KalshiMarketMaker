import argparse
import time
from typing import Dict, List

from dotenv import load_dotenv

from ..factories import create_api
from ..logging_utils import build_logger


def filter_orders(orders: List[Dict], side: str = None, action: str = None) -> List[Dict]:
    filtered = orders
    if side:
        filtered = [order for order in filtered if order.get("side") == side]
    if action:
        filtered = [order for order in filtered if order.get("action") == action]
    return filtered


def parse_position(raw_position) -> int:
    try:
        return int(float(raw_position))
    except (TypeError, ValueError):
        return 0


def cancel_order_batch(api, orders: List[Dict], logger, dry_run: bool) -> Dict[str, int]:
    canceled = 0
    failed = 0

    for order in orders:
        order_id = order.get("order_id")
        if order_id is None:
            failed += 1
            logger.error("Skipping order without order_id")
            continue

        if dry_run:
            logger.warning(f"[DRY-RUN] Would cancel order {order_id}")
            continue

        try:
            success = api.cancel_order(order_id)
            if success:
                canceled += 1
            else:
                failed += 1
                logger.error(f"Cancel returned no reduction for order {order_id}")
        except Exception as cancel_exception:
            failed += 1
            logger.error(f"Failed to cancel order {order_id}: {cancel_exception}")

    return {"canceled": canceled, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description="Cancel resting Kalshi orders (all tickers by default)")
    parser.add_argument("--ticker", type=str, default=None, help="Optional market ticker filter")
    parser.add_argument("--side", type=str, choices=["yes", "no"], default=None, help="Optional side filter")
    parser.add_argument("--action", type=str, choices=["buy", "sell"], default=None, help="Optional action filter")
    parser.add_argument("--max-cancels", type=int, default=None, help="Optional max number of orders to cancel")
    parser.add_argument(
        "--liquidate-all",
        action="store_true",
        help="After canceling resting orders, submit flattening orders for all non-zero positions",
    )
    parser.add_argument(
        "--max-liquidations",
        type=int,
        default=None,
        help="Optional max number of position liquidation orders",
    )
    parser.add_argument(
        "--liquidation-expiration-seconds",
        type=int,
        default=30,
        help="Expiration horizon for liquidation orders",
    )
    parser.add_argument(
        "--liquidation-rounds",
        type=int,
        default=8,
        help="Maximum liquidation rounds (cancel + reprice + submit + recheck)",
    )
    parser.add_argument(
        "--liquidation-round-sleep-seconds",
        type=float,
        default=1.5,
        help="Sleep between liquidation rounds",
    )
    parser.add_argument(
        "--liquidation-price-offset-cents",
        type=int,
        default=1,
        help="Extra aggressiveness in cents when pricing liquidation orders",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview matching orders without canceling")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    load_dotenv()
    logger = build_logger("CancelAllOrders", args.log_level)
    api = create_api({}, logger, market_ticker="DYNAMIC")

    try:
        resting_orders = api.list_all_resting_orders(ticker=args.ticker)
        filtered_orders = filter_orders(resting_orders, side=args.side, action=args.action)

        if args.max_cancels is not None and args.max_cancels >= 0:
            filtered_orders = filtered_orders[: args.max_cancels]

        logger.info(
            f"Matched {len(filtered_orders)} resting orders (from {len(resting_orders)} total resting orders)."
        )

        if filtered_orders:
            for order in filtered_orders:
                logger.info(
                    f"Order {order.get('order_id')} | ticker={order.get('ticker', 'UNKNOWN')} "
                    f"| side={order.get('side', 'unknown')} | action={order.get('action', 'unknown')}"
                )
        else:
            logger.info("No matching resting orders found.")

        cancel_summary = cancel_order_batch(api, filtered_orders, logger, args.dry_run)
        logger.info(
            f"Cancellation complete. canceled={cancel_summary['canceled']}, failed={cancel_summary['failed']}, "
            f"total_attempted={len(filtered_orders)}"
        )

        if not args.liquidate_all:
            return

        max_rounds = max(1, args.liquidation_rounds)
        round_sleep = max(0.0, args.liquidation_round_sleep_seconds)
        price_offset_cents = max(0, args.liquidation_price_offset_cents)
        expiration_seconds = max(1, args.liquidation_expiration_seconds)
        remaining_liquidation_budget = (
            args.max_liquidations if args.max_liquidations is not None and args.max_liquidations >= 0 else None
        )

        total_submitted = 0
        total_failed = 0

        logger.warning(
            "Starting aggressive liquidation mode: repeated cancel + reprice + submit until flat or max rounds"
        )

        for round_index in range(1, max_rounds + 1):
            if remaining_liquidation_budget is not None and remaining_liquidation_budget <= 0:
                logger.warning("Liquidation budget exhausted before flat positions")
                break

            logger.warning(f"Liquidation round {round_index}/{max_rounds}")

            round_resting_orders = api.list_all_resting_orders(ticker=args.ticker)
            if round_resting_orders:
                logger.warning(f"Round {round_index}: canceling {len(round_resting_orders)} resting orders before flattening")
                cancel_order_batch(api, round_resting_orders, logger, args.dry_run)

            positions = api.list_all_positions()
            liquidation_candidates = []

            for position in positions:
                ticker = position.get("ticker")
                if not ticker:
                    continue
                if args.ticker and ticker != args.ticker:
                    continue

                signed_position = parse_position(position.get("position", 0))
                if signed_position == 0:
                    continue

                market = api.get_market(ticker)
                market_data = market.get("market", {})

                def _cents(dollars_key: str, cents_key: str):
                    dollars = market_data.get(dollars_key)
                    if dollars is not None:
                        return int(round(float(dollars) * 100))
                    cents = market_data.get(cents_key)
                    return int(cents) if cents is not None else None

                if signed_position > 0:
                    action = "sell"
                    side = "yes"
                    best_bid = _cents("yes_bid_dollars", "yes_bid")
                    if best_bid is None:
                        logger.error(f"Skipping liquidation for {ticker}: missing yes_bid")
                        continue
                    price_cents = max(1, best_bid - price_offset_cents)
                    quantity = signed_position
                else:
                    action = "buy"
                    side = "yes"
                    best_ask = _cents("yes_ask_dollars", "yes_ask")
                    if best_ask is None:
                        logger.error(f"Skipping liquidation for {ticker}: missing yes_ask")
                        continue
                    price_cents = min(99, best_ask + price_offset_cents)
                    quantity = abs(signed_position)

                liquidation_candidates.append(
                    {
                        "ticker": ticker,
                        "action": action,
                        "side": side,
                        "price": float(price_cents) / 100,
                        "quantity": quantity,
                        "signed_position": signed_position,
                    }
                )

            if not liquidation_candidates:
                logger.warning("No non-zero positions found. Liquidation complete.")
                break

            if remaining_liquidation_budget is not None:
                liquidation_candidates = liquidation_candidates[:remaining_liquidation_budget]

            for candidate in liquidation_candidates:
                logger.warning(
                    f"Round {round_index}: liquidate ticker={candidate['ticker']} pos={candidate['signed_position']} "
                    f"via {candidate['action']} {candidate['side']} qty={candidate['quantity']} @ {candidate['price']:.2f}"
                )

            if args.dry_run:
                logger.warning("Dry-run enabled: no liquidation orders executed")
                break

            expiration_ts = int(time.time()) + expiration_seconds

            for candidate in liquidation_candidates:
                try:
                    order_id = api.place_order_for_ticker(
                        ticker=candidate["ticker"],
                        action=candidate["action"],
                        side=candidate["side"],
                        price=candidate["price"],
                        quantity=candidate["quantity"],
                        expiration_ts=expiration_ts,
                    )
                    total_submitted += 1
                    logger.warning(f"Submitted liquidation order {order_id} for {candidate['ticker']}")
                except Exception as liquidation_exception:
                    total_failed += 1
                    logger.error(
                        f"Failed liquidation for {candidate['ticker']}: {liquidation_exception}"
                    )

            if remaining_liquidation_budget is not None:
                remaining_liquidation_budget -= len(liquidation_candidates)

            if round_index < max_rounds:
                time.sleep(round_sleep)

        final_positions = api.list_all_positions()
        remaining_non_zero = []
        for position in final_positions:
            ticker = position.get("ticker")
            if not ticker:
                continue
            if args.ticker and ticker != args.ticker:
                continue
            signed_position = parse_position(position.get("position", 0))
            if signed_position != 0:
                remaining_non_zero.append((ticker, signed_position))

        if remaining_non_zero:
            logger.error(f"Liquidation ended with remaining inventory: {remaining_non_zero}")
        else:
            logger.warning("Liquidation ended flat for selected scope")

        logger.warning(
            f"Liquidation summary. submitted={total_submitted}, failed={total_failed}, remaining_positions={len(remaining_non_zero)}"
        )
    finally:
        api.logout()


if __name__ == "__main__":
    main()
