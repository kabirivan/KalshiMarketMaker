import time
from concurrent.futures import TimeoutError
from typing import Dict, Optional

import requests

from ..factories import create_api
from ..logging_utils import build_logger
from ..selection.scoring import safe_float


def _book_side_cents(market_data: Dict, dollars_key: str, cents_key: str) -> Optional[int]:
    dollars = market_data.get(dollars_key)
    if dollars is not None:
        try:
            return int(round(float(dollars) * 100))
        except (TypeError, ValueError):
            return None
    cents = market_data.get(cents_key)
    if cents is None:
        return None
    try:
        return int(cents)
    except (TypeError, ValueError):
        return None


def cancel_resting_orders_for_ticker(
    ticker: str,
    dynamic_config: Dict,
    logger,
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
) -> bool:
    cleanup_logger = build_logger(f"Cleanup_{ticker}", dynamic_config.get("log_level", "INFO"))
    api = create_api(dynamic_config.get("api", {}), cleanup_logger, market_ticker=ticker)

    try:
        for attempt in range(1, max_attempts + 1):
            try:
                orders = api.get_orders()
                if not orders:
                    cleanup_logger.info(f"No resting orders to cancel for {ticker}")
                    return True

                cleanup_logger.warning(
                    f"Found {len(orders)} resting orders for {ticker}. Cancel attempt {attempt}/{max_attempts}"
                )
                for order in orders:
                    order_id = order.get("order_id")
                    if order_id is None:
                        continue
                    try:
                        api.cancel_order(order_id)
                    except requests.exceptions.RequestException as request_exception:
                        cleanup_logger.error(f"Failed to cancel order {order_id} for {ticker}: {request_exception}")

                time.sleep(backoff_seconds)
            except requests.exceptions.RequestException as request_exception:
                cleanup_logger.error(f"Order cleanup request failed for {ticker}: {request_exception}")
                time.sleep(backoff_seconds)

        remaining = api.get_orders()
        if remaining:
            logger.error(f"Cleanup incomplete for {ticker}: {len(remaining)} resting orders still present")
            return False
        return True
    except requests.exceptions.RequestException as request_exception:
        logger.error(f"Final cleanup verification failed for {ticker}: {request_exception}")
        return False
    finally:
        api.logout()


def liquidate_position_for_ticker(
    ticker: str,
    dynamic_config: Dict,
    logger,
    max_rounds: int = 3,
    round_sleep_seconds: float = 1.5,
    price_offset_cents: int = 1,
    expiration_seconds: int = 30,
) -> bool:
    cleanup_logger = build_logger(f"Liquidate_{ticker}", dynamic_config.get("log_level", "INFO"))
    api = create_api(dynamic_config.get("api", {}), cleanup_logger, market_ticker=ticker)

    try:
        for round_index in range(1, max_rounds + 1):
            try:
                position = api.get_position()
            except requests.exceptions.RequestException as request_exception:
                cleanup_logger.error(f"Position fetch failed for {ticker}: {request_exception}")
                time.sleep(round_sleep_seconds)
                continue

            if position == 0:
                if round_index == 1:
                    cleanup_logger.info(f"No inventory to liquidate for {ticker}")
                else:
                    cleanup_logger.warning(f"Position for {ticker} flattened after {round_index - 1} rounds")
                return True

            try:
                market_response = api.get_market(ticker)
            except requests.exceptions.RequestException as request_exception:
                cleanup_logger.error(f"Market fetch failed for {ticker}: {request_exception}")
                time.sleep(round_sleep_seconds)
                continue

            market_data = market_response.get("market", {})

            if position > 0:
                action = "sell"
                best_bid = _book_side_cents(market_data, "yes_bid_dollars", "yes_bid")
                if best_bid is None or best_bid <= 0:
                    cleanup_logger.error(f"Skipping liquidation round for {ticker}: no valid yes_bid")
                    time.sleep(round_sleep_seconds)
                    continue
                price_cents = max(1, best_bid - price_offset_cents)
                quantity = position
            else:
                action = "buy"
                best_ask = _book_side_cents(market_data, "yes_ask_dollars", "yes_ask")
                if best_ask is None or best_ask <= 0:
                    cleanup_logger.error(f"Skipping liquidation round for {ticker}: no valid yes_ask")
                    time.sleep(round_sleep_seconds)
                    continue
                price_cents = min(99, best_ask + price_offset_cents)
                quantity = abs(position)

            cleanup_logger.warning(
                f"Liquidation round {round_index}/{max_rounds} for {ticker}: pos={position} "
                f"submit {action} yes qty={quantity} @ {price_cents / 100:.2f}"
            )

            try:
                expiration_ts = int(time.time()) + max(1, expiration_seconds)
                order_id = api.place_order_for_ticker(
                    ticker=ticker,
                    action=action,
                    side="yes",
                    price=price_cents / 100,
                    quantity=quantity,
                    expiration_ts=expiration_ts,
                )
                cleanup_logger.warning(f"Submitted liquidation order {order_id} for {ticker}")
            except requests.exceptions.RequestException as request_exception:
                cleanup_logger.error(f"Failed liquidation submit for {ticker}: {request_exception}")

            if round_index < max_rounds:
                time.sleep(round_sleep_seconds)

        try:
            final_position = api.get_position()
        except requests.exceptions.RequestException as request_exception:
            logger.error(f"Final position check failed for {ticker}: {request_exception}")
            return False

        if final_position == 0:
            return True

        logger.error(
            f"Liquidation exhausted for {ticker}: remaining position={final_position} after {max_rounds} rounds"
        )
        return False
    finally:
        api.logout()


def stop_worker_then_cancel(
    ticker: str,
    stop_event,
    future,
    dynamic_config: Dict,
    logger,
) -> bool:
    selector_cfg = dynamic_config.get("market_selector", {})
    shutdown_timeout_seconds = safe_float(
        selector_cfg.get("worker_shutdown_timeout_seconds", 15),
        15.0,
    )

    stop_event.set()
    try:
        future.result(timeout=shutdown_timeout_seconds)
    except TimeoutError:
        logger.error(f"Worker {ticker} did not stop within {shutdown_timeout_seconds:.1f}s; deferring cleanup")
        return False
    except Exception as worker_exception:
        logger.error(f"Worker {ticker} exited with error during shutdown: {worker_exception}")

    orders_clean = cancel_resting_orders_for_ticker(ticker, dynamic_config, logger)

    liquidate_enabled = bool(selector_cfg.get("drain_liquidate_position", True))
    if not liquidate_enabled:
        return orders_clean

    liq_rounds = int(selector_cfg.get("drain_liquidation_rounds", 3))
    liq_sleep = safe_float(selector_cfg.get("drain_liquidation_round_sleep_seconds", 1.5), 1.5)
    liq_offset = int(selector_cfg.get("drain_liquidation_price_offset_cents", 1))
    liq_expiration = int(selector_cfg.get("drain_liquidation_expiration_seconds", 30))

    position_clean = liquidate_position_for_ticker(
        ticker,
        dynamic_config,
        logger,
        max_rounds=liq_rounds,
        round_sleep_seconds=liq_sleep,
        price_offset_cents=liq_offset,
        expiration_seconds=liq_expiration,
    )

    if position_clean and not orders_clean:
        orders_clean = cancel_resting_orders_for_ticker(ticker, dynamic_config, logger)

    return orders_clean and position_clean
