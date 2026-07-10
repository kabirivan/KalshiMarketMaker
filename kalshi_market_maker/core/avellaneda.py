import math
import statistics
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import requests

from .interfaces import AbstractTradingAPI


def _is_market_closed_error(exception: Exception) -> bool:
    if not isinstance(exception, requests.exceptions.HTTPError):
        return False
    response = getattr(exception, "response", None)
    if response is None or response.status_code != 409:
        return False
    try:
        payload = response.json()
    except (ValueError, AttributeError):
        return False
    error_obj = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error_obj, dict):
        return False
    return error_obj.get("code") == "market_closed"


class AvellanedaMarketMaker:
    def __init__(
        self,
        logger,
        api: AbstractTradingAPI,
        gamma: float,
        k: float,
        sigma: float,
        T: float,
        max_position: int,
        order_expiration: int,
        min_spread: float = 0.01,
        position_limit_buffer: float = 0.1,
        inventory_skew_factor: float = 0.01,
        trade_side: str = "yes",
        max_global_contracts: Optional[int] = None,
        max_contracts_per_market: Optional[int] = None,
        reserve_contracts_buffer: int = 0,
        shared_risk_state: Optional[Dict] = None,
        fee_rate: float = 0.07,
        fee_safety_buffer: float = 0.01,
        adverse_selection_buffer: float = 0.0,
        sigma_window_ticks: int = 30,
        sigma_min: float = 0.005,
        sigma_scale: float = 1.0,
        sigma_spike_threshold: float = 0.0,
        sigma_spike_widening_factor: float = 0.0,
        k_min: float = 10.0,
        k_max: float = 500.0,
        k_depth_reference: float = 200.0,
        event_ticker: Optional[str] = None,
        max_contracts_per_event: Optional[int] = None,
        halt_before_close_seconds: float = 0.0,
    ):
        self.api = api
        self.logger = logger
        self.base_gamma = gamma
        # Static config values kept as fallbacks; dynamic estimates override at each tick.
        self.k_config = k
        self.sigma_config = sigma
        self.k = k
        self.sigma = sigma
        self.T = T
        self.max_position = max_position
        self.order_expiration = order_expiration
        self.min_spread = min_spread
        self.position_limit_buffer = position_limit_buffer
        self.inventory_skew_factor = inventory_skew_factor
        self.trade_side = trade_side
        self.max_global_contracts = max_global_contracts
        self.max_contracts_per_market = max_contracts_per_market
        self.reserve_contracts_buffer = max(0, int(reserve_contracts_buffer))
        self.shared_risk_state = shared_risk_state or {"active_markets": 1}
        self.fee_rate = fee_rate
        self.fee_safety_buffer = fee_safety_buffer
        self.adverse_selection_buffer = max(0.0, float(adverse_selection_buffer))
        # Phase 1: per-market adaptive params.
        self.sigma_window_ticks = max(5, int(sigma_window_ticks))
        self.sigma_min = sigma_min
        self.sigma_scale = sigma_scale
        self.sigma_spike_threshold = max(0.0, float(sigma_spike_threshold))
        self.sigma_spike_widening_factor = max(0.0, float(sigma_spike_widening_factor))
        self.k_min = k_min
        self.k_max = k_max
        self.k_depth_reference = max(1.0, float(k_depth_reference))
        self.mid_history = deque(maxlen=self.sigma_window_ticks)
        # None means "use elapsed_time / self.T"; otherwise this fraction is used directly.
        self._time_remaining_override: Optional[float] = None
        self.event_ticker = event_ticker
        self.max_contracts_per_event = max_contracts_per_event
        self.halt_before_close_seconds = max(0.0, float(halt_before_close_seconds))
        # Cached at first tick when we first observe close_time; the model horizon is then
        # min(T_config, seconds_to_close_at_start) so the time-decay factor is meaningful
        # even for markets closer than T_config seconds away.
        self._effective_horizon_seconds: Optional[float] = None

    def run(self, dt: float, stop_event=None):
        start_time = time.time()
        while time.time() - start_time < self.T:
            if stop_event is not None and stop_event.is_set():
                self.logger.info("Stop signal received, shutting down market maker loop")
                break

            current_time = time.time() - start_time
            try:
                mid_prices = self.api.get_price()
                mid_price = mid_prices[self.trade_side]
                inventory = self.api.get_position()

                # Phase 1: adapt σ, k, and time-remaining fraction from live market data.
                self.mid_history.append(mid_price)
                self.sigma = self._estimate_sigma()
                self.k = self._estimate_k(mid_prices)
                close_time_str = mid_prices.get("close_time")
                self._time_remaining_override = self._compute_time_remaining_fraction(close_time_str)

                # Fix #2: proactive halt when the market is dangerously close to resolution.
                # The selector's min_time_to_close_seconds is a per-cycle filter (5-min cadence);
                # this per-tick guard prevents quoting into the last minutes where event risk
                # dominates and adverse selection is uncontrollable.
                if self.halt_before_close_seconds > 0:
                    seconds_to_close = self._compute_seconds_to_close(close_time_str)
                    if seconds_to_close is not None and seconds_to_close < self.halt_before_close_seconds:
                        self.logger.warning(
                            f"HALT_NEAR_CLOSE: seconds_to_close={seconds_to_close:.0f}s "
                            f"< halt_before_close_seconds={self.halt_before_close_seconds:.0f}s. "
                            f"Cancelling resting orders and exiting quote loop; drain will handle inventory."
                        )
                        for order in self.api.get_orders():
                            try:
                                self.api.cancel_order(order["order_id"])
                            except Exception as cancel_error:
                                self.logger.error(f"HALT_NEAR_CLOSE cancel failed: {cancel_error}")
                        break

                reservation_price = self.calculate_reservation_price(mid_price, inventory, current_time)
                bid_price, ask_price = self.calculate_asymmetric_quotes(mid_price, inventory, current_time)
                current_orders = self.api.get_orders()
                buy_size, sell_size = self.calculate_order_sizes(inventory, current_orders)

                t_rem_str = (
                    f" t_rem={self._time_remaining_override:.3f}"
                    if self._time_remaining_override is not None
                    else ""
                )
                seconds_to_close_now = self._compute_seconds_to_close(close_time_str)
                ttc_str = (
                    f" ttc={seconds_to_close_now:.0f}s"
                    if seconds_to_close_now is not None
                    else ""
                )
                horizon_str = (
                    f" T_eff={self._effective_horizon_seconds:.0f}s"
                    if self._effective_horizon_seconds is not None
                    else ""
                )
                self.logger.info(
                    f"t={current_time:.2f}s mid={mid_price:.4f} inventory={inventory} "
                    f"reservation={reservation_price:.4f} bid={bid_price:.4f} ask={ask_price:.4f} "
                    f"σ={self.sigma:.4f} k={self.k:.1f}{t_rem_str}{ttc_str}{horizon_str}"
                )

                self.manage_orders(bid_price, ask_price, buy_size, sell_size, current_orders)
            except requests.exceptions.HTTPError as http_error:
                if _is_market_closed_error(http_error):
                    self.logger.warning(
                        "Market is closed; exiting worker loop cleanly so the selector can rotate it out"
                    )
                    break
                raise
            time.sleep(dt)

        self.logger.info("Avellaneda market maker finished running")

    def _estimate_sigma(self) -> float:
        # Realized per-tick std of mid moves. Needs at least a few samples;
        # falls back to configured sigma during warm-up.
        min_samples = max(5, self.sigma_window_ticks // 3)
        if len(self.mid_history) < min_samples:
            return self.sigma_config
        mids = list(self.mid_history)
        diffs = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
        if len(diffs) < 2:
            return self.sigma_config
        try:
            realized = statistics.stdev(diffs)
        except statistics.StatisticsError:
            realized = 0.0
        return max(self.sigma_min, realized * self.sigma_scale)

    def _estimate_k(self, book_snapshot: Dict) -> float:
        # Higher book depth (contracts resting near mid) implies more competing MMs
        # and faster order arrival, so k grows. Linearly interpolate to [k_min, k_max].
        bid_size = book_snapshot.get("yes_bid_size", 0) or 0
        ask_size = book_snapshot.get("yes_ask_size", 0) or 0
        depth_avg = (float(bid_size) + float(ask_size)) / 2.0
        ratio = min(1.0, depth_avg / self.k_depth_reference)
        return self.k_min + (self.k_max - self.k_min) * ratio

    def _compute_seconds_to_close(self, close_time_str) -> Optional[float]:
        if not close_time_str:
            return None
        try:
            close_dt = datetime.fromisoformat(str(close_time_str).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
        return close_dt.timestamp() - time.time()

    def _compute_time_remaining_fraction(self, close_time_str) -> Optional[float]:
        # Fraction of the effective model horizon still available before market close.
        # Effective horizon = min(T_config, seconds_to_close_at_worker_start), cached lazily.
        # For a market closing in 20 min with T_config=8h, the fraction decays from 1.0
        # (worker start) to 0.0 (market close) rather than crawling 0.041 → 0.0.
        seconds_to_close = self._compute_seconds_to_close(close_time_str)
        if seconds_to_close is None:
            return None
        if seconds_to_close <= 0:
            return 0.0
        if self._effective_horizon_seconds is None:
            self._effective_horizon_seconds = min(float(self.T), max(1.0, seconds_to_close))
        return max(0.0, min(1.0, seconds_to_close / self._effective_horizon_seconds))

    def calculate_asymmetric_quotes(self, mid_price: float, inventory: int, elapsed_time: float) -> Tuple[float, float]:
        reservation_price = self.calculate_reservation_price(mid_price, inventory, elapsed_time)
        base_spread = self.calculate_optimal_spread(elapsed_time, inventory, mid_price)

        effective_max_position = self.get_effective_max_position()
        position_ratio = inventory / effective_max_position
        spread_adjustment = base_spread * abs(position_ratio) * 3

        if inventory > 0:
            bid_spread = base_spread / 2 + spread_adjustment
            ask_spread = max(base_spread / 2 - spread_adjustment, self.min_spread / 2)
        else:
            bid_spread = max(base_spread / 2 - spread_adjustment, self.min_spread / 2)
            ask_spread = base_spread / 2 + spread_adjustment

        bid_price = max(0.01, min(mid_price, reservation_price - bid_spread))
        ask_price = min(0.99, max(mid_price, reservation_price + ask_spread))

        return bid_price, ask_price

    def _resolve_time_remaining(self, elapsed_time: float) -> float:
        if self._time_remaining_override is not None:
            return self._time_remaining_override
        return max(0.0, 1 - elapsed_time / self.T)

    def calculate_reservation_price(self, mid_price: float, inventory: int, elapsed_time: float) -> float:
        dynamic_gamma = self.calculate_dynamic_gamma(inventory)
        inventory_skew = -inventory * self.inventory_skew_factor * mid_price
        time_remaining = self._resolve_time_remaining(elapsed_time)
        return mid_price + inventory_skew - inventory * dynamic_gamma * (self.sigma ** 2) * time_remaining

    def calculate_optimal_spread(self, elapsed_time: float, inventory: int, mid_price: float = 0.5) -> float:
        dynamic_gamma = self.calculate_dynamic_gamma(inventory)
        time_remaining = self._resolve_time_remaining(elapsed_time)
        base_spread = (
            dynamic_gamma * (self.sigma ** 2) * time_remaining
            + (2 / dynamic_gamma) * math.log(1 + (dynamic_gamma / self.k))
        )
        effective_max_position = self.get_effective_max_position()
        position_ratio = min(1.0, abs(inventory) / effective_max_position)
        spread_multiplier = 1 + 2.0 * position_ratio
        # Fee-aware floor: Kalshi charges ~ceil(fee_rate * price * (1 - price)) cents per side per contract.
        # Roundtrip fee = 2 sides; add a fee safety buffer and an adverse-selection cushion
        # to price in the information asymmetry cost of being top-of-book.
        fee_per_side = math.ceil(self.fee_rate * mid_price * (1 - mid_price) * 100) / 100
        fee_min_spread = 2 * fee_per_side + self.fee_safety_buffer + self.adverse_selection_buffer
        effective_min = max(self.min_spread, fee_min_spread)
        # Vol-spike widening: when *realized* sigma jumps above threshold, widen effective_min
        # proportionally so we stay safe during regime shifts even if fee_min dominates otherwise.
        # Skipped during warm-up because sigma_config seeded here may be a wide default.
        min_samples = max(5, self.sigma_window_ticks // 3)
        past_warmup = len(self.mid_history) >= min_samples
        if past_warmup and self.sigma_spike_threshold > 0 and self.sigma > self.sigma_spike_threshold:
            spike_ratio = (self.sigma - self.sigma_spike_threshold) / self.sigma_spike_threshold
            effective_min *= 1.0 + self.sigma_spike_widening_factor * spike_ratio
        return max(base_spread * spread_multiplier, effective_min)

    def calculate_dynamic_gamma(self, inventory: int) -> float:
        effective_max_position = self.get_effective_max_position()
        position_ratio = abs(inventory) / effective_max_position
        return self.base_gamma * (1 + (position_ratio**2) * 4)

    def get_effective_max_position(self) -> int:
        if self.max_contracts_per_market is not None:
            configured_market_cap = max(1, int(self.max_contracts_per_market))
        else:
            configured_market_cap = max(1, int(self.max_position))

        if self.max_global_contracts is None:
            return configured_market_cap

        active_markets = max(1, int(self.shared_risk_state.get("active_markets", 1)))
        global_budget = max(1, int(self.max_global_contracts) - self.reserve_contracts_buffer)
        equal_weight_cap = max(1, global_budget // active_markets)
        return max(1, min(configured_market_cap, equal_weight_cap))

    @staticmethod
    def _extract_event_ticker(market_ticker: str) -> str:
        # Kalshi market tickers are "<EVENT>-<STRIKE>"; the event ticker is the
        # market ticker with the last hyphen-delimited segment removed. For
        # KXNATGASD-26JUL1317-T2.895 the event is KXNATGASD-26JUL1317.
        if not market_ticker or "-" not in market_ticker:
            return market_ticker or ""
        return market_ticker.rsplit("-", 1)[0]

    def get_global_remaining_capacity(self) -> int:
        if self.max_global_contracts is None and self.max_contracts_per_event is None:
            return 10**9

        try:
            positions = self.api.list_all_positions()

            def raw_signed(position_row: Dict) -> int:
                raw = position_row.get("position_fp", position_row.get("position", 0)) or 0
                try:
                    return int(float(raw))
                except (TypeError, ValueError):
                    return 0

            total_abs_position = sum(abs(raw_signed(position_row)) for position_row in positions)

            remainings = []
            if self.max_global_contracts is not None:
                global_remaining = (
                    int(self.max_global_contracts)
                    - self.reserve_contracts_buffer
                    - total_abs_position
                )
                remainings.append(max(0, global_remaining))

            # Fix #3: cap total exposure per event to prevent correlated risk from
            # accumulating across strikes of the same underlying (e.g. multiple
            # KXNATGASD-<date> strikes all reprice on the same natgas move).
            if self.max_contracts_per_event is not None and self.event_ticker:
                per_event_abs = 0
                for position_row in positions:
                    ticker = position_row.get("ticker", "") or ""
                    if self._extract_event_ticker(ticker) == self.event_ticker:
                        per_event_abs += abs(raw_signed(position_row))
                event_remaining = int(self.max_contracts_per_event) - per_event_abs
                remainings.append(max(0, event_remaining))

            return min(remainings) if remainings else 10**9
        except Exception as global_exception:
            self.logger.error(f"Global risk snapshot failed, blocking new risk: {global_exception}")
            return 0

    def extract_pending_exposure(self, current_orders: List[Dict]) -> Tuple[int, int]:
        pending_buy = 0
        pending_sell = 0

        for order in current_orders:
            if order.get("side") != self.trade_side:
                continue
            raw_remaining = order.get("remaining_count_fp", order.get("remaining_count", 0)) or 0
            remaining = int(float(raw_remaining))
            if order.get("action") == "buy":
                pending_buy += remaining
            elif order.get("action") == "sell":
                pending_sell += remaining

        return pending_buy, pending_sell

    def calculate_order_sizes(self, inventory: int, current_orders: List[Dict]) -> Tuple[int, int]:
        effective_max_position = self.get_effective_max_position()
        pending_buy, pending_sell = self.extract_pending_exposure(current_orders)
        effective_inventory = inventory + pending_buy - pending_sell

        local_remaining_capacity = max(0, effective_max_position - abs(effective_inventory))
        global_remaining_capacity = self.get_global_remaining_capacity()

        base_size = max(1, int(effective_max_position * self.position_limit_buffer))
        accumulation_size = min(base_size, local_remaining_capacity, global_remaining_capacity)
        if global_remaining_capacity <= 0:
            accumulation_size = 0

        reduction_size = max(1, min(effective_max_position, max(base_size, abs(effective_inventory))))

        if effective_inventory > 0:
            buy_size = accumulation_size
            sell_size = reduction_size
        elif effective_inventory < 0:
            buy_size = reduction_size
            sell_size = accumulation_size
        else:
            buy_size = accumulation_size
            sell_size = accumulation_size

        return buy_size, sell_size

    def manage_orders(
        self,
        bid_price: float,
        ask_price: float,
        buy_size: int,
        sell_size: int,
        current_orders: Optional[List[Dict]] = None,
    ):
        if current_orders is None:
            current_orders = self.api.get_orders()

        buy_orders: List[Dict] = []
        sell_orders: List[Dict] = []

        for order in current_orders:
            if order["side"] == self.trade_side:
                if order["action"] == "buy":
                    buy_orders.append(order)
                elif order["action"] == "sell":
                    sell_orders.append(order)

        self.handle_order_side("buy", buy_orders, bid_price, buy_size)
        self.handle_order_side("sell", sell_orders, ask_price, sell_size)

    def handle_order_side(self, action: str, orders: List[Dict], desired_price: float, desired_size: int):
        keep_order = None

        for order in orders:
            if self.trade_side == "yes":
                price_dollars = order.get("yes_price_dollars")
                current_price = (
                    float(price_dollars) if price_dollars is not None else float(order.get("yes_price", 0)) / 100
                )
            else:
                price_dollars = order.get("no_price_dollars")
                current_price = (
                    float(price_dollars) if price_dollars is not None else float(order.get("no_price", 0)) / 100
                )
            raw_remaining = order.get("remaining_count_fp", order.get("remaining_count", 0)) or 0
            remaining = int(float(raw_remaining))
            if (
                keep_order is None
                and abs(current_price - desired_price) < 0.01
                and remaining == desired_size
            ):
                keep_order = order
            else:
                self.api.cancel_order(order["order_id"])

        if desired_size <= 0:
            return

        current_price = self.api.get_price()[self.trade_side]
        should_place = (action == "buy" and desired_price < current_price) or (
            action == "sell" and desired_price > current_price
        )

        if keep_order is None and should_place:
            self.api.place_order(
                action,
                self.trade_side,
                desired_price,
                desired_size,
                int(time.time()) + self.order_expiration,
            )
