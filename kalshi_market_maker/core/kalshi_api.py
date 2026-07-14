import base64
import logging
import random
import time
import uuid
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from .interfaces import AbstractTradingAPI


class KalshiTradingAPI(AbstractTradingAPI):
    def __init__(
        self,
        api_key_id: str,
        private_key_path: str,
        market_ticker: str,
        base_url: str,
        logger: logging.Logger,
    ):
        if not api_key_id:
            raise ValueError("KALSHI_API_KEY_ID environment variable is required")
        if not private_key_path:
            raise ValueError("KALSHI_PRIVATE_KEY_PATH environment variable is required")

        self.api_key_id = api_key_id
        self.private_key_path = private_key_path
        self.market_ticker = market_ticker
        self.logger = logger
        self.base_url = base_url.rstrip("/")
        self.private_key = self.load_private_key()
        self.logger.info("Kalshi API client initialized")

    def load_private_key(self):
        with open(self.private_key_path, "rb") as private_key_file:
            return serialization.load_pem_private_key(private_key_file.read(), password=None)

    def logout(self):
        return None

    def _create_signature(self, timestamp: str, method: str, path: str) -> str:
        sign_path = path.split("?")[0]
        message = f"{timestamp}{method.upper()}{sign_path}".encode("utf-8")
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def get_headers(self, method: str, path: str):
        timestamp = str(int(time.time() * 1000))
        signature = self._create_signature(timestamp, method, path)
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }

    def make_request(
        self,
        method: str,
        path: str,
        params: Dict = None,
        data: Dict = None,
        max_retries: int = 5,
    ):
        url = f"{self.base_url}{path}"
        parsed_path = urlparse(url).path
        retryable_codes = {429, 500, 502, 503, 504}

        for attempt in range(max_retries + 1):
            headers = self.get_headers(method, parsed_path)
            try:
                response = requests.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=data,
                    timeout=15,
                )

                if response.status_code in retryable_codes and attempt < max_retries:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after is not None:
                        try:
                            delay_seconds = float(retry_after)
                        except ValueError:
                            delay_seconds = 0.0
                    else:
                        delay_seconds = 0.0

                    backoff = max(delay_seconds, 0.5 * (2**attempt)) + random.uniform(0, 0.25)
                    self.logger.warning(
                        f"Retryable response {response.status_code} for {method} {path}; retrying in {backoff:.2f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(backoff)
                    continue

                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as request_exception:
                # Only retry on network/timeout errors or retryable HTTP status codes.
                # 4xx (400/401/403/404/409/410) means a client-side problem — retrying is pointless
                # and produces log noise / duplicate order risk.
                is_http_error = isinstance(request_exception, requests.exceptions.HTTPError)
                status_code = (
                    request_exception.response.status_code
                    if is_http_error and request_exception.response is not None
                    else None
                )
                if is_http_error and status_code not in retryable_codes:
                    self.logger.error(f"Non-retryable HTTP error {status_code} for {method} {path}: {request_exception}")
                    if request_exception.response is not None:
                        self.logger.error(f"Response content: {request_exception.response.text}")
                    raise

                if attempt < max_retries:
                    backoff = 0.5 * (2**attempt) + random.uniform(0, 0.25)
                    self.logger.warning(
                        f"Request exception for {method} {path}: {request_exception}. Retrying in {backoff:.2f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(backoff)
                    continue

                self.logger.error(f"Request failed: {request_exception}")
                if hasattr(request_exception, "response") and request_exception.response is not None:
                    self.logger.error(f"Response content: {request_exception.response.text}")
                raise

    def get_position(self) -> int:
        path = "/portfolio/positions"
        params = {"ticker": self.market_ticker, "settlement_status": "unsettled"}
        response = self.make_request("GET", path, params=params)
        positions = response.get("market_positions", [])

        total_position = 0
        for position in positions:
            if position.get("ticker") == self.market_ticker:
                # Kalshi V2 renamed `position` (int) -> `position_fp` (decimal string).
                raw = position.get("position_fp", position.get("position", 0)) or 0
                total_position += int(float(raw))

        return total_position

    def get_price(self) -> Dict[str, float]:
        path = f"/markets/{self.market_ticker}"
        data = self.make_request("GET", path)
        market = data["market"]

        def read_side(dollars_key: str, cents_key: str) -> float:
            dollars = market.get(dollars_key)
            if dollars is not None:
                return float(dollars)
            return float(market.get(cents_key, 0)) / 100

        def read_size(fp_key: str) -> float:
            value = market.get(fp_key)
            try:
                return float(value) if value is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        yes_bid = read_side("yes_bid_dollars", "yes_bid")
        yes_ask = read_side("yes_ask_dollars", "yes_ask")
        no_bid = read_side("no_bid_dollars", "no_bid")
        no_ask = read_side("no_ask_dollars", "no_ask")

        yes_mid_price = round((yes_bid + yes_ask) / 2, 2)
        no_mid_price = round((no_bid + no_ask) / 2, 2)

        return {
            "yes": yes_mid_price,
            "no": no_mid_price,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "yes_bid_size": read_size("yes_bid_size_fp"),
            "yes_ask_size": read_size("yes_ask_size_fp"),
            "close_time": market.get("close_time"),
        }

    def place_order(self, action: str, side: str, price: float, quantity: int, expiration_ts: int = None) -> str:
        return self.place_order_for_ticker(
            ticker=self.market_ticker,
            action=action,
            side=side,
            price=price,
            quantity=quantity,
            expiration_ts=expiration_ts,
        )

    def place_order_for_ticker(
        self,
        ticker: str,
        action: str,
        side: str,
        price: float,
        quantity: int,
        expiration_ts: int = None,
    ) -> str:
        # Kalshi V2: POST /portfolio/events/orders, side=bid|ask, price in dollar-decimal strings.
        if side != "yes":
            raise ValueError(
                f"place_order_for_ticker under Kalshi V2 API currently only supports side='yes'; got side={side!r}"
            )
        action_lower = action.lower()
        if action_lower == "buy":
            v2_side = "bid"
        elif action_lower == "sell":
            v2_side = "ask"
        else:
            raise ValueError(f"Unsupported action {action!r}; expected 'buy' or 'sell'")

        path = "/portfolio/events/orders"
        # Round to nearest cent — Kalshi rejects prices outside the market's price_step (0.01).
        price_cents = max(1, min(99, int(round(price * 100))))
        data = {
            "ticker": ticker,
            "side": v2_side,
            "count": str(int(quantity)),
            "price": f"{price_cents / 100:.4f}",
            "time_in_force": "good_till_canceled",
            "self_trade_prevention_type": "maker",
            "client_order_id": str(uuid.uuid4()),
        }
        if expiration_ts is not None:
            data["expiration_time"] = int(expiration_ts)

        response = self.make_request("POST", path, data=data)
        return str(response["order_id"])

    def get_market(self, ticker: str) -> Dict:
        path = f"/markets/{ticker}"
        return self.make_request("GET", path)

    def list_all_positions(
        self,
        page_limit: int = 200,
        max_pages: int = 20,
        count_filter: str = "position",
    ) -> List[Dict]:
        positions: List[Dict] = []
        cursor = None
        pages = 0

        safe_page_limit = max(1, min(1000, page_limit))
        safe_max_pages = max(1, max_pages)

        while True:
            path = "/portfolio/positions"
            params = {"limit": safe_page_limit, "count_filter": count_filter}
            if cursor:
                params["cursor"] = cursor

            response = self.make_request("GET", path, params=params)
            batch = response.get("market_positions", [])
            # Kalshi V2 uses `position_fp` (decimal string); inject legacy `position` (int)
            # so downstream code that reads it still works.
            for entry in batch:
                if "position" not in entry:
                    raw = entry.get("position_fp", 0) or 0
                    try:
                        entry["position"] = int(float(raw))
                    except (TypeError, ValueError):
                        entry["position"] = 0
            positions.extend(batch)

            pages += 1
            cursor = response.get("cursor")

            if not cursor or pages >= safe_max_pages:
                break

        return positions

    def cancel_order(self, order_id: int) -> bool:
        # Kalshi V2 cancel path; reduced_by is a decimal string.
        # Cancel is idempotent by intent: if the order is already gone
        # (filled, expired, or previously cancelled), Kalshi returns 404
        # not_found. Treat that as success — the desired state (order not
        # resting) is already achieved. Raising here would crash the worker
        # on a routine race between get_orders() and cancel_order().
        path = f"/portfolio/events/orders/{order_id}"
        try:
            response = self.make_request("DELETE", path)
        except requests.exceptions.HTTPError as http_error:
            response_obj = getattr(http_error, "response", None)
            if response_obj is not None and response_obj.status_code == 404:
                self.logger.warning(
                    f"cancel_order: order {order_id} already gone (404); treating as success"
                )
                return False
            raise
        try:
            reduced = float(response.get("reduced_by", "0"))
        except (TypeError, ValueError):
            reduced = 0.0
        return reduced > 0

    def get_orders(self, ticker: Optional[str] = None, status: str = "resting") -> List[Dict]:
        path = "/portfolio/orders"
        effective_ticker = self.market_ticker if ticker is None else ticker
        params = {"status": status}
        if effective_ticker:
            params["ticker"] = effective_ticker

        response = self.make_request("GET", path, params=params)
        return response.get("orders", [])

    def list_all_resting_orders(
        self,
        ticker: Optional[str] = None,
        page_limit: int = 200,
        max_pages: int = 20,
    ) -> List[Dict]:
        return self.list_all_orders_by_status(
            status="resting",
            ticker=ticker,
            page_limit=page_limit,
            max_pages=max_pages,
        )

    def list_all_orders_by_status(
        self,
        status: str,
        ticker: Optional[str] = None,
        page_limit: int = 200,
        max_pages: int = 20,
    ) -> List[Dict]:
        orders: List[Dict] = []
        cursor = None
        pages = 0

        safe_page_limit = max(1, min(1000, page_limit))
        safe_max_pages = max(1, max_pages)

        while True:
            path = "/portfolio/orders"
            params = {"status": status, "limit": safe_page_limit}
            if ticker:
                params["ticker"] = ticker
            if cursor:
                params["cursor"] = cursor

            response = self.make_request("GET", path, params=params)
            batch = response.get("orders", [])
            orders.extend(batch)

            pages += 1
            cursor = response.get("cursor")

            if not cursor or pages >= safe_max_pages:
                break

        return orders

    def list_markets(
        self,
        status: str = "open",
        limit: int = 1000,
        cursor: str = None,
        series_ticker: str = None,
        mve_filter: str = "exclude",
    ) -> Dict:
        path = "/markets"
        params = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        if series_ticker:
            params["series_ticker"] = series_ticker
        if mve_filter:
            params["mve_filter"] = mve_filter
        return self.make_request("GET", path, params=params)

    def list_all_open_markets(
        self,
        series_ticker: str = None,
        mve_filter: str = "exclude",
        page_limit: int = 250,
        max_pages: int = 5,
        max_markets: int = 1250,
    ) -> List[Dict]:
        markets: List[Dict] = []
        cursor = None
        pages = 0

        safe_page_limit = max(1, min(1000, page_limit))
        safe_max_pages = max(1, max_pages)
        safe_max_markets = max(1, max_markets)

        while True:
            response = self.list_markets(
                status="open",
                limit=safe_page_limit,
                cursor=cursor,
                series_ticker=series_ticker,
                mve_filter=mve_filter,
            )
            batch = response.get("markets", [])
            markets.extend(batch)
            pages += 1
            cursor = response.get("cursor")

            if len(markets) >= safe_max_markets or pages >= safe_max_pages or not cursor:
                break

        return markets[:safe_max_markets]
