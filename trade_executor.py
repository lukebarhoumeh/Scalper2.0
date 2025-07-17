# trade_executor.py – Phase 2.1
from __future__ import annotations
import uuid
from collections import defaultdict
from typing import Dict

from coinbase.rest import RESTClient
from config import COINBASE_API_KEY, COINBASE_API_SECRET, PAPER_TRADING
from market_data import get_best_bid_ask


class TradeExecutor:
    """
    Places real or simulated market orders, tracks inventory & realised P n L.
    """

    def __init__(self, logger):
        self._client: RESTClient | None = None
        self._logger = logger

        self._inventory: Dict[str, float] = {}          # base units (±)
        self._cashflow: Dict[str, float]  = defaultdict(float)  # USD per product
        self._realised: Dict[str, float]  = defaultdict(float)  # USD per product
        self._running_pnl: float = 0.0

    # ── inventory helpers ───────────────────────────────────────────────────
    def position_base(self, product_id: str) -> float:
        return self._inventory.get(product_id, 0.0)

    def position_usd(self, product_id: str) -> float:
        base = self.position_base(product_id)
        bid, ask = get_best_bid_ask(product_id)
        return base * ((bid + ask) / 2.0)

    def running_pnl(self) -> float:
        return self._running_pnl

    def pnl_breakdown(self) -> Dict[str, float]:
        return dict(self._realised)

    # ── public order api ────────────────────────────────────────────────────
    def market_buy(self, product_id: str, usd_notional: float, strategy: str) -> str:
        return self._place("BUY", product_id, usd_notional, strategy)

    def market_sell(self, product_id: str, usd_notional: float, strategy: str) -> str:
        return self._place("SELL", product_id, usd_notional, strategy)

    # ── internal ------------------------------------------------------------
    def _place(self, side: str, product_id: str, usd: float, strategy: str) -> str:
        bid, ask = get_best_bid_ask(product_id)
        price = ask if side == "BUY" else bid
        qty_base = usd / price

        if PAPER_TRADING:
            order_id = f"SIM-{uuid.uuid4()}"
        else:
            self._ensure_client()
            if side == "BUY":
                resp = self._client.market_order_buy(product_id=product_id, quote_size=usd)
            else:
                resp = self._client.market_order_sell(product_id=product_id, quote_size=usd)
            order_id = resp.order_id  # type: ignore[attr-defined]

        self._update_inventory(product_id, side, qty_base, usd)
        self._logger.log(product_id, side, qty_base, price, strategy, self._running_pnl, order_id)
        return order_id

    def _update_inventory(self, product_id: str, side: str, qty_base: float, usd: float) -> None:
        sign = 1 if side == "BUY" else -1
        self._inventory[product_id] = self._inventory.get(product_id, 0.0) + sign * qty_base
        self._cashflow[product_id]  += -usd if side == "BUY" else usd

        # when position flips to (or through) zero → realise PnL
        if abs(self._inventory[product_id]) < 1e-10:
            realised = self._cashflow[product_id]
            self._realised[product_id] += realised
            self._running_pnl += realised
            self._cashflow[product_id] = 0.0

    def _ensure_client(self) -> None:
        if self._client is None:
            self._client = RESTClient(api_key=COINBASE_API_KEY, api_secret=COINBASE_API_SECRET)
