from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional
from collections import deque
from datetime import datetime, timezone
import uuid


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    NEW = "new"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"


@dataclass(slots=True)
class Tick:
    symbol: str
    price: float
    bid: float
    ask: float
    ts: datetime


@dataclass(slots=True)
class Order:
    symbol: str
    side: Side
    qty: float
    type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = OrderStatus.NEW
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None


@dataclass(slots=True)
class Fill:
    order_id: str
    symbol: str
    side: Side
    price: float
    qty: float
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class Position:
    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    side: Side = Side.BUY  # long exposure represented as BUY side
    opened_ts: Optional[datetime] = None
    updated_ts: Optional[datetime] = None

    def update_market_price(self, last_price: float) -> None:
        self.unrealized_pnl = (last_price - self.avg_price) * self.qty
        self.updated_ts = datetime.now(timezone.utc)

    def apply_fill(self, fill: Fill) -> float:
        """Apply a fill to this position and return realized PnL from the fill.

        Long-only netting: BUY increases qty and re-averages price; SELL reduces qty
        and realizes PnL on the sold quantity.
        """
        realized_pnl: float = 0.0
        if fill.side == Side.BUY:
            new_qty = self.qty + fill.qty
            if new_qty <= 1e-12:
                # Fully closed or negligible size
                self.qty = 0.0
                self.avg_price = 0.0
            else:
                self.avg_price = (self.avg_price * self.qty + fill.price * fill.qty) / new_qty
                self.qty = new_qty
                if self.opened_ts is None:
                    self.opened_ts = fill.ts
        else:
            # SELL reduces position; realize PnL on sold quantity
            sold_qty = min(fill.qty, self.qty)
            realized_pnl = (fill.price - self.avg_price) * sold_qty
            self.qty -= sold_qty
            if self.qty <= 1e-12:
                self.qty = 0.0
                self.avg_price = 0.0
                self.opened_ts = None
        self.updated_ts = fill.ts
        return realized_pnl


@dataclass(slots=True)
class Account:
    cash: float
    equity: float
    realized_pnl: float
    positions: Dict[str, Position] = field(default_factory=dict)


@dataclass(slots=True)
class TradeRecord:
    order: Order
    fill: Fill
    realized_pnl: float


@dataclass(slots=True)
class Signal:
    symbol: str
    side: Side
    confidence: float
    reason: str
    qty: float


@dataclass(slots=True)
class PriceHistory:
    symbol: str
    window: int
    prices: Deque[float] = field(default_factory=deque)

    def add(self, price: float) -> None:
        self.prices.append(price)
        while len(self.prices) > self.window:
            self.prices.popleft()

    def as_list(self) -> List[float]:
        return list(self.prices)


