from __future__ import annotations

from typing import Dict, List, Optional
from datetime import datetime, timezone

from .models import Account, Order, OrderStatus, OrderType, Fill, Position, Side
from .metadata import get_symbol_metadata


class PaperBroker:
    """Simple paper broker that executes market orders at mid price and
    limit orders when price crosses. Keeps balances and positions coherent.
    """

    def __init__(self, account: Account):
        self.account = account
        self.last_prices: Dict[str, float] = {}
        self.fills: List[Fill] = []
        # trailing stops and TP/SL trackers per symbol
        self.trailing_high: Dict[str, float] = {}
        self.trailing_low: Dict[str, float] = {}
        self.take_profit: Dict[str, float] = {}
        self.stop_loss: Dict[str, float] = {}

    def update_price(self, symbol: str, bid: float, ask: float) -> None:
        mid = (bid + ask) / 2.0
        self.last_prices[symbol] = mid
        pos = self.account.positions.get(symbol)
        if pos:
            pos.update_market_price(mid)
            self._refresh_equity()
            # Update trailing levels
            if pos.qty > 0:
                self.trailing_high[symbol] = max(self.trailing_high.get(symbol, mid), mid)
                self.trailing_low[symbol] = min(self.trailing_low.get(symbol, mid), mid)

    def _round_to_increment(self, value: float, increment: float) -> float:
        if increment <= 0:
            return value
        steps = round(value / increment)
        return steps * increment

    def place_order(self, order: Order) -> Optional[Fill]:
        price = self.last_prices.get(order.symbol)
        if price is None:
            return None

        # Enforce min notional and lot size/tick size
        meta = get_symbol_metadata(order.symbol)
        # Direction-aware rounding for qty
        if order.side == Side.BUY:
            order.qty = max((order.qty // meta.lot_size) * meta.lot_size, 0.0)
        else:
            # Selling: round down to lot size too, to avoid over-selling
            order.qty = max((order.qty // meta.lot_size) * meta.lot_size, 0.0)
        notional = order.qty * price
        if notional < meta.min_notional_usd or order.qty <= 0.0:
            return None

        exec_price = price
        if order.type == OrderType.LIMIT and order.price is not None:
            # Direction-aware price rounding to tick size
            if order.side == Side.BUY:
                order.price = ((order.price // meta.tick_size) * meta.tick_size)
            else:
                order.price = (((order.price + meta.tick_size - 1e-12) // meta.tick_size) * meta.tick_size)
            if order.side == Side.BUY and price > order.price:
                return None
            if order.side == Side.SELL and price < order.price:
                return None
            exec_price = order.price

        # Prevent selling more than current long position (no shorting in this model)
        if order.side == Side.SELL:
            pos = self.account.positions.get(order.symbol)
            available = pos.qty if pos else 0.0
            if available <= 1e-12:
                return None
            if order.qty > available:
                order.qty = available

        order.status = OrderStatus.FILLED
        fill = Fill(order_id=order.id, symbol=order.symbol, side=order.side, price=exec_price, qty=order.qty)
        self._apply_fill(fill)
        self.fills.append(fill)
        # Register TP/SL levels from order
        if order.side == Side.BUY:
            if order.take_profit:
                self.take_profit[order.symbol] = float(order.take_profit)
            if order.stop_loss:
                self.stop_loss[order.symbol] = float(order.stop_loss)
        return fill

    def _apply_fill(self, fill: Fill) -> None:
        pos = self.account.positions.get(fill.symbol)
        if pos is None:
            pos = Position(symbol=fill.symbol)
            self.account.positions[fill.symbol] = pos

        realized = pos.apply_fill(fill)
        # Apply fees to realized PnL for SELLs when position reduces
        meta = get_symbol_metadata(fill.symbol)
        fee_rate = meta.fee_bps / 1e4
        fees = fee_rate * fill.price * fill.qty
        if fill.side == Side.BUY:
            self.account.cash -= fill.qty * fill.price
            self.account.cash -= fees
        else:
            # Only credit cash for the quantity actually netted from the position
            net_qty = min(fill.qty, pos.qty + fill.qty) if pos else 0.0
            self.account.cash += net_qty * fill.price
            self.account.cash -= fees
        self.account.realized_pnl += realized - (fees if fill.side == Side.SELL else 0.0)
        self._refresh_equity()

    def _refresh_equity(self) -> None:
        equity = self.account.cash
        for sym, pos in self.account.positions.items():
            price = self.last_prices.get(sym)
            if price is None:
                continue
            equity += pos.qty * price
        self.account.equity = equity

    def check_exits(self, trailing_stop_bps: int) -> List[Fill]:
        """Check TP/SL/trailing exits for all symbols. If triggered, submit sell to flatten.
        Returns list of exit fills.
        """
        exits: List[Fill] = []
        for sym, pos in list(self.account.positions.items()):
            if pos.qty <= 1e-12:
                continue
            price = self.last_prices.get(sym)
            if price is None:
                continue
            # Take profit / Stop loss
            tp = self.take_profit.get(sym)
            sl = self.stop_loss.get(sym)
            do_exit = False
            if tp and price >= tp:
                do_exit = True
            if sl and price <= sl:
                do_exit = True
            # Trailing stop from trailing_high
            high = self.trailing_high.get(sym, price)
            trail = high * (1 - trailing_stop_bps / 1e4)
            if price <= trail:
                do_exit = True
            if do_exit:
                order = Order(symbol=sym, side=Side.SELL, qty=pos.qty, type=OrderType.MARKET)
                fill = self.place_order(order)
                if fill:
                    exits.append(fill)
                    # clear trackers
                    self.take_profit.pop(sym, None)
                    self.stop_loss.pop(sym, None)
                    self.trailing_high.pop(sym, None)
                    self.trailing_low.pop(sym, None)
        return exits


