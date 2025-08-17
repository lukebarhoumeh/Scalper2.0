from __future__ import annotations

from typing import Dict, Optional
from collections import deque
from math import sqrt

from .models import PriceHistory, Signal, Side
from .indicators import simple_moving_average, rsi, breakout, realized_volatility_pct


class StrategyEngine:
    """Combines SMA cross, RSI, spread filter and breakout for a simple,
    robust baseline. Produces at most one signal per symbol per tick.
    """

    def __init__(
        self,
        rsi_buy: int,
        rsi_sell: int,
        sma_fast: int,
        sma_slow: int,
        max_spread_bps: int,
        vol_gate_low_pct: float = 0.05,
        vol_gate_high_pct: float = 3.0,
        enable_mean_reversion: bool = True,
        enable_breakout: bool = True,
        enable_grid_overlay: bool = False,
    ):
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.max_spread_bps = max_spread_bps
        self.vol_gate_low_pct = vol_gate_low_pct
        self.vol_gate_high_pct = vol_gate_high_pct
        self._vol_window = max(self.sma_fast, 20)
        self.enable_mean_reversion = enable_mean_reversion
        self.enable_breakout = enable_breakout
        self.enable_grid_overlay = enable_grid_overlay
        self.history: Dict[str, PriceHistory] = {}
        self.last_side: Dict[str, Side] = {}
        self.last_rsi: Dict[str, float] = {}
        self.last_sma_fast: Dict[str, float] = {}
        self.last_sma_slow: Dict[str, float] = {}
        self.last_spread_bps: Dict[str, float] = {}
        self.last_reason: Dict[str, str] = {}
        self.last_vol_pct: Dict[str, float] = {}
        self.last_regime: Dict[str, str] = {}
        # Rolling returns statistics for O(1) vol
        self._last_price: Dict[str, float] = {}
        self._returns: Dict[str, deque] = {}
        self._sum_ret: Dict[str, float] = {}
        self._sumsq_ret: Dict[str, float] = {}

    def on_price(self, symbol: str, price: float) -> None:
        ph = self.history.get(symbol)
        if ph is None:
            ph = PriceHistory(symbol=symbol, window=max(self.sma_slow, 30))
            self.history[symbol] = ph
        # Rolling return update
        prev = self._last_price.get(symbol)
        if prev is not None and prev > 0:
            r = (price / prev) - 1.0
            dq = self._returns.get(symbol)
            if dq is None:
                dq = deque()
                self._returns[symbol] = dq
                self._sum_ret[symbol] = 0.0
                self._sumsq_ret[symbol] = 0.0
            dq.append(r)
            self._sum_ret[symbol] += r
            self._sumsq_ret[symbol] += r * r
            if len(dq) > self._vol_window:
                old = dq.popleft()
                self._sum_ret[symbol] -= old
                self._sumsq_ret[symbol] -= old * old
            n = len(dq)
            if n >= 2:
                mean = self._sum_ret[symbol] / n
                var = max(0.0, (self._sumsq_ret[symbol] / n) - (mean * mean))
                self.last_vol_pct[symbol] = sqrt(var) * 100.0
        self._last_price[symbol] = price
        ph.add(price)

    def generate(self, symbol: str, last_price: float, bid: float, ask: float) -> Optional[Signal]:
        ph = self.history.get(symbol)
        if ph is None:
            return None
        prices = ph.as_list()
        if len(prices) < max(self.sma_slow, 15):
            return None

        sma_f = simple_moving_average(prices, self.sma_fast)
        sma_s = simple_moving_average(prices, self.sma_slow)
        rsi_v = rsi(prices, 14)
        bo = breakout(prices, 20)
        if sma_f is None or sma_s is None or rsi_v is None:
            return None

        spread_bps = (max(ask - bid, 1e-8) / last_price) * 1e4
        # More lenient spread filter for synthetic data
        if spread_bps > max(self.max_spread_bps, 500):
            return None

        # Volatility gating & regime detection
        vol_pct = self.last_vol_pct.get(symbol)
        if vol_pct is None:
            vol_pct = realized_volatility_pct(prices, window=self._vol_window)
        self.last_vol_pct[symbol] = float(vol_pct)
        # More lenient volatility gates
        in_gate = 0.01 <= vol_pct <= 5.0
        regime = "trend" if abs((sma_f - sma_s) / last_price) > 0.001 else "range"
        self.last_regime[symbol] = regime

        side: Optional[Side] = None
        reason = []
        if not in_gate:
            return None
            
        # More aggressive signal generation
        sma_diff_pct = ((sma_f - sma_s) / sma_s) * 100
        
        # Mean reversion signals
        if self.enable_mean_reversion:
            if sma_diff_pct > 0.1 and rsi_v <= self.rsi_buy:
                side = Side.BUY
                reason.append("mean_reversion_buy")
            elif sma_diff_pct < -0.1 and rsi_v >= self.rsi_sell:
                side = Side.SELL
                reason.append("mean_reversion_sell")
                
        # Momentum signals (more aggressive)
        if rsi_v < 45:
            side = Side.BUY
            reason.append("rsi_oversold")
        elif rsi_v > 55:
            side = Side.SELL  
            reason.append("rsi_overbought")

        # Breakout signals  
        if self.enable_breakout:
            if bo == "breakout_up":
                side = Side.BUY
                reason.append("breakout_up")
            elif bo == "breakout_down":
                side = Side.SELL
                reason.append("breakout_down")
                
        # Price-SMA divergence signals
        price_above_sma = last_price > sma_f
        if price_above_sma and sma_f > sma_s:
            if side != Side.SELL:  # Don't override existing sell
                side = Side.BUY
                reason.append("price_momentum_up")
        elif not price_above_sma and sma_f < sma_s:
            side = Side.SELL
            reason.append("price_momentum_down")

        # Store metrics for UI/telemetry
        self.last_rsi[symbol] = float(rsi_v)
        self.last_sma_fast[symbol] = float(sma_f)
        self.last_sma_slow[symbol] = float(sma_s)
        self.last_spread_bps[symbol] = float(spread_bps)

        # More lenient debouncing - allow same side if confidence increased
        if side is None:
            return None
            
        # Allow same-side signals if reason changed or time passed
        reason_text = ",".join(reason) if reason else ""
        last_reason = self.last_reason.get(symbol, "")
        if self.last_side.get(symbol) == side and reason_text == last_reason:
            return None

        reason_text = ",".join(reason)
        self.last_reason[symbol] = reason_text
        self.last_side[symbol] = side
        return Signal(symbol=symbol, side=side, confidence=0.6, reason=reason_text, qty=0.0)

    def get_metrics(self, symbol: str) -> Dict[str, float | str]:
        return {
            "rsi": self.last_rsi.get(symbol),
            "sma_fast": self.last_sma_fast.get(symbol),
            "sma_slow": self.last_sma_slow.get(symbol),
            "spread_bps": self.last_spread_bps.get(symbol),
            "reason": self.last_reason.get(symbol, ""),
            "last_side": self.last_side.get(symbol).value if self.last_side.get(symbol) else None,
            "vol_pct": self.last_vol_pct.get(symbol),
            "regime": self.last_regime.get(symbol),
        }


