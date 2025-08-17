# strategy_engine.py  – Phase 2
"""
High-level orchestration of multiple trading strategies for ScalperBot 2.0.

Features
────────
• Cool-down timer (per-coin)                     → avoids rapid-fire orders
• Inventory cap & side-aware trading            → bounded exposure
• Vol-scaled trade size                         → risk-proportional sizing
• Config-driven parameters (see .env)           → no code edits required
"""

from __future__ import annotations

import logging, signal, threading, time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List

import pandas as pd
import ta                         # TA-Lib-style indicators

from config import TRADE_COINS, POLL_INTERVAL_SEC
from market_data import (
    get_best_bid_ask,
    get_historic_candles,
    realised_volatility,
    mid_price,
)
from trade_executor import TradeExecutor
from adaptive_utils import (
    CooldownTracker,
    vol_scaled_size,
    inventory_allows,
)

# ────────────────────────── logging ──────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
_LOG = logging.getLogger("StrategyEngine")

# ────────────────────────── data models ──────────────────────
class Action(Enum):
    BUY = auto()
    SELL = auto()


@dataclass(frozen=True)
class MarketSnapshot:
    product_id: str
    bid: float
    ask: float
    last: float
    candles: pd.DataFrame
    realised_vol: float  # % annualised


# ────────────────────────── base strategy ────────────────────
class BaseStrategy:
    """
    Abstract strategy; subclasses must implement `evaluate()`.
    """

    name: str  # must be set by subclass

    def __init__(self, executor: TradeExecutor):
        self._exe = executor
        if not getattr(self, "name", None):
            raise ValueError("Strategy subclass must define a 'name' attribute")

    def evaluate(self, snap: MarketSnapshot) -> Action | None:  # pragma: no cover
        raise NotImplementedError

    # helper for subclasses
    def _submit(self, action: Action, snap: MarketSnapshot, usd_size: float) -> None:
        if action == Action.BUY:
            self._exe.market_buy(snap.product_id, usd_size, self.name)
        else:
            self._exe.market_sell(snap.product_id, usd_size, self.name)
        _LOG.info("[%s] %s %s for $%.2f", self.name, action.name, snap.product_id, usd_size)


# ────────────────────────── Scalper v2 ───────────────────────
class ScalperStrategy(BaseStrategy):
    """
    Mean-reversion scalper with:
      • cool-down
      • vol filter
      • spread threshold
      • inventory cap
      • vol-scaled sizing
    """

    name = "scalper_v2"

    def __init__(
        self,
        executor: TradeExecutor,
        sma_window: int = 20,
        vol_thresh: float = 10.0,
        spread_thresh_pct: float = 0.20,
    ):
        super().__init__(executor)
        self.sma_window = sma_window
        self.vol_thresh = vol_thresh
        self.spread_thresh_pct = spread_thresh_pct
        self._cooldown = CooldownTracker()  # default window from config

    def evaluate(self, snap: MarketSnapshot) -> Action | None:
        if len(snap.candles) < self.sma_window:
            return None

        if snap.realised_vol > self.vol_thresh:
            return None  # too turbulent

        if not self._cooldown.ready(snap.product_id):
            return None  # still cooling down

        sma = snap.candles["close"].tail(self.sma_window).mean()
        mid = (snap.bid + snap.ask) / 2.0
        dev_pct = (mid - sma) / sma * 100.0

        action: Action | None = None
        if dev_pct <= -self.spread_thresh_pct:
            action = Action.BUY
        elif dev_pct >= self.spread_thresh_pct:
            action = Action.SELL

        if action:
            usd_size = vol_scaled_size(snap.realised_vol)

            # inventory gate
            if not inventory_allows(self._exe, snap.product_id, action.name, usd_size):
                return None

            self._submit(action, snap, usd_size)
            self._cooldown.stamp(snap.product_id)
        return action


# ────────────────────────── Breakout ATR ─────────────────────
class BreakoutStrategy(BaseStrategy):
    """
    Donchian-style breakout with ATR buffer
    """

    name = "breakout_atr_v1"

    def __init__(
        self,
        executor: TradeExecutor,
        lookback: int = 50,
        atr_window: int = 14,
        atr_multiplier: float = 1.5,
    ):
        super().__init__(executor)
        self.lb = lookback
        self.win = atr_window
        self.k = atr_multiplier
        self._last: Dict[str, Action | None] = {}  # last signal per product

    def evaluate(self, snap: MarketSnapshot) -> Action | None:
        df = snap.candles
        if len(df) < max(self.lb, self.win) + 1:
            return None

        atr = ta.volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], self.win
        ).average_true_range().iloc[-1]

        ch_high = df["high"].tail(self.lb).max() + self.k * atr
        ch_low = df["low"].tail(self.lb).min() - self.k * atr
        price = df["close"].iloc[-1]

        act = Action.BUY if price > ch_high else Action.SELL if price < ch_low else None
        if act and self._last.get(snap.product_id) != act:
            usd_size = vol_scaled_size(snap.realised_vol)
            if inventory_allows(self._exe, snap.product_id, act.name, usd_size):
                self._submit(act, snap, usd_size)
                self._last[snap.product_id] = act
                return act
        return None


# ────────────────────────── engine core ──────────────────────
class StrategyEngine:
    """
    Coordinates snapshot collection and strategy evaluation on a fixed schedule.
    """

    def __init__(self, executor: TradeExecutor, poll_interval: int | None = None):
        self._exe = executor
        self._poll = poll_interval or POLL_INTERVAL_SEC
        self._strategies: List[BaseStrategy] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # public API
    def register(self, strategy: BaseStrategy) -> None:
        self._strategies.append(strategy)
        _LOG.info("Registered strategy: %s", strategy.name)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Engine already running")
        _LOG.info("Starting StrategyEngine polling every %d s", self._poll)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        signal.signal(signal.SIGINT, lambda *_: self.stop())

    def stop(self) -> None:
        _LOG.info("Stopping StrategyEngine …")
        self._stop.set()
        if self._thread:
            self._thread.join()
        _LOG.info("Engine stopped.")

    # internal
    def _run_loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self._tick()
            except Exception:                                # any bug → log, continue
                _LOG.exception("Unhandled error in engine tick")
            time.sleep(max(0.0, self._poll - (time.time() - t0)))

    def _tick(self) -> None:
        for symbol in TRADE_COINS:
            pid = f"{symbol}-USD"
            try:
                snap = self._collect_snapshot(pid)
            except Exception as exc:                         # per-symbol isolation
                _LOG.warning("Snapshot failure for %s: %s", pid, exc, exc_info=False)
                continue
            for strat in self._strategies:
                try:
                    strat.evaluate(snap)
                except Exception as exc:
                    _LOG.warning("[%s] evaluation error: %s", strat.name, exc, exc_info=False)

    # helper: build a MarketSnapshot
    @staticmethod
    def _collect_snapshot(product_id: str) -> MarketSnapshot:
        bid, ask = get_best_bid_ask(product_id)
        candles  = get_historic_candles(product_id, granularity_sec=60, lookback=300)
        vol      = realised_volatility(candles)
        return MarketSnapshot(product_id, bid, ask, mid_price(bid, ask), candles, vol)
