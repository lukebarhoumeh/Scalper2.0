from __future__ import annotations

import asyncio
from typing import Dict, List
from datetime import datetime, timezone, timedelta

from .config import BotConfig
from .logger import setup_logging
from .state import initialize_account, load_state, save_state, clear_state_file
from .models import Order, OrderType, Side, TradeRecord, Signal
from .market import MarketDataFeed
from .broker import PaperBroker
from .strategy import StrategyEngine
from .risk import (
    RiskLimits,
    compute_position_size_usd,
    can_increase_exposure,
    compute_vol_targeted_qty,
    current_total_exposure_usd,
    current_symbol_exposure_usd,
    current_alt_exposure_usd,
)
from .ui import TerminalUI
from .ledger import Ledger
from .analytics import compute_kpis
from .dynamic_config import DynamicConfigManager, StrategyParams
from .advanced_strategy import AdvancedStrategyEngine
from .profit_manager import ProfitManager
from .execution_optimizer import ExecutionOptimizer, SmartOrderRouter
from .performance_optimizer import PerformanceOptimizer
from .ml_predictor import MLPredictor, MarketFeatures
from .kelly_sizing import KellySizer


class App:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.log, self.audit = setup_logging(cfg.logs_dir)
        # Always start fresh if configured
        if cfg.reset_state_on_start:
            clear_state_file(cfg.state_path)
        acct = initialize_account(cfg.starting_cash_usd)
        self.account = load_state(cfg.state_path, acct)
        self.market = MarketDataFeed(cfg.trading_pairs, cfg.poll_interval_sec, source=cfg.data_source, request_rate_limit=cfg.max_requests_per_sec)
        self.broker = PaperBroker(self.account)
        self.strategy = StrategyEngine(
            rsi_buy=cfg.rsi_buy,
            rsi_sell=cfg.rsi_sell,
            sma_fast=cfg.sma_fast,
            sma_slow=cfg.sma_slow,
            max_spread_bps=cfg.spread_bps_max,
            enable_mean_reversion=cfg.enable_mean_reversion,
            enable_breakout=cfg.enable_breakout,
            enable_grid_overlay=cfg.enable_grid_overlay,
        )
        self.ui = TerminalUI()
        self.recent_trades: List[TradeRecord] = []
        self._shutdown = asyncio.Event()
        self._ticks_processed: int = 0
        self._trades_count: int = 0
        self._last_tick_time: Dict[str, datetime] = {}
        self._starting_equity: float = self.account.equity
        self._pause_trading: bool = False
        self.ledger = Ledger()
        
        # Advanced components
        self.advanced_strategy = AdvancedStrategyEngine()
        self.profit_manager = ProfitManager(
            initial_capital=cfg.starting_cash_usd,
            min_working_capital=cfg.starting_cash_usd * 0.5,
            profit_threshold_pct=0.05,  # 5% profit triggers evaluation
            base_withdrawal_rate=0.40,  # 40% of profits
            withdrawal_cooldown_hours=4  # 4 hours between withdrawals
        )
        self._last_daily_pnl = 0.0
        self._day_start_equity = self.account.equity
        
        # Performance optimization
        self.performance_optimizer = PerformanceOptimizer(target_latency_us=100)
        self.execution_optimizer = ExecutionOptimizer()
        self.smart_router = SmartOrderRouter()
        
        # Machine learning
        self.ml_predictor = MLPredictor(min_confidence=0.7)
        
        # Kelly sizing
        self.kelly_sizer = KellySizer(max_kelly_fraction=0.25)
        # Dynamic config manager (bounded auto-tuning)
        self._dyn = DynamicConfigManager(
            initial=StrategyParams(
                rsi_buy=cfg.rsi_buy,
                rsi_sell=cfg.rsi_sell,
                sma_fast=cfg.sma_fast,
                sma_slow=cfg.sma_slow,
                max_spread_bps=cfg.spread_bps_max,
                take_profit_bps=cfg.take_profit_bps,
                stop_loss_bps=cfg.stop_loss_bps,
                trailing_stop_bps=cfg.trailing_stop_bps,
                pos_size_multiplier=1.0,
            ),
            history_path="logs/param_history.json",
            cooldown_updates=20,
            activity_target_tph=cfg.activity_target_tph,
        )

    async def run(self) -> None:
        self.log.info("Starting ScalperBot 2.0 app loop")
        # Queues for micro-loops (bounded for backpressure)
        self._tick_q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._order_q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        # Background tasks
        tasks = [
            asyncio.create_task(self._data_loop(), name="data_loop"),
            asyncio.create_task(self._strategy_loop(), name="strategy_loop"),
            asyncio.create_task(self._execution_loop(), name="execution_loop"),
            asyncio.create_task(self._risk_exit_loop(), name="risk_exit_loop"),
            asyncio.create_task(self._ui_loop(), name="ui_loop"),
            asyncio.create_task(self._periodic_save(), name="save_loop"),
            asyncio.create_task(self._periodic_ledger_flush(), name="ledger_flush"),
            asyncio.create_task(self._periodic_profit_check(), name="profit_check"),
        ]
        try:
            await self._shutdown.wait()
        finally:
            for t in tasks:
                t.cancel()
            for t in tasks:
                try:
                    await t
                except Exception:
                    pass
            save_state(self.cfg.state_path, self.account)
            self.log.info("Shutdown complete")

    async def _periodic_save(self) -> None:
        try:
            while not self._shutdown.is_set():
                await asyncio.sleep(60)
                save_state(self.cfg.state_path, self.account)
        except asyncio.CancelledError:
            return

    def request_shutdown(self) -> None:
        self._shutdown.set()

    async def _periodic_ledger_flush(self) -> None:
        try:
            while not self._shutdown.is_set():
                await asyncio.sleep(self.cfg.ledger_flush_sec)
                self.ledger.flush_to_csv("logs/ledger.csv")
        except asyncio.CancelledError:
            return
            
    async def _periodic_profit_check(self) -> None:
        """Periodically check for profit withdrawal opportunities"""
        try:
            while not self._shutdown.is_set():
                await asyncio.sleep(60)  # Check every minute
                
                # Update daily P&L
                current_equity = self.account.equity
                daily_pnl = current_equity - self._day_start_equity
                
                # Check if day rolled over (simplified - in production use proper timezone)
                now = datetime.now(timezone.utc)
                if now.hour == 0 and now.minute < 1:
                    self.profit_manager.update_daily_performance(daily_pnl)
                    self._day_start_equity = current_equity
                    self._last_daily_pnl = 0.0
                else:
                    self._last_daily_pnl = daily_pnl
                
                # Evaluate profit withdrawal
                plan = self.profit_manager.evaluate_profits(self.account)
                if plan and plan.withdrawal > 0:
                    # Execute withdrawal
                    if self.profit_manager.execute_withdrawal(self.account, plan):
                        self.log.info(f"💰 Profit withdrawal: ${plan.withdrawal:.2f} - {plan.reason}")
                        self.audit.info(f"WITHDRAWAL|{datetime.now(timezone.utc).isoformat()}|${plan.withdrawal:.2f}|{plan.reason}")
                        
                        # Force save state after withdrawal
                        save_state(self.cfg.state_path, self.account)
                        
        except asyncio.CancelledError:
            return

    async def _data_loop(self) -> None:
        try:
            async for tick in self.market.ticks():
                # Update broker prices and strategy history quickly
                self.broker.update_price(tick.symbol, tick.bid, tick.ask)
                self.strategy.on_price(tick.symbol, tick.price)
                self._ticks_processed += 1
                self._last_tick_time[tick.symbol] = tick.ts
                # Backpressure if queue is full
                await self._tick_q.put(tick)
                if self._shutdown.is_set():
                    break
            # If the feed ends (finite historical), request shutdown
            self._shutdown.set()
        except asyncio.CancelledError:
            return

    async def _strategy_loop(self) -> None:
        try:
            while not self._shutdown.is_set():
                tick = await self._tick_q.get()
                # Apply dynamic params to strategy before generating
                dp = self._dyn.get_params()
                self.strategy.rsi_buy = dp.rsi_buy
                self.strategy.rsi_sell = dp.rsi_sell
                self.strategy.sma_fast = dp.sma_fast
                self.strategy.sma_slow = dp.sma_slow
                self.strategy.max_spread_bps = dp.max_spread_bps
                if self._pause_trading:
                    continue
                    
                # Get signals from multiple sources
                # Basic strategy
                basic_sig = self.strategy.generate(tick.symbol, tick.price, tick.bid, tick.ask)
                
                # Prepare data
                ph = self.strategy.history.get(tick.symbol)
                prices = ph.as_list() if ph else []
                volumes = [100.0] * len(prices)  # Synthetic volume for now
                
                # Advanced strategy signals
                advanced_sigs = self.advanced_strategy.generate_signals(
                    tick.symbol, prices, volumes, tick.bid, tick.ask, tick.price
                )
                
                # ML predictions
                ml_sig = None
                if len(prices) >= 60:
                    features = self.ml_predictor.extract_features(
                        tick.symbol, prices, volumes,
                        {'bid': tick.bid, 'ask': tick.ask, 'bid_size': 1.0, 'ask_size': 1.0},
                        []  # No recent trades for synthetic
                    )
                    ml_prediction = self.ml_predictor.predict(tick.symbol, features)
                    if ml_prediction and ml_prediction.confidence >= 0.75:
                        ml_sig = Signal(
                            symbol=tick.symbol,
                            side=Side.BUY if ml_prediction.direction == "buy" else Side.SELL,
                            confidence=ml_prediction.confidence,
                            reason=f"ml_{ml_prediction.model_name}",
                            qty=0.0
                        )
                
                # Combine signals with priority
                sig = None
                if ml_sig and ml_sig.confidence >= 0.8:
                    sig = ml_sig
                elif advanced_sigs and advanced_sigs[0].confidence >= 0.7:
                    sig = advanced_sigs[0]
                    sig.symbol = tick.symbol
                elif basic_sig:
                    sig = basic_sig
                if sig is None:
                    continue
                limits = RiskLimits(
                    risk_per_trade=self.cfg.risk_per_trade,
                    max_total_exposure_usd=self.cfg.max_total_exposure_usd,
                    max_daily_loss_percent=self.cfg.max_daily_loss_percent,
                    max_symbol_usd=self.cfg.max_position_usd,
                    alt_exposure_usd_cap=self.cfg.alt_exposure_usd_cap,
                )
                last_price = self.broker.last_prices.get(sig.symbol)
                if last_price is None or last_price <= 0:
                    continue
                ph = self.strategy.history.get(sig.symbol)
                prices = ph.as_list() if ph else []
                strat_vol = self.strategy.last_vol_pct.get(sig.symbol) if hasattr(self.strategy, 'last_vol_pct') else None
                qty, _ = compute_vol_targeted_qty(
                    self.account,
                    limits,
                    last_price,
                    prices,
                    vol_window=max(self.cfg.sma_fast, 20),
                    precomputed_vol_pct=strat_vol,
                )
                qty *= self._dyn.pos_multiplier()
                usd = qty * last_price
                sig.qty = qty
                # Use Kelly sizing for optimal position size
                stop_loss_price = last_price * (1 - (dp.stop_loss_bps / 1e4))
                
                kelly_size = self.kelly_sizer.calculate_position_size(
                    symbol=sig.symbol,
                    strategy=sig.reason,
                    account_equity=self.account.equity,
                    current_price=last_price,
                    stop_loss_price=stop_loss_price,
                    confidence=sig.confidence
                )
                
                # Update position size with Kelly recommendation
                kelly_risk_fraction = kelly_size.conservative_fraction
                kelly_position_usd = kelly_size.position_dollars
                
                # Apply Kelly sizing while respecting caps
                usd = min(kelly_position_usd, usd)  # Use smaller of vol-targeted or Kelly
                sig.qty = usd / last_price
                
                # Clamp to available caps
                total_remaining = max(0.0, limits.max_total_exposure_usd - current_total_exposure_usd(self.account, self.broker.last_prices))
                symbol_remaining = max(0.0, limits.max_symbol_usd - current_symbol_exposure_usd(self.account, self.broker.last_prices, sig.symbol))
                alt_remaining = float('inf')
                base = sig.symbol.split("-")[0].upper()
                if base != "BTC":
                    alt_remaining = max(0.0, limits.alt_exposure_usd_cap - current_alt_exposure_usd(self.account, self.broker.last_prices))
                allowed_usd = max(0.0, min(total_remaining, symbol_remaining, alt_remaining))
                if allowed_usd <= 0.0:
                    continue
                if usd > allowed_usd:
                    usd = allowed_usd
                    sig.qty = max(allowed_usd / last_price, 0.0)
                # Final exposure check
                if not can_increase_exposure(self.account, self.broker.last_prices, limits, usd, sig.symbol):
                    continue
                tp = last_price * (1 + (dp.take_profit_bps / 1e4) * (1 if sig.side == Side.BUY else -1))
                sl = last_price * (1 - (dp.stop_loss_bps / 1e4) * (1 if sig.side == Side.BUY else -1))
                order = Order(
                    symbol=sig.symbol,
                    side=sig.side,
                    qty=qty,
                    type=OrderType.MARKET,
                    take_profit=tp,
                    stop_loss=sl,
                )
                await self._order_q.put(order)
        except asyncio.CancelledError:
            return

    async def _execution_loop(self) -> None:
        try:
            while not self._shutdown.is_set():
                order: Order = await self._order_q.get()
                
                # Optimize execution with smart routing
                market_data = {
                    'bid': self.broker.last_prices.get(order.symbol, order.price or 0) * 0.999,
                    'ask': self.broker.last_prices.get(order.symbol, order.price or 0) * 1.001,
                    'volatility': self.strategy.last_vol_pct.get(order.symbol, 1.0),
                    'momentum': 0.0,  # Simplified
                    'avg_volume': 1000
                }
                
                # Get optimized child orders
                child_orders = await self.smart_router.route_order(order, market_data)
                
                # Execute child orders
                for child_order in child_orders:
                    fill = self.broker.place_order(child_order)
                    if fill is not None:
                        tr = TradeRecord(order=child_order, fill=fill, realized_pnl=self.account.realized_pnl)
                        self.recent_trades.append(tr)
                        self._trades_count += 1
                        self.audit.info(f"{fill.ts.isoformat()}|{fill.symbol}|{fill.side.value}|{fill.qty}|{fill.price}")
                        
                        # Record in ledger
                        try:
                            last_mid = self.broker.last_prices.get(fill.symbol, fill.price)
                            self.ledger.add_fill(child_order, fill, last_mid, last_mid, self.account.realized_pnl)
                        except Exception:
                            pass
                        
                        # Update Kelly sizer with result
                        # This would need outcome tracking in production
                        won = fill.side == Side.SELL  # Simplified
                        r_multiple = 0.5 if won else -0.5  # Simplified
                        self.kelly_sizer.update_performance(
                            fill.symbol, order.reason if hasattr(order, 'reason') else 'unknown',
                            won, r_multiple
                        )
        except asyncio.CancelledError:
            return

    async def _risk_exit_loop(self) -> None:
        try:
            while not self._shutdown.is_set():
                await asyncio.sleep(max(0.1, self.cfg.poll_interval_sec))
                # Stale watchdog
                now = datetime.now(timezone.utc)
                stale_symbols = [s for s in self.cfg.trading_pairs if (now - self._last_tick_time.get(s, now)) > timedelta(seconds=max(5, int(self.cfg.poll_interval_sec * 4)))]
                self._pause_trading = bool(stale_symbols)
                # Holding-time exit
                max_hold_minutes = 30
                for sym, pos in list(self.account.positions.items()):
                    if pos.qty <= 0 or not pos.opened_ts:
                        continue
                    if (now - pos.opened_ts) > timedelta(minutes=max_hold_minutes):
                        await self._order_q.put(Order(symbol=sym, side=Side.SELL, qty=pos.qty, type=OrderType.MARKET))
                # TP/SL/Trailing exits
                exit_fills = self.broker.check_exits(self.cfg.trailing_stop_bps)
                for ef in exit_fills:
                    tr = TradeRecord(order=Order(symbol=ef.symbol, side=ef.side, qty=ef.qty), fill=ef, realized_pnl=self.account.realized_pnl)
                    self.recent_trades.append(tr)
                    self._trades_count += 1
                    self.audit.info(f"{ef.ts.isoformat()}|{ef.symbol}|{ef.side.value}|{ef.qty}|{ef.price}")
                # Markout breaker
                try:
                    self.ledger.update_markouts(self.broker.last_prices, now)
                    recent = [e for e in self.ledger.entries if (now - e.ts).total_seconds() <= 10 and e.markout_5s_bps is not None]
                    if len(recent) >= 5:
                        avg_markout = sum(e.markout_5s_bps for e in recent if e.markout_5s_bps is not None) / max(1, len(recent))
                        if avg_markout < -5.0:
                            self._pause_trading = True
                            self.log.warning(f"Markout breaker triggered avg={avg_markout:.1f}bps; pausing")
                except Exception:
                    pass
        except asyncio.CancelledError:
            return

    async def _ui_loop(self) -> None:
        try:
            while not self._shutdown.is_set():
                await asyncio.sleep(max(0.2, 1.0 / max(0.1, self.cfg.ui_refresh_hz)))
                # Exposure
                exposure = 0.0
                for sym, pos in self.account.positions.items():
                    price = self.broker.last_prices.get(sym)
                    if price is not None:
                        exposure += abs(pos.qty * price)
                alt_exposure = 0.0
                for sym, pos in self.account.positions.items():
                    price = self.broker.last_prices.get(sym)
                    if price is None:
                        continue
                    base = sym.split("-")[0].upper()
                    if base != "BTC":
                        alt_exposure += abs(pos.qty * price)
                # KPIs
                k = compute_kpis(self.recent_trades)
                # Dynamic tuning (bounded, periodic)
                try:
                    avg_spread = sum(self.strategy.last_spread_bps.values()) / max(1, len(self.strategy.last_spread_bps)) if self.strategy.last_spread_bps else None
                    avg_vol = sum(self.strategy.last_vol_pct.values()) / max(1, len(self.strategy.last_vol_pct)) if getattr(self.strategy, 'last_vol_pct', None) else None
                    recent_markouts = [e.markout_5s_bps for e in self.ledger.entries if e.markout_5s_bps is not None]
                    avg_markout_5s = (sum(recent_markouts) / len(recent_markouts)) if recent_markouts else None
                    runtime_hours = max(1e-6, self._ticks_processed * self.cfg.poll_interval_sec / 3600.0)
                    tph = self._trades_count / runtime_hours if self._trades_count > 0 else 0.0
                    self._dyn.maybe_update(
                        session_win_rate_pct=k.win_rate_pct,
                        session_sharpe=k.sharpe,
                        avg_markout_5s_bps=avg_markout_5s,
                        avg_spread_bps=avg_spread,
                        avg_vol_pct=avg_vol,
                        activity_tph=tph,
                    )
                except Exception:
                    pass
                stats = {
                    "ticks": float(self._ticks_processed),
                    "trades": float(self._trades_count),
                    "win_rate": k.win_rate_pct,
                    "exposure": exposure,
                    "last_tick_age": max(
                        0.0,
                        max(
                            (
                                (datetime.now(timezone.utc) - self._last_tick_time.get(sym, datetime.now(timezone.utc))).total_seconds()
                                for sym in self.cfg.trading_pairs
                            ),
                            default=0.0,
                        ),
                    ),
                    "connection": self.cfg.data_source,
                    "util_total": (exposure / self.cfg.max_total_exposure_usd) if self.cfg.max_total_exposure_usd > 0 else 0.0,
                    "util_alt": (alt_exposure / self.cfg.alt_exposure_usd_cap) if self.cfg.alt_exposure_usd_cap > 0 else 0.0,
                }
                ind = {sym: self.strategy.get_metrics(sym) for sym in self.cfg.trading_pairs}
                self.ui.render(self.account, self.recent_trades, stats, ind)
        except asyncio.CancelledError:
            return


