"""
trade_executor_production.py - Elite Production HFT Trade Executor
Features:
- Fresh start daily (no position carryover)
- $1000 daily capital limit with tracking
- Smart order routing with limit orders
- Advanced slippage protection
- Real-time P&L tracking
"""

from __future__ import annotations
import uuid
import time
import json
import threading
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

from coinbase.rest import RESTClient
from market_data import get_best_bid_ask, MarketDataManager
from risk_manager import check_trade_risk

# Import from production config
try:
    from config_production_hft import *
except ImportError:
    from config import *

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT_IOC = "limit_ioc"     # Immediate or Cancel
    LIMIT_GTT = "limit_gtt"     # Good Till Time
    LIMIT_POST = "limit_post"   # Post-only (maker)

@dataclass
class ExecutionMetrics:
    """Track execution quality metrics"""
    total_trades: int = 0
    successful_trades: int = 0
    rejected_trades: int = 0
    total_slippage_bps: float = 0.0
    avg_execution_ms: float = 0.0
    daily_volume_usd: float = 0.0
    daily_pnl: float = 0.0
    daily_capital_used: float = 0.0

class ProductionTradeExecutor:
    """
    Production-grade trade executor with professional risk management
    """

    def __init__(self, logger):
        self._client: Optional[RESTClient] = None
        self._logger = logger
        
        # Market data manager
        self._market_data = MarketDataManager()
        
        # Position tracking - starts fresh daily
        self._inventory: Dict[str, float] = {}              # Base units per product
        self._avg_entry_price: Dict[str, float] = {}        # VWAP entry prices
        self._cashflow: Dict[str, float] = defaultdict(float)
        self._realised_pnl: Dict[str, float] = defaultdict(float)
        
        # Daily capital management
        self._daily_capital_limit = MAX_DAILY_CAPITAL
        self._daily_capital_used = 0.0
        self._daily_pnl = 0.0
        self._trading_day_start = self._get_trading_day_start()
        
        # Execution tracking
        self._execution_metrics = ExecutionMetrics()
        self._order_history = deque(maxlen=1000)
        self._pending_orders: Dict[str, dict] = {}
        
        # Performance tracking
        self._trade_latencies = deque(maxlen=100)
        self._slippage_history = defaultdict(lambda: deque(maxlen=50))
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Fresh start check
        self._check_fresh_start()
        
        self._logger.info(f"Production Trade Executor initialized. Daily limit: ${self._daily_capital_limit}")
    
    def _get_trading_day_start(self) -> datetime:
        """Get the start of current trading day (UTC)"""
        now = datetime.now(timezone.utc)
        return now.replace(hour=RESET_HOUR_UTC, minute=0, second=0, microsecond=0)
    
    def _check_fresh_start(self):
        """Check if we need to start fresh for the day"""
        if START_FRESH_DAILY:
            current_time = datetime.now(timezone.utc)
            if current_time >= self._trading_day_start:
                self._reset_daily_tracking()
                self._logger.info("Starting fresh for new trading day")
    
    def _reset_daily_tracking(self):
        """Reset all daily tracking (called at start of each day)"""
        with self._lock:
            # Clear positions if configured
            if CLEAR_POSITIONS_ON_START:
                self._inventory.clear()
                self._avg_entry_price.clear()
                self._cashflow.clear()
            
            # Reset daily metrics
            self._daily_capital_used = 0.0
            self._daily_pnl = 0.0
            self._execution_metrics = ExecutionMetrics()
            
            # Log reset
            self._logger.info(f"Daily reset completed at {datetime.now(timezone.utc)}")
    
    def can_trade(self, size_usd: float) -> Tuple[bool, str]:
        """Check if we have capital available for this trade"""
        with self._lock:
            # Check daily capital limit
            if self._daily_capital_used + size_usd > self._daily_capital_limit:
                available = self._daily_capital_limit - self._daily_capital_used
                return False, f"Daily capital limit reached. Available: ${available:.2f}"
            
            # Check daily loss limit
            if self._daily_pnl < -RISK_LIMITS["max_daily_loss_usd"]:
                return False, f"Daily loss limit hit: ${-self._daily_pnl:.2f}"
            
            return True, "OK"
    
    def market_buy(self, product_id: str, usd_notional: float, strategy: str) -> Optional[str]:
        """Execute smart buy order"""
        return self._place_smart_order("BUY", product_id, usd_notional, strategy)
    
    def market_sell(self, product_id: str, usd_notional: float, strategy: str) -> Optional[str]:
        """Execute smart sell order"""
        return self._place_smart_order("SELL", product_id, usd_notional, strategy)
    
    def _place_smart_order(
        self, 
        side: str, 
        product_id: str, 
        usd_notional: float, 
        strategy: str,
        urgency: str = "normal"
    ) -> Optional[str]:
        """
        Smart order routing with limit order preference
        """
        exec_start = time.perf_counter()
        
        with self._lock:
            # Check capital availability
            can_trade, reason = self.can_trade(usd_notional)
            if not can_trade:
                logger.warning(f"Trade rejected: {reason}")
                self._execution_metrics.rejected_trades += 1
                return None
            
            # Get current market
            try:
                bid, ask = get_best_bid_ask(product_id)
                mid = (bid + ask) / 2
                
                # Pre-trade risk check
                risk_check = check_trade_risk(
                    product_id, side, usd_notional, bid, ask, mid
                )
                
                if not risk_check.approved:
                    logger.warning(f"Risk check failed: {risk_check.reason}")
                    self._execution_metrics.rejected_trades += 1
                    return None
                
                # Smart order type selection
                order_type, limit_price = self._select_order_type(
                    side, bid, ask, risk_check, urgency
                )
                
                # Calculate quantity
                exec_price = limit_price if limit_price else (ask if side == "BUY" else bid)
                qty_base = usd_notional / exec_price
                
                # Execute order
                order_id, fill_price, fill_qty, status = self._execute_order(
                    side, product_id, qty_base, order_type, limit_price
                )
                
                if status == "filled":
                    # Update tracking
                    self._update_position(product_id, side, fill_qty, fill_price, usd_notional)
                    
                    # Record execution
                    exec_time_ms = (time.perf_counter() - exec_start) * 1000
                    slippage_bps = abs(fill_price - mid) / mid * 10000
                    
                    self._record_execution(
                        product_id, side, fill_qty, fill_price, 
                        slippage_bps, exec_time_ms, strategy
                    )
                    
                    # Log trade to CSV using trade logger
                    if hasattr(self, '_trade_logger') and self._trade_logger:
                        self._trade_logger.log(
                            product_id,
                            side.lower(),
                            fill_qty,
                            fill_price,
                            strategy,
                            self._daily_pnl,
                            order_id
                        )
                    
                    # Log to standard logger
                    self._logger.info(f"Trade executed: {order_id} - {side} {fill_qty} {product_id} @ ${fill_price}")
                    
                    return order_id
                else:
                    logger.warning(f"Order failed: {status}")
                    self._execution_metrics.rejected_trades += 1
                    return None
                    
            except Exception as e:
                logger.error(f"Order execution failed: {e}")
                self._execution_metrics.rejected_trades += 1
                return None
    
    def _select_order_type(
        self, 
        side: str, 
        bid: float, 
        ask: float,
        risk_check,
        urgency: str
    ) -> Tuple[OrderType, Optional[float]]:
        """Intelligent order type selection"""
        
        # High urgency or high risk = market order
        if urgency == "urgent" or risk_check.risk_score > 70:
            return OrderType.MARKET, None
        
        # Prefer limit orders for better execution
        if ORDER_ROUTING["prefer_limit_orders"]:
            spread_bps = (ask - bid) / bid * 10000
            
            # For tight spreads, use post-only
            if spread_bps < 5 and urgency == "patient":
                if side == "BUY":
                    limit_price = bid  # Join the bid
                else:
                    limit_price = ask  # Join the ask
                return OrderType.LIMIT_POST, limit_price
            
            # For normal conditions, use IOC with slight improvement
            else:
                improvement_bps = ORDER_ROUTING["limit_price_improvement_bps"]
                if side == "BUY":
                    limit_price = bid * (1 + improvement_bps / 10000)
                else:
                    limit_price = ask * (1 - improvement_bps / 10000)
                return OrderType.LIMIT_IOC, limit_price
        
        return OrderType.MARKET, None
    
    def _execute_order(
        self,
        side: str,
        product_id: str,
        qty_base: float,
        order_type: OrderType,
        limit_price: Optional[float]
    ) -> Tuple[str, float, float, str]:
        """Execute order with proper error handling"""
        
        if PAPER_TRADING:
            # Simulate execution
            fill_price = limit_price if limit_price else get_best_bid_ask(product_id)[1 if side == "BUY" else 0]
            
            # Simulate realistic slippage
            import random
            slippage_factor = 1 + random.uniform(-0.0005, 0.0005)  # ±5 bps
            fill_price *= slippage_factor
            
            order_id = f"SIM-{uuid.uuid4()}"
            return order_id, fill_price, qty_base, "filled"
        
        else:
            # Real order execution
            self._ensure_client()
            
            try:
                if order_type == OrderType.MARKET:
                    order = self._client.market_order(
                        product_id=product_id,
                        side=side.lower(),
                        size=str(qty_base)
                    )
                else:
                    order = self._client.limit_order(
                        product_id=product_id,
                        side=side.lower(),
                        size=str(qty_base),
                        limit_price=str(limit_price),
                        post_only=(order_type == OrderType.LIMIT_POST),
                        time_in_force="IOC" if order_type == OrderType.LIMIT_IOC else "GTT"
                    )
                
                # Wait for fill (with timeout)
                timeout = 5.0
                start = time.time()
                
                while time.time() - start < timeout:
                    order_status = self._client.get_order(order["id"])
                    if order_status["status"] in ["filled", "cancelled", "expired"]:
                        break
                    time.sleep(0.1)
                
                if order_status["status"] == "filled":
                    return (
                        order["id"],
                        float(order_status["average_filled_price"]),
                        float(order_status["filled_size"]),
                        "filled"
                    )
                else:
                    return order["id"], 0, 0, order_status["status"]
                    
            except Exception as e:
                logger.error(f"Order execution error: {e}")
                return str(uuid.uuid4()), 0, 0, "error"
    
    def _update_position(
        self, 
        product_id: str, 
        side: str, 
        qty: float, 
        price: float,
        usd_value: float
    ):
        """Update position tracking with VWAP"""
        sign = 1 if side == "BUY" else -1
        
        # Update inventory
        current_position = self._inventory.get(product_id, 0.0)
        new_position = current_position + (sign * qty)
        
        # Update VWAP
        if product_id not in self._avg_entry_price:
            self._avg_entry_price[product_id] = price
        else:
            if sign == 1:  # Buying - update VWAP
                total_value = (current_position * self._avg_entry_price[product_id]) + (qty * price)
                total_qty = current_position + qty
                if total_qty > 0:
                    self._avg_entry_price[product_id] = total_value / total_qty
            else:  # Selling - calculate P&L
                if current_position > 0:
                    pnl = qty * (price - self._avg_entry_price[product_id])
                    self._realised_pnl[product_id] += pnl
                    self._daily_pnl += pnl
        
        # Update position
        if abs(new_position) < 1e-10:
            # Position closed
            self._inventory[product_id] = 0
            self._avg_entry_price.pop(product_id, None)
        else:
            self._inventory[product_id] = new_position
        
        # Update capital tracking
        self._daily_capital_used += usd_value
        self._execution_metrics.daily_capital_used = self._daily_capital_used
        self._execution_metrics.daily_pnl = self._daily_pnl
    
    def _record_execution(
        self,
        product_id: str,
        side: str,
        qty: float,
        price: float,
        slippage_bps: float,
        exec_time_ms: float,
        strategy: str
    ):
        """Record execution metrics"""
        self._execution_metrics.total_trades += 1
        self._execution_metrics.successful_trades += 1
        self._execution_metrics.total_slippage_bps += slippage_bps
        self._execution_metrics.daily_volume_usd += qty * price
        
        # Update moving averages
        self._trade_latencies.append(exec_time_ms)
        self._execution_metrics.avg_execution_ms = sum(self._trade_latencies) / len(self._trade_latencies)
        
        # Track slippage by product
        self._slippage_history[product_id].append(slippage_bps)
        
        # Store order details
        self._order_history.append({
            'timestamp': datetime.now(timezone.utc),
            'product_id': product_id,
            'side': side,
            'qty': qty,
            'price': price,
            'slippage_bps': slippage_bps,
            'exec_time_ms': exec_time_ms,
            'strategy': strategy
        })
    
    def position_base(self, product_id: str) -> float:
        """Get current position in base currency"""
        return self._inventory.get(product_id, 0.0)
    
    def position_usd(self, product_id: str) -> float:
        """Get current position value in USD"""
        base = self.position_base(product_id)
        if base == 0:
            return 0.0
        
        bid, ask = get_best_bid_ask(product_id)
        mid = (bid + ask) / 2
        return base * mid
    
    def get_total_exposure(self) -> float:
        """Get total portfolio exposure in USD"""
        total = 0.0
        for product_id in self._inventory:
            total += abs(self.position_usd(product_id))
        return total
    
    def get_daily_stats(self) -> Dict:
        """Get comprehensive daily statistics"""
        with self._lock:
            # Check if we need to reset for new day
            now = datetime.now(timezone.utc)
            if now >= self._trading_day_start + timedelta(days=1):
                self._trading_day_start = self._get_trading_day_start()
                self._reset_daily_tracking()
            
            return {
                'timestamp': now.isoformat(),
                'daily_pnl': self._daily_pnl,
                'daily_capital_used': self._daily_capital_used,
                'daily_capital_remaining': self._daily_capital_limit - self._daily_capital_used,
                'total_exposure': self.get_total_exposure(),
                'total_trades': self._execution_metrics.total_trades,
                'successful_trades': self._execution_metrics.successful_trades,
                'rejected_trades': self._execution_metrics.rejected_trades,
                'avg_slippage_bps': (
                    self._execution_metrics.total_slippage_bps / 
                    self._execution_metrics.successful_trades
                    if self._execution_metrics.successful_trades > 0 else 0
                ),
                'avg_execution_ms': self._execution_metrics.avg_execution_ms,
                'win_rate': self._calculate_win_rate(),
                'positions': self._get_position_summary()
            }
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from order history"""
        if not self._order_history:
            return 0.0
        
        wins = sum(1 for order in self._order_history if order.get('pnl', 0) > 0)
        return wins / len(self._order_history) if self._order_history else 0.0
    
    def _get_position_summary(self) -> Dict:
        """Get summary of all current positions"""
        summary = {}
        for product_id, position in self._inventory.items():
            if position != 0:
                summary[product_id] = {
                    'base': position,
                    'usd': self.position_usd(product_id),
                    'avg_entry': self._avg_entry_price.get(product_id, 0),
                    'unrealized_pnl': self._calculate_unrealized_pnl(product_id)
                }
        return summary
    
    def _calculate_unrealized_pnl(self, product_id: str) -> float:
        """Calculate unrealized P&L for a position"""
        position = self._inventory.get(product_id, 0)
        if position == 0:
            return 0.0
        
        avg_entry = self._avg_entry_price.get(product_id, 0)
        if avg_entry == 0:
            return 0.0
        
        bid, ask = get_best_bid_ask(product_id)
        current_price = bid if position > 0 else ask  # Use exit price
        
        return position * (current_price - avg_entry)
    
    def _ensure_client(self):
        """Ensure Coinbase client is initialized"""
        if self._client is None:
            self._client = RESTClient(
                api_key=COINBASE_API_KEY,
                api_secret=COINBASE_API_SECRET
            )
    
    def get_order_book(self, product_id: str, level: int = 2) -> Dict:
        """Get order book data using market data manager"""
        return self._market_data.get_order_book(product_id, level)
    
    def shutdown(self):
        """Clean shutdown with position summary"""
        self._logger.info("Shutting down Trade Executor...")
        
        stats = self.get_daily_stats()
        self._logger.info(f"Final daily stats: {json.dumps(stats, indent=2)}")
        
        # Log any open positions
        if self._inventory:
            self._logger.warning(f"Open positions at shutdown: {self._inventory}")

# Global instance for easy access
_executor_instance = None

def get_trade_executor(logger) -> ProductionTradeExecutor:
    """Get or create the global trade executor instance"""
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = ProductionTradeExecutor(logger)
    return _executor_instance 