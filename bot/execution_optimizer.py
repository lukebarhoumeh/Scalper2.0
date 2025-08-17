"""Smart Order Execution Optimizer

This module implements advanced order execution strategies to minimize slippage,
optimize timing, and maximize fill rates for HFT trading.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import numpy as np
from collections import deque
import asyncio

from .models import Order, OrderType, Side, Fill


@dataclass
class ExecutionMetrics:
    """Metrics for execution quality"""
    avg_slippage_bps: float
    fill_rate: float
    avg_time_to_fill_ms: float
    adverse_selection_bps: float
    implementation_shortfall_bps: float


@dataclass
class OrderTiming:
    """Optimal timing for order execution"""
    execute_now: bool
    delay_ms: int
    urgency_score: float  # 0-1
    reason: str


@dataclass
class OrderSlicing:
    """Order slicing recommendation"""
    slices: List[float]  # Percentages of total order
    intervals_ms: List[int]  # Time between slices
    aggressive: bool  # Use aggressive pricing


class ExecutionOptimizer:
    """
    Advanced execution algorithms for optimal order placement.
    
    Features:
    - Adaptive order timing based on market microstructure
    - Dynamic order slicing for large orders
    - Spread-aware limit order placement
    - Anti-gaming logic to avoid predictable patterns
    - Real-time execution quality monitoring
    """
    
    def __init__(self, 
                 min_order_interval_ms: int = 100,
                 max_spread_cross_bps: int = 10,
                 slicing_threshold_usd: float = 500):
        
        self.min_order_interval_ms = min_order_interval_ms
        self.max_spread_cross_bps = max_spread_cross_bps
        self.slicing_threshold_usd = slicing_threshold_usd
        
        # Execution history for learning
        self.execution_history: deque = deque(maxlen=1000)
        self.spread_history: Dict[str, deque] = {}
        self.volume_profile: Dict[str, List[float]] = {}
        
        # Timing patterns to avoid detection
        self.last_order_times: Dict[str, datetime] = {}
        self.jitter_range_ms = 50
        
        # Performance tracking
        self.fills_by_hour: Dict[int, List[Fill]] = {}
        self.slippage_by_urgency: Dict[str, List[float]] = {}
    
    def optimize_order_timing(self, order: Order, current_spread: float, 
                            volatility: float, momentum: float) -> OrderTiming:
        """
        Determine optimal timing for order execution.
        
        Considers:
        - Current spread width
        - Recent volatility
        - Price momentum
        - Time since last order (anti-gaming)
        - Historical fill patterns
        """
        
        symbol = order.symbol
        now = datetime.now(timezone.utc)
        
        # Check minimum interval to avoid detection
        if symbol in self.last_order_times:
            time_since_last = (now - self.last_order_times[symbol]).total_seconds() * 1000
            if time_since_last < self.min_order_interval_ms:
                delay = int(self.min_order_interval_ms - time_since_last + np.random.randint(0, self.jitter_range_ms))
                return OrderTiming(
                    execute_now=False,
                    delay_ms=delay,
                    urgency_score=0.3,
                    reason="minimum_interval_anti_gaming"
                )
        
        # Calculate urgency score
        urgency = self._calculate_urgency(order, volatility, momentum)
        
        # Spread analysis
        if symbol not in self.spread_history:
            self.spread_history[symbol] = deque(maxlen=100)
        self.spread_history[symbol].append(current_spread)
        
        avg_spread = np.mean(self.spread_history[symbol])
        spread_percentile = (sum(1 for s in self.spread_history[symbol] if s < current_spread) / 
                           len(self.spread_history[symbol]))
        
        # Timing decision logic
        if urgency > 0.8:
            # High urgency - execute immediately
            return OrderTiming(
                execute_now=True,
                delay_ms=0,
                urgency_score=urgency,
                reason="high_urgency"
            )
        
        elif spread_percentile < 0.3 and urgency > 0.5:
            # Good spread + moderate urgency
            return OrderTiming(
                execute_now=True,
                delay_ms=0,
                urgency_score=urgency,
                reason="favorable_spread"
            )
        
        elif spread_percentile > 0.7 and urgency < 0.5:
            # Bad spread + low urgency - wait
            delay = min(500, int(200 / urgency)) + np.random.randint(-self.jitter_range_ms, self.jitter_range_ms)
            return OrderTiming(
                execute_now=False,
                delay_ms=delay,
                urgency_score=urgency,
                reason="unfavorable_spread_waiting"
            )
        
        else:
            # Default - slight randomized delay
            delay = np.random.randint(50, 150)
            return OrderTiming(
                execute_now=delay < 100,
                delay_ms=delay if delay >= 100 else 0,
                urgency_score=urgency,
                reason="standard_execution"
            )
    
    def optimize_order_slicing(self, order: Order, avg_volume: float, 
                             volatility: float, urgency: float) -> Optional[OrderSlicing]:
        """
        Determine if order should be sliced and how.
        
        Large orders are split to minimize market impact.
        """
        
        order_value = order.qty * order.price if order.price else 0
        
        # Check if slicing needed
        if order_value < self.slicing_threshold_usd:
            return None
        
        # Calculate participation rate based on urgency
        if urgency > 0.8:
            participation_rate = 0.15  # 15% of volume
        elif urgency > 0.5:
            participation_rate = 0.10
        else:
            participation_rate = 0.05
        
        # Determine number of slices
        target_slice_value = avg_volume * participation_rate
        num_slices = max(2, min(10, int(order_value / target_slice_value)))
        
        # Create slice distribution
        if volatility > 2.0:  # High volatility
            # Front-load execution
            slices = self._create_frontloaded_slices(num_slices)
            intervals = [100] * (num_slices - 1)  # Fast execution
            aggressive = True
        else:
            # Even distribution with randomization
            slices = self._create_randomized_slices(num_slices)
            intervals = [np.random.randint(200, 500) for _ in range(num_slices - 1)]
            aggressive = urgency > 0.6
        
        return OrderSlicing(
            slices=slices,
            intervals_ms=intervals,
            aggressive=aggressive
        )
    
    def calculate_optimal_limit_price(self, side: Side, current_bid: float, 
                                    current_ask: float, urgency: float) -> float:
        """
        Calculate optimal limit price considering spread and urgency.
        
        Returns price that balances execution probability with cost.
        """
        
        spread = current_ask - current_bid
        mid = (current_bid + current_ask) / 2
        
        if urgency > 0.9:
            # Very urgent - cross the spread
            return current_ask if side == Side.BUY else current_bid
        
        elif urgency > 0.7:
            # Urgent - go to touch
            return current_bid if side == Side.BUY else current_ask
        
        elif urgency > 0.5:
            # Moderate - inside spread
            if side == Side.BUY:
                return current_bid + spread * 0.25
            else:
                return current_ask - spread * 0.25
        
        else:
            # Patient - join the queue
            if side == Side.BUY:
                return current_bid
            else:
                return current_ask
    
    def should_use_iceberg(self, order: Order, avg_order_size: float) -> Tuple[bool, Optional[float]]:
        """
        Determine if iceberg order should be used.
        
        Returns (should_use, visible_percentage)
        """
        
        if order.qty > avg_order_size * 3:
            # Large order - use iceberg
            visible_pct = min(0.3, avg_order_size / order.qty)
            return True, visible_pct
        
        return False, None
    
    def track_execution(self, order: Order, fill: Fill, 
                       pre_trade_mid: float, post_trade_mid: float) -> None:
        """Track execution quality for learning"""
        
        # Calculate slippage
        if fill.side == Side.BUY:
            slippage_bps = ((fill.price - pre_trade_mid) / pre_trade_mid) * 10000
        else:
            slippage_bps = ((pre_trade_mid - fill.price) / pre_trade_mid) * 10000
        
        # Calculate adverse selection (post-trade price movement)
        if fill.side == Side.BUY:
            adverse_bps = ((pre_trade_mid - post_trade_mid) / pre_trade_mid) * 10000
        else:
            adverse_bps = ((post_trade_mid - pre_trade_mid) / pre_trade_mid) * 10000
        
        # Store metrics
        execution_data = {
            'timestamp': fill.ts,
            'symbol': fill.symbol,
            'side': fill.side,
            'qty': fill.qty,
            'price': fill.price,
            'slippage_bps': slippage_bps,
            'adverse_selection_bps': adverse_bps,
            'hour': fill.ts.hour
        }
        
        self.execution_history.append(execution_data)
        
        # Update performance tracking
        hour = fill.ts.hour
        if hour not in self.fills_by_hour:
            self.fills_by_hour[hour] = []
        self.fills_by_hour[hour].append(fill)
    
    def get_execution_analytics(self, symbol: Optional[str] = None) -> ExecutionMetrics:
        """Get execution quality metrics"""
        
        if not self.execution_history:
            return ExecutionMetrics(0, 0, 0, 0, 0)
        
        # Filter by symbol if provided
        data = self.execution_history
        if symbol:
            data = [d for d in data if d['symbol'] == symbol]
        
        if not data:
            return ExecutionMetrics(0, 0, 0, 0, 0)
        
        # Calculate metrics
        slippages = [d['slippage_bps'] for d in data]
        adverse_selections = [d['adverse_selection_bps'] for d in data]
        
        avg_slippage = np.mean(slippages)
        avg_adverse = np.mean(adverse_selections)
        implementation_shortfall = avg_slippage + avg_adverse
        
        return ExecutionMetrics(
            avg_slippage_bps=avg_slippage,
            fill_rate=1.0,  # Simplified - in reality track unfilled orders
            avg_time_to_fill_ms=50,  # Simplified
            adverse_selection_bps=avg_adverse,
            implementation_shortfall_bps=implementation_shortfall
        )
    
    def get_optimal_trading_hours(self) -> List[int]:
        """Identify hours with best execution quality"""
        
        hour_quality = {}
        
        for hour, fills in self.fills_by_hour.items():
            if len(fills) < 10:
                continue
            
            # Calculate average execution quality for this hour
            # Simplified - in reality would track slippage by hour
            hour_quality[hour] = len(fills)
        
        # Return top hours
        sorted_hours = sorted(hour_quality.items(), key=lambda x: x[1], reverse=True)
        return [hour for hour, _ in sorted_hours[:8]]
    
    def _calculate_urgency(self, order: Order, volatility: float, momentum: float) -> float:
        """Calculate execution urgency score"""
        
        urgency = 0.5  # Base urgency
        
        # Momentum alignment
        if (order.side == Side.BUY and momentum > 0) or (order.side == Side.SELL and momentum < 0):
            urgency += 0.2
        else:
            urgency -= 0.1
        
        # Volatility adjustment
        if volatility > 2.0:
            urgency += 0.2  # Execute quickly in volatile markets
        elif volatility < 0.5:
            urgency -= 0.1  # Can be patient in calm markets
        
        # Stop loss orders are always urgent
        if hasattr(order, 'is_stop_loss') and order.is_stop_loss:
            urgency = 0.95
        
        return max(0.1, min(1.0, urgency))
    
    def _create_frontloaded_slices(self, num_slices: int) -> List[float]:
        """Create front-loaded slice distribution"""
        
        # Exponentially decreasing slices
        base = 0.5
        slices = [base ** i for i in range(num_slices)]
        total = sum(slices)
        
        return [s / total for s in slices]
    
    def _create_randomized_slices(self, num_slices: int) -> List[float]:
        """Create randomized slice distribution"""
        
        # Start with equal slices
        slices = [1.0 / num_slices] * num_slices
        
        # Add random variation
        variations = np.random.uniform(0.8, 1.2, num_slices)
        slices = [s * v for s, v in zip(slices, variations)]
        
        # Normalize
        total = sum(slices)
        return [s / total for s in slices]


class SmartOrderRouter:
    """
    Routes orders intelligently based on market conditions.
    
    In production, this would route between:
    - Different exchanges
    - Lit vs dark pools
    - Market maker vs taker strategies
    """
    
    def __init__(self):
        self.execution_optimizer = ExecutionOptimizer()
        self.venue_performance: Dict[str, ExecutionMetrics] = {}
    
    async def route_order(self, order: Order, market_data: Dict) -> List[Order]:
        """
        Route order optimally, potentially splitting across venues.
        
        Returns list of child orders.
        """
        
        # Extract market data
        current_bid = market_data.get('bid', 0)
        current_ask = market_data.get('ask', 0)
        volatility = market_data.get('volatility', 1.0)
        momentum = market_data.get('momentum', 0.0)
        avg_volume = market_data.get('avg_volume', 1000)
        
        # Get timing recommendation
        timing = self.execution_optimizer.optimize_order_timing(
            order, current_ask - current_bid, volatility, momentum
        )
        
        if not timing.execute_now:
            # Delay execution
            await asyncio.sleep(timing.delay_ms / 1000)
        
        # Check if slicing needed
        urgency = timing.urgency_score
        slicing = self.execution_optimizer.optimize_order_slicing(
            order, avg_volume, volatility, urgency
        )
        
        if slicing:
            # Create sliced orders
            child_orders = []
            remaining_qty = order.qty
            
            for i, slice_pct in enumerate(slicing.slices):
                slice_qty = order.qty * slice_pct
                
                # Adjust last slice for rounding
                if i == len(slicing.slices) - 1:
                    slice_qty = remaining_qty
                
                # Calculate limit price for slice
                limit_price = self.execution_optimizer.calculate_optimal_limit_price(
                    order.side, current_bid, current_ask, urgency
                )
                
                child_order = Order(
                    symbol=order.symbol,
                    side=order.side,
                    qty=slice_qty,
                    type=OrderType.LIMIT,
                    price=limit_price,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit
                )
                
                child_orders.append(child_order)
                remaining_qty -= slice_qty
            
            return child_orders
        
        else:
            # Single order - just optimize price
            if order.type == OrderType.MARKET:
                # Convert to aggressive limit for better control
                limit_price = self.execution_optimizer.calculate_optimal_limit_price(
                    order.side, current_bid, current_ask, urgency
                )
                
                order.type = OrderType.LIMIT
                order.price = limit_price
            
            return [order]
