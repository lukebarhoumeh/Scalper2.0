"""
risk_manager.py - HFT-grade pre-trade risk checks and slippage protection

Critical for protecting capital in fast markets:
- Slippage estimation and limits
- Position concentration checks  
- Market quality validation
- Real-time P&L tracking
"""

from __future__ import annotations
import time
import threading
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from collections import defaultdict, deque
import numpy as np

from config import INVENTORY_CAP_USD

_LOG = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Real-time risk metrics for a product."""
    position_usd: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    trade_count: int = 0
    win_rate: float = 0.0
    avg_slippage_bps: float = 0.0
    max_position_usd: float = 0.0
    last_update: float = 0.0


@dataclass 
class PreTradeCheck:
    """Result of pre-trade risk check."""
    approved: bool
    reason: str = ""
    estimated_slippage_bps: float = 0.0
    estimated_fill_price: float = 0.0
    risk_score: float = 0.0  # 0-100, higher = riskier


class HFTRiskManager:
    """
    Comprehensive risk management for HFT trading.
    All checks must pass before trade execution.
    """
    
    def __init__(
        self,
        max_slippage_bps: float = 10.0,  # 10 basis points max
        max_spread_bps: float = 20.0,    # 20 bps max spread
        position_limit_usd: float = INVENTORY_CAP_USD,
        max_loss_per_day_usd: float = 100.0,
        min_liquidity_usd: float = 10000.0,  # Min order book depth
    ):
        self._max_slippage_bps = max_slippage_bps
        self._max_spread_bps = max_spread_bps
        self._position_limit = position_limit_usd
        self._max_daily_loss = max_loss_per_day_usd
        self._min_liquidity = min_liquidity_usd
        
        # Track metrics per product
        self._metrics: Dict[str, RiskMetrics] = defaultdict(RiskMetrics)
        
        # Daily P&L tracking
        self._daily_pnl = 0.0
        self._day_start_time = time.time()
        
        # Slippage tracking for adaptive limits
        self._slippage_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Thread safety
        self._lock = threading.RLock()
        
    def check_pre_trade(
        self, 
        product_id: str,
        side: str,  # "BUY" or "SELL"
        size_usd: float,
        bid: float,
        ask: float,
        mid: float,
        recent_trades: Optional[List[float]] = None
    ) -> PreTradeCheck:
        """
        Comprehensive pre-trade risk check.
        This is the main gate before any trade execution.
        """
        with self._lock:
            # Check 1: Spread quality
            spread_bps = ((ask - bid) / mid) * 10000
            if spread_bps > self._max_spread_bps:
                return PreTradeCheck(
                    approved=False,
                    reason=f"Spread too wide: {spread_bps:.1f}bps > {self._max_spread_bps}bps",
                    risk_score=100
                )
            
            # Check 2: Daily loss limit
            if self._daily_pnl < -self._max_daily_loss:
                return PreTradeCheck(
                    approved=False,
                    reason=f"Daily loss limit hit: ${-self._daily_pnl:.2f}",
                    risk_score=100
                )
            
            # Check 3: Position limits
            metrics = self._metrics[product_id]
            current_position = metrics.position_usd
            
            if side == "BUY":
                new_position = current_position + size_usd
                if new_position > self._position_limit:
                    return PreTradeCheck(
                        approved=False,
                        reason=f"Position limit breach: ${new_position:.2f} > ${self._position_limit:.2f}",
                        risk_score=90
                    )
            
            # Check 4: Estimate slippage
            slippage_estimate = self._estimate_slippage(
                product_id, side, size_usd, bid, ask, recent_trades
            )
            
            if slippage_estimate > self._max_slippage_bps:
                return PreTradeCheck(
                    approved=False,
                    reason=f"Slippage too high: {slippage_estimate:.1f}bps > {self._max_slippage_bps}bps",
                    estimated_slippage_bps=slippage_estimate,
                    risk_score=80
                )
            
            # Check 5: Market quality (volatility spike detection)
            if recent_trades and len(recent_trades) > 5:
                price_std = np.std(recent_trades)
                if price_std / mid > 0.002:  # 0.2% threshold
                    return PreTradeCheck(
                        approved=False,
                        reason="Market too volatile - price instability detected",
                        risk_score=70
                    )
            
            # Calculate estimated fill price
            if side == "BUY":
                base_price = ask
                slippage_mult = 1 + (slippage_estimate / 10000)
            else:
                base_price = bid  
                slippage_mult = 1 - (slippage_estimate / 10000)
            
            estimated_fill = base_price * slippage_mult
            
            # Calculate risk score (0-100)
            risk_score = self._calculate_risk_score(
                spread_bps, slippage_estimate, current_position, size_usd
            )
            
            return PreTradeCheck(
                approved=True,
                reason="All checks passed",
                estimated_slippage_bps=slippage_estimate,
                estimated_fill_price=estimated_fill,
                risk_score=risk_score
            )
    
    def _estimate_slippage(
        self,
        product_id: str,
        side: str,
        size_usd: float,
        bid: float,
        ask: float,
        recent_trades: Optional[List[float]] = None
    ) -> float:
        """
        Estimate slippage based on order size and market conditions.
        Uses historical slippage data when available.
        """
        # Base slippage from spread
        spread_bps = ((ask - bid) / ((ask + bid) / 2)) * 10000
        base_slippage = spread_bps * 0.5  # Assume we cross half the spread
        
        # Size impact (larger orders have more slippage)
        # Rough estimate: 1bp per $1000 of size
        size_impact = (size_usd / 1000) * 1.0
        
        # Historical adjustment
        hist_slippage = self._get_historical_slippage(product_id)
        if hist_slippage > 0:
            # Blend historical with estimate
            base_slippage = (base_slippage + hist_slippage) / 2
        
        # Volatility adjustment
        if recent_trades and len(recent_trades) > 2:
            volatility_factor = np.std(recent_trades) / np.mean(recent_trades)
            volatility_mult = 1 + (volatility_factor * 100)  # Scale up for volatility
        else:
            volatility_mult = 1.0
        
        total_slippage = (base_slippage + size_impact) * volatility_mult
        
        # Ensure we return a plain float, not numpy type
        result = min(total_slippage, 100.0)
        return float(result)  # Cap at 100bps (1%)
    
    def _get_historical_slippage(self, product_id: str) -> float:
        """Get average historical slippage for a product."""
        history = self._slippage_history[product_id]
        if not history:
            return 0.0
        return sum(history) / len(history)
    
    def _calculate_risk_score(
        self,
        spread_bps: float,
        slippage_bps: float,
        current_position: float,
        size_usd: float
    ) -> float:
        """Calculate overall risk score (0-100)."""
        # Spread risk (0-30 points)
        spread_score = min(30, (spread_bps / self._max_spread_bps) * 30)
        
        # Slippage risk (0-30 points)
        slippage_score = min(30, (slippage_bps / self._max_slippage_bps) * 30)
        
        # Position risk (0-40 points)
        position_after = abs(current_position + size_usd)
        position_score = min(40, (position_after / self._position_limit) * 40)
        
        return spread_score + slippage_score + position_score
    
    def record_trade(
        self,
        product_id: str,
        side: str,
        size_usd: float,
        expected_price: float,
        actual_price: float,
        pnl: float = 0.0
    ) -> None:
        """Record trade execution for risk tracking."""
        with self._lock:
            metrics = self._metrics[product_id]
            
            # Update position
            if side == "BUY":
                metrics.position_usd += size_usd
            else:
                metrics.position_usd -= size_usd
            
            # Track slippage
            slippage_bps = abs(actual_price - expected_price) / expected_price * 10000
            self._slippage_history[product_id].append(slippage_bps)
            
            # Update metrics
            metrics.trade_count += 1
            metrics.realized_pnl += pnl
            self._daily_pnl += pnl
            
            if abs(metrics.position_usd) > metrics.max_position_usd:
                metrics.max_position_usd = abs(metrics.position_usd)
            
            # Update average slippage
            metrics.avg_slippage_bps = self._get_historical_slippage(product_id)
            metrics.last_update = time.time()
            
            # Log if slippage was high
            if slippage_bps > self._max_slippage_bps:
                _LOG.warning(
                    f"High slippage on {product_id} {side}: "
                    f"{slippage_bps:.1f}bps (expected: {expected_price:.4f}, actual: {actual_price:.4f})"
                )
    
    def get_risk_summary(self) -> Dict[str, dict]:
        """Get current risk metrics for all products."""
        with self._lock:
            summary = {}
            total_position = 0.0
            
            for product_id, metrics in self._metrics.items():
                if metrics.trade_count > 0:
                    summary[product_id] = {
                        'position_usd': metrics.position_usd,
                        'realized_pnl': metrics.realized_pnl,
                        'trade_count': metrics.trade_count,
                        'avg_slippage_bps': metrics.avg_slippage_bps,
                        'max_position_usd': metrics.max_position_usd,
                    }
                    total_position += abs(metrics.position_usd)
            
            summary['_total'] = {
                'total_position_usd': total_position,
                'daily_pnl': self._daily_pnl,
                'position_utilization': (total_position / self._position_limit) * 100,
            }
            
            return summary
    
    def reset_daily_metrics(self) -> None:
        """Reset daily P&L tracking (call at start of trading day)."""
        with self._lock:
            self._daily_pnl = 0.0
            self._day_start_time = time.time()
            _LOG.info("Daily risk metrics reset")


# Global risk manager instance
_RISK_MANAGER = HFTRiskManager()


def check_trade_risk(
    product_id: str,
    side: str,
    size_usd: float,
    bid: float,
    ask: float,
    mid: float
) -> PreTradeCheck:
    """Global interface for pre-trade risk checks."""
    return _RISK_MANAGER.check_pre_trade(
        product_id, side, size_usd, bid, ask, mid
    )


def record_execution(
    product_id: str,
    side: str,
    size_usd: float,
    expected_price: float,
    actual_price: float,
    pnl: float = 0.0
) -> None:
    """Record trade execution for risk tracking."""
    _RISK_MANAGER.record_trade(
        product_id, side, size_usd, expected_price, actual_price, pnl
    ) 