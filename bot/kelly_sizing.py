"""Kelly Criterion and Advanced Position Sizing

This module implements the Kelly Criterion and other advanced position sizing
methods to optimize bet size based on edge and win probability.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import deque
import math


@dataclass
class TradingEdge:
    """Quantified trading edge"""
    win_rate: float  # Probability of winning
    avg_win: float   # Average win size (in R multiples)
    avg_loss: float  # Average loss size (in R multiples)
    sharpe_ratio: float
    sample_size: int
    confidence_interval: Tuple[float, float]  # 95% CI for win rate


@dataclass
class PositionSize:
    """Optimal position size recommendation"""
    kelly_fraction: float  # Full Kelly
    conservative_fraction: float  # Fractional Kelly
    risk_dollars: float
    position_dollars: float
    shares: float
    confidence: float
    method: str
    reason: str


class KellySizer:
    """
    Advanced position sizing using Kelly Criterion and related methods.
    
    Features:
    - Full and fractional Kelly calculations
    - Dynamic Kelly fraction based on confidence
    - Drawdown-adjusted sizing
    - Regime-specific adjustments
    - Multi-strategy Kelly optimization
    """
    
    def __init__(self,
                 max_kelly_fraction: float = 0.25,  # Never use more than 25% Kelly
                 kelly_confidence_threshold: int = 30,  # Min trades for Kelly
                 base_kelly_divisor: float = 4.0):  # Use Kelly/4 as default
        
        self.max_kelly_fraction = max_kelly_fraction
        self.kelly_confidence_threshold = kelly_confidence_threshold
        self.base_kelly_divisor = base_kelly_divisor
        
        # Track performance by strategy/symbol
        self.performance_history: Dict[str, deque] = {}
        self.edge_calculations: Dict[str, TradingEdge] = {}
        
        # Risk adjustments
        self.current_drawdown = 0.0
        self.max_historical_drawdown = 0.0
        self.volatility_regime = "normal"
    
    def calculate_position_size(self, 
                              symbol: str,
                              strategy: str,
                              account_equity: float,
                              current_price: float,
                              stop_loss_price: float,
                              confidence: float = 0.6) -> PositionSize:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Args:
            symbol: Trading symbol
            strategy: Strategy name for edge tracking
            account_equity: Total account value
            current_price: Current asset price
            stop_loss_price: Stop loss price
            confidence: Signal confidence (0-1)
        
        Returns:
            PositionSize with recommendations
        """
        
        # Get trading edge for this strategy/symbol
        edge_key = f"{symbol}_{strategy}"
        edge = self._calculate_edge(edge_key)
        
        if not edge or edge.sample_size < 5:
            # Not enough data - use minimum size
            return self._minimum_position_size(
                account_equity, current_price, stop_loss_price,
                "Insufficient data for Kelly sizing"
            )
        
        # Calculate Kelly fraction
        kelly_f = self._kelly_formula(edge)
        
        # Apply confidence adjustment
        confidence_adjusted_kelly = kelly_f * confidence
        
        # Apply safety constraints
        safe_kelly = self._apply_safety_constraints(
            confidence_adjusted_kelly, edge, account_equity
        )
        
        # Calculate position metrics
        risk_per_share = abs(current_price - stop_loss_price)
        risk_fraction = safe_kelly
        risk_dollars = account_equity * risk_fraction
        shares = risk_dollars / risk_per_share
        position_dollars = shares * current_price
        
        # Ensure position size constraints
        max_position = account_equity * 0.2  # Max 20% in one position
        if position_dollars > max_position:
            position_dollars = max_position
            shares = position_dollars / current_price
            risk_dollars = shares * risk_per_share
            risk_fraction = risk_dollars / account_equity
        
        return PositionSize(
            kelly_fraction=kelly_f,
            conservative_fraction=safe_kelly,
            risk_dollars=risk_dollars,
            position_dollars=position_dollars,
            shares=shares,
            confidence=confidence,
            method="kelly_criterion",
            reason=self._build_sizing_reason(edge, kelly_f, safe_kelly)
        )
    
    def calculate_multi_strategy_allocation(self,
                                          strategies: List[str],
                                          account_equity: float) -> Dict[str, float]:
        """
        Optimize allocation across multiple strategies using Kelly.
        
        Returns optimal capital allocation percentages.
        """
        
        # Get edges for all strategies
        edges = {}
        for strategy in strategies:
            edge = self._calculate_edge(strategy)
            if edge and edge.sample_size >= self.kelly_confidence_threshold:
                edges[strategy] = edge
        
        if not edges:
            # Equal allocation if no edges
            equal_alloc = 1.0 / len(strategies)
            return {s: equal_alloc for s in strategies}
        
        # Calculate Kelly fractions
        kelly_fractions = {}
        for strategy, edge in edges.items():
            kelly_fractions[strategy] = self._kelly_formula(edge)
        
        # Normalize to sum to 1 (or less for safety)
        total_kelly = sum(kelly_fractions.values())
        max_total = 0.5  # Never allocate more than 50% total
        
        if total_kelly > max_total:
            # Scale down proportionally
            scale_factor = max_total / total_kelly
            kelly_fractions = {s: k * scale_factor for s, k in kelly_fractions.items()}
        
        # Fill in non-edge strategies with minimum allocation
        min_allocation = 0.02  # 2% minimum
        for strategy in strategies:
            if strategy not in kelly_fractions:
                kelly_fractions[strategy] = min_allocation
        
        # Normalize again
        total = sum(kelly_fractions.values())
        if total > 1.0:
            kelly_fractions = {s: k / total for s, k in kelly_fractions.items()}
        
        return kelly_fractions
    
    def update_performance(self, 
                          symbol: str,
                          strategy: str,
                          won: bool,
                          r_multiple: float) -> None:
        """
        Update performance history for edge calculation.
        
        Args:
            symbol: Trading symbol
            strategy: Strategy name
            won: Whether trade was profitable
            r_multiple: Result in R multiples (profit/initial_risk)
        """
        
        edge_key = f"{symbol}_{strategy}"
        
        if edge_key not in self.performance_history:
            self.performance_history[edge_key] = deque(maxlen=1000)
        
        self.performance_history[edge_key].append({
            'won': won,
            'r_multiple': r_multiple,
            'timestamp': np.datetime64('now')
        })
        
        # Recalculate edge
        self._calculate_edge(edge_key)
    
    def set_market_conditions(self, 
                            drawdown: float,
                            volatility_regime: str) -> None:
        """Update market conditions for position sizing adjustments"""
        
        self.current_drawdown = drawdown
        self.max_historical_drawdown = max(self.max_historical_drawdown, drawdown)
        self.volatility_regime = volatility_regime
    
    def get_edge_report(self) -> Dict[str, TradingEdge]:
        """Get current edges for all tracked strategies"""
        return self.edge_calculations.copy()
    
    def _kelly_formula(self, edge: TradingEdge) -> float:
        """
        Calculate Kelly fraction: f = (p*b - q)/b
        where:
        - p = probability of winning
        - q = probability of losing (1-p)
        - b = ratio of win to loss amounts
        """
        
        p = edge.win_rate
        q = 1 - p
        
        # Handle edge cases
        if edge.avg_loss == 0:
            return 0.0
        
        b = edge.avg_win / edge.avg_loss
        
        # Kelly formula
        if b == 0:
            return 0.0
        
        kelly = (p * b - q) / b
        
        # Kelly can be negative (no edge) or very large (huge edge)
        # Constrain to reasonable bounds
        return max(0.0, min(kelly, self.max_kelly_fraction))
    
    def _calculate_edge(self, edge_key: str) -> Optional[TradingEdge]:
        """Calculate trading edge from historical performance"""
        
        if edge_key not in self.performance_history:
            return None
        
        history = list(self.performance_history[edge_key])
        
        if len(history) < 5:
            return None
        
        # Calculate metrics
        wins = [h for h in history if h['won']]
        losses = [h for h in history if not h['won']]
        
        win_rate = len(wins) / len(history)
        
        # Calculate average R multiples
        avg_win = np.mean([w['r_multiple'] for w in wins]) if wins else 0.0
        avg_loss = abs(np.mean([l['r_multiple'] for l in losses])) if losses else 1.0
        
        # Calculate Sharpe ratio (simplified)
        all_returns = [h['r_multiple'] for h in history]
        if len(all_returns) > 1:
            sharpe = np.mean(all_returns) / (np.std(all_returns) + 1e-6) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Calculate confidence interval for win rate
        # Using Wilson score interval
        n = len(history)
        z = 1.96  # 95% confidence
        
        p_hat = win_rate
        denominator = 1 + z**2/n
        centre = (p_hat + z**2/(2*n)) / denominator
        margin = z * np.sqrt((p_hat*(1-p_hat)/n + z**2/(4*n**2))) / denominator
        
        ci_lower = max(0, centre - margin)
        ci_upper = min(1, centre + margin)
        
        edge = TradingEdge(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            sharpe_ratio=sharpe,
            sample_size=len(history),
            confidence_interval=(ci_lower, ci_upper)
        )
        
        self.edge_calculations[edge_key] = edge
        
        return edge
    
    def _apply_safety_constraints(self, 
                                kelly_f: float,
                                edge: TradingEdge,
                                account_equity: float) -> float:
        """Apply various safety constraints to Kelly fraction"""
        
        # Start with base fractional Kelly
        safe_f = kelly_f / self.base_kelly_divisor
        
        # Reduce during drawdown
        if self.current_drawdown > 0.1:  # 10% drawdown
            drawdown_multiplier = 1 - (self.current_drawdown / 0.5)  # Linear reduction
            drawdown_multiplier = max(0.2, drawdown_multiplier)  # At least 20% of normal
            safe_f *= drawdown_multiplier
        
        # Reduce in high volatility
        if self.volatility_regime == "extreme":
            safe_f *= 0.5
        elif self.volatility_regime == "high":
            safe_f *= 0.75
        
        # Reduce if edge confidence is low
        if edge.sample_size < self.kelly_confidence_threshold:
            confidence_multiplier = edge.sample_size / self.kelly_confidence_threshold
            safe_f *= confidence_multiplier
        
        # Reduce if confidence interval is wide
        ci_width = edge.confidence_interval[1] - edge.confidence_interval[0]
        if ci_width > 0.3:  # Wide CI
            safe_f *= 0.7
        
        # Never risk more than 2% on a single trade
        max_risk_fraction = 0.02
        safe_f = min(safe_f, max_risk_fraction)
        
        # Scale based on Sharpe ratio
        if edge.sharpe_ratio < 0.5:
            safe_f *= 0.5
        elif edge.sharpe_ratio > 2.0:
            safe_f *= 1.2
        
        return safe_f
    
    def _minimum_position_size(self,
                             account_equity: float,
                             current_price: float,
                             stop_loss_price: float,
                             reason: str) -> PositionSize:
        """Calculate minimum position size when Kelly can't be used"""
        
        # Use fixed fractional (0.5% risk)
        risk_fraction = 0.005
        risk_dollars = account_equity * risk_fraction
        
        risk_per_share = abs(current_price - stop_loss_price)
        shares = risk_dollars / risk_per_share
        position_dollars = shares * current_price
        
        return PositionSize(
            kelly_fraction=0.0,
            conservative_fraction=risk_fraction,
            risk_dollars=risk_dollars,
            position_dollars=position_dollars,
            shares=shares,
            confidence=0.5,
            method="fixed_fractional",
            reason=reason
        )
    
    def _build_sizing_reason(self, 
                           edge: TradingEdge,
                           kelly_f: float,
                           safe_f: float) -> str:
        """Build descriptive reason for position sizing"""
        
        reasons = []
        
        # Edge quality
        if edge.win_rate > 0.6:
            reasons.append(f"strong edge ({edge.win_rate:.1%} win rate)")
        elif edge.win_rate < 0.45:
            reasons.append(f"weak edge ({edge.win_rate:.1%} win rate)")
        
        # Kelly adjustment
        reduction_factor = safe_f / kelly_f if kelly_f > 0 else 0
        if reduction_factor < 0.5:
            reasons.append(f"conservative Kelly ({reduction_factor:.1f}x)")
        
        # Sample size
        if edge.sample_size < self.kelly_confidence_threshold:
            reasons.append(f"limited data (n={edge.sample_size})")
        
        # Market conditions
        if self.current_drawdown > 0.1:
            reasons.append(f"drawdown adjusted ({self.current_drawdown:.1%})")
        
        if self.volatility_regime != "normal":
            reasons.append(f"{self.volatility_regime} volatility")
        
        return " - ".join(reasons) if reasons else "standard sizing"


class OptimalF:
    """
    Ralph Vince's Optimal F position sizing method.
    More aggressive than Kelly but with higher variance.
    """
    
    def __init__(self, lookback_trades: int = 100):
        self.lookback_trades = lookback_trades
        self.trade_history: deque = deque(maxlen=lookback_trades)
    
    def calculate_optimal_f(self, 
                          trade_results: List[float]) -> float:
        """
        Calculate Optimal F using historical trade results.
        
        Args:
            trade_results: List of trade P&L amounts
            
        Returns:
            Optimal fraction to risk
        """
        
        if len(trade_results) < 10:
            return 0.01  # Default minimum
        
        # Find worst loss
        worst_loss = abs(min(trade_results))
        
        if worst_loss == 0:
            return 0.01
        
        # Test different f values
        best_f = 0.01
        best_twr = 0.0  # Terminal Wealth Relative
        
        for f in np.arange(0.01, 0.50, 0.01):
            twr = 1.0
            
            for result in trade_results:
                hpp = result / worst_loss  # Holding Period Profit
                twr *= (1 + f * hpp)
                
                if twr <= 0:
                    break
            
            if twr > best_twr:
                best_twr = twr
                best_f = f
        
        # Apply safety factor
        return best_f * 0.5  # Use half of Optimal F


class VolatilityAdjustedSizing:
    """
    Position sizing based on volatility targets.
    Similar to risk parity approaches.
    """
    
    def __init__(self, 
                 target_volatility: float = 0.15,  # 15% annual
                 lookback_days: int = 20):
        
        self.target_volatility = target_volatility
        self.lookback_days = lookback_days
        self.volatility_history: Dict[str, deque] = {}
    
    def calculate_position_size(self,
                              symbol: str,
                              account_equity: float,
                              current_volatility: float) -> float:
        """
        Calculate position size to achieve target volatility.
        
        Returns:
            Position size as fraction of equity
        """
        
        # Annualize current volatility
        annual_vol = current_volatility * np.sqrt(252)
        
        if annual_vol == 0:
            return 0.0
        
        # Calculate position size
        position_fraction = self.target_volatility / annual_vol
        
        # Apply constraints
        position_fraction = min(position_fraction, 0.2)  # Max 20% per position
        position_fraction = max(position_fraction, 0.01)  # Min 1%
        
        return position_fraction
