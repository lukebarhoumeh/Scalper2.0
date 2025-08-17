#!/usr/bin/env python3
"""
Elite HFT Profit Preservation Engine
====================================
Sophisticated profit management system with:
- Dynamic profit preservation (18-25%+)
- Auto-scaling capital allocation
- Trend continuation detection
- $500K net profit targeting
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging
from collections import deque
import json

logger = logging.getLogger(__name__)

@dataclass
class ProfitTarget:
    """Profit preservation configuration"""
    minimum_take_rate: float = 0.18  # 18% minimum
    target_take_rate: float = 0.25   # 25% target
    preservation_rate: float = 0.25   # Preserve 25% of profits
    trend_continuation_threshold: float = 0.25  # Continue if >25%
    trailing_stop_distance: float = 0.15  # 15% trailing stop
    
@dataclass
class CapitalScaling:
    """Capital scaling configuration for $500K target"""
    target_net_profit: float = 500_000
    annual_periods: int = 252  # Trading days
    daily_target_return: float = 0.35  # 35% average (between 25-50%)
    risk_adjustment_factor: float = 0.7  # Account for losing days
    
    def calculate_required_capital(self) -> Dict[str, float]:
        """Calculate required starting capital to reach target"""
        # Compound interest formula reversed
        # Target = Principal * (1 + rate)^periods
        # Principal = Target / (1 + rate)^periods
        
        adjusted_daily_return = self.daily_target_return * self.risk_adjustment_factor
        
        # Different timeframe calculations
        timeframes = {
            '1_month': 21,
            '3_months': 63,
            '6_months': 126,
            '1_year': 252
        }
        
        results = {}
        for period_name, days in timeframes.items():
            compound_factor = (1 + adjusted_daily_return) ** days
            required_capital = self.target_net_profit / (compound_factor - 1)
            results[period_name] = {
                'required_capital': required_capital,
                'expected_return': required_capital * (compound_factor - 1),
                'days': days,
                'daily_avg_profit': self.target_net_profit / days
            }
            
        return results

class ProfitPreservationEngine:
    """
    Advanced profit preservation and capital management system
    """
    
    def __init__(self, initial_capital: float = 1000.0):
        self.initial_capital = initial_capital
        self.trading_capital = initial_capital
        self.preserved_capital = 0.0
        self.total_net_profit = 0.0
        
        # Configuration
        self.profit_config = ProfitTarget()
        self.capital_config = CapitalScaling()
        
        # Performance tracking
        self.trade_history = deque(maxlen=1000)
        self.profit_curve = []
        self.preservation_log = []
        
        # Dynamic adjustment parameters
        self.performance_window = 20  # trades
        self.adjustment_threshold = 0.1  # 10% deviation
        
        # Trend detection
        self.momentum_window = 5
        self.trend_strength_threshold = 0.6
        
        logger.info(f"Profit Preservation Engine initialized with ${initial_capital}")
        
    def process_trade_closure(self, 
                            product_id: str,
                            entry_price: float,
                            exit_price: float,
                            position_size: float,
                            side: str,
                            market_data: Dict) -> Dict:
        """
        Process a closed trade with dynamic profit preservation
        """
        # Calculate profit
        if side == 'BUY':
            profit_pct = (exit_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - exit_price) / entry_price
            
        profit_usd = position_size * profit_pct
        
        # Record trade
        trade_record = {
            'timestamp': datetime.now(timezone.utc),
            'product_id': product_id,
            'profit_pct': profit_pct,
            'profit_usd': profit_usd,
            'preserved': 0.0,
            'reinvested': 0.0,
            'action': 'none'
        }
        
        # Process based on profit level
        if profit_pct >= self.profit_config.minimum_take_rate:
            # Determine preservation strategy
            preservation_result = self._determine_preservation_strategy(
                profit_pct, product_id, market_data
            )
            
            if preservation_result['action'] == 'preserve':
                # Calculate preservation
                preservation_rate = preservation_result['rate']
                preserved_amount = profit_usd * preservation_rate
                reinvest_amount = profit_usd * (1 - preservation_rate)
                
                # Update capitals
                self.preserved_capital += preserved_amount
                self.trading_capital += reinvest_amount
                self.total_net_profit += profit_usd
                
                # Update trade record
                trade_record.update({
                    'preserved': preserved_amount,
                    'reinvested': reinvest_amount,
                    'action': 'preserved',
                    'preservation_rate': preservation_rate
                })
                
                logger.info(f"Preserved ${preserved_amount:.2f} from ${profit_usd:.2f} profit")
                
            elif preservation_result['action'] == 'continue':
                # Let winner run with trailing stop
                trade_record['action'] = 'trailing'
                trade_record['trailing_stop'] = preservation_result['trailing_stop']
                
        else:
            # No preservation, full reinvestment
            self.trading_capital += profit_usd
            self.total_net_profit += profit_usd
            trade_record['reinvested'] = profit_usd
            
        # Record and analyze
        self.trade_history.append(trade_record)
        self._update_performance_metrics()
        
        return trade_record
        
    def _determine_preservation_strategy(self, 
                                       profit_pct: float,
                                       product_id: str,
                                       market_data: Dict) -> Dict:
        """
        Intelligently determine preservation strategy based on:
        - Profit level
        - Market trend
        - Recent performance
        """
        result = {'action': 'preserve', 'rate': self.profit_config.preservation_rate}
        
        # Check if profit exceeds continuation threshold
        if profit_pct >= self.profit_config.trend_continuation_threshold:
            # Analyze trend strength
            trend_strength = self._analyze_trend_strength(product_id, market_data)
            
            if trend_strength > self.trend_strength_threshold:
                # Strong trend - let winner run
                result['action'] = 'continue'
                result['trailing_stop'] = profit_pct - self.profit_config.trailing_stop_distance
                logger.info(f"Strong trend detected ({trend_strength:.2f}), continuing position")
                
        # Dynamic preservation rate based on profit level
        if result['action'] == 'preserve':
            if profit_pct >= 0.50:  # 50%+ profit
                result['rate'] = 0.40  # Preserve 40%
            elif profit_pct >= 0.35:  # 35%+ profit
                result['rate'] = 0.30  # Preserve 30%
            else:
                result['rate'] = self.profit_config.preservation_rate  # Default 25%
                
        return result
        
    def _analyze_trend_strength(self, product_id: str, market_data: Dict) -> float:
        """
        Analyze trend strength using multiple indicators
        """
        try:
            # Get recent price data
            candles = market_data.get('candles', pd.DataFrame())
            if candles.empty:
                return 0.0
                
            # Calculate momentum indicators
            close_prices = candles['close'].values[-self.momentum_window:]
            
            # Price momentum
            returns = np.diff(close_prices) / close_prices[:-1]
            momentum_score = np.mean(returns > 0)
            
            # Volume confirmation
            volumes = candles['volume'].values[-self.momentum_window:]
            volume_trend = np.corrcoef(range(len(volumes)), volumes)[0, 1]
            
            # RSI trend
            rsi_values = market_data.get('rsi', [])
            if rsi_values:
                rsi_trend = 1.0 if rsi_values[-1] > 50 else 0.0
            else:
                rsi_trend = 0.5
                
            # Combine scores
            trend_strength = (momentum_score * 0.5 + 
                            (volume_trend + 1) / 2 * 0.3 + 
                            rsi_trend * 0.2)
                            
            return trend_strength
            
        except Exception as e:
            logger.error(f"Trend analysis error: {e}")
            return 0.0
            
    def _update_performance_metrics(self):
        """
        Update performance metrics and adjust parameters dynamically
        """
        if len(self.trade_history) < self.performance_window:
            return
            
        # Calculate recent performance
        recent_trades = list(self.trade_history)[-self.performance_window:]
        avg_profit = np.mean([t['profit_pct'] for t in recent_trades])
        win_rate = np.mean([t['profit_pct'] > 0 for t in recent_trades])
        
        # Adjust preservation rates based on performance
        if avg_profit > 0.30 and win_rate > 0.7:
            # Excellent performance - be more aggressive
            self.profit_config.minimum_take_rate = max(0.15, 
                self.profit_config.minimum_take_rate - 0.01)
        elif avg_profit < 0.10 or win_rate < 0.4:
            # Poor performance - be more conservative
            self.profit_config.minimum_take_rate = min(0.25, 
                self.profit_config.minimum_take_rate + 0.01)
                
        # Update profit curve
        self.profit_curve.append({
            'timestamp': datetime.now(timezone.utc),
            'total_profit': self.total_net_profit,
            'preserved': self.preserved_capital,
            'trading': self.trading_capital,
            'win_rate': win_rate,
            'avg_profit': avg_profit
        })
        
    def get_capital_allocation(self) -> Dict[str, float]:
        """
        Get current capital allocation recommendations
        """
        total_capital = self.trading_capital + self.preserved_capital
        
        # Risk-based position sizing
        recent_volatility = self._calculate_recent_volatility()
        risk_multiplier = 1.0 / (1.0 + recent_volatility)
        
        # Scale position size based on proximity to profit target
        progress_to_target = self.total_net_profit / self.capital_config.target_net_profit
        
        if progress_to_target < 0.5:
            # First half - be more aggressive
            aggressiveness = 1.2
        elif progress_to_target < 0.8:
            # Getting closer - normal risk
            aggressiveness = 1.0
        else:
            # Near target - protect gains
            aggressiveness = 0.8
            
        base_position_size = self.trading_capital * 0.1  # 10% per position
        adjusted_position_size = base_position_size * risk_multiplier * aggressiveness
        
        return {
            'trading_capital': self.trading_capital,
            'preserved_capital': self.preserved_capital,
            'total_capital': total_capital,
            'recommended_position_size': adjusted_position_size,
            'max_positions': int(self.trading_capital / adjusted_position_size),
            'progress_to_target': progress_to_target,
            'risk_multiplier': risk_multiplier
        }
        
    def _calculate_recent_volatility(self) -> float:
        """Calculate recent trading volatility"""
        if len(self.trade_history) < 10:
            return 0.5  # Default medium volatility
            
        recent_profits = [t['profit_pct'] for t in list(self.trade_history)[-20:]]
        return np.std(recent_profits)
        
    def get_status_report(self) -> Dict:
        """Get comprehensive status report"""
        allocation = self.get_capital_allocation()
        capital_requirements = self.capital_config.calculate_required_capital()
        
        return {
            'current_status': {
                'total_net_profit': self.total_net_profit,
                'preserved_capital': self.preserved_capital,
                'trading_capital': self.trading_capital,
                'total_trades': len(self.trade_history),
                'progress_to_500k': f"{(self.total_net_profit / 500000) * 100:.1f}%"
            },
            'capital_allocation': allocation,
            'profit_targets': {
                'minimum_take': f"{self.profit_config.minimum_take_rate * 100:.1f}%",
                'target_take': f"{self.profit_config.target_take_rate * 100:.1f}%",
                'preservation_rate': f"{self.profit_config.preservation_rate * 100:.1f}%"
            },
            'capital_requirements_to_500k': capital_requirements,
            'performance_metrics': self._get_performance_summary()
        }
        
    def _get_performance_summary(self) -> Dict:
        """Get performance summary statistics"""
        if not self.trade_history:
            return {}
            
        all_profits = [t['profit_pct'] for t in self.trade_history]
        winning_trades = [p for p in all_profits if p > 0]
        
        return {
            'total_trades': len(all_profits),
            'win_rate': f"{(len(winning_trades) / len(all_profits)) * 100:.1f}%",
            'avg_profit': f"{np.mean(all_profits) * 100:.2f}%",
            'best_trade': f"{max(all_profits) * 100:.2f}%",
            'worst_trade': f"{min(all_profits) * 100:.2f}%",
            'sharpe_ratio': self._calculate_sharpe_ratio(all_profits)
        }
        
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        return np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)