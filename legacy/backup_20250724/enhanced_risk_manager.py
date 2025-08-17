#!/usr/bin/env python3
"""
Enhanced Risk Manager - Dynamic Position Sizing & ATR-Based Risk Management
===========================================================================
Implements recommendations from the profitability report
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import pandas_ta as ta  # Using pandas-ta instead of TA-Lib
import logging
from datetime import datetime, timedelta

from config_unified import (
    ATR_PERIOD, DEFAULT_RISK_REWARD_RATIO,
    MAX_SPREAD_PERCENT, VOLUME_MA_PERIOD
)

logger = logging.getLogger(__name__)

@dataclass
class PositionInfo:
    """Enhanced position tracking with trailing stops"""
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    high_watermark: float
    size: float
    side: str  # 'long' or 'short'
    strategy: str
    
    def update_trailing_stop(self, current_price: float, atr: float):
        """Update trailing stop based on price movement"""
        if self.side == 'long' and current_price > self.high_watermark:
            self.high_watermark = current_price
            # Trail stop at 1.0x ATR below high watermark
            new_stop = current_price - atr
            self.stop_loss = max(self.stop_loss, new_stop)
            
        elif self.side == 'short' and current_price < self.high_watermark:
            self.high_watermark = current_price
            # Trail stop at 1.0x ATR above low watermark
            new_stop = current_price + atr
            self.stop_loss = min(self.stop_loss, new_stop)
    
    def should_exit(self, current_price: float, current_time: datetime, 
                   max_hold_minutes: int = 30) -> Tuple[bool, str]:
        """Check if position should be exited"""
        # Stop loss hit
        if self.side == 'long' and current_price <= self.stop_loss:
            return True, "stop_loss"
        elif self.side == 'short' and current_price >= self.stop_loss:
            return True, "stop_loss"
            
        # Take profit hit
        if self.side == 'long' and current_price >= self.take_profit:
            return True, "take_profit"
        elif self.side == 'short' and current_price <= self.take_profit:
            return True, "take_profit"
            
        # Time-based exit
        elapsed = (current_time - self.entry_time).total_seconds() / 60
        if elapsed > max_hold_minutes:
            return True, "time_exit"
            
        return False, ""


class EnhancedRiskManager:
    """Advanced risk management with dynamic sizing and ATR-based stops"""
    
    def __init__(self):
        self.positions: Dict[str, PositionInfo] = {}
        self.volatility_cache: Dict[str, float] = {}
        self.atr_cache: Dict[str, float] = {}
        
    def calculate_atr(self, candles: pd.DataFrame, period: int = ATR_PERIOD) -> float:
        """Calculate ATR using pandas-ta"""
        if len(candles) < period + 1:
            return 0.0
            
        # Use pandas-ta for ATR calculation
        atr_result = ta.atr(
            high=candles['high'],
            low=candles['low'],
            close=candles['close'],
            length=period
        )
        
        if atr_result is not None and len(atr_result) > 0:
            return float(atr_result.iloc[-1])
        return 0.0
        
    def determine_volatility_regime(self, atr: float, price: float) -> str:
        """Classify market volatility regime"""
        atr_percent = (atr / price) * 100
        
        if atr_percent > 5.0:
            return "extreme"
        elif atr_percent > 3.0:
            return "high"
        elif atr_percent > 1.5:
            return "normal"
        else:
            return "low"
            
    def calculate_dynamic_risk(self, volatility_regime: str, base_risk: float = 0.01) -> float:
        """Adjust risk percentage based on volatility"""
        risk_adjustments = {
            "extreme": 0.3,   # 30% of base risk
            "high": 0.5,      # 50% of base risk
            "normal": 1.0,    # 100% of base risk
            "low": 1.2        # 120% of base risk
        }
        
        return base_risk * risk_adjustments.get(volatility_regime, 1.0)
        
    def get_atr_multiplier(self, volatility_regime: str, strategy_type: str = "scalp") -> float:
        """Get ATR multiplier for stop loss based on regime and strategy"""
        multipliers = {
            "scalp": {
                "extreme": 2.5,
                "high": 2.0,
                "normal": 1.5,
                "low": 1.2
            },
            "swing": {
                "extreme": 3.5,
                "high": 3.0,
                "normal": 2.5,
                "low": 2.0
            }
        }
        
        return multipliers.get(strategy_type, multipliers["scalp"]).get(volatility_regime, 1.5)
        
    def calculate_position_size(self, 
                              account_equity: float,
                              entry_price: float,
                              atr: float,
                              volatility_regime: str,
                              confidence: float = 1.0,
                              strategy_type: str = "scalp") -> Dict[str, float]:
        """
        Calculate dynamic position size with all parameters
        
        Returns:
            Dict with 'size', 'stop_loss', 'take_profit', 'risk_amount'
        """
        # Get dynamic risk percentage
        risk_pct = self.calculate_dynamic_risk(volatility_regime)
        
        # Adjust for confidence (0.7 to 1.5)
        risk_pct *= min(max(confidence, 0.7), 1.5)
        
        # Get ATR multiplier
        atr_mult = self.get_atr_multiplier(volatility_regime, strategy_type)
        
        # Calculate stop distance
        stop_distance = atr * atr_mult
        
        # Calculate position size
        dollar_risk = account_equity * risk_pct
        position_size = dollar_risk / stop_distance
        
        # Calculate stop loss and take profit
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + (stop_distance * DEFAULT_RISK_REWARD_RATIO)
        
        return {
            'size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': dollar_risk,
            'atr_multiplier': atr_mult,
            'risk_percent': risk_pct * 100
        }
        
    def check_spread_quality(self, bid: float, ask: float) -> Tuple[bool, float]:
        """Check if spread is acceptable for trading"""
        if bid <= 0 or ask <= 0:
            return False, 0.0
            
        spread_pct = (ask - bid) / bid
        return spread_pct <= MAX_SPREAD_PERCENT, spread_pct
        
    def should_increase_position(self, 
                               win_rate: float, 
                               recent_pnl: List[float],
                               strategy_sharpe: float) -> float:
        """Determine position size multiplier based on performance"""
        # Base multiplier
        multiplier = 1.0
        
        # Win rate adjustment
        if win_rate > 0.65:
            multiplier *= 1.2
        elif win_rate < 0.45:
            multiplier *= 0.8
            
        # Recent performance
        if len(recent_pnl) >= 5:
            recent_avg = np.mean(recent_pnl[-5:])
            if recent_avg > 0:
                multiplier *= 1.1
            else:
                multiplier *= 0.9
                
        # Sharpe ratio adjustment
        if strategy_sharpe > 1.5:
            multiplier *= 1.15
        elif strategy_sharpe < 0.5:
            multiplier *= 0.85
            
        # Cap multiplier
        return min(max(multiplier, 0.5), 1.5)
        
    def create_position(self,
                       product_id: str,
                       entry_price: float,
                       size: float,
                       stop_loss: float,
                       take_profit: float,
                       side: str,
                       strategy: str) -> PositionInfo:
        """Create and track a new position"""
        position = PositionInfo(
            entry_time=datetime.now(),
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            high_watermark=entry_price,
            size=size,
            side=side,
            strategy=strategy
        )
        
        self.positions[product_id] = position
        logger.info(f"Created {side} position for {product_id}: "
                   f"entry={entry_price:.4f}, stop={stop_loss:.4f}, "
                   f"target={take_profit:.4f}, size={size:.4f}")
        
        return position
        
    def update_positions(self, market_data: Dict[str, Dict]) -> List[Dict]:
        """Update all positions and return exit signals"""
        exit_signals = []
        
        for product_id, position in list(self.positions.items()):
            if product_id not in market_data:
                continue
                
            current_price = market_data[product_id]['price']
            atr = market_data[product_id].get('atr', 0)
            
            # Update trailing stop
            if atr > 0:
                position.update_trailing_stop(current_price, atr)
                
            # Check exit conditions
            should_exit, reason = position.should_exit(
                current_price, datetime.now()
            )
            
            if should_exit:
                exit_signals.append({
                    'product_id': product_id,
                    'position': position,
                    'exit_price': current_price,
                    'reason': reason
                })
                
        return exit_signals
        
    def remove_position(self, product_id: str):
        """Remove position after exit"""
        if product_id in self.positions:
            del self.positions[product_id]
            
    def get_position_summary(self) -> Dict:
        """Get summary of all positions"""
        total_positions = len(self.positions)
        long_positions = sum(1 for p in self.positions.values() if p.side == 'long')
        short_positions = total_positions - long_positions
        
        return {
            'total': total_positions,
            'long': long_positions,
            'short': short_positions,
            'products': list(self.positions.keys())
        }


# Global instance
_risk_manager = None

def get_enhanced_risk_manager() -> EnhancedRiskManager:
    """Get or create global enhanced risk manager"""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = EnhancedRiskManager()
    return _risk_manager 