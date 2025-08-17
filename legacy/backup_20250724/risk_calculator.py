#!/usr/bin/env python3
"""
Risk Calculator for Dynamic Position Sizing
==========================================
Production-grade risk management calculations for HFT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import logging

from market_data import get_candles
from config_unified import (
    RISK_PERCENT, ATR_PERIOD, ATR_MULTIPLIER,
    MAX_TOTAL_EXPOSURE, DEFAULT_RISK_REWARD_RATIO
)

logger = logging.getLogger(__name__)


class RiskCalculator:
    """Dynamic risk and position size calculator"""
    
    def __init__(self):
        self.atr_cache: Dict[str, Dict] = {}  # Cache ATR calculations
        self.cache_ttl = 300  # 5 minutes cache
        
    def calculate_position_size(
        self, 
        account_balance: float, 
        risk_percent: float, 
        stop_loss_distance: float
    ) -> float:
        """
        Calculate position size based on account risk.
        
        Formula: position_size = (account_balance * risk_percent) / stop_loss_distance
        
        Args:
            account_balance: Total account balance in USD
            risk_percent: Risk percentage per trade (e.g., 0.01 for 1%)
            stop_loss_distance: Dollar distance to stop loss
            
        Returns:
            Position size in USD
        """
        if stop_loss_distance <= 0:
            logger.warning("Invalid stop loss distance, using minimum")
            stop_loss_distance = account_balance * 0.001  # 0.1% minimum
            
        position_size = (account_balance * risk_percent) / stop_loss_distance
        
        # Apply safety limits
        max_position = account_balance * MAX_TOTAL_EXPOSURE
        position_size = min(position_size, max_position)
        
        return round(position_size, 2)
        
    def calculate_atr(self, symbol: str, period: int = ATR_PERIOD) -> Optional[float]:
        """
        Calculate Average True Range for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            period: ATR period (default: 14)
            
        Returns:
            ATR value or None if calculation fails
        """
        # Check cache first
        cache_key = f"{symbol}_{period}"
        cached = self.atr_cache.get(cache_key)
        if cached and (datetime.now(timezone.utc).timestamp() - cached['timestamp'] < self.cache_ttl):
            return cached['value']
            
        try:
            # Fetch candles (need period + 1 for TR calculation)
            candles = get_candles(symbol, granularity=3600, limit=period + 1)  # 1-hour candles
            
            if not candles or len(candles) < period:
                logger.warning(f"Insufficient candle data for ATR calculation: {symbol}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            
            # Calculate True Range
            df['hl'] = df['high'] - df['low']
            df['hc'] = abs(df['high'] - df['close'].shift(1))
            df['lc'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
            
            # Calculate ATR
            atr = df['tr'].rolling(window=period).mean().iloc[-1]
            
            # Cache result
            self.atr_cache[cache_key] = {
                'value': atr,
                'timestamp': datetime.now(timezone.utc).timestamp()
            }
            
            return atr
            
        except Exception as e:
            logger.error(f"Failed to calculate ATR for {symbol}: {e}")
            return None
            
    def determine_stop_loss_distance(self, symbol: str, current_price: float) -> float:
        """
        Determine appropriate stop loss distance using ATR.
        
        Args:
            symbol: Trading pair
            current_price: Current market price
            
        Returns:
            Stop loss distance in dollars
        """
        atr = self.calculate_atr(symbol)
        
        if atr is None:
            # Fallback to percentage-based stop loss (1% default)
            return current_price * 0.01
            
        # Stop loss = ATR * multiplier
        return atr * ATR_MULTIPLIER
        
    def calculate_take_profit_distance(
        self, 
        stop_loss_distance: float, 
        risk_reward_ratio: float = DEFAULT_RISK_REWARD_RATIO
    ) -> float:
        """
        Calculate take profit distance based on risk/reward ratio.
        
        Args:
            stop_loss_distance: Stop loss distance in dollars
            risk_reward_ratio: Risk to reward ratio (default 1:1)
            
        Returns:
            Take profit distance in dollars
        """
        return stop_loss_distance * risk_reward_ratio
        
    def calculate_volatility_adjusted_size(
        self,
        base_position_size: float,
        symbol: str,
        current_volatility: Optional[float] = None
    ) -> float:
        """
        Adjust position size based on current volatility.
        
        Higher volatility = smaller position size
        
        Args:
            base_position_size: Base calculated position size
            symbol: Trading pair
            current_volatility: Current volatility (will calculate if not provided)
            
        Returns:
            Adjusted position size
        """
        if current_volatility is None:
            atr = self.calculate_atr(symbol)
            if atr is None:
                return base_position_size
                
            # Get current price for volatility calculation
            try:
                candles = get_candles(symbol, granularity=3600, limit=1)
                if candles and candles[0]:
                    current_price = candles[0]['close']
                    current_volatility = atr / current_price
                else:
                    return base_position_size
            except:
                return base_position_size
                
        # Volatility adjustment factor (inverse relationship)
        # If volatility is 2%, factor = 0.01 / 0.02 = 0.5 (half size)
        # If volatility is 0.5%, factor = 0.01 / 0.005 = 2.0 (double size, capped)
        target_volatility = 0.01  # 1% target
        adjustment_factor = min(2.0, max(0.25, target_volatility / current_volatility))
        
        return round(base_position_size * adjustment_factor, 2)
        
    def check_position_limits(
        self,
        current_positions: Dict[str, float],
        new_position_size: float,
        symbol: str,
        account_balance: float
    ) -> Tuple[bool, str]:
        """
        Check if new position respects risk limits.
        
        Args:
            current_positions: Dict of symbol -> position size in USD
            new_position_size: Proposed new position size
            symbol: Symbol for new position
            account_balance: Total account balance
            
        Returns:
            Tuple of (allowed, reason)
        """
        # Check total exposure
        total_exposure = sum(current_positions.values()) + new_position_size
        max_allowed_exposure = account_balance * MAX_TOTAL_EXPOSURE
        
        if total_exposure > max_allowed_exposure:
            return False, f"Total exposure would exceed limit: ${total_exposure:.2f} > ${max_allowed_exposure:.2f}"
            
        # Check per-symbol exposure (max 30% in one symbol)
        symbol_exposure = current_positions.get(symbol, 0) + new_position_size
        max_symbol_exposure = account_balance * 0.3
        
        if symbol_exposure > max_symbol_exposure:
            return False, f"Symbol exposure would exceed limit: ${symbol_exposure:.2f} > ${max_symbol_exposure:.2f}"
            
        # Check number of concurrent positions (max 5)
        if len(current_positions) >= 5 and symbol not in current_positions:
            return False, "Maximum number of concurrent positions (5) reached"
            
        return True, "Position allowed"
        

class VolatilityManager:
    """Manage volatility calculations and adjustments"""
    
    def __init__(self):
        self.volatility_cache: Dict[str, Dict] = {}
        self.cache_ttl = 60  # 1 minute cache
        
    def calculate_average_volume(self, symbol: str, period: int = 20) -> Optional[float]:
        """
        Calculate average volume over specified period.
        
        Args:
            symbol: Trading pair
            period: Number of periods for moving average
            
        Returns:
            Average volume or None
        """
        try:
            candles = get_candles(symbol, granularity=3600, limit=period)  # 1-hour candles
            
            if not candles or len(candles) < period:
                return None
                
            volumes = [c['volume'] for c in candles]
            return np.mean(volumes)
            
        except Exception as e:
            logger.error(f"Failed to calculate average volume for {symbol}: {e}")
            return None
            
    def calculate_spread_percent(self, best_bid: float, best_ask: float) -> float:
        """
        Calculate bid-ask spread as percentage of mid price.
        
        Args:
            best_bid: Best bid price
            best_ask: Best ask price
            
        Returns:
            Spread percentage (e.g., 0.002 for 0.2%)
        """
        if best_bid <= 0 or best_ask <= 0:
            return float('inf')
            
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        return spread / mid_price
        
    def is_spread_acceptable(self, best_bid: float, best_ask: float, max_spread_pct: float = 0.002) -> bool:
        """
        Check if spread is within acceptable limits.
        
        Args:
            best_bid: Best bid price
            best_ask: Best ask price
            max_spread_pct: Maximum acceptable spread percentage
            
        Returns:
            True if spread is acceptable
        """
        spread_pct = self.calculate_spread_percent(best_bid, best_ask)
        return spread_pct <= max_spread_pct
        
    def calculate_strategy_weights(
        self,
        current_volatility: float,
        base_scalping_weight: float = 0.6,
        base_breakout_weight: float = 0.4
    ) -> Dict[str, float]:
        """
        Dynamically adjust strategy weights based on market conditions.
        
        High volatility -> favor breakout
        Low volatility -> favor scalping
        
        Args:
            current_volatility: Current market volatility (ATR/Price ratio)
            base_scalping_weight: Base weight for scalping strategy
            base_breakout_weight: Base weight for breakout strategy
            
        Returns:
            Dict with adjusted weights
        """
        # Volatility thresholds
        low_vol_threshold = 0.01   # 1%
        high_vol_threshold = 0.03  # 3%
        
        if current_volatility <= low_vol_threshold:
            # Low volatility: favor scalping
            scalping_mult = 1.5
            breakout_mult = 0.5
        elif current_volatility >= high_vol_threshold:
            # High volatility: favor breakout
            scalping_mult = 0.5
            breakout_mult = 1.5
        else:
            # Normal volatility: linear interpolation
            vol_ratio = (current_volatility - low_vol_threshold) / (high_vol_threshold - low_vol_threshold)
            scalping_mult = 1.5 - (1.0 * vol_ratio)
            breakout_mult = 0.5 + (1.0 * vol_ratio)
            
        # Calculate adjusted weights
        scalping_weight = base_scalping_weight * scalping_mult
        breakout_weight = base_breakout_weight * breakout_mult
        
        # Normalize to sum to 1.0
        total_weight = scalping_weight + breakout_weight
        
        return {
            'scalping': scalping_weight / total_weight,
            'breakout': breakout_weight / total_weight
        }


# Global instances
risk_calculator = RiskCalculator()
volatility_manager = VolatilityManager() 