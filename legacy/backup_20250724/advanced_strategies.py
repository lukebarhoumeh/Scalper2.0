#!/usr/bin/env python3
"""
Advanced Trading Strategies - VWAP, Keltner Channels, ALMA
==========================================================
Implementation of sophisticated scalping strategies
"""

import numpy as np
import pandas as pd
import pandas_ta as ta  # Using pandas-ta instead of TA-Lib
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime, timezone
import logging

from strategy_engine_production import (
    ProductionStrategy, TradingSignal, MarketSnapshot, Action
)
from config_unified import TRADING_PAIRS

logger = logging.getLogger(__name__)


class VWAPMACDStrategy(ProductionStrategy):
    """
    VWAP + MACD Scalping Strategy
    - Enter when price crosses VWAP with MACD confirmation
    - Exit on MACD histogram flip or stop/target
    """
    
    name = "VWAP_MACD_Scalper"
    
    def __init__(self, executor):
        super().__init__(executor)
        self.config = {
            "vwap_period": 30,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "min_macd_diff": 0.001,  # Minimum MACD difference for signal
            "cooldown_seconds": 180
        }
        
    def calculate_vwap(self, candles: pd.DataFrame, period: int) -> float:
        """Calculate Volume Weighted Average Price"""
        if len(candles) < period:
            return 0.0
            
        recent = candles.tail(period)
        typical_price = (recent['high'] + recent['low'] + recent['close']) / 3
        vwap = np.sum(typical_price * recent['volume']) / np.sum(recent['volume'])
        
        return float(vwap)
        
    def evaluate(self, snapshot: MarketSnapshot) -> Optional[TradingSignal]:
        """Evaluate VWAP + MACD strategy"""
        
        # Check cooldown
        if not self._check_cooldown(snapshot.product_id):
            return None
            
        # Skip if spread too wide
        if snapshot.spread_bps > 20:  # 20 basis points max
            return None
            
        # Calculate VWAP
        vwap = self.calculate_vwap(snapshot.candles, self.config["vwap_period"])
        if vwap <= 0:
            return None
            
        # Get MACD values using pandas-ta
        close_prices = snapshot.candles['close']
        
        if len(close_prices) < self.config["macd_slow"] + self.config["macd_signal"]:
            return None
            
        macd_result = ta.macd(
            close_prices,
            fast=self.config["macd_fast"],
            slow=self.config["macd_slow"],
            signal=self.config["macd_signal"]
        )
        
        if macd_result is None or len(macd_result) < 2:
            return None
            
        # Extract MACD values
        macd_line = macd_result[f'MACD_{self.config["macd_fast"]}_{self.config["macd_slow"]}_{self.config["macd_signal"]}']
        signal_line = macd_result[f'MACDs_{self.config["macd_fast"]}_{self.config["macd_slow"]}_{self.config["macd_signal"]}']
        
        # Current values
        current_price = snapshot.last
        prev_price = close_prices.iloc[-2]
        macd_current = macd_line.iloc[-1]
        signal_current = signal_line.iloc[-1]
        macd_prev = macd_line.iloc[-2]
        signal_prev = signal_line.iloc[-2]
        
        # VWAP cross detection
        vwap_cross_up = prev_price <= vwap and current_price > vwap
        vwap_cross_down = prev_price >= vwap and current_price < vwap
        
        # MACD confirmation
        macd_bullish = (macd_current > signal_current and 
                       macd_prev <= signal_prev and
                       abs(macd_current - signal_current) > self.config["min_macd_diff"])
                       
        macd_bearish = (macd_current < signal_current and 
                       macd_prev >= signal_prev and
                       abs(macd_current - signal_current) > self.config["min_macd_diff"])
        
        # Generate signals
        if vwap_cross_up and macd_bullish:
            # Bullish signal
            confidence = self._calculate_confidence(snapshot, "bullish", vwap)
            size = self._calculate_position_size(snapshot, confidence)
            
            self._update_cooldown(snapshot.product_id)
            
            return TradingSignal(
                product_id=snapshot.product_id,
                action=Action.BUY,
                confidence=confidence,
                size_usd=size,
                indicators={
                    'vwap': vwap,
                    'price': current_price,
                    'macd': macd_current,
                    'macd_signal': signal_current,
                    'volume_ratio': snapshot.volume_ratio
                },
                volatility_regime=snapshot.volatility_regime,
                strategy_source=self.name,
                timestamp=datetime.now(timezone.utc),
                risk_score=self._calculate_risk_score(snapshot)
            )
            
        elif vwap_cross_down and macd_bearish:
            # Check if we have position to sell
            current_position = self._executor.position_base(snapshot.product_id)
            if current_position > 0:
                confidence = self._calculate_confidence(snapshot, "bearish", vwap)
                size = min(
                    self._calculate_position_size(snapshot, confidence),
                    current_position * snapshot.last
                )
                
                self._update_cooldown(snapshot.product_id)
                
                return TradingSignal(
                    product_id=snapshot.product_id,
                    action=Action.SELL,
                    confidence=confidence,
                    size_usd=size,
                    indicators={
                        'vwap': vwap,
                        'price': current_price,
                        'macd': macd_current,
                        'macd_signal': signal_current,
                        'volume_ratio': snapshot.volume_ratio
                    },
                    volatility_regime=snapshot.volatility_regime,
                    strategy_source=self.name,
                    timestamp=datetime.now(timezone.utc),
                    risk_score=self._calculate_risk_score(snapshot)
                )
                
        return None
        
    def _calculate_confidence(self, snapshot: MarketSnapshot, direction: str, vwap: float) -> float:
        """Calculate signal confidence"""
        confidence = 0.5  # Base confidence
        
        # Volume confirmation
        if snapshot.volume_ratio > 1.5:
            confidence += 0.2
        elif snapshot.volume_ratio < 0.5:
            confidence -= 0.1
            
        # Price distance from VWAP
        vwap_distance = abs(snapshot.last - vwap) / vwap
        if vwap_distance < 0.002:  # Very close to VWAP
            confidence += 0.1
        elif vwap_distance > 0.01:  # Far from VWAP
            confidence -= 0.1
            
        # Volatility adjustment
        if snapshot.volatility_regime == "normal":
            confidence += 0.1
        elif snapshot.volatility_regime in ["high", "extreme"]:
            confidence -= 0.2
            
        return min(max(confidence, 0.3), 0.9)
        
    def _calculate_position_size(self, snapshot: MarketSnapshot, confidence: float) -> float:
        """Calculate position size based on confidence and volatility"""
        base_size = 50.0  # Base size in USD
        
        # Confidence adjustment
        size = base_size * (0.8 + 0.4 * confidence)
        
        # Volatility adjustment
        if snapshot.volatility_regime == "low":
            size *= 1.2
        elif snapshot.volatility_regime == "high":
            size *= 0.7
        elif snapshot.volatility_regime == "extreme":
            size *= 0.5
            
        return min(size, 150.0)  # Cap at $150
        
    def _calculate_risk_score(self, snapshot: MarketSnapshot) -> float:
        """Calculate risk score for the signal"""
        risk = 0.5  # Base risk
        
        # Spread risk
        risk += snapshot.spread_bps / 100
        
        # Volatility risk
        if snapshot.volatility_regime == "extreme":
            risk += 0.3
        elif snapshot.volatility_regime == "high":
            risk += 0.2
            
        return min(risk, 1.0)


class KeltnerRSIStrategy(ProductionStrategy):
    """
    Keltner Channel + RSI Strategy
    - Enter when price closes outside channel with RSI confirmation
    - Exit at opposite channel or RSI extremes
    """
    
    name = "Keltner_RSI_Scalper"
    
    def __init__(self, executor):
        super().__init__(executor)
        self.config = {
            "keltner_period": 20,
            "keltner_mult": 2.0,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_neutral": 50,
            "cooldown_seconds": 180
        }
        
    def calculate_keltner_channels(self, candles: pd.DataFrame) -> tuple:
        """Calculate Keltner Channels using pandas-ta"""
        period = self.config["keltner_period"]
        mult = self.config["keltner_mult"]
        
        if len(candles) < period:
            return 0.0, 0.0, 0.0
            
        # Calculate Keltner Channels using pandas-ta
        kc_result = ta.kc(
            high=candles['high'],
            low=candles['low'],
            close=candles['close'],
            length=period,
            scalar=mult
        )
        
        if kc_result is None or len(kc_result) < 1:
            return 0.0, 0.0, 0.0
            
        # Extract channel values
        lower = kc_result[f'KCLl_{period}_{mult}'].iloc[-1]
        middle = kc_result[f'KCBm_{period}_{mult}'].iloc[-1]
        upper = kc_result[f'KCUu_{period}_{mult}'].iloc[-1]
        
        return float(upper), float(middle), float(lower)
        
    def evaluate(self, snapshot: MarketSnapshot) -> Optional[TradingSignal]:
        """Evaluate Keltner + RSI strategy"""
        
        # Check cooldown
        if not self._check_cooldown(snapshot.product_id):
            return None
            
        # Skip if spread too wide
        if snapshot.spread_bps > 25:
            return None
            
        # Calculate indicators
        upper, middle, lower = self.calculate_keltner_channels(snapshot.candles)
        
        if upper <= 0 or lower <= 0:
            return None
            
        # Current price and RSI
        current_price = snapshot.last
        current_rsi = snapshot.rsi
        
        # Determine position relative to channels
        above_upper = current_price > upper
        below_lower = current_price < lower
        
        # Generate signals
        if below_lower and current_rsi < self.config["rsi_neutral"]:
            # Bullish signal - price below lower channel with RSI confirmation
            confidence = self._calculate_confidence(
                snapshot, "bullish", current_rsi, lower, upper
            )
            
            size = self._calculate_position_size(snapshot, confidence)
            
            self._update_cooldown(snapshot.product_id)
            
            return TradingSignal(
                product_id=snapshot.product_id,
                action=Action.BUY,
                confidence=confidence,
                size_usd=size,
                indicators={
                    'keltner_upper': upper,
                    'keltner_middle': middle,
                    'keltner_lower': lower,
                    'rsi': current_rsi,
                    'price': current_price
                },
                volatility_regime=snapshot.volatility_regime,
                strategy_source=self.name,
                timestamp=datetime.now(timezone.utc),
                risk_score=self._calculate_risk_score(snapshot)
            )
            
        elif above_upper and current_rsi > self.config["rsi_neutral"]:
            # Bearish signal - price above upper channel with RSI confirmation
            current_position = self._executor.position_base(snapshot.product_id)
            
            if current_position > 0:
                confidence = self._calculate_confidence(
                    snapshot, "bearish", current_rsi, lower, upper
                )
                
                size = min(
                    self._calculate_position_size(snapshot, confidence),
                    current_position * snapshot.last
                )
                
                self._update_cooldown(snapshot.product_id)
                
                return TradingSignal(
                    product_id=snapshot.product_id,
                    action=Action.SELL,
                    confidence=confidence,
                    size_usd=size,
                    indicators={
                        'keltner_upper': upper,
                        'keltner_middle': middle,
                        'keltner_lower': lower,
                        'rsi': current_rsi,
                        'price': current_price
                    },
                    volatility_regime=snapshot.volatility_regime,
                    strategy_source=self.name,
                    timestamp=datetime.now(timezone.utc),
                    risk_score=self._calculate_risk_score(snapshot)
                )
                
        return None
        
    def _calculate_confidence(self, snapshot: MarketSnapshot, direction: str, 
                            rsi: float, lower: float, upper: float) -> float:
        """Calculate signal confidence"""
        confidence = 0.5
        
        # RSI strength
        if direction == "bullish":
            if rsi < self.config["rsi_oversold"]:
                confidence += 0.2
            confidence += (self.config["rsi_neutral"] - rsi) / 100
        else:
            if rsi > self.config["rsi_overbought"]:
                confidence += 0.2
            confidence += (rsi - self.config["rsi_neutral"]) / 100
            
        # Channel width (volatility)
        channel_width = (upper - lower) / lower
        if channel_width < 0.02:  # Narrow channel
            confidence -= 0.1
        elif channel_width > 0.05:  # Wide channel
            confidence += 0.1
            
        # Volume confirmation
        if snapshot.volume_ratio > 1.2:
            confidence += 0.1
            
        return min(max(confidence, 0.3), 0.9)
        
    def _calculate_position_size(self, snapshot: MarketSnapshot, confidence: float) -> float:
        """Dynamic position sizing"""
        base_size = 60.0
        
        size = base_size * (0.7 + 0.6 * confidence)
        
        # Volatility adjustment
        if snapshot.volatility_regime == "low":
            size *= 1.3
        elif snapshot.volatility_regime == "high":
            size *= 0.6
        elif snapshot.volatility_regime == "extreme":
            size *= 0.4
            
        return min(size, 150.0)
        
    def _calculate_risk_score(self, snapshot: MarketSnapshot) -> float:
        """Calculate risk score"""
        risk = 0.4
        
        # Spread risk
        risk += snapshot.spread_bps / 80
        
        # RSI extreme risk
        if snapshot.rsi > 80 or snapshot.rsi < 20:
            risk += 0.2
            
        # Volatility risk
        if snapshot.volatility_regime == "extreme":
            risk += 0.3
        elif snapshot.volatility_regime == "high":
            risk += 0.2
            
        return min(risk, 1.0)


class ALMAStochasticStrategy(ProductionStrategy):
    """
    ALMA + Stochastic Strategy
    - ALMA for trend, Stochastic for timing
    - Enter on ALMA cross with Stochastic confirmation
    """
    
    name = "ALMA_Stochastic_Scalper"
    
    def __init__(self, executor):
        super().__init__(executor)
        self.config = {
            "alma_period": 21,
            "alma_offset": 0.85,
            "alma_sigma": 6.0,
            "stoch_k": 14,
            "stoch_d": 3,
            "stoch_smooth": 3,
            "stoch_overbought": 80,
            "stoch_oversold": 20,
            "cooldown_seconds": 180
        }
        
    def calculate_alma(self, prices: np.ndarray, period: int, 
                      offset: float, sigma: float) -> float:
        """Calculate Arnaud Legoux Moving Average"""
        if len(prices) < period:
            return 0.0
            
        # ALMA weights calculation
        m = offset * (period - 1)
        s = period / sigma
        
        weights = np.zeros(period)
        for i in range(period):
            weights[i] = np.exp(-((i - m) ** 2) / (2 * s ** 2))
            
        weights = weights / np.sum(weights)
        
        # Apply weights to recent prices
        recent_prices = prices[-period:]
        alma = np.sum(recent_prices * weights)
        
        return float(alma)
        
    def evaluate(self, snapshot: MarketSnapshot) -> Optional[TradingSignal]:
        """Evaluate ALMA + Stochastic strategy"""
        
        # Check cooldown
        if not self._check_cooldown(snapshot.product_id):
            return None
            
        # Skip if spread too wide
        if snapshot.spread_bps > 20:
            return None
            
        # Calculate ALMA
        close_prices = snapshot.candles['close'].values
        
        alma_current = self.calculate_alma(
            close_prices,
            self.config["alma_period"],
            self.config["alma_offset"],
            self.config["alma_sigma"]
        )
        
        if alma_current <= 0 or len(close_prices) < self.config["alma_period"] + 1:
            return None
            
        # Previous ALMA for crossover detection
        alma_prev = self.calculate_alma(
            close_prices[:-1],
            self.config["alma_period"],
            self.config["alma_offset"],
            self.config["alma_sigma"]
        )
        
        # Calculate Stochastic using pandas-ta
        stoch_result = ta.stoch(
            high=snapshot.candles['high'],
            low=snapshot.candles['low'],
            close=snapshot.candles['close'],
            k=self.config["stoch_k"],
            d=self.config["stoch_d"],
            smooth_k=self.config["stoch_smooth"]
        )
        
        if stoch_result is None or len(stoch_result) < 2:
            return None
            
        # Extract Stochastic values
        k_values = stoch_result[f'STOCHk_{self.config["stoch_k"]}_{self.config["stoch_d"]}_{self.config["stoch_smooth"]}']
        d_values = stoch_result[f'STOCHd_{self.config["stoch_k"]}_{self.config["stoch_d"]}_{self.config["stoch_smooth"]}']
        
        # Current values
        current_price = snapshot.last
        prev_price = close_prices[-2]
        stoch_k = k_values.iloc[-1]
        stoch_k_prev = k_values.iloc[-2]
        stoch_d = d_values.iloc[-1]
        
        # ALMA crossover detection
        alma_cross_up = prev_price <= alma_prev and current_price > alma_current
        alma_cross_down = prev_price >= alma_prev and current_price < alma_current
        
        # Stochastic confirmation
        stoch_bullish = (stoch_k < self.config["stoch_oversold"] and 
                        stoch_k > stoch_k_prev)
                        
        stoch_bearish = (stoch_k > self.config["stoch_overbought"] and 
                        stoch_k < stoch_k_prev)
        
        # Generate signals
        if alma_cross_up and stoch_bullish:
            # Bullish signal
            confidence = self._calculate_confidence(
                snapshot, "bullish", stoch_k, alma_current
            )
            
            size = self._calculate_position_size(snapshot, confidence)
            
            self._update_cooldown(snapshot.product_id)
            
            return TradingSignal(
                product_id=snapshot.product_id,
                action=Action.BUY,
                confidence=confidence,
                size_usd=size,
                indicators={
                    'alma': alma_current,
                    'price': current_price,
                    'stoch_k': stoch_k,
                    'stoch_d': stoch_d,
                    'volume_ratio': snapshot.volume_ratio
                },
                volatility_regime=snapshot.volatility_regime,
                strategy_source=self.name,
                timestamp=datetime.now(timezone.utc),
                risk_score=self._calculate_risk_score(snapshot)
            )
            
        elif alma_cross_down and stoch_bearish:
            # Bearish signal
            current_position = self._executor.position_base(snapshot.product_id)
            
            if current_position > 0:
                confidence = self._calculate_confidence(
                    snapshot, "bearish", stoch_k, alma_current
                )
                
                size = min(
                    self._calculate_position_size(snapshot, confidence),
                    current_position * snapshot.last
                )
                
                self._update_cooldown(snapshot.product_id)
                
                return TradingSignal(
                    product_id=snapshot.product_id,
                    action=Action.SELL,
                    confidence=confidence,
                    size_usd=size,
                    indicators={
                        'alma': alma_current,
                        'price': current_price,
                        'stoch_k': stoch_k,
                        'stoch_d': stoch_d,
                        'volume_ratio': snapshot.volume_ratio
                    },
                    volatility_regime=snapshot.volatility_regime,
                    strategy_source=self.name,
                    timestamp=datetime.now(timezone.utc),
                    risk_score=self._calculate_risk_score(snapshot)
                )
                
        return None
        
    def _calculate_confidence(self, snapshot: MarketSnapshot, direction: str,
                            stoch_k: float, alma: float) -> float:
        """Calculate signal confidence"""
        confidence = 0.5
        
        # Stochastic strength
        if direction == "bullish":
            confidence += (self.config["stoch_oversold"] - stoch_k) / 40
        else:
            confidence += (stoch_k - self.config["stoch_overbought"]) / 40
            
        # Price distance from ALMA
        alma_distance = abs(snapshot.last - alma) / alma
        if alma_distance < 0.002:
            confidence += 0.15
        elif alma_distance > 0.01:
            confidence -= 0.1
            
        # Volume confirmation
        if snapshot.volume_ratio > 1.3:
            confidence += 0.15
            
        # Volatility adjustment
        if snapshot.volatility_regime == "normal":
            confidence += 0.1
        elif snapshot.volatility_regime in ["high", "extreme"]:
            confidence -= 0.15
            
        return min(max(confidence, 0.3), 0.9)
        
    def _calculate_position_size(self, snapshot: MarketSnapshot, confidence: float) -> float:
        """Dynamic position sizing"""
        base_size = 55.0
        
        size = base_size * (0.8 + 0.4 * confidence)
        
        # Volatility adjustment
        if snapshot.volatility_regime == "low":
            size *= 1.25
        elif snapshot.volatility_regime == "high":
            size *= 0.65
        elif snapshot.volatility_regime == "extreme":
            size *= 0.45
            
        return min(size, 150.0)
        
    def _calculate_risk_score(self, snapshot: MarketSnapshot) -> float:
        """Calculate risk score"""
        risk = 0.45
        
        # Spread risk
        risk += snapshot.spread_bps / 100
        
        # Stochastic extreme risk
        if snapshot.rsi > 85 or snapshot.rsi < 15:
            risk += 0.25
            
        # Volatility risk
        if snapshot.volatility_regime == "extreme":
            risk += 0.3
        elif snapshot.volatility_regime == "high":
            risk += 0.2
            
        return min(risk, 1.0)


def combine_strategy_signals(signals: List[TradingSignal], 
                           weights: Dict[str, float]) -> Optional[TradingSignal]:
    """
    Combine multiple strategy signals with weighted voting
    """
    if not signals:
        return None
        
    # Group by product and action
    product_signals = {}
    
    for signal in signals:
        key = (signal.product_id, signal.action)
        if key not in product_signals:
            product_signals[key] = []
        product_signals[key].append(signal)
        
    # Find strongest combined signal
    best_score = 0
    best_signal = None
    
    for (product_id, action), sigs in product_signals.items():
        # Calculate weighted score
        total_score = 0
        total_weight = 0
        
        for sig in sigs:
            weight = weights.get(sig.strategy_source, 0.25)
            total_score += sig.confidence * weight
            total_weight += weight
            
        if total_weight > 0:
            avg_score = total_score / total_weight
            
            # Need at least 2 strategies to agree or very high confidence
            if (len(sigs) >= 2 or avg_score > 0.8) and avg_score > best_score:
                best_score = avg_score
                
                # Create combined signal
                best_signal = TradingSignal(
                    product_id=product_id,
                    action=action,
                    confidence=avg_score,
                    size_usd=np.mean([s.size_usd for s in sigs]),
                    indicators={
                        'strategies': [s.strategy_source for s in sigs],
                        'individual_confidences': [s.confidence for s in sigs]
                    },
                    volatility_regime=sigs[0].volatility_regime,
                    strategy_source="Combined",
                    timestamp=datetime.now(timezone.utc),
                    risk_score=np.mean([s.risk_score for s in sigs])
                )
                
    return best_signal 