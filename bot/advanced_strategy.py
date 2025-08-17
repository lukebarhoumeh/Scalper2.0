"""Advanced Multi-Strategy Trading System"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .models import Signal, Side, PriceHistory
from .indicators import simple_moving_average, rsi, breakout
from .advanced_indicators import (
    vwap, keltner_channels, bollinger_bands, macd, 
    stochastic_oscillator, money_flow_index, supertrend,
    williams_r, commodity_channel_index
)


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    SQUEEZE = "squeeze"


class TrendStrength(Enum):
    STRONG = "strong"
    MODERATE = "moderate" 
    WEAK = "weak"
    NONE = "none"


@dataclass
class MarketContext:
    """Complete market context for decision making"""
    regime: MarketRegime
    trend_strength: TrendStrength
    volatility_percentile: float
    volume_profile: str  # "increasing", "decreasing", "stable"
    microstructure_bias: float  # -1 to 1
    momentum_score: float  # -100 to 100
    mean_reversion_score: float  # 0 to 100


class AdvancedStrategyEngine:
    """Multi-strategy engine with market regime adaptation"""
    
    def __init__(self, 
                 lookback_period: int = 100,
                 min_confidence: float = 0.6,
                 max_correlation: float = 0.7):
        self.lookback_period = lookback_period
        self.min_confidence = min_confidence
        self.max_correlation = max_correlation
        
        # Strategy weights by regime
        self.regime_weights = {
            MarketRegime.TRENDING_UP: {
                "momentum": 0.4, "breakout": 0.3, "vwap": 0.2, "mean_reversion": 0.1
            },
            MarketRegime.TRENDING_DOWN: {
                "momentum": 0.4, "breakout": 0.3, "vwap": 0.2, "mean_reversion": 0.1
            },
            MarketRegime.RANGING: {
                "mean_reversion": 0.4, "bollinger": 0.3, "rsi": 0.2, "stochastic": 0.1
            },
            MarketRegime.VOLATILE: {
                "keltner": 0.3, "atr_breakout": 0.3, "supertrend": 0.2, "defensive": 0.2
            },
            MarketRegime.SQUEEZE: {
                "breakout": 0.5, "momentum": 0.3, "volume": 0.2
            }
        }
        
        # Track historical context
        self.context_history: Dict[str, List[MarketContext]] = {}
        self.signal_history: Dict[str, List[Signal]] = {}
        
    def analyze_market_context(self, symbol: str, prices: List[float], 
                             volumes: List[float], spread: float) -> MarketContext:
        """Comprehensive market analysis"""
        if len(prices) < self.lookback_period:
            return MarketContext(
                regime=MarketRegime.RANGING,
                trend_strength=TrendStrength.NONE,
                volatility_percentile=50.0,
                volume_profile="stable",
                microstructure_bias=0.0,
                momentum_score=0.0,
                mean_reversion_score=50.0
            )
        
        # Detect regime
        regime = self._detect_regime(prices)
        
        # Measure trend strength
        trend_strength = self._measure_trend_strength(prices)
        
        # Calculate volatility percentile
        vol_percentile = self._calculate_volatility_percentile(prices)
        
        # Analyze volume profile
        volume_profile = self._analyze_volume_profile(volumes)
        
        # Microstructure analysis
        micro_bias = self._analyze_microstructure(prices, spread)
        
        # Momentum scoring
        momentum = self._calculate_momentum_score(prices)
        
        # Mean reversion potential
        mr_score = self._calculate_mean_reversion_score(prices)
        
        return MarketContext(
            regime=regime,
            trend_strength=trend_strength,
            volatility_percentile=vol_percentile,
            volume_profile=volume_profile,
            microstructure_bias=micro_bias,
            momentum_score=momentum,
            mean_reversion_score=mr_score
        )
    
    def generate_signals(self, symbol: str, prices: List[float], volumes: List[float],
                        bid: float, ask: float, last_price: float) -> List[Signal]:
        """Generate signals from multiple strategies"""
        
        # Need enough data
        if len(prices) < 50:
            return []
        
        # Get market context
        spread = ask - bid
        context = self.analyze_market_context(symbol, prices, volumes, spread)
        
        # Store context
        if symbol not in self.context_history:
            self.context_history[symbol] = []
        self.context_history[symbol].append(context)
        
        signals = []
        
        # Run strategies based on regime
        weights = self.regime_weights[context.regime]
        
        # Momentum strategies
        if weights.get("momentum", 0) > 0:
            momentum_signal = self._momentum_strategy(prices, context)
            if momentum_signal:
                momentum_signal.confidence *= weights["momentum"]
                signals.append(momentum_signal)
        
        # Mean reversion strategies  
        if weights.get("mean_reversion", 0) > 0:
            mr_signal = self._mean_reversion_strategy(prices, context)
            if mr_signal:
                mr_signal.confidence *= weights["mean_reversion"]
                signals.append(mr_signal)
        
        # Breakout strategies
        if weights.get("breakout", 0) > 0:
            breakout_signal = self._breakout_strategy(prices, volumes, context)
            if breakout_signal:
                breakout_signal.confidence *= weights["breakout"]
                signals.append(breakout_signal)
        
        # VWAP strategies
        if weights.get("vwap", 0) > 0 and len(volumes) >= 20:
            vwap_signal = self._vwap_strategy(prices, volumes, last_price, context)
            if vwap_signal:
                vwap_signal.confidence *= weights["vwap"]
                signals.append(vwap_signal)
        
        # Bollinger band strategies
        if weights.get("bollinger", 0) > 0:
            bb_signal = self._bollinger_strategy(prices, context)
            if bb_signal:
                bb_signal.confidence *= weights["bollinger"]
                signals.append(bb_signal)
        
        # Filter and combine signals
        strong_signals = [s for s in signals if s.confidence >= self.min_confidence * 0.5]
        
        if not strong_signals:
            return []
        
        # Check correlation and combine
        return self._combine_signals(strong_signals, symbol)
    
    def _detect_regime(self, prices: List[float]) -> MarketRegime:
        """Detect current market regime"""
        
        # Use multiple timeframes
        short_sma = simple_moving_average(prices, 10)
        medium_sma = simple_moving_average(prices, 20)
        long_sma = simple_moving_average(prices, 50)
        
        if not all([short_sma, medium_sma, long_sma]):
            return MarketRegime.RANGING
        
        # Calculate directional movement
        price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        recent_changes = price_changes[-20:]
        
        up_moves = sum(1 for c in recent_changes if c > 0)
        down_moves = sum(1 for c in recent_changes if c < 0)
        
        # Volatility analysis
        volatility = np.std(recent_changes)
        avg_volatility = np.std(price_changes)
        
        # Trend detection
        if short_sma > medium_sma > long_sma:
            if volatility < avg_volatility * 0.8:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.VOLATILE
        elif short_sma < medium_sma < long_sma:
            if volatility < avg_volatility * 0.8:
                return MarketRegime.TRENDING_DOWN
            else:
                return MarketRegime.VOLATILE
        elif volatility < avg_volatility * 0.5:
            return MarketRegime.SQUEEZE
        else:
            return MarketRegime.RANGING
    
    def _measure_trend_strength(self, prices: List[float]) -> TrendStrength:
        """Measure strength of current trend"""
        
        if len(prices) < 20:
            return TrendStrength.NONE
        
        # Linear regression slope
        x = np.arange(len(prices[-20:]))
        y = np.array(prices[-20:])
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize by average price
        avg_price = np.mean(y)
        normalized_slope = (slope / avg_price) * 100
        
        # R-squared for trend quality
        y_pred = np.polyval(np.polyfit(x, y, 1), x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Classify strength
        if abs(normalized_slope) > 2.0 and r_squared > 0.7:
            return TrendStrength.STRONG
        elif abs(normalized_slope) > 1.0 and r_squared > 0.5:
            return TrendStrength.MODERATE
        elif abs(normalized_slope) > 0.5:
            return TrendStrength.WEAK
        else:
            return TrendStrength.NONE
    
    def _calculate_volatility_percentile(self, prices: List[float]) -> float:
        """Calculate current volatility as percentile of historical"""
        
        if len(prices) < 30:
            return 50.0
        
        # Calculate rolling volatilities
        volatilities = []
        for i in range(20, len(prices)):
            window = prices[i-20:i]
            vol = np.std([window[j] - window[j-1] for j in range(1, len(window))])
            volatilities.append(vol)
        
        if not volatilities:
            return 50.0
        
        current_vol = volatilities[-1]
        percentile = (sum(1 for v in volatilities if v < current_vol) / len(volatilities)) * 100
        
        return percentile
    
    def _analyze_volume_profile(self, volumes: List[float]) -> str:
        """Analyze volume trends"""
        
        if len(volumes) < 20:
            return "stable"
        
        recent_avg = np.mean(volumes[-10:])
        previous_avg = np.mean(volumes[-20:-10])
        
        ratio = recent_avg / (previous_avg + 1e-10)
        
        if ratio > 1.3:
            return "increasing"
        elif ratio < 0.7:
            return "decreasing"
        else:
            return "stable"
    
    def _analyze_microstructure(self, prices: List[float], spread: float) -> float:
        """Analyze microstructure for short-term bias"""
        
        if len(prices) < 10:
            return 0.0
        
        # Analyze recent price movements vs spread
        recent_moves = []
        for i in range(-10, 0):
            if i < -1:
                move = prices[i] - prices[i-1]
                recent_moves.append(move)
        
        if not recent_moves:
            return 0.0
        
        # Calculate microstructure score
        avg_move = np.mean(recent_moves)
        avg_abs_move = np.mean([abs(m) for m in recent_moves])
        
        if avg_abs_move == 0:
            return 0.0
        
        # Directional bias adjusted by spread
        bias = (avg_move / avg_abs_move) * (1 - min(spread / avg_abs_move, 0.5))
        
        return np.clip(bias, -1, 1)
    
    def _calculate_momentum_score(self, prices: List[float]) -> float:
        """Calculate comprehensive momentum score"""
        
        if len(prices) < 30:
            return 0.0
        
        scores = []
        
        # Rate of change
        roc_5 = ((prices[-1] - prices[-6]) / prices[-6]) * 100
        roc_10 = ((prices[-1] - prices[-11]) / prices[-11]) * 100
        roc_20 = ((prices[-1] - prices[-21]) / prices[-21]) * 100
        
        # Weighted ROC
        weighted_roc = (roc_5 * 0.5 + roc_10 * 0.3 + roc_20 * 0.2)
        scores.append(np.clip(weighted_roc * 10, -100, 100))
        
        # RSI momentum
        rsi_val = rsi(prices, 14) or 50
        rsi_momentum = (rsi_val - 50) * 2  # -100 to 100
        scores.append(rsi_momentum)
        
        # Moving average alignment
        sma_5 = simple_moving_average(prices, 5)
        sma_10 = simple_moving_average(prices, 10)
        sma_20 = simple_moving_average(prices, 20)
        
        if sma_5 and sma_10 and sma_20:
            if sma_5 > sma_10 > sma_20:
                ma_score = 50
            elif sma_5 < sma_10 < sma_20:
                ma_score = -50
            else:
                ma_score = 0
            scores.append(ma_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_mean_reversion_score(self, prices: List[float]) -> float:
        """Calculate mean reversion potential"""
        
        if len(prices) < 20:
            return 50.0
        
        current_price = prices[-1]
        sma_20 = simple_moving_average(prices, 20)
        
        if not sma_20:
            return 50.0
        
        # Distance from mean
        distance = ((current_price - sma_20) / sma_20) * 100
        
        # Bollinger band position
        upper, middle, lower = bollinger_bands(prices, 20, 2.0)
        
        if upper and lower:
            bb_width = upper - lower
            bb_position = (current_price - lower) / bb_width if bb_width > 0 else 0.5
            
            # Mean reversion score (0-100)
            if bb_position > 0.95:  # Near upper band
                mr_score = 90
            elif bb_position < 0.05:  # Near lower band  
                mr_score = 90
            elif 0.4 <= bb_position <= 0.6:  # Near middle
                mr_score = 20
            else:
                mr_score = 50 + abs(bb_position - 0.5) * 80
        else:
            mr_score = 50
        
        return mr_score
    
    def _momentum_strategy(self, prices: List[float], context: MarketContext) -> Optional[Signal]:
        """Pure momentum strategy"""
        
        if context.momentum_score > 30 and context.trend_strength in [TrendStrength.STRONG, TrendStrength.MODERATE]:
            # Bullish momentum
            confidence = 0.6 + (context.momentum_score / 100) * 0.3
            return Signal(
                symbol="",  # Will be filled by caller
                side=Side.BUY,
                confidence=confidence,
                reason="momentum_bullish",
                qty=0.0
            )
        elif context.momentum_score < -30 and context.trend_strength in [TrendStrength.STRONG, TrendStrength.MODERATE]:
            # Bearish momentum
            confidence = 0.6 + (abs(context.momentum_score) / 100) * 0.3
            return Signal(
                symbol="",
                side=Side.SELL,
                confidence=confidence,
                reason="momentum_bearish",
                qty=0.0
            )
        
        return None
    
    def _mean_reversion_strategy(self, prices: List[float], context: MarketContext) -> Optional[Signal]:
        """Mean reversion strategy"""
        
        if context.mean_reversion_score < 80:
            return None
        
        # Check if we're at extremes
        upper, middle, lower = bollinger_bands(prices, 20, 2.0)
        
        if not all([upper, middle, lower]):
            return None
        
        current_price = prices[-1]
        rsi_val = rsi(prices, 14) or 50
        
        if current_price >= upper and rsi_val > 70:
            # Overbought - sell signal
            confidence = 0.6 + (context.mean_reversion_score / 100) * 0.3
            return Signal(
                symbol="",
                side=Side.SELL,
                confidence=confidence,
                reason="mean_reversion_overbought",
                qty=0.0
            )
        elif current_price <= lower and rsi_val < 30:
            # Oversold - buy signal
            confidence = 0.6 + (context.mean_reversion_score / 100) * 0.3
            return Signal(
                symbol="",
                side=Side.BUY,
                confidence=confidence,
                reason="mean_reversion_oversold", 
                qty=0.0
            )
        
        return None
    
    def _breakout_strategy(self, prices: List[float], volumes: List[float], 
                          context: MarketContext) -> Optional[Signal]:
        """Breakout detection strategy"""
        
        if len(prices) < 50:
            return None
        
        # Find recent high/low
        recent_high = max(prices[-20:])
        recent_low = min(prices[-20:])
        current_price = prices[-1]
        
        # Check for breakout with volume confirmation
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
        current_volume = volumes[-1] if volumes else 0
        
        volume_surge = current_volume > avg_volume * 1.5 if avg_volume > 0 else False
        
        # Breakout conditions
        if current_price > recent_high and volume_surge:
            # Upside breakout
            confidence = 0.7
            if context.regime == MarketRegime.SQUEEZE:
                confidence += 0.2
            
            return Signal(
                symbol="",
                side=Side.BUY,
                confidence=confidence,
                reason="breakout_upside",
                qty=0.0
            )
        elif current_price < recent_low and volume_surge:
            # Downside breakout
            confidence = 0.7
            if context.regime == MarketRegime.SQUEEZE:
                confidence += 0.2
            
            return Signal(
                symbol="",
                side=Side.SELL,
                confidence=confidence,
                reason="breakout_downside",
                qty=0.0
            )
        
        return None
    
    def _vwap_strategy(self, prices: List[float], volumes: List[float], 
                      current_price: float, context: MarketContext) -> Optional[Signal]:
        """VWAP-based strategy"""
        
        vwap_val = vwap(prices, volumes, 20)
        
        if not vwap_val:
            return None
        
        # Price relative to VWAP
        distance = ((current_price - vwap_val) / vwap_val) * 100
        
        # VWAP acts as dynamic support/resistance
        if distance < -1.0 and context.momentum_score > 0:
            # Price below VWAP with positive momentum
            confidence = 0.65 + min(abs(distance) * 0.05, 0.25)
            return Signal(
                symbol="",
                side=Side.BUY,
                confidence=confidence,
                reason="vwap_support_bounce",
                qty=0.0
            )
        elif distance > 1.0 and context.momentum_score < 0:
            # Price above VWAP with negative momentum
            confidence = 0.65 + min(distance * 0.05, 0.25)
            return Signal(
                symbol="",
                side=Side.SELL,
                confidence=confidence,
                reason="vwap_resistance_rejection",
                qty=0.0
            )
        
        return None
    
    def _bollinger_strategy(self, prices: List[float], context: MarketContext) -> Optional[Signal]:
        """Bollinger Bands strategy"""
        
        upper, middle, lower = bollinger_bands(prices, 20, 2.0)
        
        if not all([upper, middle, lower]):
            return None
        
        current_price = prices[-1]
        band_width = (upper - lower) / middle
        
        # Squeeze detection
        is_squeeze = band_width < 0.02  # Less than 2% width
        
        if is_squeeze and context.regime == MarketRegime.SQUEEZE:
            # Wait for breakout direction
            return None
        
        # Band touches with momentum filter
        if current_price <= lower and context.momentum_score > -30:
            confidence = 0.7
            return Signal(
                symbol="",
                side=Side.BUY,
                confidence=confidence,
                reason="bollinger_lower_touch",
                qty=0.0
            )
        elif current_price >= upper and context.momentum_score < 30:
            confidence = 0.7
            return Signal(
                symbol="",
                side=Side.SELL,
                confidence=confidence,
                reason="bollinger_upper_touch",
                qty=0.0
            )
        
        return None
    
    def _combine_signals(self, signals: List[Signal], symbol: str) -> List[Signal]:
        """Combine and filter signals"""
        
        if not signals:
            return []
        
        # Group by side
        buy_signals = [s for s in signals if s.side == Side.BUY]
        sell_signals = [s for s in signals if s.side == Side.SELL]
        
        combined = []
        
        # Process buy signals
        if buy_signals:
            # Average confidence
            avg_confidence = np.mean([s.confidence for s in buy_signals])
            
            # Only proceed if multiple strategies agree or very high confidence
            if len(buy_signals) >= 2 or avg_confidence >= 0.8:
                reasons = [s.reason for s in buy_signals]
                combined_signal = Signal(
                    symbol=symbol,
                    side=Side.BUY,
                    confidence=min(avg_confidence * 1.1, 0.95),  # Boost for agreement
                    reason="+".join(reasons),
                    qty=0.0
                )
                combined.append(combined_signal)
        
        # Process sell signals
        if sell_signals:
            avg_confidence = np.mean([s.confidence for s in sell_signals])
            
            if len(sell_signals) >= 2 or avg_confidence >= 0.8:
                reasons = [s.reason for s in sell_signals]
                combined_signal = Signal(
                    symbol=symbol,
                    side=Side.SELL,
                    confidence=min(avg_confidence * 1.1, 0.95),
                    reason="+".join(reasons),
                    qty=0.0
                )
                combined.append(combined_signal)
        
        return combined
