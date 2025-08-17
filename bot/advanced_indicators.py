"""Advanced Technical Indicators for HFT Trading"""

from typing import List, Optional, Tuple
import numpy as np
from collections import deque
import math


def vwap(prices: List[float], volumes: List[float], period: int = 20) -> Optional[float]:
    """Volume Weighted Average Price"""
    if len(prices) < period or len(volumes) < period:
        return None
    
    recent_prices = prices[-period:]
    recent_volumes = volumes[-period:]
    
    if sum(recent_volumes) == 0:
        return None
        
    return sum(p * v for p, v in zip(recent_prices, recent_volumes)) / sum(recent_volumes)


def keltner_channels(highs: List[float], lows: List[float], closes: List[float], 
                    period: int = 20, mult: float = 2.0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Keltner Channels (upper, middle, lower)"""
    if len(highs) < period or len(lows) < period or len(closes) < period:
        return None, None, None
    
    # Calculate EMA of close
    ema = exponential_moving_average(closes, period)
    if ema is None:
        return None, None, None
    
    # Calculate Average True Range
    atr_values = []
    for i in range(-period, 0):
        if i == -period:
            tr = highs[i] - lows[i]
        else:
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            tr = max(high_low, high_close, low_close)
        atr_values.append(tr)
    
    atr = sum(atr_values) / len(atr_values)
    
    upper = ema + (mult * atr)
    lower = ema - (mult * atr)
    
    return upper, ema, lower


def exponential_moving_average(values: List[float], period: int) -> Optional[float]:
    """Calculate EMA"""
    if len(values) < period:
        return None
    
    multiplier = 2 / (period + 1)
    ema = sum(values[-period:]) / period  # Start with SMA
    
    # Calculate EMA for the last value
    for i in range(-period + 1, 0):
        ema = (values[i] - ema) * multiplier + ema
    
    return ema


def stochastic_oscillator(highs: List[float], lows: List[float], closes: List[float],
                         k_period: int = 14, d_period: int = 3) -> Tuple[Optional[float], Optional[float]]:
    """Stochastic Oscillator (%K, %D)"""
    if len(highs) < k_period or len(lows) < k_period or len(closes) < k_period:
        return None, None
    
    # Calculate %K
    recent_highs = highs[-k_period:]
    recent_lows = lows[-k_period:]
    current_close = closes[-1]
    
    highest_high = max(recent_highs)
    lowest_low = min(recent_lows)
    
    if highest_high == lowest_low:
        k_percent = 50.0
    else:
        k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
    
    # For %D, we need historical %K values - simplified version
    # In production, you'd maintain a rolling buffer of %K values
    d_percent = k_percent  # Simplified - normally this would be SMA of %K
    
    return k_percent, d_percent


def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Bollinger Bands (upper, middle, lower)"""
    if len(prices) < period:
        return None, None, None
    
    recent = prices[-period:]
    middle = sum(recent) / period
    
    # Calculate standard deviation
    variance = sum((x - middle) ** 2 for x in recent) / period
    std = math.sqrt(variance)
    
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    
    return upper, middle, lower


def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """MACD (macd_line, signal_line, histogram)"""
    if len(prices) < slow + signal:
        return None, None, None
    
    ema_fast = exponential_moving_average(prices, fast)
    ema_slow = exponential_moving_average(prices, slow)
    
    if ema_fast is None or ema_slow is None:
        return None, None, None
    
    macd_line = ema_fast - ema_slow
    
    # Simplified signal line (should be EMA of MACD line)
    signal_line = macd_line * 0.9  # Simplified
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def money_flow_index(highs: List[float], lows: List[float], closes: List[float], 
                    volumes: List[float], period: int = 14) -> Optional[float]:
    """Money Flow Index - volume-weighted RSI"""
    if len(highs) < period + 1 or len(volumes) < period + 1:
        return None
    
    positive_flow = 0.0
    negative_flow = 0.0
    
    for i in range(-period, 0):
        typical_price = (highs[i] + lows[i] + closes[i]) / 3
        prev_typical = (highs[i-1] + lows[i-1] + closes[i-1]) / 3
        
        money_flow = typical_price * volumes[i]
        
        if typical_price > prev_typical:
            positive_flow += money_flow
        else:
            negative_flow += money_flow
    
    if negative_flow == 0:
        return 100.0
    
    money_ratio = positive_flow / negative_flow
    mfi = 100 - (100 / (1 + money_ratio))
    
    return mfi


def average_true_range(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    """Average True Range for volatility measurement"""
    if len(highs) < period + 1:
        return None
    
    true_ranges = []
    for i in range(-period, 0):
        if i == -period:
            tr = highs[i] - lows[i]
        else:
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            tr = max(high_low, high_close, low_close)
        true_ranges.append(tr)
    
    return sum(true_ranges) / len(true_ranges)


def williams_r(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    """Williams %R - momentum indicator"""
    if len(highs) < period:
        return None
    
    highest_high = max(highs[-period:])
    lowest_low = min(lows[-period:])
    current_close = closes[-1]
    
    if highest_high == lowest_low:
        return -50.0
    
    williams = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
    
    return williams


def commodity_channel_index(highs: List[float], lows: List[float], closes: List[float], period: int = 20) -> Optional[float]:
    """CCI - identifies cyclical trends"""
    if len(highs) < period:
        return None
    
    typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs[-period:], lows[-period:], closes[-period:])]
    sma = sum(typical_prices) / period
    
    mean_deviation = sum(abs(tp - sma) for tp in typical_prices) / period
    
    if mean_deviation == 0:
        return 0.0
    
    cci = (typical_prices[-1] - sma) / (0.015 * mean_deviation)
    
    return cci


def supertrend(highs: List[float], lows: List[float], closes: List[float], 
               period: int = 7, multiplier: float = 3.0) -> Tuple[Optional[float], Optional[str]]:
    """Supertrend indicator for trend following"""
    if len(highs) < period:
        return None, None
    
    atr = average_true_range(highs, lows, closes, period)
    if atr is None:
        return None, None
    
    hl_avg = (highs[-1] + lows[-1]) / 2
    
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    # Simplified trend determination
    if closes[-1] <= upper_band:
        return upper_band, "sell"
    else:
        return lower_band, "buy"
