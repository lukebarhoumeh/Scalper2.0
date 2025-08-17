from __future__ import annotations

from typing import List, Optional


def simple_moving_average(values: List[float], window: int) -> Optional[float]:
    if len(values) < window or window <= 0:
        return None
    return sum(values[-window:]) / float(window)


def rsi(values: List[float], period: int = 14) -> Optional[float]:
    if period <= 0 or len(values) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        change = values[i] - values[i - 1]
        if change >= 0:
            gains += change
        else:
            losses -= change
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100 - (100 / (1 + rs))


def breakout(values: List[float], window: int = 20) -> Optional[str]:
    if len(values) < window:
        return None
    recent = values[-1]
    highest = max(values[-window:])
    lowest = min(values[-window:])
    if recent >= highest:
        return "breakout_up"
    if recent <= lowest:
        return "breakout_down"
    return None


def realized_volatility_pct(values: List[float], window: int = 20) -> float:
    """Std of simple returns over window in percent."""
    if len(values) < window + 1:
        return 0.0
    rets: List[float] = []
    start = len(values) - window
    for i in range(start + 1, len(values)):
        p0 = values[i - 1] if values[i - 1] != 0 else 1e-12
        p1 = values[i] if values[i] != 0 else 1e-12
        rets.append((p1 / p0) - 1.0)
    if len(rets) <= 1:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return float((var ** 0.5) * 100.0)


