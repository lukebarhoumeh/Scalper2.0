"""
realtime_volatility.py - HFT-grade streaming volatility calculation

Replaces slow historical volatility with real-time EWMA updates:
- Updates on every tick (sub-millisecond)
- Adapts quickly to regime changes
- Multiple time horizons for different strategies
- Volatility regime detection
"""

from __future__ import annotations
import time
import threading
import logging
from typing import Dict, Optional, Tuple, List
from collections import defaultdict, deque
import numpy as np
from dataclasses import dataclass

_LOG = logging.getLogger(__name__)


@dataclass
class VolatilityMetrics:
    """Real-time volatility metrics for a product."""
    spot_vol: float  # Instantaneous volatility
    ewma_vol_5m: float  # 5-minute EWMA
    ewma_vol_1h: float  # 1-hour EWMA
    ewma_vol_1d: float  # 1-day EWMA
    vol_of_vol: float  # Volatility of volatility
    regime: str  # "low", "normal", "high", "extreme"
    last_price: float
    last_update: float


class RealtimeVolatilityTracker:
    """
    High-performance streaming volatility calculator.
    Updates volatility estimates on every price tick.
    """
    
    def __init__(self):
        # EWMA parameters (halflife in seconds)
        self._halflife_5m = 300  # 5 minutes
        self._halflife_1h = 3600  # 1 hour
        self._halflife_1d = 86400  # 1 day
        
        # Calculate decay factors
        self._alpha_5m = 1 - np.exp(-np.log(2) / self._halflife_5m)
        self._alpha_1h = 1 - np.exp(-np.log(2) / self._halflife_1h)
        self._alpha_1d = 1 - np.exp(-np.log(2) / self._halflife_1d)
        
        # State tracking per product
        self._last_prices: Dict[str, float] = {}
        self._last_timestamps: Dict[str, float] = {}
        self._ewma_vars: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            '5m': 0.0, '1h': 0.0, '1d': 0.0
        })
        
        # Vol of vol tracking
        self._vol_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Regime thresholds (annualized %)
        self._regime_thresholds = {
            'low': 10,
            'normal': 30,
            'high': 50,
            'extreme': 100
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self._update_times: deque = deque(maxlen=1000)
    
    def update(self, product_id: str, price: float, timestamp: Optional[float] = None) -> VolatilityMetrics:
        """
        Update volatility estimates with new price tick.
        This is the main interface - call on every price update.
        """
        update_start = time.perf_counter()
        
        with self._lock:
            if timestamp is None:
                timestamp = time.time()
            
            # Initialize if first price
            if product_id not in self._last_prices:
                self._last_prices[product_id] = price
                self._last_timestamps[product_id] = timestamp
                return VolatilityMetrics(
                    spot_vol=0.0,
                    ewma_vol_5m=0.0,
                    ewma_vol_1h=0.0,
                    ewma_vol_1d=0.0,
                    vol_of_vol=0.0,
                    regime="normal",
                    last_price=price,
                    last_update=timestamp
                )
            
            # Calculate return
            last_price = self._last_prices[product_id]
            last_time = self._last_timestamps[product_id]
            
            if price <= 0 or last_price <= 0:
                _LOG.warning(f"Invalid price for {product_id}: {price}")
                return self._get_current_metrics(product_id)
            
            # Log return
            log_return = np.log(price / last_price)
            
            # Time adjustment for irregular intervals
            time_diff = timestamp - last_time
            if time_diff <= 0:
                return self._get_current_metrics(product_id)
            
            # Instantaneous variance (squared return)
            instant_var = log_return ** 2
            
            # Update EWMA variances with time adjustment
            vars_dict = self._ewma_vars[product_id]
            
            # Adjust alphas for actual time interval
            alpha_5m_adj = 1 - (1 - self._alpha_5m) ** time_diff
            alpha_1h_adj = 1 - (1 - self._alpha_1h) ** time_diff
            alpha_1d_adj = 1 - (1 - self._alpha_1d) ** time_diff
            
            # Update EWMA variances
            if vars_dict['5m'] == 0:
                # Initialize with first observation
                vars_dict['5m'] = instant_var
                vars_dict['1h'] = instant_var
                vars_dict['1d'] = instant_var
            else:
                vars_dict['5m'] = alpha_5m_adj * instant_var + (1 - alpha_5m_adj) * vars_dict['5m']
                vars_dict['1h'] = alpha_1h_adj * instant_var + (1 - alpha_1h_adj) * vars_dict['1h']
                vars_dict['1d'] = alpha_1d_adj * instant_var + (1 - alpha_1d_adj) * vars_dict['1d']
            
            # Convert to annualized volatility (%)
            annualization_factor = np.sqrt(365 * 24 * 3600)  # Crypto trades 24/7
            
            spot_vol = np.sqrt(instant_var) * annualization_factor * 100
            vol_5m = np.sqrt(vars_dict['5m']) * annualization_factor * 100
            vol_1h = np.sqrt(vars_dict['1h']) * annualization_factor * 100
            vol_1d = np.sqrt(vars_dict['1d']) * annualization_factor * 100
            
            # Track vol history for vol of vol
            self._vol_history[product_id].append(vol_5m)
            
            # Calculate vol of vol
            vol_of_vol = self._calculate_vol_of_vol(product_id)
            
            # Determine regime
            regime = self._determine_regime(vol_5m)
            
            # Update state
            self._last_prices[product_id] = price
            self._last_timestamps[product_id] = timestamp
            
            # Track performance
            update_time_us = (time.perf_counter() - update_start) * 1_000_000
            self._update_times.append(update_time_us)
            
            if update_time_us > 100:  # Warn if >100 microseconds
                _LOG.warning(f"Slow vol update: {update_time_us:.0f}us")
            
            return VolatilityMetrics(
                spot_vol=spot_vol,
                ewma_vol_5m=vol_5m,
                ewma_vol_1h=vol_1h,
                ewma_vol_1d=vol_1d,
                vol_of_vol=vol_of_vol,
                regime=regime,
                last_price=price,
                last_update=timestamp
            )
    
    def _calculate_vol_of_vol(self, product_id: str) -> float:
        """Calculate volatility of volatility."""
        vol_history = list(self._vol_history[product_id])
        
        if len(vol_history) < 10:
            return 0.0
        
        # Standard deviation of recent volatilities
        return float(np.std(vol_history))
    
    def _determine_regime(self, current_vol: float) -> str:
        """Determine volatility regime."""
        if current_vol < self._regime_thresholds['low']:
            return "low"
        elif current_vol < self._regime_thresholds['normal']:
            return "normal"
        elif current_vol < self._regime_thresholds['high']:
            return "high"
        else:
            return "extreme"
    
    def _get_current_metrics(self, product_id: str) -> VolatilityMetrics:
        """Get current metrics without update."""
        vars_dict = self._ewma_vars[product_id]
        annualization_factor = np.sqrt(365 * 24 * 3600)
        
        return VolatilityMetrics(
            spot_vol=0.0,
            ewma_vol_5m=np.sqrt(vars_dict['5m']) * annualization_factor * 100,
            ewma_vol_1h=np.sqrt(vars_dict['1h']) * annualization_factor * 100,
            ewma_vol_1d=np.sqrt(vars_dict['1d']) * annualization_factor * 100,
            vol_of_vol=self._calculate_vol_of_vol(product_id),
            regime=self._determine_regime(np.sqrt(vars_dict['5m']) * annualization_factor * 100),
            last_price=self._last_prices.get(product_id, 0.0),
            last_update=self._last_timestamps.get(product_id, 0.0)
        )
    
    def get_vol_for_sizing(self, product_id: str, horizon: str = "5m") -> float:
        """Get volatility for position sizing (default 5-minute)."""
        with self._lock:
            vars_dict = self._ewma_vars[product_id]
            if horizon in vars_dict and vars_dict[horizon] > 0:
                annualization_factor = np.sqrt(365 * 24 * 3600)
                return np.sqrt(vars_dict[horizon]) * annualization_factor * 100
            return 15.0  # Default fallback
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        if not self._update_times:
            return {
                'avg_update_time_us': 0,
                'p99_update_time_us': 0,
                'products_tracked': 0
            }
        
        times = list(self._update_times)
        return {
            'avg_update_time_us': np.mean(times),
            'p99_update_time_us': np.percentile(times, 99),
            'products_tracked': len(self._last_prices)
        }


# Global volatility tracker
_VOL_TRACKER = RealtimeVolatilityTracker()


def update_volatility(product_id: str, price: float, timestamp: Optional[float] = None) -> VolatilityMetrics:
    """Update volatility with new price tick."""
    return _VOL_TRACKER.update(product_id, price, timestamp)


def get_current_volatility(product_id: str, horizon: str = "5m") -> float:
    """Get current volatility estimate for a given horizon."""
    return _VOL_TRACKER.get_vol_for_sizing(product_id, horizon)


def get_volatility_metrics(product_id: str) -> VolatilityMetrics:
    """Get full volatility metrics for a product."""
    return _VOL_TRACKER._get_current_metrics(product_id)


# Background performance logger
def _log_vol_stats():
    """Log volatility calculation performance."""
    while True:
        time.sleep(60)
        try:
            stats = _VOL_TRACKER.get_performance_stats()
            _LOG.info(
                f"Volatility Tracker: "
                f"avg_update={stats['avg_update_time_us']:.0f}us, "
                f"p99={stats['p99_update_time_us']:.0f}us, "
                f"products={stats['products_tracked']}"
            )
        except Exception as e:
            _LOG.debug(f"Vol stats error: {e}")


threading.Thread(target=_log_vol_stats, daemon=True).start() 