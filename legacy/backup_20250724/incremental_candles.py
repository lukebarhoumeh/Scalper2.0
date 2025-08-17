"""
incremental_candles.py - HFT-grade incremental candle management

Instead of fetching 300 candles every tick, we maintain a rolling window
and only fetch new candles. This reduces latency from ~100ms to <10ms.
"""

from __future__ import annotations
import time
import threading
import logging
from collections import deque, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, Deque
import numpy as np
import pandas as pd

from market_data import get_historical_candles

_LOG = logging.getLogger(__name__)


class IncrementalCandleManager:
    """
    HFT-optimized candle management with incremental updates.
    Maintains rolling windows in memory and only fetches new data.
    """
    
    def __init__(self, window_size: int = 200, granularity_sec: int = 60):
        self._window_size = window_size
        self._granularity_sec = granularity_sec
        
        # Product -> deque of candles (newest first)
        self._candle_buffers: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=window_size))
        
        # Product -> last candle timestamp
        self._last_timestamps: Dict[str, datetime] = {}
        
        # Thread safety
        self._locks: Dict[str, threading.RLock] = defaultdict(threading.RLock)
        
        # Performance metrics
        self._fetch_times: deque = deque(maxlen=100)
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Stagger control - prevent all products from updating at once
        self._stagger_counter = 0
        self._products_per_tick = 1  # Only update 1 product per tick
        self._product_update_order = []
        
        # Cooldown control - prevent too frequent API calls
        self._last_fetch_times: Dict[str, float] = {}
        self._fetch_cooldown_sec = 5  # Minimum 5 seconds between fetches per product
        
    def get_candles(self, product_id: str, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get candles with incremental updates. 
        Returns DataFrame sorted by time (oldest first).
        """
        with self._locks[product_id]:
            start_time = time.perf_counter()
            
            # Check cooldown - prevent too frequent API calls
            current_time = time.time()
            if (product_id in self._last_fetch_times and 
                current_time - self._last_fetch_times[product_id] < self._fetch_cooldown_sec and
                not force_refresh):
                # Return cached data if in cooldown
                self._cache_hits += 1
                return self._buffer_to_dataframe(product_id)
            
            # Check if we need initial load
            if product_id not in self._last_timestamps or force_refresh:
                df = self._initial_load(product_id)
            else:
                df = self._incremental_update(product_id)
            
            # Update last fetch time
            self._last_fetch_times[product_id] = current_time
            
            # Track performance
            fetch_time_ms = (time.perf_counter() - start_time) * 1000
            self._fetch_times.append(fetch_time_ms)
            
            if fetch_time_ms > 20:  # Warn if slow
                _LOG.warning(f"Slow candle fetch for {product_id}: {fetch_time_ms:.1f}ms")
            
            return df
    
    def _initial_load(self, product_id: str) -> pd.DataFrame:
        """Initial full candle load."""
        _LOG.info(f"Initial candle load for {product_id}")
        self._cache_misses += 1
        
        # Fetch full window - use a safe number of candles (max 100 to avoid API limit)
        safe_window = min(self._window_size, 100)
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(seconds=self._granularity_sec * safe_window)
        
        df = get_historical_candles(
            product_id, 
            granularity_sec=self._granularity_sec,
            start=start_time,
            end=end_time
        )
        
        if df.empty:
            return df
        
        # Store in buffer (newest first for efficient updates)
        self._candle_buffers[product_id].clear()
        for _, row in df.iloc[::-1].iterrows():  # Reverse to get newest first
            self._candle_buffers[product_id].append(row.to_dict())
        
        # Extract the max timestamp value - ensure it's a datetime
        if not df.empty:
            max_time = df['time'].max()
            # Convert to timestamp if needed
            if hasattr(max_time, 'to_pydatetime'):
                max_time = max_time.to_pydatetime()
            elif not isinstance(max_time, datetime):
                max_time = pd.to_datetime(max_time).to_pydatetime()
            self._last_timestamps[product_id] = max_time
        
        return df
    
    def _incremental_update(self, product_id: str) -> pd.DataFrame:
        """Fetch only new candles since last update."""
        last_ts = self._last_timestamps[product_id]
        now = datetime.now(timezone.utc)
        
        # How many candles are we missing?
        time_diff = now - last_ts
        expected_new_candles = int(time_diff.total_seconds() / self._granularity_sec)
        
        if expected_new_candles <= 0:
            # No new candles expected, return cached data
            self._cache_hits += 1
            return self._buffer_to_dataframe(product_id)
        
        if expected_new_candles > 20:  # Reduced threshold for faster recovery
            _LOG.warning(f"Too many candles missing ({expected_new_candles}), doing full refresh")
            return self._initial_load(product_id)
        
        # Fetch only recent candles
        _LOG.debug(f"Fetching {expected_new_candles} new candles for {product_id}")
        
        # Fetch with some overlap to ensure we don't miss any - limit to safe amount
        lookback = min(expected_new_candles + 3, 30)  # Reduced max candles for faster updates
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(seconds=self._granularity_sec * lookback)
        
        recent_df = get_historical_candles(
            product_id,
            granularity_sec=self._granularity_sec,
            start=start_time,
            end=end_time
        )
        
        if recent_df.empty:
            return self._buffer_to_dataframe(product_id)
        
        # Add only truly new candles to buffer
        buffer = self._candle_buffers[product_id]
        new_count = 0
        
        for _, row in recent_df.iloc[::-1].iterrows():  # Process newest first
            candle_time = row['time']
            if candle_time > last_ts:
                buffer.appendleft(row.to_dict())  # Add to front
                new_count += 1
                self._last_timestamps[product_id] = max(
                    self._last_timestamps[product_id], 
                    candle_time
                )
        
        if new_count > 0:
            _LOG.debug(f"Added {new_count} new candles to {product_id}")
            self._cache_misses += 1
        else:
            self._cache_hits += 1
        
        return self._buffer_to_dataframe(product_id)
    
    def _buffer_to_dataframe(self, product_id: str) -> pd.DataFrame:
        """Convert buffer to DataFrame, oldest first."""
        buffer = self._candle_buffers[product_id]
        if not buffer:
            return pd.DataFrame()
        
        # Convert deque to list in reverse order (oldest first)
        candles_list = list(buffer)[::-1]
        return pd.DataFrame(candles_list)
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        if not self._fetch_times:
            avg_fetch_time = 0
            p95_fetch_time = 0
        else:
            fetch_times = list(self._fetch_times)
            avg_fetch_time = sum(fetch_times) / len(fetch_times)
            p95_fetch_time = np.percentile(fetch_times, 95)
        
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'avg_fetch_time_ms': avg_fetch_time,
            'p95_fetch_time_ms': p95_fetch_time,
            'cache_hit_rate': hit_rate * 100,
            'total_requests': total_requests,
            'products_tracked': len(self._candle_buffers)
        }
    
    def clear_product(self, product_id: str) -> None:
        """Clear cache for a specific product."""
        with self._locks[product_id]:
            self._candle_buffers.pop(product_id, None)
            self._last_timestamps.pop(product_id, None)
    
    def clear_all(self) -> None:
        """Clear all cached data."""
        for product_id in list(self._candle_buffers.keys()):
            self.clear_product(product_id)


# Global instance for sharing across modules
_CANDLE_MANAGER = IncrementalCandleManager()


def get_incremental_candles(product_id: str, window_size: int = 200) -> pd.DataFrame:
    """
    Get candles using incremental updates.
    This is the main interface for other modules.
    """
    return _CANDLE_MANAGER.get_candles(product_id)


def get_candle_stats() -> dict:
    """Get performance statistics for candle fetching."""
    return _CANDLE_MANAGER.get_stats()


# Background stats logger
def _log_stats_periodically():
    """Log performance stats every minute."""
    while True:
        time.sleep(60)
        try:
            stats = get_candle_stats()
            _LOG.info(
                f"Candle Manager Stats: "
                f"avg_fetch={stats['avg_fetch_time_ms']:.1f}ms, "
                f"p95={stats['p95_fetch_time_ms']:.1f}ms, "
                f"hit_rate={stats['cache_hit_rate']:.1f}%, "
                f"requests={stats['total_requests']}"
            )
        except Exception as e:
            _LOG.debug(f"Stats logging error: {e}")


# Start background stats thread
threading.Thread(target=_log_stats_periodically, daemon=True).start() 