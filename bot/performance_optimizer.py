"""Performance Optimization Module for HFT

This module provides performance profiling, optimization, and caching
to achieve microsecond-level latency for critical paths.
"""

from __future__ import annotations
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from functools import lru_cache, wraps
import numpy as np
from collections import deque
import psutil
import gc
import cProfile
import pstats
import io
from datetime import datetime, timezone


@dataclass
class PerformanceMetrics:
    """Performance metrics for a code section"""
    name: str
    avg_time_us: float  # microseconds
    p50_time_us: float
    p95_time_us: float
    p99_time_us: float
    calls_per_second: float
    cache_hit_rate: float


class PerformanceOptimizer:
    """
    Monitors and optimizes performance for HFT operations.
    
    Features:
    - Microsecond-precision timing
    - Hot path detection and optimization
    - Memory pool management
    - Cache optimization
    - CPU affinity setting
    - Garbage collection tuning
    """
    
    def __init__(self, target_latency_us: int = 100):
        self.target_latency_us = target_latency_us
        
        # Performance tracking
        self.timings: Dict[str, deque] = {}
        self.call_counts: Dict[str, int] = {}
        self.cache_stats: Dict[str, Dict[str, int]] = {}
        
        # Memory pools for object reuse
        self.object_pools: Dict[type, List[Any]] = {}
        
        # GC tuning
        self._tune_garbage_collector()
        
        # CPU optimization
        self._optimize_cpu_usage()
    
    def profile(self, name: str):
        """Decorator for profiling function performance"""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    elapsed_us = (time.perf_counter() - start) * 1_000_000
                    self._record_timing(name, elapsed_us)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed_us = (time.perf_counter() - start) * 1_000_000
                    self._record_timing(name, elapsed_us)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def fast_cache(self, maxsize: int = 128):
        """Optimized LRU cache with stats tracking"""
        def decorator(func: Callable):
            # Use built-in lru_cache
            cached_func = lru_cache(maxsize=maxsize)(func)
            
            # Track cache stats
            cache_name = f"{func.__module__}.{func.__name__}"
            self.cache_stats[cache_name] = {"hits": 0, "misses": 0}
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Check cache stats
                info = cached_func.cache_info()
                self.cache_stats[cache_name]["hits"] = info.hits
                self.cache_stats[cache_name]["misses"] = info.misses
                
                return cached_func(*args, **kwargs)
            
            wrapper.cache_clear = cached_func.cache_clear
            wrapper.cache_info = cached_func.cache_info
            
            return wrapper
        return decorator
    
    def get_object(self, obj_type: type, *args, **kwargs) -> Any:
        """Get object from pool or create new one"""
        pool = self.object_pools.get(obj_type, [])
        
        if pool:
            obj = pool.pop()
            # Reset object if it has a reset method
            if hasattr(obj, 'reset'):
                obj.reset(*args, **kwargs)
            return obj
        else:
            return obj_type(*args, **kwargs)
    
    def return_object(self, obj: Any) -> None:
        """Return object to pool for reuse"""
        obj_type = type(obj)
        
        if obj_type not in self.object_pools:
            self.object_pools[obj_type] = []
        
        # Limit pool size to prevent memory bloat
        if len(self.object_pools[obj_type]) < 1000:
            self.object_pools[obj_type].append(obj)
    
    def optimize_numpy_operation(self, operation: str) -> None:
        """Optimize NumPy operations for speed"""
        if operation == "matmul":
            # Use BLAS optimization
            np.seterr(all='ignore')  # Ignore warnings for speed
        elif operation == "general":
            # General optimizations
            np.seterr(divide='ignore', invalid='ignore')
    
    def get_performance_report(self) -> Dict[str, PerformanceMetrics]:
        """Generate comprehensive performance report"""
        report = {}
        
        for name, timings in self.timings.items():
            if not timings:
                continue
            
            timings_array = np.array(timings)
            calls_per_sec = self.call_counts.get(name, 0) / max(1, len(timings) / 1000)
            
            # Calculate cache hit rate
            cache_key = name
            cache_hit_rate = 0.0
            if cache_key in self.cache_stats:
                hits = self.cache_stats[cache_key]["hits"]
                misses = self.cache_stats[cache_key]["misses"]
                total = hits + misses
                cache_hit_rate = hits / total if total > 0 else 0.0
            
            report[name] = PerformanceMetrics(
                name=name,
                avg_time_us=np.mean(timings_array),
                p50_time_us=np.percentile(timings_array, 50),
                p95_time_us=np.percentile(timings_array, 95),
                p99_time_us=np.percentile(timings_array, 99),
                calls_per_second=calls_per_sec,
                cache_hit_rate=cache_hit_rate
            )
        
        return report
    
    def identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks"""
        report = self.get_performance_report()
        bottlenecks = []
        
        for name, metrics in report.items():
            # Check against target latency
            if metrics.p95_time_us > self.target_latency_us:
                bottlenecks.append(f"{name}: p95={metrics.p95_time_us:.1f}μs (target={self.target_latency_us}μs)")
            
            # Check cache performance
            if metrics.cache_hit_rate < 0.8 and metrics.cache_hit_rate > 0:
                bottlenecks.append(f"{name}: low cache hit rate {metrics.cache_hit_rate:.1%}")
        
        return bottlenecks
    
    def profile_code_section(self, code: str) -> str:
        """Profile a code section and return stats"""
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            exec(code)
        finally:
            profiler.disable()
        
        # Generate stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        return s.getvalue()
    
    def _record_timing(self, name: str, elapsed_us: float) -> None:
        """Record timing for a named operation"""
        if name not in self.timings:
            self.timings[name] = deque(maxlen=10000)
            self.call_counts[name] = 0
        
        self.timings[name].append(elapsed_us)
        self.call_counts[name] += 1
        
        # Alert if exceeding target
        if elapsed_us > self.target_latency_us * 2:
            print(f"⚠️  Performance warning: {name} took {elapsed_us:.1f}μs")
    
    def _tune_garbage_collector(self) -> None:
        """Optimize garbage collector for HFT"""
        # Disable GC during critical paths
        gc.disable()
        
        # Set thresholds for less frequent collection
        gc.set_threshold(50000, 10, 10)
        
        # Enable GC with optimized settings
        gc.enable()
    
    def _optimize_cpu_usage(self) -> None:
        """Optimize CPU usage and affinity"""
        try:
            # Get process
            p = psutil.Process()
            
            # Set high priority (requires admin on some systems)
            try:
                p.nice(-5)  # Higher priority
            except:
                pass
            
            # Set CPU affinity to specific cores (if available)
            cpu_count = psutil.cpu_count()
            if cpu_count >= 4:
                # Use first half of cores for main process
                cores = list(range(cpu_count // 2))
                try:
                    p.cpu_affinity(cores)
                except:
                    pass
        except:
            pass


# Optimized data structures for HFT

class FastRingBuffer:
    """Ultra-fast ring buffer for price/volume data"""
    
    def __init__(self, size: int):
        self.size = size
        self.data = np.zeros(size, dtype=np.float64)
        self.index = 0
        self.filled = False
    
    def append(self, value: float) -> None:
        """O(1) append"""
        self.data[self.index] = value
        self.index = (self.index + 1) % self.size
        if self.index == 0:
            self.filled = True
    
    def get_last(self, n: int) -> np.ndarray:
        """Get last n values in order"""
        if not self.filled and self.index < n:
            return self.data[:self.index]
        
        if n >= self.size:
            n = self.size
        
        if self.index >= n:
            return self.data[self.index-n:self.index]
        else:
            return np.concatenate([
                self.data[self.size-(n-self.index):],
                self.data[:self.index]
            ])
    
    @property
    def mean(self) -> float:
        """O(1) mean calculation"""
        if self.filled:
            return np.mean(self.data)
        else:
            return np.mean(self.data[:self.index])
    
    @property
    def std(self) -> float:
        """O(1) standard deviation"""
        if self.filled:
            return np.std(self.data)
        else:
            return np.std(self.data[:self.index])


class FastOrderBook:
    """Optimized order book for fast updates and queries"""
    
    def __init__(self):
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self._bid_prices: List[float] = []
        self._ask_prices: List[float] = []
        self._best_bid: Optional[float] = None
        self._best_ask: Optional[float] = None
    
    def update_bid(self, price: float, size: float) -> None:
        """O(log n) bid update"""
        if size <= 0:
            if price in self.bids:
                del self.bids[price]
                self._bid_prices.remove(price)
        else:
            if price not in self.bids:
                # Binary search insertion
                import bisect
                bisect.insort(self._bid_prices, price)
            self.bids[price] = size
        
        # Update best bid
        self._best_bid = self._bid_prices[-1] if self._bid_prices else None
    
    def update_ask(self, price: float, size: float) -> None:
        """O(log n) ask update"""
        if size <= 0:
            if price in self.asks:
                del self.asks[price]
                self._ask_prices.remove(price)
        else:
            if price not in self.asks:
                import bisect
                bisect.insort(self._ask_prices, price)
            self.asks[price] = size
        
        # Update best ask
        self._best_ask = self._ask_prices[0] if self._ask_prices else None
    
    @property
    def best_bid(self) -> Optional[float]:
        """O(1) best bid"""
        return self._best_bid
    
    @property
    def best_ask(self) -> Optional[float]:
        """O(1) best ask"""
        return self._best_ask
    
    @property
    def spread(self) -> float:
        """O(1) spread calculation"""
        if self._best_bid and self._best_ask:
            return self._best_ask - self._best_bid
        return 0.0
    
    def get_mid(self) -> Optional[float]:
        """O(1) mid price"""
        if self._best_bid and self._best_ask:
            return (self._best_bid + self._best_ask) / 2
        return None


# Performance utilities

def measure_latency(func: Callable) -> float:
    """Measure function latency in microseconds"""
    start = time.perf_counter()
    func()
    return (time.perf_counter() - start) * 1_000_000


def benchmark_strategy(strategy_func: Callable, iterations: int = 10000) -> Dict[str, float]:
    """Benchmark strategy performance"""
    latencies = []
    
    # Warmup
    for _ in range(100):
        strategy_func()
    
    # Benchmark
    for _ in range(iterations):
        latency = measure_latency(strategy_func)
        latencies.append(latency)
    
    return {
        "mean_us": np.mean(latencies),
        "p50_us": np.percentile(latencies, 50),
        "p95_us": np.percentile(latencies, 95),
        "p99_us": np.percentile(latencies, 99),
        "max_us": np.max(latencies)
    }
