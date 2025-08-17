# ─────────────────── market_data.py ──────────────────────────────────────────
from __future__ import annotations

import json, logging, threading, time, os
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple, Union, Optional, List
import queue

import numpy as np
import pandas as pd
import requests, websocket
from coinbase.rest import RESTClient
from dateutil import parser
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import signal

from config import (
    COINBASE_API_KEY,
    COINBASE_API_SECRET,
    REST_RATE_LIMIT_PER_S,
    TRADE_COINS,
    USE_WS_FEED,
)

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
_LOG = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Timeout wrapper for REST API calls
# ──────────────────────────────────────────────────────────────────────────────
class TimeoutException(Exception):
    pass

def with_timeout(timeout_seconds):
    """Decorator to add timeout to functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import functools
            import threading
            
            result = [TimeoutException(f"{func.__name__} timed out after {timeout_seconds}s")]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)
            
            if thread.is_alive():
                # Thread still running, timeout occurred
                raise TimeoutException(f"{func.__name__} timed out after {timeout_seconds}s")
            
            if isinstance(result[0], Exception):
                raise result[0]
            
            return result[0]
        
        return wrapper
    return decorator

# ────────────────────────── REST Client Pool for HFT ──────────────────────
class RESTClientPool:
    """Thread-safe connection pool with health monitoring and auto-recovery."""
    
    def __init__(self, pool_size: int = 10):  # Increased pool size
        self._pool = queue.Queue(maxsize=pool_size)
        self._pool_size = pool_size
        self._lock = threading.RLock()
        self._created_clients = []
        self._last_health_check = time.time()
        self._health_check_interval = 60  # 1 minute - more frequent checks
        self._client_ages = {}  # Track client creation times
        self._max_client_age = 300  # 5 minutes max lifetime - prevent stale connections
        self._emergency_clients = set()  # Track emergency clients
        self._max_emergency_clients = 5  # Limit emergency clients
        
        # Initialize pool
        for _ in range(pool_size):
            client = self._create_client()
            self._pool.put(client)
    
    def _create_client(self) -> RESTClient:
        """Create a new REST client with proper configuration."""
        from config import COINBASE_API_KEY, COINBASE_API_SECRET
        rest = RESTClient(api_key=COINBASE_API_KEY, api_secret=COINBASE_API_SECRET, verbose=False)
        
        # Track creation time
        client_id = id(rest)
        self._client_ages[client_id] = time.time()
        self._created_clients.append(rest)
        
        return rest
    
    def _is_client_stale(self, client: RESTClient) -> bool:
        """Check if client is too old and should be recycled."""
        client_id = id(client)
        age = time.time() - self._client_ages.get(client_id, 0)
        return age > self._max_client_age
    
    def acquire(self, timeout: float = 2.0) -> RESTClient:
        """Get a client from the pool with timeout."""
        try:
            # Periodic health check
            if time.time() - self._last_health_check > self._health_check_interval:
                self._perform_health_check()
            
            client = self._pool.get(timeout=timeout)
            
            # Check if client needs recycling
            if self._is_client_stale(client):
                _LOG.debug("Recycling stale REST client")
                client = self._recycle_client(client)
            
            return client
            
        except queue.Empty:
            # Pool exhausted - create emergency client with limits
            with self._lock:
                if len(self._emergency_clients) >= self._max_emergency_clients:
                    # Try to recycle an old emergency client
                    _LOG.warning("Emergency client limit reached, waiting for available client")
                    try:
                        # Wait a bit longer for a regular client
                        client = self._pool.get(timeout=5.0)
                        return client
                    except queue.Empty:
                        # Still no client, force create one
                        _LOG.error("Force creating emergency client despite limit")
                
                _LOG.warning("REST client pool exhausted, creating emergency client")
                emergency_client = self._create_client()
                self._emergency_clients.add(id(emergency_client))
                return emergency_client
    
    def release(self, client: RESTClient) -> None:
        """Return a client to the pool with safety checks."""
        try:
            # Check if client is healthy before returning to pool
            client_id = id(client)
            
            # Handle emergency clients
            if client_id in self._emergency_clients:
                _LOG.debug("Discarding emergency REST client")
                # Clean up tracking
                self._emergency_clients.discard(client_id)
                if client_id in self._client_ages:
                    del self._client_ages[client_id]
                return
                
            # Don't return stale clients
            if self._is_client_stale(client):
                _LOG.debug("Discarding stale REST client")
                if client_id in self._client_ages:
                    del self._client_ages[client_id]
                return
                
            # Return healthy client to pool
            self._pool.put_nowait(client)
        except queue.Full:
            _LOG.warning("REST client pool full, discarding client")
            # Clean up tracking
            client_id = id(client)
            if client_id in self._client_ages:
                del self._client_ages[client_id]
    
    def _recycle_client(self, old_client: RESTClient) -> RESTClient:
        """Replace an old client with a new one."""
        try:
            # Remove old client tracking
            client_id = id(old_client)
            if client_id in self._client_ages:
                del self._client_ages[client_id]
            
            # Create new client
            return self._create_client()
        except Exception as e:
            _LOG.error(f"Failed to recycle client: {e}")
            return old_client  # Keep using old one if recycling fails
    
    def _perform_health_check(self) -> None:
        """Check health of all clients in pool."""
        with self._lock:
            self._last_health_check = time.time()
            _LOG.debug(f"REST client pool health check: {self._pool.qsize()}/{self._pool_size} available")

# Global client pool - increased size to prevent exhaustion
_CLIENT_POOL = RESTClientPool(pool_size=30)

# ───────────────────────── Rate limiter with token bucket ────────────────────
class RateLimiter:
    """Thread-safe token bucket rate limiter for HFT."""
    
    def __init__(self, rate_per_second: int):
        self._rate = rate_per_second
        self._tokens = rate_per_second
        self._last_update = time.time()
        self._lock = threading.Lock()
        
    def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens, returns wait time if rate limited."""
        with self._lock:
            now = time.time()
            # Refill tokens
            elapsed = now - self._last_update
            self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
            self._last_update = now
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0  # No wait needed
            else:
                # Calculate wait time
                wait_time = (tokens - self._tokens) / self._rate
                return wait_time

_RATE_LIMITER = RateLimiter(REST_RATE_LIMIT_PER_S)

def _rate_limited() -> None:
    """HFT-optimized rate limiting."""
    wait_time = _RATE_LIMITER.acquire()
    if wait_time > 0:
        time.sleep(wait_time)

# ───────────────────────── WebSocket best‑bid/ask cache ──────────────────────
_BBO: Dict[str, Tuple[float, float, float]] = defaultdict(lambda: (np.nan, np.nan, 0.0))  # {product_id: (bid, ask, timestamp)}
_WS_CLIENT = None
_WS_THREAD = None
_WS_RECONNECT_COUNT = 0
_WS_MAX_RECONNECT_DELAY = 60  # Max 60 seconds between reconnect attempts
_WS_LAST_MESSAGE_TIME = time.time()

def _ws_loop(products: list[str]) -> None:
    """WebSocket feed with automatic reconnection and health monitoring."""
    global _WS_CLIENT, _WS_RECONNECT_COUNT, _WS_LAST_MESSAGE_TIME
    
    reconnect_delay = 1  # Start with 1 second delay
    
    while True:
        try:
            _LOG.info(f"Starting WebSocket connection (attempt {_WS_RECONNECT_COUNT + 1})")
            _WS_LAST_MESSAGE_TIME = time.time()
            
            def on_msg(_, msg: str):
                global _WS_LAST_MESSAGE_TIME
                _WS_LAST_MESSAGE_TIME = time.time()
                
                try:
                    data = json.loads(msg)
                    if data.get("channel") == "ticker" and "product_id" in data:
                        bid = float(data.get("best_bid", 0))
                        ask = float(data.get("best_ask", 0))
                        if bid > 0 and ask > 0:
                            _BBO[data["product_id"]] = (bid, ask, time.time())
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    _LOG.debug(f"WS message parse error: {e}")
    
            def on_open(ws):
                # Subscribe to ticker channel for all products
                sub_msg = {
                    "type": "subscribe",
                    "product_ids": [f"{p}-USD" for p in products],
                    "channel": "ticker",
                }
                ws.send(json.dumps(sub_msg))
                _LOG.info("WebSocket connected and subscribed")
                global _WS_RECONNECT_COUNT
                _WS_RECONNECT_COUNT = 0  # Reset on successful connection
                reconnect_delay = 1  # Reset delay

            def on_error(ws, error):
                _LOG.warning(f"WebSocket error: {error}")
                
            def on_close(ws, close_status_code, close_msg):
                _LOG.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
            
            # Create WebSocket client
            api_key = COINBASE_API_KEY
            api_secret = COINBASE_API_SECRET
            
            ws_client = websocket.WebSocketApp(
                "wss://advanced-trade-ws.coinbase.com",
                on_open=on_open,
                on_message=on_msg,
                on_error=on_error,
                on_close=on_close
            )
            
            _WS_CLIENT = ws_client
            
            # Run WebSocket (this blocks until connection closes)
            ws_client.run_forever(ping_interval=20, ping_timeout=10)
            
        except Exception as e:
            _LOG.error(f"WebSocket connection failed: {e}")
        
        # Connection lost - implement exponential backoff
        _WS_RECONNECT_COUNT += 1
        _LOG.info(f"Reconnecting in {reconnect_delay}s...")
        time.sleep(reconnect_delay)
        
        # Exponential backoff with max delay
        reconnect_delay = min(reconnect_delay * 2, _WS_MAX_RECONNECT_DELAY)

# Start WebSocket thread with health monitoring
def start_websocket_feed(products: list[str]) -> None:
    """Start WebSocket feed with health monitoring."""
    global _WS_THREAD
    
    if USE_WS_FEED and not _WS_THREAD:
        _WS_THREAD = threading.Thread(target=_ws_loop, args=(products,), daemon=True)
        _WS_THREAD.start()
        
        # Start health monitor
        health_thread = threading.Thread(target=_ws_health_monitor, daemon=True)
        health_thread.start()

def _ws_health_monitor() -> None:
    """Monitor WebSocket health and force reconnect if needed."""
    while True:
        try:
            time.sleep(30)  # Check every 30 seconds
            
            time_since_last_msg = time.time() - _WS_LAST_MESSAGE_TIME
            
            if time_since_last_msg > 60:  # No messages for 1 minute
                _LOG.warning(f"No WebSocket messages for {time_since_last_msg:.0f}s")
                
                # Force reconnect by closing current connection
                if _WS_CLIENT:
                    try:
                        _WS_CLIENT.close()
                    except:
                        pass
                        
        except Exception as e:
            _LOG.error(f"WS health monitor error: {e}")

# Initialize WebSocket feed on module load
if USE_WS_FEED:
    from config import TRADE_COINS
    start_websocket_feed(TRADE_COINS)

# ───────────────────────── helpers & public API ──────────────────────────────
_RETRYABLE = (requests.exceptions.RequestException, ConnectionError)

def _retry():
    return retry(
        wait=wait_exponential(multiplier=0.4, min=0.8, max=8),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )

def mid_price(bid: float, ask: float) -> float:
    return (bid + ask) / 2.0

@_retry()
@with_timeout(5.0)  # 5 second timeout for REST calls
def _rest_best_bid_ask(product_id: str) -> Tuple[float, float]:
    _rate_limited()
    client = _CLIENT_POOL.acquire()
    try:
        response = client.get_best_bid_ask(product_ids=[product_id])
        if not response or not response.pricebooks:
            raise ValueError(f"No price data for {product_id}")
        pb = response.pricebooks[0]
        if not pb.bids or not pb.asks:
            raise ValueError(f"No bid/ask data for {product_id}")
        return float(pb.bids[0].price), float(pb.asks[0].price)
    finally:
        _CLIENT_POOL.release(client)

def get_best_bid_ask(product_id: str, max_staleness_seconds: float = 1.0) -> Tuple[float, float]:
    """Get best bid/ask with staleness check for HFT reliability."""
    bid, ask, timestamp = _BBO.get(product_id, (np.nan, np.nan, 0.0))
    
    # Check staleness
    is_stale = (time.time() - timestamp) > max_staleness_seconds
    
    if np.isnan(bid) or np.isnan(ask) or is_stale:
        if is_stale and USE_WS_FEED:
            _LOG.debug(f"Stale WS data for {product_id}, fetching fresh via REST")
        bid, ask = _rest_best_bid_ask(product_id)  # cold‑start / WS gap‑filler
        _BBO[product_id] = (bid, ask, time.time())
    return bid, ask

def get_last_price(product_id: str) -> float:
    bid, ask = get_best_bid_ask(product_id)
    return mid_price(bid, ask)

# ───────────────────────── Candle cache for efficiency ──────────────────────
class CandleCache:
    """In-memory candle cache to reduce REST calls."""
    
    def __init__(self, max_age_seconds: int = 30):
        self._cache: Dict[Tuple[str, int], Tuple[pd.DataFrame, float]] = {}
        self._lock = threading.Lock()
        self._max_age = max_age_seconds
    
    def get(self, product_id: str, granularity_sec: int) -> Optional[pd.DataFrame]:
        """Get cached candles if fresh enough."""
        with self._lock:
            key = (product_id, granularity_sec)
            if key in self._cache:
                df, timestamp = self._cache[key]
                if time.time() - timestamp < self._max_age:
                    return df.copy()  # Return copy to avoid mutations
            return None
    
    def put(self, product_id: str, granularity_sec: int, df: pd.DataFrame) -> None:
        """Cache candles."""
        with self._lock:
            key = (product_id, granularity_sec)
            self._cache[key] = (df.copy(), time.time())
    
    def clear_stale(self) -> None:
        """Remove stale entries."""
        with self._lock:
            now = time.time()
            self._cache = {
                k: v for k, v in self._cache.items()
                if now - v[1] < self._max_age
            }

_CANDLE_CACHE = CandleCache()

@_retry()
@with_timeout(3.0)  # Reduced timeout for faster response
def get_historical_candles(
    product_id: str, 
    granularity_sec: int, 
    start: datetime, 
    end: datetime
) -> pd.DataFrame:
    """Get candles with caching for HFT efficiency."""
    # Check cache first
    cached = _CANDLE_CACHE.get(product_id, granularity_sec)
    if cached is not None and len(cached) >= (end - start).total_seconds() / granularity_sec:
        return cached
    
    _rate_limited()
    gran_map = {
        60: "ONE_MINUTE", 300: "FIVE_MINUTE", 900: "FIFTEEN_MINUTE",
        1800: "THIRTY_MINUTE", 3600: "ONE_HOUR", 7200: "TWO_HOUR",
        21600: "SIX_HOUR", 86400: "ONE_DAY",
    }
    gran = gran_map[granularity_sec]
    # Use the provided start and end times directly
    
    client = _CLIENT_POOL.acquire()
    try:
        resp = client.get_candles(
            product_id=product_id,
            start=str(int(start.timestamp())),
            end=str(int(end.timestamp())),
            granularity=gran,
        )
        
        if not resp.candles:
            # Create empty DataFrame with proper column specification
            return pd.DataFrame(data=[], columns=["time","open","high","low","close","volume"])
        
        # Parse candle data with proper null handling
        candle_data = []
        for c in resp.candles:
            candle_data.append({
                "time": datetime.fromtimestamp(int(c.start) if c.start else 0, tz=timezone.utc),
                "open": float(c.open) if c.open else 0.0,
                "high": float(c.high) if c.high else 0.0,
                "low": float(c.low) if c.low else 0.0,
                "close": float(c.close) if c.close else 0.0,
                "volume": float(c.volume) if c.volume else 0.0,
            })
        
        df = pd.DataFrame(candle_data).sort_values("time").reset_index(drop=True)
        
        # Cache the result
        _CANDLE_CACHE.put(product_id, granularity_sec, df)
        
        return df
    finally:
        _CLIENT_POOL.release(client)

def realised_volatility(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    log_ret = np.log1p(df["close"].pct_change().dropna())
    ann = 365*24*60*60 / (df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds()
    return float(np.sqrt(log_ret.var() * ann) * 100)

# Periodic cache cleanup
def _cache_cleanup_loop():
    """Background thread to clean stale cache entries."""
    while True:
        time.sleep(60)  # Every minute
        try:
            _CANDLE_CACHE.clear_stale()
        except Exception as e:
            _LOG.debug(f"Cache cleanup error: {e}")

threading.Thread(target=_cache_cleanup_loop, daemon=True).start()

# ──────────────────────────────────────────────────────────────────────────────
# Market Data Manager for Adaptive System
# ──────────────────────────────────────────────────────────────────────────────

class MarketDataManager:
    """
    Unified market data interface for the adaptive trading system.
    Provides access to order book, recent trades, and historical candles.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_order_book(self, product_id: str, level: int = 2) -> Dict:
        """
        Get order book data. For now, returns mock data using best bid/ask.
        In production, this would fetch full order book from WebSocket.
        """
        # Ensure level is an integer
        try:
            level = int(level) if level is not None else 2
        except (ValueError, TypeError):
            level = 2  # Default to level 2 if conversion fails
        
        try:
            bid, ask = get_best_bid_ask(product_id)
            return {
                'bids': [[bid, 1.0]],  # Mock depth
                'asks': [[ask, 1.0]],  # Mock depth
                'product_id': product_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            self.logger.warning(f"Failed to get order book for {product_id}: {e}")
            return {
                'bids': [],
                'asks': [],
                'product_id': product_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def get_recent_trades(self, product_id: str, limit: int = 100) -> List[Dict]:
        """
        Get recent trades. For now, returns mock data using last price.
        In production, this would fetch from WebSocket trade feed.
        """
        try:
            last_price = get_last_price(product_id)
            return [{
                'price': last_price,
                'size': 1.0,  # Mock size
                'side': 'buy',  # Mock side
                'time': datetime.now(timezone.utc).isoformat(),
                'trade_id': f"mock_{int(time.time())}"
            }]
        except Exception as e:
            self.logger.warning(f"Failed to get recent trades for {product_id}: {e}")
            return []
    
    def get_candles(self, product_id: str, granularity_sec: int = 300, 
                   lookback_periods: int = 100) -> Optional[pd.DataFrame]:
        """
        Get historical candles using the existing get_historical_candles function.
        """
        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(seconds=granularity_sec * lookback_periods)
            return get_historical_candles(product_id, granularity_sec, start, end)
        except Exception as e:
            self.logger.warning(f"Failed to get candles for {product_id}: {e}")
            return None
    
    def get_market_snapshot(self, product_id: str) -> Dict:
        """
        Get a complete market snapshot including bid/ask and last price.
        """
        try:
            bid, ask = get_best_bid_ask(product_id)
            last_price = get_last_price(product_id)
            return {
                'product_id': product_id,
                'bid': bid,
                'ask': ask,
                'last_price': last_price,
                'spread': ask - bid,
                'mid_price': (bid + ask) / 2,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            self.logger.warning(f"Failed to get market snapshot for {product_id}: {e}")
            return {
                'product_id': product_id,
                'bid': 0.0,
                'ask': 0.0,
                'last_price': 0.0,
                'spread': 0.0,
                'mid_price': 0.0,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }


# ──────────────────────────────────────────────────────────────────────────────
# Standalone function for external imports
# ──────────────────────────────────────────────────────────────────────────────
def get_candles(product_id: str, granularity: int = 3600, limit: int = 24) -> List[Dict]:
    """
    Get OHLCV candle data for a given product.
    
    Args:
        product_id: The product ID (e.g., 'BTC-USD')
        granularity: Candle granularity in seconds (default: 3600 = 1 hour)
        limit: Number of candles to fetch (default: 24)
        
    Returns:
        List of candle dictionaries with keys: time, open, high, low, close, volume
    """
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(seconds=granularity * limit)
        
        df = get_historical_candles(product_id, granularity, start, end)
        
        if df is None or df.empty:
            return []
            
        # Convert DataFrame to list of dicts
        candles = []
        for _, row in df.iterrows():
            candles.append({
                'time': row['time'].isoformat() if pd.notna(row['time']) else None,
                'open': float(row['open']) if pd.notna(row['open']) else 0.0,
                'high': float(row['high']) if pd.notna(row['high']) else 0.0,
                'low': float(row['low']) if pd.notna(row['low']) else 0.0,
                'close': float(row['close']) if pd.notna(row['close']) else 0.0,
                'volume': float(row['volume']) if pd.notna(row['volume']) else 0.0
            })
            
        return candles
        
    except Exception as e:
        _LOG.error(f"Failed to get candles for {product_id}: {e}")
        return []