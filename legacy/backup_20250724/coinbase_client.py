#!/usr/bin/env python3
"""
Production Coinbase Client with Rate Limiting
============================================
Advanced client for both REST and WebSocket connections
"""

import asyncio
import time
import threading
import logging
import json
from typing import Dict, List, Optional, Callable, Any, Tuple
from collections import deque
from datetime import datetime, timezone
import websocket
from coinbase.rest import RESTClient
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from config_unified import (
    COINBASE_API_KEY, COINBASE_API_SECRET,
    MAX_REQUESTS_PER_SECOND, WS_RECONNECT_TIMEOUT
)

logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe rate limiter for API calls"""
    
    def __init__(self, max_per_second: int):
        self.max_per_second = max_per_second
        self.requests = deque()
        self.lock = threading.Lock()
        
    def acquire(self):
        """Wait if necessary to respect rate limit"""
        with self.lock:
            now = time.time()
            
            # Remove old requests
            while self.requests and self.requests[0] < now - 1.0:
                self.requests.popleft()
                
            # Check if we need to wait
            if len(self.requests) >= self.max_per_second:
                sleep_time = 1.0 - (now - self.requests[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            # Record this request
            self.requests.append(time.time())


class CoinbaseRESTPool:
    """Connection pool for REST clients"""
    
    def __init__(self, pool_size: int = 5):
        self.pool_size = pool_size
        self.clients = []
        self.available = []
        self.lock = threading.Lock()
        
        # Initialize pool
        for _ in range(pool_size):
            client = RESTClient(
                api_key=COINBASE_API_KEY,
                api_secret=COINBASE_API_SECRET
            )
            self.clients.append(client)
            self.available.append(client)
            
    def acquire(self) -> RESTClient:
        """Get a client from the pool"""
        with self.lock:
            while not self.available:
                time.sleep(0.01)
            return self.available.pop()
            
    def release(self, client: RESTClient):
        """Return a client to the pool"""
        with self.lock:
            if client not in self.available:
                self.available.append(client)


class CoinbaseWebSocketClient:
    """WebSocket client with auto-reconnect and error handling"""
    
    def __init__(self, product_ids: List[str], callbacks: Dict[str, Callable]):
        self.product_ids = product_ids
        self.callbacks = callbacks
        self.ws = None
        self.running = False
        self.reconnect_delay = 1.0
        self.max_reconnect_delay = 60.0
        self.thread = None
        
    def start(self):
        """Start WebSocket connection"""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop WebSocket connection"""
        self.running = False
        if self.ws:
            self.ws.close()
            
    def _run(self):
        """Main WebSocket loop with reconnection"""
        while self.running:
            try:
                self._connect()
                self.reconnect_delay = 1.0  # Reset on successful connection
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                time.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
                
    def _connect(self):
        """Establish WebSocket connection"""
        self.ws = websocket.WebSocketApp(
            "wss://ws-feed.exchange.coinbase.com",
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        self.ws.run_forever()
        
    def _on_open(self, ws):
        """Subscribe to channels on connection"""
        logger.info("WebSocket connected")
        
        # Subscribe to level2 and ticker channels
        subscribe_message = {
            "type": "subscribe",
            "product_ids": self.product_ids,
            "channels": ["level2", "ticker", "matches"]
        }
        ws.send(json.dumps(subscribe_message))
        
    def _on_message(self, ws, message):
        """Handle incoming messages"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type in self.callbacks:
                self.callbacks[msg_type](data)
            elif msg_type == 'subscriptions':
                logger.info(f"Subscribed to channels: {data.get('channels', [])}")
            elif msg_type == 'error':
                logger.error(f"WebSocket error message: {data}")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            
    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
        
    def _on_close(self, ws):
        """Handle connection close"""
        logger.info("WebSocket disconnected")


class CoinbaseClient:
    """Unified client for REST and WebSocket operations"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter(MAX_REQUESTS_PER_SECOND)
        self.rest_pool = CoinbaseRESTPool()
        self.ws_client = None
        
        # Market data cache
        self.order_books = {}
        self.last_trades = {}
        self.lock = threading.Lock()
        
    def start_websocket(self, product_ids: List[str]):
        """Start WebSocket feed for products"""
        callbacks = {
            'l2update': self._handle_l2_update,
            'snapshot': self._handle_snapshot,
            'ticker': self._handle_ticker,
            'match': self._handle_match
        }
        
        self.ws_client = CoinbaseWebSocketClient(product_ids, callbacks)
        self.ws_client.start()
        
    def stop_websocket(self):
        """Stop WebSocket feed"""
        if self.ws_client:
            self.ws_client.stop()
            
    def _handle_l2_update(self, data):
        """Handle level 2 order book updates"""
        product_id = data.get('product_id')
        if not product_id:
            return
            
        with self.lock:
            if product_id not in self.order_books:
                self.order_books[product_id] = {'bids': {}, 'asks': {}}
                
            book = self.order_books[product_id]
            
            for side, price, size in data.get('changes', []):
                price = float(price)
                size = float(size)
                
                if side == 'buy':
                    if size == 0:
                        book['bids'].pop(price, None)
                    else:
                        book['bids'][price] = size
                elif side == 'sell':
                    if size == 0:
                        book['asks'].pop(price, None)
                    else:
                        book['asks'][price] = size
                        
    def _handle_snapshot(self, data):
        """Handle order book snapshot"""
        product_id = data.get('product_id')
        if not product_id:
            return
            
        with self.lock:
            book = {'bids': {}, 'asks': {}}
            
            for bid in data.get('bids', []):
                price = float(bid[0])
                size = float(bid[1])
                book['bids'][price] = size
                
            for ask in data.get('asks', []):
                price = float(ask[0])
                size = float(ask[1])
                book['asks'][price] = size
                
            self.order_books[product_id] = book
            
    def _handle_ticker(self, data):
        """Handle ticker updates"""
        product_id = data.get('product_id')
        if product_id:
            with self.lock:
                self.last_trades[product_id] = {
                    'price': float(data.get('price', 0)),
                    'time': data.get('time'),
                    'best_bid': float(data.get('best_bid', 0)),
                    'best_ask': float(data.get('best_ask', 0))
                }
                
    def _handle_match(self, data):
        """Handle trade matches"""
        # Could store recent trades for analysis
        pass
        
    def get_best_bid_ask(self, product_id: str) -> Tuple[float, float]:
        """Get best bid/ask from WebSocket feed or REST"""
        # Try WebSocket data first
        with self.lock:
            if product_id in self.order_books:
                book = self.order_books[product_id]
                best_bid = max(book['bids'].keys()) if book['bids'] else 0
                best_ask = min(book['asks'].keys()) if book['asks'] else 0
                if best_bid > 0 and best_ask > 0:
                    return best_bid, best_ask
                    
        # Fallback to REST
        return self._get_best_bid_ask_rest(product_id)
        
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception)
    )
    def _get_best_bid_ask_rest(self, product_id: str) -> Tuple[float, float]:
        """Get best bid/ask via REST API"""
        self.rate_limiter.acquire()
        
        client = self.rest_pool.acquire()
        try:
            response = client.get_product_book(product_id=product_id, level=1)
            
            if response and hasattr(response, 'bids') and hasattr(response, 'asks'):
                bids = response.bids
                asks = response.asks
                
                best_bid = float(bids[0][0]) if bids else 0
                best_ask = float(asks[0][0]) if asks else 0
                
                return best_bid, best_ask
            else:
                return 0, 0
                
        finally:
            self.rest_pool.release(client)
            
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception)
    )
    def place_order(self, **kwargs) -> Optional[Dict]:
        """Place an order with retry logic"""
        self.rate_limiter.acquire()
        
        client = self.rest_pool.acquire()
        try:
            response = client.market_order_buy(**kwargs) if kwargs.get('side') == 'buy' else client.market_order_sell(**kwargs)
            
            if response and hasattr(response, 'order_id'):
                return {
                    'order_id': response.order_id,
                    'product_id': response.product_id,
                    'side': response.side,
                    'status': response.status
                }
            return None
            
        except Exception as e:
            if "429" in str(e):
                logger.warning("Rate limit hit, backing off")
                time.sleep(5)
            raise
        finally:
            self.rest_pool.release(client)
            
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception)
    )
    def get_accounts(self) -> List[Dict]:
        """Get account information"""
        self.rate_limiter.acquire()
        
        client = self.rest_pool.acquire()
        try:
            response = client.get_accounts()
            accounts = []
            
            if response and hasattr(response, 'accounts'):
                for account in response.accounts:
                    accounts.append({
                        'currency': account.currency,
                        'available': float(account.available_balance.value),
                        'hold': float(account.hold.value) if account.hold else 0
                    })
                    
            return accounts
            
        finally:
            self.rest_pool.release(client)
            
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception)
    )
    def get_candles(self, product_id: str, granularity: int, start: str, end: str) -> List[Dict]:
        """Get historical candles"""
        self.rate_limiter.acquire()
        
        client = self.rest_pool.acquire()
        try:
            # Map granularity to Coinbase format
            gran_map = {
                60: "ONE_MINUTE",
                300: "FIVE_MINUTE",
                900: "FIFTEEN_MINUTE",
                3600: "ONE_HOUR",
                21600: "SIX_HOUR",
                86400: "ONE_DAY"
            }
            
            response = client.get_candles(
                product_id=product_id,
                start=start,
                end=end,
                granularity=gran_map.get(granularity, "ONE_HOUR")
            )
            
            candles = []
            if response and hasattr(response, 'candles'):
                for candle in response.candles:
                    candles.append({
                        'time': datetime.fromtimestamp(int(candle.start), tz=timezone.utc),
                        'open': float(candle.open),
                        'high': float(candle.high),
                        'low': float(candle.low),
                        'close': float(candle.close),
                        'volume': float(candle.volume)
                    })
                    
            return candles
            
        finally:
            self.rest_pool.release(client)


# Global client instance
_coinbase_client = None


def get_coinbase_client() -> CoinbaseClient:
    """Get or create global Coinbase client"""
    global _coinbase_client
    if _coinbase_client is None:
        _coinbase_client = CoinbaseClient()
    return _coinbase_client 