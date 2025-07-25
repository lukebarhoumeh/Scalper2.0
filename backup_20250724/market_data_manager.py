#!/usr/bin/env python3
"""
Market Data Manager - Centralized market data handling
Prevents REST client exhaustion and provides real-time updates
"""

import time
import threading
import logging
from typing import Dict, Tuple, Optional
from collections import defaultdict
from datetime import datetime, timezone
import asyncio

from market_data import get_best_bid_ask, get_candles


class MarketDataManager:
    """Centralized market data management with caching"""
    
    def __init__(self):
        self.logger = logging.getLogger("MarketDataManager")
        
        # Price cache
        self.price_cache = {}
        self.cache_ttl = 2.0  # 2 second cache
        self.cache_lock = threading.Lock()
        
        # Market overview data
        self.market_overview = {
            'BTC-USD': {'price': 0, 'change_24h': 0, 'volume_24h': 0},
            'ETH-USD': {'price': 0, 'change_24h': 0, 'volume_24h': 0},
            'SOL-USD': {'price': 0, 'change_24h': 0, 'volume_24h': 0}
        }
        
        # Update tracking
        self.last_overview_update = 0
        self.overview_update_interval = 30  # Update every 30 seconds
        
        # Connection management
        self.active_requests = 0
        self.max_concurrent_requests = 5
        self.request_lock = threading.Lock()
        
        # Start background updater
        self.running = True
        self.update_thread = threading.Thread(target=self._background_updater, daemon=True)
        self.update_thread.start()
        
    def get_price(self, product_id: str) -> Tuple[float, float]:
        """Get bid/ask prices with caching"""
        with self.cache_lock:
            cached = self.price_cache.get(product_id)
            if cached and time.time() - cached['timestamp'] < self.cache_ttl:
                return cached['bid'], cached['ask']
        
        # Fetch new price
        try:
            with self.request_lock:
                if self.active_requests >= self.max_concurrent_requests:
                    # Return cached even if stale
                    if cached:
                        return cached['bid'], cached['ask']
                    return 0, 0
                    
                self.active_requests += 1
                
            try:
                bid, ask = get_best_bid_ask(product_id)
                
                # Update cache
                with self.cache_lock:
                    self.price_cache[product_id] = {
                        'bid': bid,
                        'ask': ask,
                        'timestamp': time.time()
                    }
                    
                return bid, ask
                
            finally:
                with self.request_lock:
                    self.active_requests -= 1
                    
        except Exception as e:
            self.logger.error(f"Failed to get price for {product_id}: {e}")
            return 0, 0
            
    def get_market_overview(self) -> Dict:
        """Get market overview data"""
        return self.market_overview.copy()
        
    def _background_updater(self):
        """Background thread to update market overview"""
        while self.running:
            try:
                current_time = time.time()
                
                # Update overview periodically
                if current_time - self.last_overview_update > self.overview_update_interval:
                    self._update_market_overview()
                    self.last_overview_update = current_time
                    
                # Clean old cache entries
                self._clean_cache()
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Background updater error: {e}")
                time.sleep(5)
                
    def _update_market_overview(self):
        """Update market overview data"""
        for product_id in self.market_overview.keys():
            try:
                # Rate limit check
                with self.request_lock:
                    if self.active_requests >= self.max_concurrent_requests:
                        continue
                    self.active_requests += 1
                    
                try:
                    # Get current price
                    bid, ask = get_best_bid_ask(product_id)
                    current_price = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
                    
                    # Get 24h data
                    candles = get_candles(product_id, granularity=3600, limit=24)
                    if candles and len(candles) > 0:
                        open_price = candles[0]['open']
                        volume_24h = sum(c['volume'] for c in candles)
                        change_24h = ((current_price - open_price) / open_price * 100) if open_price > 0 else 0
                    else:
                        volume_24h = 0
                        change_24h = 0
                        
                    # Update overview
                    self.market_overview[product_id] = {
                        'price': current_price,
                        'change_24h': change_24h,
                        'volume_24h': volume_24h,
                        'last_update': time.time()
                    }
                    
                finally:
                    with self.request_lock:
                        self.active_requests -= 1
                        
                # Small delay between requests
                time.sleep(0.2)
                
            except Exception as e:
                self.logger.error(f"Failed to update overview for {product_id}: {e}")
                
    def _clean_cache(self):
        """Clean old cache entries"""
        with self.cache_lock:
            current_time = time.time()
            to_remove = []
            
            for product_id, data in self.price_cache.items():
                if current_time - data['timestamp'] > 60:  # Remove entries older than 1 minute
                    to_remove.append(product_id)
                    
            for product_id in to_remove:
                del self.price_cache[product_id]
                
    def stop(self):
        """Stop the background updater"""
        self.running = False
        if self.update_thread.is_alive():
            self.update_thread.join(timeout=5)


# Global instance
_market_data_manager = None


def get_market_data_manager() -> MarketDataManager:
    """Get or create the global market data manager"""
    global _market_data_manager
    if _market_data_manager is None:
        _market_data_manager = MarketDataManager()
    return _market_data_manager 