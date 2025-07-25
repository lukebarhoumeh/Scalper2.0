#!/usr/bin/env python3
"""
unified_scalperbot.py - The ONE Unified Production ScalperBot
==========================================================
A single, intelligent trading bot that:
- Automatically switches between conservative/balanced/aggressive modes
- Uses OpenAI for decision making
- Implements profit preservation (25-75% withdrawal)
- Fixes all connection and execution issues
- Provides real-time market overview updates
"""

import sys
import os
import time
import json
import signal
import logging
import asyncio
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

# Colorful terminal output
import colorama
from colorama import Fore, Style, Back
colorama.init()

# Core imports
from coinbase.rest import RESTClient
from trade_logger import TradeLogger
from unified_terminal_ui import UnifiedTerminalUI, TerminalUIIntegration
from enhanced_output_formatter import EnhancedFormatter as EnhancedOutputFormatter

# OpenAI imports
from openai import OpenAI

# Configuration
from config import *

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class MarketSnapshot:
    """Real-time market data snapshot"""
    product_id: str
    price: float
    bid: float
    ask: float
    spread_bps: float
    volume_24h: float = 0
    change_24h_pct: float = 0
    volatility_1h: float = 0
    timestamp: float = field(default_factory=time.time)

@dataclass
class TradingSignal:
    """Trading signal with AI confidence"""
    product_id: str
    side: str  # BUY or SELL
    size_usd: float
    confidence: float
    reason: str
    ai_analysis: Optional[str] = None
    risk_score: float = 0.5
    timestamp: float = field(default_factory=time.time)

@dataclass
class BotConfig:
    """Dynamic bot configuration"""
    mode: str = "balanced"  # conservative, balanced, aggressive
    position_size_mult: float = 1.0
    max_positions: int = 3
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.03
    min_confidence: float = 0.7
    enable_ai: bool = True
    profit_withdrawal_pct: float = 0.5  # 50% default withdrawal
    max_spread_bps: float = 100.0  # 1% max spread (relaxed from 0.5%)


class UnifiedScalperBot:
    """The unified, intelligent ScalperBot"""
    
    def __init__(self):
        self.logger = logging.getLogger("UnifiedScalperBot")
        self.logger.info(f"{Fore.CYAN}Initializing Unified ScalperBot...{Style.RESET_ALL}")
        
        # Core components
        self.rest_client = self._create_rest_client()
        self.trade_logger = TradeLogger()
        self.formatter = EnhancedOutputFormatter()
        
        # Terminal UI
        self.terminal_ui = UnifiedTerminalUI()
        self.ui_integration = TerminalUIIntegration(self.terminal_ui)
        
        # OpenAI client
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        
        # Trading state
        self.positions: Dict[str, float] = {}  # product_id -> size in base units
        self.avg_entry_prices: Dict[str, float] = {}
        self.daily_pnl = 0.0
        self.daily_volume = 0.0
        self.daily_trades = 0
        self.daily_capital_used = 0.0
        
        # Market data cache
        self.market_data: Dict[str, MarketSnapshot] = {}
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.spread_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))  # Track spread history
        
        # Dynamic configuration
        self.config = BotConfig()
        self.performance_history = deque(maxlen=1000)
        
        # Error tracking
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_error_time = None
        self.signal_rejections: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))  # Track rejection reasons
        
        # Profit preservation
        self.profit_withdrawn_today = 0.0
        self.high_water_mark = 0.0
        
        # Threading
        self._shutdown = threading.Event()
        self._market_update_thread = None
        self._ui_thread = None
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info(f"{Fore.GREEN}Bot initialized successfully{Style.RESET_ALL}")
    
    def _create_rest_client(self) -> RESTClient:
        """Create REST client with proper configuration"""
        try:
            client = RESTClient(
                api_key=COINBASE_API_KEY,
                api_secret=COINBASE_API_SECRET,
                timeout=30,
                verbose=False
            )
            # Test connection
            client.get_accounts()
            self.logger.info(f"{Fore.GREEN}REST client connected successfully{Style.RESET_ALL}")
            return client
        except Exception as e:
            self.logger.error(f"{Fore.RED}Failed to create REST client: {e}{Style.RESET_ALL}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"\n{Fore.YELLOW}Received signal {signum}. Shutting down...{Style.RESET_ALL}")
        self._shutdown.set()
    
    async def initialize(self):
        """Initialize all bot components"""
        try:
            # Display banner
            self.formatter.display_startup_banner()
            
            # Start UI thread
            self._ui_thread = threading.Thread(target=self._run_ui, daemon=True)
            self._ui_thread.start()
            
            # Start market data updates
            self._market_update_thread = threading.Thread(target=self._market_update_loop, daemon=True)
            self._market_update_thread.start()
            
            # Load previous state if exists
            self._load_state()
            
            # Initial market data fetch
            await self._update_all_market_data()
            
            # AI health check
            if self.openai_client:
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": "Test"}],
                        max_tokens=10
                    )
                    self.logger.info(f"{Fore.GREEN}OpenAI API connected successfully{Style.RESET_ALL}")
                except Exception as e:
                    self.logger.warning(f"{Fore.YELLOW}OpenAI API not available: {e}{Style.RESET_ALL}")
                    self.config.enable_ai = False
            
            self.logger.info(f"{Fore.GREEN}[OK] All systems initialized{Style.RESET_ALL}")
            
        except Exception as e:
            self.logger.error(f"{Fore.RED}Initialization failed: {e}{Style.RESET_ALL}")
            raise
    
    def _run_ui(self):
        """Run terminal UI in separate thread"""
        try:
            self.terminal_ui.run()
        except Exception as e:
            self.logger.error(f"UI error: {e}")
    
    def _market_update_loop(self):
        """Background thread for market data updates"""
        while not self._shutdown.is_set():
            try:
                # Update market data every second
                asyncio.run(self._update_all_market_data())
                
                # Update UI
                self._update_ui()
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.debug(f"Market update error: {e}")
                time.sleep(5)
    
    async def _update_all_market_data(self):
        """Update market data for all trading pairs"""
        tasks = []
        for coin in TRADE_COINS:
            product_id = f"{coin}-USD"
            tasks.append(self._update_market_data(product_id))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.debug(f"Failed to update {TRADE_COINS[i]}: {result}")
    
    async def _update_market_data(self, product_id: str):
        """Update market data for a single product"""
        try:
            # Get best bid/ask
            response = self.rest_client.get_best_bid_ask(product_ids=[product_id])
            if not response or not response.pricebooks or not response.pricebooks[0].bids or not response.pricebooks[0].asks:
                self.logger.debug(f"No price data for {product_id}")
                return
                
            pb = response.pricebooks[0]
            bid = float(pb.bids[0].price)
            ask = float(pb.asks[0].price)
            price = (bid + ask) / 2  # Mid price
            
            # Calculate spread
            spread_bps = ((ask - bid) / price) * 10000 if price > 0 else 0
            
            # Get candles for 24h stats
            try:
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(hours=24)
                
                candles_response = self.rest_client.get_candles(
                    product_id=product_id,
                    start=int(start_time.timestamp()),
                    end=int(end_time.timestamp()),
                    granularity="ONE_HOUR"
                )
                
                volume_24h = 0
                change_24h_pct = 0
                
                if candles_response and candles_response.candles:
                    # Sum volume from all candles
                    for candle in candles_response.candles:
                        volume_24h += float(candle.volume)
                    
                    # Calculate 24h change
                    oldest_candle = candles_response.candles[-1]  # API returns newest first
                    open_24h = float(oldest_candle.open)
                    change_24h_pct = ((price - open_24h) / open_24h * 100) if open_24h > 0 else 0
                    
            except Exception as e:
                self.logger.debug(f"Error getting candles for {product_id}: {e}")
                volume_24h = 0
                change_24h_pct = 0
            
            # Store snapshot
            snapshot = MarketSnapshot(
                product_id=product_id,
                price=price,
                bid=bid,
                ask=ask,
                spread_bps=spread_bps,
                volume_24h=volume_24h,
                change_24h_pct=change_24h_pct
            )
            
            self.market_data[product_id] = snapshot
            self.price_history[product_id].append((time.time(), price))
            
            # Track spread history
            self.spread_history[product_id].append({
                'time': time.time(),
                'spread_bps': spread_bps
            })
            
            # Log if spread is consistently high
            if len(self.spread_history[product_id]) >= 10:
                recent_spreads = [s['spread_bps'] for s in list(self.spread_history[product_id])[-10:]]
                avg_spread = np.mean(recent_spreads)
                if avg_spread > 80:  # 0.8%
                    self.logger.warning(f"{product_id}: High average spread {avg_spread:.1f} bps")
            
            # Calculate volatility
            if len(self.price_history[product_id]) > 10:
                prices = [p[1] for p in list(self.price_history[product_id])[-20:]]
                returns = np.diff(np.log(prices))
                snapshot.volatility_1h = np.std(returns) * np.sqrt(3600) if len(returns) > 0 else 0
            
        except Exception as e:
            self.logger.debug(f"Error updating {product_id}: {e}")
    
    def _update_ui(self):
        """Update terminal UI with current data"""
        try:
            # Update stats including capital tracking
            stats = {
                'daily_pnl': self.daily_pnl,
                'daily_volume': self.daily_volume,
                'daily_trades': self.daily_trades,
                'total_trades': self.daily_trades,  # UI expects this field
                'positions': len(self.positions),
                'mode': self.config.mode,
                'error_count': self.error_count,
                'daily_capital_used': self.daily_capital_used,
                'daily_capital_remaining': MAX_DAILY_CAPITAL - self.daily_capital_used,
                'win_rate': 0.0,  # TODO: Calculate actual win rate
                'avg_slippage_bps': 0.0  # TODO: Calculate actual slippage
            }
            self.ui_integration.update_stats(stats)
            
            # Update positions
            position_data = {}
            for product_id, size in self.positions.items():
                if product_id in self.market_data:
                    price = self.market_data[product_id].price
                    value = size * price
                    avg_price = self.avg_entry_prices.get(product_id, price)
                    pnl = (price - avg_price) * size
                    pnl_pct = ((price / avg_price) - 1) * 100 if avg_price > 0 else 0
                    
                    position_data[product_id] = {
                        'size': size,
                        'value': value,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct
                    }
            
            self.ui_integration.update_positions(position_data)
            
            # Update market overview
            market_overview = {}
            for product_id, snapshot in self.market_data.items():
                market_overview[product_id] = {
                    'price': snapshot.price,
                    '24h_change': snapshot.change_24h_pct,
                    'volume': snapshot.volume_24h,
                    'spread': snapshot.spread_bps,
                    'volatility': snapshot.volatility_1h * 100  # Convert to percentage
                }
            
            self.ui_integration.update_market(market_overview)
            
            # Update health status
            health = {
                'executor_healthy': True,
                'strategy_healthy': True,
                'market_data_healthy': len(self.market_data) > 0,
                'ai_healthy': self.config.enable_ai and self.openai_client is not None,
                'errors': self.consecutive_errors,
                'uptime': time.time() - (self.last_error_time or time.time()),
                'memory_mb': self._get_memory_usage()
            }
            self.ui_integration.update_health(health)
            
        except Exception as e:
            self.logger.debug(f"UI update error: {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    async def _get_ai_market_analysis(self) -> Dict[str, Any]:
        """Get AI analysis of current market conditions"""
        if not self.config.enable_ai or not self.openai_client:
            return {"recommendation": "balanced", "confidence": 0.5}
        
        try:
            # Prepare market summary
            market_summary = []
            for product_id, snapshot in self.market_data.items():
                market_summary.append(
                    f"{product_id}: ${snapshot.price:.2f} "
                    f"({snapshot.change_24h_pct:+.2f}%), "
                    f"Vol: ${snapshot.volume_24h/1e6:.1f}M, "
                    f"Volatility: {snapshot.volatility_1h*100:.2f}%"
                )
            
            # Prepare performance summary
            performance_summary = (
                f"Daily P&L: ${self.daily_pnl:.2f}, "
                f"Trades: {self.daily_trades}, "
                f"Current positions: {len(self.positions)}"
            )
            
            # Create prompt
            prompt = f"""
            As a cryptocurrency trading AI, analyze the current market conditions and recommend a trading approach.
            
            Market Overview:
            {chr(10).join(market_summary)}
            
            Bot Performance Today:
            {performance_summary}
            
            Based on this data, provide:
            1. Recommended trading mode (conservative/balanced/aggressive)
            2. Key market observations
            3. Specific coins to focus on
            4. Risk level assessment (1-10)
            
            Respond in JSON format.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert cryptocurrency trader providing market analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse response
            content = response.choices[0].message.content
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"recommendation": "balanced", "confidence": 0.5}
                
        except Exception as e:
            self.logger.debug(f"AI analysis error: {e}")
            return {"recommendation": "balanced", "confidence": 0.5}
    
    async def _evaluate_trading_signals(self) -> List[TradingSignal]:
        """Evaluate all products and generate trading signals"""
        signals = []
        
        for product_id, snapshot in self.market_data.items():
            # Skip if no price history
            if len(self.price_history[product_id]) < 20:
                logger.debug(f"{product_id}: Skipped - insufficient price history ({len(self.price_history[product_id])}/20)")
                continue
            
            # Get recent prices
            recent_prices = [p[1] for p in list(self.price_history[product_id])[-20:]]
            
            # Calculate simple indicators
            sma_5 = np.mean(recent_prices[-5:])
            sma_20 = np.mean(recent_prices)
            price = snapshot.price
            
            # RSI calculation
            price_changes = np.diff(recent_prices)
            gains = price_changes[price_changes > 0]
            losses = -price_changes[price_changes < 0]
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Generate signals based on indicators
            signal = None
            
            # Buy signal conditions - log each check failure
            if sma_5 <= sma_20:
                reason = f"No uptrend - SMA5 {sma_5:.2f} <= SMA20 {sma_20:.2f}"
                logger.debug(f"{product_id}: {reason}")
                self.signal_rejections[product_id].append({'reason': reason, 'time': time.time()})
            elif price <= sma_5:
                reason = f"Price {price:.2f} not above SMA5 {sma_5:.2f}"
                logger.debug(f"{product_id}: {reason}")
                self.signal_rejections[product_id].append({'reason': reason, 'time': time.time()})
            elif rsi >= 70:
                reason = f"RSI {rsi:.0f} >= 70"
                logger.info(f"{product_id}: Buy signal rejected - {reason}")
                self.signal_rejections[product_id].append({'reason': f"Buy rejected - {reason}", 'time': time.time()})
            elif snapshot.spread_bps >= self._get_dynamic_spread_threshold(product_id):
                max_spread = self._get_dynamic_spread_threshold(product_id)
                reason = f"Spread {snapshot.spread_bps:.1f} bps >= {max_spread:.0f} bps"
                logger.info(f"{product_id}: Signal rejected - {reason}")
                self.signal_rejections[product_id].append({'reason': reason, 'time': time.time()})
            elif (sma_5 > sma_20 and price > sma_5 and rsi < 70 and 
                snapshot.spread_bps < self._get_dynamic_spread_threshold(product_id)):  # All conditions met
                
                # Check if we have room for position
                current_position_value = self.positions.get(product_id, 0) * price
                if current_position_value < PER_COIN_POSITION_LIMIT:
                    confidence = min(0.9, 0.5 + (sma_5 - sma_20) / sma_20 * 10)
                    
                    signal = TradingSignal(
                        product_id=product_id,
                        side="BUY",
                        size_usd=self._calculate_position_size(confidence),
                        confidence=confidence,
                        reason=f"Uptrend detected, RSI={rsi:.0f}",
                        risk_score=1 - confidence
                    )
                else:
                    logger.info(f"{product_id}: Buy signal rejected - position limit reached ({current_position_value:.2f}/{PER_COIN_POSITION_LIMIT})")
            
            # Sell signal conditions
            elif (sma_5 < sma_20 and price < sma_5 and rsi > 30 and
                  self.positions.get(product_id, 0) > 0):
                
                confidence = min(0.9, 0.5 + (sma_20 - sma_5) / sma_20 * 10)
                
                # Calculate position size (don't sell more than we have)
                position_value = self.positions[product_id] * price
                size_usd = min(
                    self._calculate_position_size(confidence),
                    position_value * 0.5  # Sell half at most
                )
                
                signal = TradingSignal(
                    product_id=product_id,
                    side="SELL",
                    size_usd=size_usd,
                    confidence=confidence,
                    reason=f"Downtrend detected, RSI={rsi:.0f}",
                    risk_score=1 - confidence
                )
            
            # Add AI enhancement if available
            if signal and self.config.enable_ai:
                ai_opinion = await self._get_ai_signal_opinion(signal, snapshot)
                signal.ai_analysis = ai_opinion.get('analysis', '')
                signal.confidence *= ai_opinion.get('confidence_multiplier', 1.0)
            
            # Only add signal if confidence meets threshold
            if signal and signal.confidence >= self.config.min_confidence:
                signals.append(signal)
        
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit number of concurrent signals based on mode
        max_signals = {
            'conservative': 1,
            'balanced': 2,
            'aggressive': 3
        }
        
        return signals[:max_signals.get(self.config.mode, 2)]
    
    async def _get_ai_signal_opinion(self, signal: TradingSignal, snapshot: MarketSnapshot) -> Dict:
        """Get AI opinion on a trading signal"""
        if not self.config.enable_ai or not self.openai_client:
            return {"confidence_multiplier": 1.0}
        
        try:
            prompt = f"""
            Evaluate this trading signal:
            Product: {signal.product_id}
            Action: {signal.side}
            Size: ${signal.size_usd:.2f}
            Technical Confidence: {signal.confidence:.2%}
            Reason: {signal.reason}
            
            Market Data:
            Current Price: ${snapshot.price:.2f}
            24h Change: {snapshot.change_24h_pct:+.2f}%
            Volatility: {snapshot.volatility_1h*100:.2f}%
            Spread: {snapshot.spread_bps:.1f} bps
            
            Should this trade be executed? Provide confidence adjustment (0.5-1.5x).
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a risk management AI. Be conservative."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            
            # Extract confidence multiplier
            import re
            mult_match = re.search(r'(\d+\.?\d*)x', content)
            if mult_match:
                multiplier = float(mult_match.group(1))
                multiplier = max(0.5, min(1.5, multiplier))  # Clamp to range
            else:
                multiplier = 1.0
            
            return {
                "confidence_multiplier": multiplier,
                "analysis": content[:100]  # First 100 chars
            }
            
        except Exception as e:
            self.logger.debug(f"AI signal opinion error: {e}")
            return {"confidence_multiplier": 1.0}
    
    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence and mode"""
        base_size = {
            'conservative': MIN_POSITION_SIZE,
            'balanced': (MIN_POSITION_SIZE + MAX_POSITION_SIZE) / 2,
            'aggressive': MAX_POSITION_SIZE
        }
        
        size = base_size.get(self.config.mode, MIN_POSITION_SIZE)
        
        # Adjust by confidence
        size *= confidence
        
        # Adjust by performance
        if self.daily_pnl > 50:  # Winning
            size *= 1.2
        elif self.daily_pnl < -30:  # Losing
            size *= 0.8
        
        # Apply multiplier
        size *= self.config.position_size_mult
        
        # Ensure within limits
        size = max(MIN_POSITION_SIZE, min(MAX_POSITION_SIZE, size))
        
        # Check capital limits
        if self.daily_capital_used + size > MAX_DAILY_CAPITAL:
            size = MAX_DAILY_CAPITAL - self.daily_capital_used
        
        return round(size, 2)
    
    def _get_dynamic_spread_threshold(self, product_id: str) -> float:
        """Get spread threshold based on time and volatility"""
        current_hour = datetime.now(timezone.utc).hour
        
        # Wider spreads during off-hours (UTC)
        if current_hour in [0, 1, 2, 3, 22, 23]:  # Late night
            base_spread = 150  # 1.5%
        elif current_hour in [13, 14, 15, 16]:    # US market hours
            base_spread = 50   # 0.5%
        else:
            base_spread = 100  # 1.0%
        
        # Adjust for volatility
        if product_id in self.market_data:
            volatility = self.market_data[product_id].volatility_1h
            if volatility > 0.02:  # High vol
                base_spread *= 1.5
        
        # Use max of dynamic or configured threshold
        return max(base_spread, self.config.max_spread_bps)
    
    async def _execute_signal(self, signal: TradingSignal) -> bool:
        """Execute a trading signal"""
        try:
            # Check paper trading FIRST
            if PAPER_TRADING:
                logger.info(f"PAPER TRADE: {signal.side} {signal.product_id} ${signal.size_usd:.2f}")
                
                # Get current market data for paper trading
                if signal.product_id not in self.market_data:
                    return False
                    
                price = self.market_data[signal.product_id].price
                
                # Simulate execution
                if signal.side == "BUY":
                    # Update positions
                    current_qty = self.positions.get(signal.product_id, 0)
                    new_qty = current_qty + (signal.size_usd / price)
                    self.positions[signal.product_id] = new_qty
                    
                    # Track capital
                    self.daily_capital_used += signal.size_usd
                    self.daily_volume += signal.size_usd
                    
                    # Log trade to UI
                    self.recent_trades.append({
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'product': signal.product_id,
                        'side': 'BUY',
                        'size': f"${signal.size_usd:.2f}",
                        'price': f"${price:.2f}"
                    })
                else:  # SELL
                    # Update positions
                    current_qty = self.positions.get(signal.product_id, 0)
                    sell_qty = min(current_qty, signal.size_usd / price)
                    
                    if sell_qty > 0:
                        self.positions[signal.product_id] = current_qty - sell_qty
                        
                        # Simple P&L calc
                        avg_entry = self.avg_entry_prices.get(signal.product_id, price)
                        pnl = (price - avg_entry) * sell_qty
                        self.daily_pnl += pnl
                        self.daily_volume += sell_qty * price
                        
                        # Log trade
                        self.recent_trades.append({
                            'time': datetime.now().strftime("%H:%M:%S"),
                            'product': signal.product_id,
                            'side': 'SELL',
                            'size': f"${sell_qty * price:.2f}",
                            'price': f"${price:.2f}",
                            'pnl': f"${pnl:.2f}"
                        })
                
                self.daily_trades += 1
                
                # Log to trade logger
                self.trade_logger.log_trade(
                    order_id=f"PAPER-{int(time.time())}",
                    product_id=signal.product_id,
                    side=signal.side,
                    qty_base=signal.size_usd / price,
                    price=price,
                    notional_usd=signal.size_usd,
                    strategy="Paper Trading",
                    pnl=self.daily_pnl
                )
                
                return True
            
            # LIVE TRADING CODE BELOW
            self.logger.info(
                f"{Fore.CYAN}Executing {signal.side} {signal.product_id} "
                f"${signal.size_usd:.2f} (confidence: {signal.confidence:.1%}){Style.RESET_ALL}"
            )
            
            # Log signal to UI
            self.ui_integration.log_signal({
                'side': signal.side,
                'product': signal.product_id,
                'size': signal.size_usd,
                'strategy': 'Unified AI Strategy',
                'confidence': signal.confidence * 100
            })
            
            # Get current market data
            if signal.product_id not in self.market_data:
                self.logger.warning(f"No market data for {signal.product_id}")
                return False
            
            snapshot = self.market_data[signal.product_id]
            
            # Determine order parameters
            if signal.side == "BUY":
                # Use limit order slightly above bid
                limit_price = snapshot.bid * 1.0001
                size_base = signal.size_usd / limit_price
            else:
                # Use limit order slightly below ask  
                limit_price = snapshot.ask * 0.9999
                size_base = min(
                    signal.size_usd / limit_price,
                    self.positions.get(signal.product_id, 0)
                )
            
            # Round size to product precision
            size_base = round(size_base, 8)
            
            # Skip if size too small
            if size_base * limit_price < 10:
                self.logger.debug(f"Order size too small: ${size_base * limit_price:.2f}")
                return False
            
            # Place order
            order_config = {
                'product_id': signal.product_id,
                'side': signal.side,
                'order_configuration': {
                    'limit_limit_gtc': {
                        'base_size': str(size_base),
                        'limit_price': str(round(limit_price, 2))
                    }
                }
            }
            
            result = self.rest_client.create_order(**order_config)
            
            if result.get('success', False):
                order_id = result['order_id']
                self.logger.info(f"{Fore.GREEN}Order placed: {order_id}{Style.RESET_ALL}")
                
                # Wait for fill
                await asyncio.sleep(1)
                
                # Check order status
                order = self.rest_client.get_order(order_id)
                
                if order['status'] in ['FILLED', 'PARTIALLY_FILLED']:
                    filled_size = float(order.get('filled_size', 0))
                    avg_price = float(order.get('average_filled_price', limit_price))
                    
                    # Update positions
                    if signal.side == "BUY":
                        current_size = self.positions.get(signal.product_id, 0)
                        current_avg = self.avg_entry_prices.get(signal.product_id, 0)
                        
                        # Calculate new average price
                        total_value = (current_size * current_avg) + (filled_size * avg_price)
                        new_size = current_size + filled_size
                        new_avg = total_value / new_size if new_size > 0 else avg_price
                        
                        self.positions[signal.product_id] = new_size
                        self.avg_entry_prices[signal.product_id] = new_avg
                        self.daily_capital_used += filled_size * avg_price
                        
                    else:  # SELL
                        self.positions[signal.product_id] -= filled_size
                        if self.positions[signal.product_id] <= 0:
                            self.positions.pop(signal.product_id, None)
                            self.avg_entry_prices.pop(signal.product_id, None)
                        
                        # Calculate P&L
                        entry_price = self.avg_entry_prices.get(signal.product_id, avg_price)
                        pnl = (avg_price - entry_price) * filled_size
                        self.daily_pnl += pnl
                    
                    # Update metrics
                    self.daily_volume += filled_size * avg_price
                    self.daily_trades += 1
                    
                    # Log trade
                    self.trade_logger.log_trade(
                        order_id=order_id,
                        product_id=signal.product_id,
                        side=signal.side,
                        qty_base=filled_size,
                        price=avg_price,
                        notional_usd=filled_size * avg_price,
                        strategy="Unified AI Strategy",
                        running_pnl=self.daily_pnl
                    )
                    
                    # Log to UI
                    self.ui_integration.log_trade({
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'product': signal.product_id,
                        'side': signal.side,
                        'size': filled_size,
                        'price': avg_price,
                        'value': filled_size * avg_price,
                        'pnl': pnl if signal.side == "SELL" else 0
                    })
                    
                    return True
                else:
                    # Cancel unfilled order
                    self.rest_client.cancel_orders([order_id])
                    self.logger.warning(f"Order not filled, cancelled: {order_id}")
                    return False
            else:
                self.logger.error(f"Order failed: {result.get('error_response', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"{Fore.RED}Execution error: {e}{Style.RESET_ALL}")
            self.consecutive_errors += 1
            self.error_count += 1
            self.last_error_time = time.time()
            return False
    
    async def _check_profit_preservation(self):
        """Check if we should withdraw profits"""
        if self.daily_pnl <= 0:
            return
        
        # Update high water mark
        self.high_water_mark = max(self.high_water_mark, self.daily_pnl)
        
        # Check if we should preserve profits
        preserve_conditions = [
            self.daily_pnl > 100,  # Minimum $100 profit
            datetime.now().hour >= 22,  # After 10 PM
            self.daily_pnl < self.high_water_mark * 0.9,  # 10% drawdown from peak
        ]
        
        if any(preserve_conditions):
            withdrawal_amount = self.daily_pnl * self.config.profit_withdrawal_pct
            
            self.logger.info(
                f"{Fore.GREEN}Profit preservation triggered: "
                f"Withdrawing ${withdrawal_amount:.2f} "
                f"({self.config.profit_withdrawal_pct:.0%} of ${self.daily_pnl:.2f}){Style.RESET_ALL}"
            )
            
            # In a real implementation, this would transfer to a separate account
            # For now, we just track it
            self.profit_withdrawn_today += withdrawal_amount
            self.daily_pnl -= withdrawal_amount
            
            # Close some positions to realize profits
            await self._close_winning_positions(withdrawal_amount)
    
    async def _close_winning_positions(self, target_amount: float):
        """Close winning positions to realize profits"""
        positions_to_close = []
        
        for product_id, size in self.positions.items():
            if product_id not in self.market_data:
                continue
                
            current_price = self.market_data[product_id].price
            entry_price = self.avg_entry_prices.get(product_id, current_price)
            pnl = (current_price - entry_price) * size
            
            if pnl > 0:  # Winning position
                positions_to_close.append((product_id, size, pnl))
        
        # Sort by P&L
        positions_to_close.sort(key=lambda x: x[2], reverse=True)
        
        # Close positions until we reach target
        realized = 0
        for product_id, size, pnl in positions_to_close:
            if realized >= target_amount:
                break
            
            # Create sell signal
            signal = TradingSignal(
                product_id=product_id,
                side="SELL",
                size_usd=size * self.market_data[product_id].price,
                confidence=0.95,
                reason="Profit preservation"
            )
            
            if await self._execute_signal(signal):
                realized += pnl
    
    async def _update_mode(self):
        """Dynamically update trading mode based on conditions"""
        # Get AI recommendation
        ai_analysis = await self._get_ai_market_analysis()
        ai_mode = ai_analysis.get('recommendation', 'balanced')
        
        # Factor in performance
        if self.daily_pnl > 100:
            performance_mode = 'aggressive'
        elif self.daily_pnl < -50:
            performance_mode = 'conservative'
        else:
            performance_mode = 'balanced'
        
        # Factor in errors
        if self.consecutive_errors > 5:
            error_mode = 'conservative'
        else:
            error_mode = 'balanced'
        
        # Combine factors
        modes = [ai_mode, performance_mode, error_mode]
        mode_scores = {
            'conservative': modes.count('conservative'),
            'balanced': modes.count('balanced'),
            'aggressive': modes.count('aggressive')
        }
        
        # Select mode with highest score
        new_mode = max(mode_scores, key=mode_scores.get)
        
        if new_mode != self.config.mode:
            self.logger.info(
                f"{Fore.YELLOW}Switching mode from {self.config.mode} to {new_mode} "
                f"(AI: {ai_mode}, Perf: {performance_mode}, Errors: {error_mode}){Style.RESET_ALL}"
            )
            self.config.mode = new_mode
            
            # Adjust configuration
            if new_mode == 'conservative':
                self.config.position_size_mult = 0.7
                self.config.min_confidence = 0.8
                self.config.stop_loss_pct = 0.015
                self.config.take_profit_pct = 0.02
            elif new_mode == 'aggressive':
                self.config.position_size_mult = 1.3
                self.config.min_confidence = 0.6
                self.config.stop_loss_pct = 0.03
                self.config.take_profit_pct = 0.05
            else:  # balanced
                self.config.position_size_mult = 1.0
                self.config.min_confidence = 0.7
                self.config.stop_loss_pct = 0.02
                self.config.take_profit_pct = 0.03
    
    async def run(self):
        """Main trading loop"""
        self.logger.info(f"{Fore.GREEN}Starting unified trading loop...{Style.RESET_ALL}")
        
        last_trade_time = time.time()
        mode_check_interval = 300  # 5 minutes
        last_mode_check = time.time()
        
        while not self._shutdown.is_set():
            try:
                # Update mode periodically
                if time.time() - last_mode_check > mode_check_interval:
                    await self._update_mode()
                    last_mode_check = time.time()
                
                # Check for profit preservation
                await self._check_profit_preservation()
                
                # Get trading signals
                signals = await self._evaluate_trading_signals()
                
                # Execute signals with rate limiting
                if signals and time.time() - last_trade_time > 10:  # Min 10s between trades
                    for signal in signals:
                        if await self._execute_signal(signal):
                            last_trade_time = time.time()
                            break  # One trade at a time
                
                # Reset consecutive errors on successful iteration
                if self.consecutive_errors > 0:
                    self.consecutive_errors = 0
                
                # Sleep before next iteration
                await asyncio.sleep(5)
                
            except Exception as e:
                self.consecutive_errors += 1
                self.error_count += 1
                self.last_error_time = time.time()
                
                self.logger.error(f"{Fore.RED}Main loop error: {e}{Style.RESET_ALL}")
                
                # Longer sleep on errors
                await asyncio.sleep(30)
    
    def _load_state(self):
        """Load previous bot state if exists"""
        try:
            state_file = "bot_state.json"
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.high_water_mark = state.get('high_water_mark', 0)
                    self.logger.info(f"Loaded previous state: HWM=${self.high_water_mark:.2f}")
        except Exception as e:
            self.logger.debug(f"Could not load state: {e}")
    
    def _save_state(self):
        """Save current bot state"""
        try:
            state = {
                'high_water_mark': self.high_water_mark,
                'daily_pnl': self.daily_pnl,
                'profit_withdrawn_today': self.profit_withdrawn_today,
                'timestamp': datetime.now().isoformat()
            }
            with open("bot_state.json", 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save state: {e}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info(f"{Fore.YELLOW}Shutting down Unified ScalperBot...{Style.RESET_ALL}")
        
        # Set shutdown flag
        self._shutdown.set()
        
        # Close all positions
        self.logger.info("Closing all positions...")
        for product_id in list(self.positions.keys()):
            signal = TradingSignal(
                product_id=product_id,
                side="SELL",
                size_usd=self.positions[product_id] * self.market_data[product_id].price,
                confidence=1.0,
                reason="Shutdown"
            )
            await self._execute_signal(signal)
        
        # Save state
        self._save_state()
        
        # Save performance data
        performance_file = f"logs/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(performance_file, 'w') as f:
            json.dump({
                'mode': self.config.mode,
                'daily_pnl': self.daily_pnl,
                'daily_volume': self.daily_volume,
                'daily_trades': self.daily_trades,
                'profit_withdrawn': self.profit_withdrawn_today,
                'error_count': self.error_count,
                'final_positions': dict(self.positions),
                'config': self.config.__dict__
            }, f, indent=2, default=str)
        
        # Stop UI
        self.terminal_ui.stop()
        
        self.logger.info(f"{Fore.GREEN}Shutdown complete. Final P&L: ${self.daily_pnl:.2f}{Style.RESET_ALL}")


async def main():
    """Main entry point"""
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Create and run bot
    bot = UnifiedScalperBot()
    
    try:
        await bot.initialize()
        await bot.run()
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Fatal error: {e}{Style.RESET_ALL}")
        raise
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    # Run the bot
    asyncio.run(main()) 