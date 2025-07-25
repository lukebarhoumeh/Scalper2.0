#!/usr/bin/env python3
"""
Unified ScalperBot v2.0 - Production-Grade HFT Trading Bot
A clean, streamlined implementation that actually trades.
"""

import asyncio
import logging
import os
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import signal

import pandas as pd
import numpy as np
from colorama import Fore, Style, init as colorama_init
from coinbase.rest import RESTClient
import openai

# Initialize colorama for Windows
colorama_init()

# Import configuration
from config_unified import (
    CB_API_KEY, CB_API_SECRET, OPENAI_API_KEY,
    TRADING_PAIRS, BASE_POSITION_SIZE, MAX_POSITION_SIZE, INVENTORY_CAP_USD,
    STARTING_CAPITAL, MAX_DAILY_CAPITAL, RSI_BUY_THRESHOLD, RSI_SELL_THRESHOLD,
    SMA_FAST, SMA_SLOW, MIN_PRICE_HISTORY, MIN_SPREAD_BPS, MAX_SPREAD_BPS,
    PAPER_TRADING, MAX_TRADES_PER_HOUR, MAX_CONSECUTIVE_LOSSES,
    PROFIT_WITHDRAWAL_PERCENT, MIN_PROFIT_FOR_WITHDRAWAL,
    validate_config
)

# Import trade logger
from trade_logger import get_trade_logger, log_trade, log_signal

# Validate configuration on startup
config_errors = validate_config()
if config_errors:
    print(f"{Fore.YELLOW}Configuration warnings: {'; '.join(config_errors)}{Style.RESET_ALL}")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('UnifiedScalperBot')

# File logger for debugging
os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler(f'logs/unified_bot_{datetime.now().strftime("%Y%m%d")}.log')
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)


class TradingSignal:
    """Represents a trading signal"""
    def __init__(self, product_id: str, action: str, confidence: float, reason: str):
        self.product_id = product_id
        self.action = action  # 'buy' or 'sell'
        self.confidence = confidence
        self.reason = reason
        self.timestamp = datetime.now(timezone.utc)


class MarketSnapshot:
    """Current market state for a product"""
    def __init__(self):
        self.bid = 0.0
        self.ask = 0.0
        self.mid = 0.0
        self.spread_bps = 0.0
        self.volume_24h = 0.0
        self.price_change_24h = 0.0
        self.volatility_1h = 0.0
        self.last_update = datetime.now(timezone.utc)


class UnifiedScalperBot:
    """Main bot class - streamlined and production-ready"""
    
    def __init__(self):
        logger.info(f"{Fore.CYAN}Initializing Unified ScalperBot v2.0...{Style.RESET_ALL}")
        
        # Core components
        self.rest_client = None
        self.openai_client = None
        self.ui = None
        self.running = False
        
        # Market data
        self.market_snapshots: Dict[str, MarketSnapshot] = {coin: MarketSnapshot() for coin in TRADING_PAIRS}
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.spread_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Trading state
        self.positions: Dict[str, Dict] = {}  # {product_id: {'size': float, 'entry_price': float}}
        self.capital_used = 0.0
        self.capital_remaining = STARTING_CAPITAL
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.errors_caught = 0
        
        # Trading history
        self.recent_trades = deque(maxlen=50)
        self.signal_rejections: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        
        # Mode management
        self.current_mode = "balanced"
        self.mode_multipliers = {
            "conservative": 0.5,
            "balanced": 1.0,
            "aggressive": 1.5
        }
        
        # Circuit breaker state
        self.trades_this_hour = deque(maxlen=MAX_TRADES_PER_HOUR)
        self.consecutive_losses = 0
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None
        
        # Trade logger
        self.trade_logger = get_trade_logger()
        
        # Initialize components
        self._init_api_clients()
        self._init_ui()
        
        logger.info(f"{Fore.GREEN}Bot initialized successfully{Style.RESET_ALL}")
    
    def _init_api_clients(self):
        """Initialize API clients"""
        # Coinbase REST client
        if not CB_API_KEY or not CB_API_SECRET:
            logger.warning(f"{Fore.YELLOW}No Coinbase API credentials found. Running in simulation mode.{Style.RESET_ALL}")
            self.rest_client = None
        else:
            try:
                self.rest_client = RESTClient(api_key=CB_API_KEY, api_secret=CB_API_SECRET)
                # Test connection
                self.rest_client.get_accounts()
                logger.info(f"{Fore.GREEN}Coinbase API connected successfully{Style.RESET_ALL}")
            except Exception as e:
                logger.error(f"Failed to connect to Coinbase API: {e}")
                self.rest_client = None
        
        # OpenAI client
        if OPENAI_API_KEY:
            try:
                openai.api_key = OPENAI_API_KEY
                self.openai_client = openai
                logger.info(f"{Fore.GREEN}OpenAI API configured{Style.RESET_ALL}")
            except Exception as e:
                logger.warning(f"OpenAI API not available: {e}")
                self.openai_client = None
    
    def _init_ui(self):
        """Initialize terminal UI"""
        try:
            from unified_terminal_ui import UnifiedTerminalUI, TerminalUIIntegration
            self.ui = UnifiedTerminalUI()
            self.ui_integration = TerminalUIIntegration(self.ui)
            
            # Start UI thread
            ui_thread = threading.Thread(target=self.ui.run, daemon=True)
            ui_thread.start()
            time.sleep(0.5)  # Let UI initialize
            
            logger.info("Terminal UI initialized")
        except Exception as e:
            logger.error(f"Failed to initialize UI: {e}")
            self.ui = None
    
    def _update_market_data(self):
        """Fetch latest market data"""
        try:
            if self.rest_client:
                for product_id in TRADING_PAIRS:
                    try:
                        # Get ticker data
                        ticker = self.rest_client.get_product(product_id)
                        
                        # Get order book for best bid/ask
                        order_book = self.rest_client.get_product_book(product_id, level=1)
                        
                        snapshot = self.market_snapshots[product_id]
                        
                        # Update snapshot
                        if order_book and 'bids' in order_book and 'asks' in order_book:
                            if order_book['bids'] and order_book['asks']:
                                snapshot.bid = float(order_book['bids'][0][0])
                                snapshot.ask = float(order_book['asks'][0][0])
                                snapshot.mid = (snapshot.bid + snapshot.ask) / 2
                                
                                # Calculate spread in basis points
                                if snapshot.mid > 0:
                                    snapshot.spread_bps = ((snapshot.ask - snapshot.bid) / snapshot.mid) * 10000
                                    self.spread_history[product_id].append(snapshot.spread_bps)
                        
                        # Update volume and price change
                        if ticker:
                            snapshot.volume_24h = float(ticker.get('volume', 0))
                            # Calculate price change from ticker data
                            if 'price' in ticker and 'open_24h' in ticker:
                                current_price = float(ticker['price'])
                                open_price = float(ticker['open_24h'])
                                if open_price > 0:
                                    snapshot.price_change_24h = ((current_price - open_price) / open_price) * 100
                        
                        # Store price history
                        if snapshot.mid > 0:
                            self.price_history[product_id].append({
                                'price': snapshot.mid,
                                'timestamp': datetime.now(timezone.utc)
                            })
                        
                        # Calculate simple volatility
                        if len(self.price_history[product_id]) > 10:
                            recent_prices = [p['price'] for p in list(self.price_history[product_id])[-20:]]
                            snapshot.volatility_1h = np.std(recent_prices) / np.mean(recent_prices) if np.mean(recent_prices) > 0 else 0
                        
                        snapshot.last_update = datetime.now(timezone.utc)
                        
                        # Log high spreads
                        if len(self.spread_history[product_id]) > 20:
                            avg_spread = np.mean(list(self.spread_history[product_id])[-20:])
                            if avg_spread > 80:  # 0.8%
                                logger.warning(f"{product_id}: High average spread {avg_spread:.1f} bps")
                    
                    except Exception as e:
                        logger.debug(f"Error updating {product_id}: {e}")
                        self.errors_caught += 1
            else:
                # Simulation mode - generate realistic market data
                self._simulate_market_data()
            
        except Exception as e:
            logger.error(f"Market data update error: {e}")
            self.errors_caught += 1
    
    def _simulate_market_data(self):
        """Generate simulated market data for testing"""
        base_prices = {
            "BTC-USD": 43000,
            "ETH-USD": 2200,
            "SOL-USD": 100,
            "DOGE-USD": 0.08,
            "AVAX-USD": 35
        }
        
        for product_id in TRADING_PAIRS:
            snapshot = self.market_snapshots[product_id]
            base_price = base_prices[product_id]
            
            # Add some random walk
            if snapshot.mid == 0:
                snapshot.mid = base_price
            else:
                change = np.random.normal(0, 0.001)  # 0.1% standard deviation
                snapshot.mid *= (1 + change)
            
            # Generate realistic bid/ask
            spread_bps = np.random.uniform(30, 80)  # 0.3% to 0.8% spread
            half_spread = (snapshot.mid * spread_bps / 20000)
            
            snapshot.bid = snapshot.mid - half_spread
            snapshot.ask = snapshot.mid + half_spread
            snapshot.spread_bps = spread_bps
            
            # Other market data
            snapshot.volume_24h = np.random.uniform(1000000, 10000000)
            snapshot.price_change_24h = np.random.uniform(-3, 3)
            snapshot.volatility_1h = np.random.uniform(0.001, 0.02)
            
            # Store price history
            self.price_history[product_id].append({
                'price': snapshot.mid,
                'timestamp': datetime.now(timezone.utc)
            })
            self.spread_history[product_id].append(snapshot.spread_bps)
            
            snapshot.last_update = datetime.now(timezone.utc)
    
    def _calculate_indicators(self, product_id: str) -> Dict:
        """Calculate technical indicators"""
        if len(self.price_history[product_id]) < MIN_PRICE_HISTORY:
            return {}
        
        prices = [p['price'] for p in list(self.price_history[product_id])]
        
        # Simple Moving Averages
        sma_fast = np.mean(prices[-SMA_FAST:]) if len(prices) >= SMA_FAST else 0
        sma_slow = np.mean(prices[-SMA_SLOW:]) if len(prices) >= SMA_SLOW else 0
        
        # RSI
        if len(prices) >= 14:
            price_changes = np.diff(prices[-15:])
            gains = price_changes[price_changes > 0]
            losses = -price_changes[price_changes < 0]
            
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100 if avg_gain > 0 else 50
        else:
            rsi = 50
        
        return {
            'sma_fast': sma_fast,
            'sma_slow': sma_slow,
            'rsi': rsi,
            'price': prices[-1]
        }
    
    def _evaluate_trading_signals(self) -> List[TradingSignal]:
        """Generate trading signals"""
        signals = []
        
        for product_id in TRADING_PAIRS:
            try:
                snapshot = self.market_snapshots[product_id]
                
                # Skip if no recent data
                if (datetime.now(timezone.utc) - snapshot.last_update).seconds > 5:
                    continue
                
                # Calculate indicators
                indicators = self._calculate_indicators(product_id)
                if not indicators:
                    self.signal_rejections[product_id].append(f"Insufficient price history (<{MIN_PRICE_HISTORY})")
                    continue
                
                # Check position limits
                current_position = self.positions.get(product_id, {}).get('size', 0)
                position_value = current_position * snapshot.mid
                
                if position_value >= MAX_POSITION_SIZE:
                    self.signal_rejections[product_id].append(f"Position limit reached (${position_value:.2f})")
                    continue
                
                # Dynamic spread threshold based on time and volatility
                spread_threshold = self._get_dynamic_spread_threshold(product_id, snapshot)
                
                # Check spread
                if snapshot.spread_bps > spread_threshold:
                    self.signal_rejections[product_id].append(f"Spread too high ({snapshot.spread_bps:.1f} > {spread_threshold:.1f} bps)")
                    continue
                
                # Generate signals based on indicators
                sma_fast = indicators['sma_fast']
                sma_slow = indicators['sma_slow']
                rsi = indicators['rsi']
                current_price = indicators['price']
                
                confidence = 0.0
                reason = ""
                
                # BUY signals
                if current_position < MAX_POSITION_SIZE / snapshot.mid:
                    if (sma_fast > sma_slow and 
                        current_price < sma_fast * 1.001 and  # Price near fast SMA
                        rsi < RSI_BUY_THRESHOLD):
                        
                        confidence = min(0.9, (RSI_BUY_THRESHOLD - rsi) / RSI_BUY_THRESHOLD)
                        reason = f"Bullish: SMA crossover, RSI={rsi:.1f}"
                        signals.append(TradingSignal(product_id, 'buy', confidence, reason))
                    
                    elif rsi < 25 and snapshot.spread_bps < spread_threshold * 0.7:  # Oversold with tight spread
                        confidence = 0.8
                        reason = f"Oversold: RSI={rsi:.1f}, tight spread"
                        signals.append(TradingSignal(product_id, 'buy', confidence, reason))
                
                # SELL signals
                if current_position > 0:
                    position_pnl = ((snapshot.bid - self.positions[product_id]['entry_price']) / 
                                   self.positions[product_id]['entry_price']) * 100
                    
                    if (sma_fast < sma_slow and 
                        current_price > sma_fast * 0.999 and
                        rsi > RSI_SELL_THRESHOLD):
                        
                        confidence = min(0.9, (rsi - RSI_SELL_THRESHOLD) / (100 - RSI_SELL_THRESHOLD))
                        reason = f"Bearish: SMA crossover, RSI={rsi:.1f}"
                        signals.append(TradingSignal(product_id, 'sell', confidence, reason))
                    
                    elif rsi > 75 and snapshot.spread_bps < spread_threshold * 0.7:  # Overbought with tight spread
                        confidence = 0.8
                        reason = f"Overbought: RSI={rsi:.1f}, tight spread"
                        signals.append(TradingSignal(product_id, 'sell', confidence, reason))
                    
                    elif position_pnl > 1.0:  # Take profit at 1%
                        confidence = 0.7
                        reason = f"Take profit: +{position_pnl:.2f}%"
                        signals.append(TradingSignal(product_id, 'sell', confidence, reason))
                    
                    elif position_pnl < -0.5:  # Stop loss at -0.5%
                        confidence = 0.9
                        reason = f"Stop loss: {position_pnl:.2f}%"
                        signals.append(TradingSignal(product_id, 'sell', confidence, reason))
                
                # Log why we didn't generate a signal
                if not any(s.product_id == product_id for s in signals):
                    rejection_reason = ""
                    if sma_fast <= sma_slow:
                        rejection_reason = f"No trend: SMA {sma_fast:.2f} <= {sma_slow:.2f}"
                    elif rsi >= RSI_BUY_THRESHOLD and rsi <= RSI_SELL_THRESHOLD:
                        rejection_reason = f"RSI neutral: {rsi:.1f}"
                    
                    if rejection_reason:
                        self.signal_rejections[product_id].append(rejection_reason)
                        
                        # Log rejected signal
                        log_signal({
                            'product_id': product_id,
                            'action': 'none',
                            'confidence': 0,
                            'reason': rejection_reason,
                            'executed': False,
                            'rejection_reason': rejection_reason,
                            'spread_bps': snapshot.spread_bps,
                            'rsi': rsi,
                            'sma_fast': sma_fast,
                            'sma_slow': sma_slow
                        })
                
            except Exception as e:
                logger.error(f"Signal evaluation error for {product_id}: {e}")
                self.errors_caught += 1
        
        return signals
    
    def _get_dynamic_spread_threshold(self, product_id: str, snapshot: MarketSnapshot) -> float:
        """Calculate dynamic spread threshold based on market conditions"""
        current_hour = datetime.now(timezone.utc).hour
        
        # Time-based adjustments (UTC)
        if 2 <= current_hour <= 8:  # Late night/early morning - wider spreads OK
            base_spread = 150
        elif 13 <= current_hour <= 21:  # US market hours - tighter spreads
            base_spread = 50
        else:
            base_spread = 100
        
        # Volatility adjustment
        if snapshot.volatility_1h > 0.02:  # High volatility
            base_spread *= 1.5
        
        # Mode adjustment
        mode_multiplier = self.mode_multipliers.get(self.current_mode, 1.0)
        adjusted_spread = base_spread * (2 - mode_multiplier)  # Aggressive mode tolerates higher spreads
        
        return min(adjusted_spread, MAX_SPREAD_BPS)
    
    def _execute_signal(self, signal: TradingSignal) -> bool:
        """Execute a trading signal"""
        try:
            # Check circuit breaker
            if not self._check_circuit_breaker():
                log_signal({
                    'product_id': signal.product_id,
                    'action': signal.action,
                    'confidence': signal.confidence,
                    'reason': signal.reason,
                    'executed': False,
                    'rejection_reason': 'Circuit breaker active'
                })
                return False
            
            snapshot = self.market_snapshots[signal.product_id]
            
            # Calculate position size
            mode_multiplier = self.mode_multipliers.get(self.current_mode, 1.0)
            position_size_usd = BASE_POSITION_SIZE * mode_multiplier * signal.confidence
            
            # Check capital constraints
            if signal.action == 'buy' and self.capital_remaining < position_size_usd:
                logger.info(f"Insufficient capital for {signal.product_id}: ${position_size_usd:.2f} > ${self.capital_remaining:.2f}")
                log_signal({
                    'product_id': signal.product_id,
                    'action': signal.action,
                    'confidence': signal.confidence,
                    'reason': signal.reason,
                    'executed': False,
                    'rejection_reason': f'Insufficient capital: ${position_size_usd:.2f} > ${self.capital_remaining:.2f}'
                })
                return False
            
            # Paper trading execution
            if PAPER_TRADING or not self.rest_client:
                success = self._execute_paper_trade(signal, position_size_usd)
                if success:
                    logger.info(f"{Fore.GREEN}PAPER TRADE: {signal.action.upper()} {signal.product_id} "
                              f"${position_size_usd:.2f} @ ${snapshot.mid:.2f} - {signal.reason}{Style.RESET_ALL}")
                return success
            
            # Live trading execution
            size = position_size_usd / snapshot.ask if signal.action == 'buy' else self.positions[signal.product_id]['size']
            
            try:
                if signal.action == 'buy':
                    order = self.rest_client.market_order_buy(
                        product_id=signal.product_id,
                        quote_size=str(position_size_usd)
                    )
                else:
                    order = self.rest_client.market_order_sell(
                        product_id=signal.product_id,
                        base_size=str(size)
                    )
                
                # Process executed order
                if order and order.get('status') == 'filled':
                    self._process_executed_order(order, signal)
                    logger.info(f"{Fore.GREEN}LIVE TRADE: {signal.action.upper()} {signal.product_id} "
                              f"${position_size_usd:.2f} - {signal.reason}{Style.RESET_ALL}")
                    return True
                else:
                    logger.warning(f"Order not filled: {order}")
                    return False
                    
            except Exception as e:
                logger.error(f"Order execution error: {e}")
                self.errors_caught += 1
                return False
                
        except Exception as e:
            logger.error(f"Signal execution error: {e}")
            self.errors_caught += 1
            return False
    
    def _execute_paper_trade(self, signal: TradingSignal, position_size_usd: float) -> bool:
        """Execute a simulated paper trade"""
        snapshot = self.market_snapshots[signal.product_id]
        
        if signal.action == 'buy':
            # Simulate buy
            execution_price = snapshot.ask
            size = position_size_usd / execution_price
            
            if signal.product_id in self.positions:
                # Add to existing position
                old_size = self.positions[signal.product_id]['size']
                old_price = self.positions[signal.product_id]['entry_price']
                new_size = old_size + size
                new_price = ((old_size * old_price) + (size * execution_price)) / new_size
                
                self.positions[signal.product_id] = {
                    'size': new_size,
                    'entry_price': new_price
                }
            else:
                # New position
                self.positions[signal.product_id] = {
                    'size': size,
                    'entry_price': execution_price
                }
            
            self.capital_used += position_size_usd
            self.capital_remaining -= position_size_usd
            
        else:  # sell
            if signal.product_id not in self.positions:
                return False
                
            # Simulate sell
            position = self.positions[signal.product_id]
            execution_price = snapshot.bid
            sell_value = position['size'] * execution_price
            
            # Calculate P&L
            cost_basis = position['size'] * position['entry_price']
            pnl = sell_value - cost_basis
            
            # Update state
            self.daily_pnl += pnl
            self.capital_used -= cost_basis
            self.capital_remaining += sell_value
            
            if pnl > 0:
                self.winning_trades += 1
            
            # Remove position
            del self.positions[signal.product_id]
        
        # Record trade
        self.total_trades += 1
        trade_pnl = pnl if signal.action == 'sell' else 0
        
        trade_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'product_id': signal.product_id,
            'action': signal.action,
            'size': size if signal.action == 'buy' else position['size'],
            'price': execution_price,
            'value': position_size_usd if signal.action == 'buy' else sell_value,
            'reason': signal.reason,
            'execution_type': 'PAPER',
            'pnl': trade_pnl,
            'cumulative_pnl': self.daily_pnl,
            'position_size_after': self.positions.get(signal.product_id, {}).get('size', 0),
            'capital_used': self.capital_used,
            'capital_remaining': self.capital_remaining
        }
        
        # Log to CSV
        log_trade(trade_data)
        
        # Add to recent trades for UI
        self.recent_trades.append({
            'timestamp': datetime.now(timezone.utc),
            'product_id': signal.product_id,
            'action': signal.action,
            'size': size if signal.action == 'buy' else position['size'],
            'price': execution_price,
            'value': position_size_usd if signal.action == 'buy' else sell_value,
            'reason': signal.reason
        })
        
        # Update circuit breaker state
        self._update_circuit_breaker(trade_pnl)
        
        # Log successful signal
        log_signal({
            'product_id': signal.product_id,
            'action': signal.action,
            'confidence': signal.confidence,
            'reason': signal.reason,
            'executed': True,
            'rejection_reason': '',
            'spread_bps': snapshot.spread_bps
        })
        
        return True
    
    def _process_executed_order(self, order: Dict, signal: TradingSignal):
        """Process a filled order"""
        # Update positions based on filled order
        filled_size = float(order.get('filled_size', 0))
        avg_price = float(order.get('executed_value', 0)) / filled_size if filled_size > 0 else 0
        
        if signal.action == 'buy':
            if signal.product_id in self.positions:
                # Update existing position
                old_size = self.positions[signal.product_id]['size']
                old_price = self.positions[signal.product_id]['entry_price']
                new_size = old_size + filled_size
                new_price = ((old_size * old_price) + (filled_size * avg_price)) / new_size
                
                self.positions[signal.product_id] = {
                    'size': new_size,
                    'entry_price': new_price
                }
            else:
                # New position
                self.positions[signal.product_id] = {
                    'size': filled_size,
                    'entry_price': avg_price
                }
            
            self.capital_used += float(order.get('executed_value', 0))
            self.capital_remaining -= float(order.get('executed_value', 0))
            
        else:  # sell
            if signal.product_id in self.positions:
                # Calculate realized P&L
                cost_basis = self.positions[signal.product_id]['size'] * self.positions[signal.product_id]['entry_price']
                sell_value = float(order.get('executed_value', 0))
                pnl = sell_value - cost_basis
                
                self.daily_pnl += pnl
                self.capital_used -= cost_basis
                self.capital_remaining += sell_value
                
                if pnl > 0:
                    self.winning_trades += 1
                
                # Remove or reduce position
                del self.positions[signal.product_id]
        
        self.total_trades += 1
        
        # Record trade
        self.recent_trades.append({
            'timestamp': datetime.now(timezone.utc),
            'product_id': signal.product_id,
            'action': signal.action,
            'size': filled_size,
            'price': avg_price,
            'value': float(order.get('executed_value', 0)),
            'reason': signal.reason,
            'order_id': order.get('id')
        })
    
    def _update_ui(self):
        """Update terminal UI with current state"""
        if not self.ui_integration:
            return
            
        try:
            # Update statistics
            win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
            
            stats = {
                'mode': self.current_mode.upper(),
                'daily_pnl': self.daily_pnl,
                'daily_capital_used': self.capital_used,
                'daily_capital_remaining': self.capital_remaining,
                'total_trades': self.total_trades,
                'win_rate': win_rate,
                'errors': self.errors_caught,
                'uptime': int(time.time() - self.start_time),
                'paper_trading': PAPER_TRADING
            }
            self.ui_integration.update_stats(stats)
            
            # Update positions
            position_list = []
            for product_id, position in self.positions.items():
                current_price = self.market_snapshots[product_id].mid
                if current_price > 0:
                    pnl = ((current_price - position['entry_price']) / position['entry_price']) * 100
                    position_list.append({
                        'product': product_id,
                        'size': position['size'],
                        'entry_price': position['entry_price'],
                        'current_price': current_price,
                        'pnl_percent': pnl,
                        'value': position['size'] * current_price
                    })
            self.ui_integration.update_positions(position_list)
            
            # Update market overview
            market_data = {}
            for product_id, snapshot in self.market_snapshots.items():
                market_data[product_id] = {
                    'price': snapshot.mid,
                    'bid': snapshot.bid,
                    'ask': snapshot.ask,
                    'spread_bps': snapshot.spread_bps,
                    'volume_24h': snapshot.volume_24h,
                    'change_24h': snapshot.price_change_24h,
                    'volatility': snapshot.volatility_1h * 100  # Convert to percentage
                }
            self.ui_integration.update_market_overview(market_data)
            
            # Update recent trades
            self.ui_integration.update_trades(list(self.recent_trades))
            
            # Update signal rejections
            self.ui_integration.update_signal_health(dict(self.signal_rejections))
            
            # Update system health
            health = {
                'rest_client': self.rest_client is not None,
                'market_data': all(s.last_update > datetime.now(timezone.utc) - timedelta(seconds=10) 
                                  for s in self.market_snapshots.values()),
                'strategy_engine': True,  # Always true in unified bot
                'ai_enabled': self.openai_client is not None,
                'errors': self.errors_caught,
                'paper_trading': PAPER_TRADING
            }
            self.ui_integration.update_health(health)
            
        except Exception as e:
            logger.error(f"UI update error: {e}")
            self.errors_caught += 1
    
    def _adjust_mode_based_on_performance(self):
        """Dynamically adjust trading mode based on performance"""
        if self.total_trades < 10:
            return  # Not enough data
            
        win_rate = self.winning_trades / self.total_trades
        
        # Performance-based mode adjustment
        if win_rate < 0.4 and self.current_mode != "conservative":
            self.current_mode = "conservative"
            logger.info(f"Switching to CONSERVATIVE mode due to low win rate ({win_rate:.1%})")
        elif win_rate > 0.6 and self.daily_pnl > 50 and self.current_mode != "aggressive":
            self.current_mode = "aggressive"
            logger.info(f"Switching to AGGRESSIVE mode due to high performance")
        elif 0.4 <= win_rate <= 0.6 and self.current_mode != "balanced":
            self.current_mode = "balanced"
            logger.info(f"Switching to BALANCED mode")
        
        # AI-based adjustment if available
        if self.openai_client and np.random.random() < 0.1:  # Check AI recommendation 10% of the time
            self._get_ai_mode_recommendation()
    
    def _get_ai_mode_recommendation(self):
        """Get AI recommendation for trading mode"""
        try:
            # Prepare market context
            market_summary = []
            for product_id, snapshot in self.market_snapshots.items():
                market_summary.append(f"{product_id}: ${snapshot.mid:.2f}, "
                                    f"spread={snapshot.spread_bps:.1f}bps, "
                                    f"vol={snapshot.volatility_1h*100:.2f}%")
            
            prompt = f"""
            Current trading performance:
            - Win rate: {self.winning_trades/max(1,self.total_trades)*100:.1f}%
            - Daily P&L: ${self.daily_pnl:.2f}
            - Total trades: {self.total_trades}
            - Current mode: {self.current_mode}
            
            Market conditions:
            {chr(10).join(market_summary)}
            
            Should I switch trading modes? Reply with only: CONSERVATIVE, BALANCED, or AGGRESSIVE
            """
            
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=10,
                temperature=0.3
            )
            
            recommendation = response.choices[0].text.strip().lower()
            if recommendation in self.mode_multipliers and recommendation != self.current_mode:
                logger.info(f"AI recommends switching to {recommendation.upper()} mode")
                self.current_mode = recommendation
                
        except Exception as e:
            logger.debug(f"AI recommendation error: {e}")
    
    def _preserve_profits(self):
        """Withdraw a portion of daily profits"""
        if self.daily_pnl > MIN_PROFIT_FOR_WITHDRAWAL:
            withdrawal_amount = self.daily_pnl * PROFIT_WITHDRAWAL_PERCENT
            
            logger.info(f"{Fore.GREEN}Preserving profits: Withdrawing ${withdrawal_amount:.2f} "
                       f"({PROFIT_WITHDRAWAL_PERCENT*100:.0f}% of ${self.daily_pnl:.2f}){Style.RESET_ALL}")
            
            # In production, this would transfer to a separate account
            # For now, we just track it
            self.capital_remaining -= withdrawal_amount
            
            # Log the withdrawal
            with open('logs/profit_withdrawals.csv', 'a') as f:
                f.write(f"{datetime.now().isoformat()},{withdrawal_amount:.2f},{self.daily_pnl:.2f}\n")
    
    def run(self):
        """Main bot loop"""
        self.running = True
        self.start_time = time.time()
        last_update = time.time()
        last_signal_check = time.time()
        last_health_check = time.time()
        
        logger.info(f"{Fore.CYAN}Starting Unified ScalperBot main loop...{Style.RESET_ALL}")
        logger.info(f"Mode: {self.current_mode.upper()}, Paper Trading: {PAPER_TRADING}")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        while self.running:
            try:
                current_time = time.time()
                
                # Update market data every second
                if current_time - last_update > 1.0:
                    self._update_market_data()
                    self._update_ui()
                    last_update = current_time
                
                # Check for trading signals every 2 seconds
                if current_time - last_signal_check > 2.0:
                    signals = self._evaluate_trading_signals()
                    
                    # Execute high confidence signals
                    for sig in signals:
                        if sig.confidence > 0.6:  # Only trade high confidence signals
                            success = self._execute_signal(sig)
                            if success:
                                time.sleep(0.5)  # Brief pause between trades
                    
                    last_signal_check = current_time
                
                # Health check and mode adjustment every minute
                if current_time - last_health_check > 60:
                    self._adjust_mode_based_on_performance()
                    
                    # Check if market is closed (simplified - would be more complex in production)
                    current_hour = datetime.now(timezone.utc).hour
                    if current_hour == 21:  # 9 PM UTC = 5 PM ET
                        self._preserve_profits()
                    
                    last_health_check = current_time
                
                # Brief sleep to prevent CPU spinning
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                self.errors_caught += 1
                time.sleep(1)  # Prevent rapid error loops
        
        self._shutdown()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def _shutdown(self):
        """Clean shutdown"""
        logger.info(f"{Fore.YELLOW}Shutting down Unified ScalperBot...{Style.RESET_ALL}")
        
        # Close all positions if configured
        if os.getenv('CLOSE_ON_EXIT', 'false').lower() == 'true' and not PAPER_TRADING:
            self._close_all_positions()
        
        # Save state
        self._save_state()
        
        # Final profit preservation
        if self.daily_pnl > 0:
            self._preserve_profits()
        
        # Stop UI
        if self.ui:
            self.ui.stop()
        
        logger.info(f"{Fore.GREEN}Shutdown complete. Final P&L: ${self.daily_pnl:.2f}{Style.RESET_ALL}")
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker should prevent trading"""
        now = datetime.now(timezone.utc)
        
        # Check if circuit breaker is active and still in effect
        if self.circuit_breaker_active and self.circuit_breaker_until:
            if now < self.circuit_breaker_until:
                return False
            else:
                # Circuit breaker expired
                self.circuit_breaker_active = False
                self.circuit_breaker_until = None
                self.consecutive_losses = 0
                logger.info(f"{Fore.GREEN}Circuit breaker deactivated{Style.RESET_ALL}")
        
        # Check hourly trade limit
        one_hour_ago = now - timedelta(hours=1)
        recent_trades = [t for t in self.trades_this_hour if t > one_hour_ago]
        
        if len(recent_trades) >= MAX_TRADES_PER_HOUR:
            logger.warning(f"{Fore.YELLOW}Hourly trade limit reached: {len(recent_trades)}/{MAX_TRADES_PER_HOUR}{Style.RESET_ALL}")
            return False
        
        # Check consecutive losses
        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            self._activate_circuit_breaker()
            return False
        
        return True
    
    def _update_circuit_breaker(self, trade_pnl: float):
        """Update circuit breaker state after a trade"""
        now = datetime.now(timezone.utc)
        
        # Record trade time
        self.trades_this_hour.append(now)
        
        # Update consecutive losses
        if trade_pnl < 0:
            self.consecutive_losses += 1
            logger.debug(f"Consecutive losses: {self.consecutive_losses}/{MAX_CONSECUTIVE_LOSSES}")
        else:
            self.consecutive_losses = 0
    
    def _activate_circuit_breaker(self):
        """Activate circuit breaker due to excessive losses"""
        self.circuit_breaker_active = True
        self.circuit_breaker_until = datetime.now(timezone.utc) + timedelta(minutes=30)
        
        logger.warning(f"{Fore.RED}CIRCUIT BREAKER ACTIVATED: Too many consecutive losses ({self.consecutive_losses}). "
                      f"Trading suspended until {self.circuit_breaker_until.strftime('%H:%M:%S')}{Style.RESET_ALL}")
    
    def _close_all_positions(self):
        """Close all open positions"""
        for product_id in list(self.positions.keys()):
            signal = TradingSignal(product_id, 'sell', 1.0, "Closing on exit")
            self._execute_signal(signal)
    
    def _save_state(self):
        """Save bot state to file"""
        state = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'mode': self.current_mode,
            'daily_pnl': self.daily_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'positions': self.positions,
            'errors_caught': self.errors_caught
        }
        
        with open('bot_state.json', 'w') as f:
            json.dump(state, f, indent=2)


def main():
    """Entry point"""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Initialize and run bot
        bot = UnifiedScalperBot()
        bot.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 