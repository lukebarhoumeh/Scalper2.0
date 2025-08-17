#!/usr/bin/env python3
"""
ScalperBot Production - The ONE Unified Bot
==========================================
Senior quant-level HFT system with:
- Dynamic AI-driven strategy selection (no manual modes)
- Smart profit preservation with withdrawals
- Bulletproof connection management
- Real-time market adaptation
"""

import asyncio
import logging
import os
import time
import threading
import json
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Any
from collections import deque
import numpy as np

# Core imports
from colorama import init, Fore, Style
init(autoreset=True)

from strategy_engine_production import ProductionStrategyEngine, ProductionStrategy
from trade_executor_production import ProductionTradeExecutor
from trade_logger import TradeLogger
from market_data import get_candles, start_websocket_feed
from risk_manager import HFTRiskManager
from ai_risk_manager import AIRiskManager
from market_intelligence import MarketIntelligence
from market_data_manager import get_market_data_manager
from unified_terminal_ui import UnifiedTerminalUI, TerminalUIIntegration
from risk_calculator import risk_calculator, volatility_manager
from coinbase_client import get_coinbase_client
import openai

# Import new modules
from enhanced_risk_manager import get_enhanced_risk_manager
from advanced_strategies import (
    VWAPMACDStrategy, KeltnerRSIStrategy, ALMAStochasticStrategy,
    combine_strategy_signals
)
from sentiment_analyzer import get_sentiment_analyzer, update_market_sentiment
from advanced_profit_manager import get_profit_manager

# Load config
from config_unified import *
openai.api_key = os.getenv("OPENAI_API_KEY")


class ProductionBot:
    """The ONE production bot - no modes, just intelligence"""
    
    def __init__(self):
        # Logging
        self.logger = self._setup_logging()
        
        # Core components
        self.executor = ProductionTradeExecutor(self.logger)
        self.trade_logger = TradeLogger()
        self.risk_manager = HFTRiskManager()
        self.ai_risk_manager = AIRiskManager()
        self.market_intelligence = MarketIntelligence()
        self.market_data_manager = get_market_data_manager()
        self.coinbase_client = get_coinbase_client()
        
        # NEW: Enhanced components
        self.enhanced_risk_manager = get_enhanced_risk_manager()
        self.profit_manager = get_profit_manager(INITIAL_CAPITAL)
        self.sentiment_analyzer = get_sentiment_analyzer()
        
        # Strategy engine (will be created after initialization)
        self.strategy_engine = None
        
        # Terminal UI
        self.terminal_ui = UnifiedTerminalUI()
        self.ui_integration = TerminalUIIntegration(self.terminal_ui)
        
        # Trading state
        self.initial_capital = INITIAL_CAPITAL
        self.working_capital = INITIAL_CAPITAL
        self.daily_loss = 0.0
        self.max_daily_loss = INITIAL_CAPITAL * MAX_DAILY_LOSS_PERCENT
        self.withdrawn_profit = 0.0
        self.total_profit = 0.0
        self.peak_balance = INITIAL_CAPITAL
        
        # Position tracking
        self.positions = {}  # product_id -> size
        self.position_entry_times = {}  # product_id -> timestamp
        
        # Performance tracking
        self.trades_today = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.errors_today = 0
        self.start_time = datetime.now(timezone.utc)
        
        # Dynamic parameters
        self.current_aggression = 0.5  # 0-1 scale
        self.strategy_weights = {
            'MultiIndicatorScalper': 0.20,
            'MomentumBreakout': 0.20,
            'VWAP_MACD_Scalper': 0.20,
            'Keltner_RSI_Scalper': 0.20,
            'ALMA_Stochastic_Scalper': 0.20
        }
        
        # Connection management
        self.connection_lock = threading.Lock()
        self.active_connections = 0
        self.max_connections = 5
        
        # Shutdown handling
        self._running = True
        self._shutdown = threading.Event()
        
    def _setup_logging(self) -> logging.Logger:
        """Production logging setup with rotation"""
        from logging.handlers import RotatingFileHandler
        
        logger = logging.getLogger("ScalperBotProduction")
        logger.setLevel(getattr(logging, LOG_LEVEL))
        
        # Rotating file handler
        fh = RotatingFileHandler(
            f'logs/production_{datetime.now().strftime("%Y%m%d")}.log',
            maxBytes=LOG_ROTATION_SIZE,
            backupCount=LOG_BACKUP_COUNT
        )
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Detailed formatter
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        fh.setFormatter(detailed_formatter)
        ch.setFormatter(simple_formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        # Also setup logging for key modules
        for module in ['risk_calculator', 'coinbase_client', 'market_data_manager']:
            module_logger = logging.getLogger(module)
            module_logger.setLevel(logging.INFO)
            module_logger.addHandler(fh)
        
        return logger
        
    async def initialize(self):
        """Initialize all systems"""
        print(f"{Fore.CYAN}{'='*60}")
        print(f"SCALPERBOT PRODUCTION - UNIFIED SYSTEM")
        print(f"{'='*60}{Style.RESET_ALL}")
        
        # Start market feeds
        self.logger.info("Starting market data feeds...")
        # Use new WebSocket client
        self.coinbase_client.start_websocket(TRADING_PAIRS)
        
        # CRITICAL FIX: Start legacy WebSocket feed for market data functions
        start_websocket_feed(TRADING_PAIRS)
        self.logger.info("Started legacy WebSocket feed for price data")
        
        await asyncio.sleep(2)
        
        # Initialize AI
        self.logger.info("Initializing AI systems...")
        await self.ai_risk_manager.initialize()
        await self.market_intelligence.initialize()
        
        # Create strategy engine
        self._create_strategy_engine()
        
        # Start UI
        threading.Thread(target=self.terminal_ui.run, daemon=True).start()
        
        print(f"{Fore.GREEN}✓ All systems ready{Style.RESET_ALL}")
        
    def _create_strategy_engine(self):
        """Create and configure strategy engine"""
        self.strategy_engine = ProductionStrategyEngine(self.executor)
        
        # Register existing strategies
        from strategy_engine_production import MultiIndicatorScalper, MomentumBreakoutStrategy
        self.strategy_engine.register(MultiIndicatorScalper(self.executor))
        self.strategy_engine.register(MomentumBreakoutStrategy(self.executor))
        
        # Register NEW advanced strategies
        self.strategy_engine.register(VWAPMACDStrategy(self.executor))
        self.strategy_engine.register(KeltnerRSIStrategy(self.executor))
        self.strategy_engine.register(ALMAStochasticStrategy(self.executor))
        
        # Hook into trades
        self._setup_trade_hooks()
        
        # Start engine
        self.strategy_engine.start()
        self.logger.info("Strategy engine started with 5 strategies")
        
    def _setup_trade_hooks(self):
        """Hook into trading for tracking"""
        original_buy = self.executor.market_buy
        original_sell = self.executor.market_sell
        
        def wrapped_buy(product_id, size, strategy):
            with self.connection_lock:
                if self.active_connections >= self.max_connections:
                    self.logger.warning("Connection limit reached, skipping trade")
                    return None
                self.active_connections += 1
                
            try:
                # Get current price for position sizing
                bid, ask = self.market_data_manager.get_price(product_id)
                current_price = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
                
                if current_price <= 0:
                    self.logger.warning(f"Invalid price for {product_id}, skipping trade")
                    return None
                
                # Get market data for enhanced risk calculation
                candles = get_candles(product_id, granularity=3600, limit=50)
                if not candles or len(candles) < 20:
                    self.logger.warning(f"Insufficient candle data for {product_id}")
                    return None
                    
                # Convert to DataFrame for ATR calculation
                import pandas as pd
                candle_df = pd.DataFrame(candles)
                
                # Calculate ATR and volatility regime
                atr = self.enhanced_risk_manager.calculate_atr(candle_df)
                volatility_regime = self.enhanced_risk_manager.determine_volatility_regime(
                    atr, current_price
                )
                
                # Check sentiment filter
                symbol = product_id.replace('-USD', '')
                if self.sentiment_analyzer.should_filter_signal('buy', symbol):
                    self.logger.info(f"Buy signal filtered by negative sentiment for {symbol}")
                    return None
                
                # Calculate dynamic position size
                position_params = self.enhanced_risk_manager.calculate_position_size(
                    account_equity=self.working_capital,
                    entry_price=current_price,
                    atr=atr,
                    volatility_regime=volatility_regime,
                    confidence=0.7,  # Base confidence, could be passed from strategy
                    strategy_type="scalp"
                )
                
                # Use dynamic size
                adjusted_size = position_params['size'] * current_price  # Convert to USD
                
                # Apply performance-based adjustment
                should_increase, multiplier = self.profit_manager.should_increase_risk()
                if should_increase:
                    adjusted_size *= multiplier
                    self.logger.info(f"Increased position size by {multiplier}x due to strong performance")
                
                # Check position limits
                allowed, reason = risk_calculator.check_position_limits(
                    self.positions,
                    adjusted_size,
                    product_id,
                    self.working_capital
                )
                
                if not allowed:
                    self.logger.warning(f"Position limit check failed: {reason}")
                    return None
                
                # Check spread
                if not volatility_manager.is_spread_acceptable(bid, ask, MAX_SPREAD_PERCENT):
                    self.logger.warning(f"Spread too wide for {product_id}: {volatility_manager.calculate_spread_percent(bid, ask):.4f}")
                    return None
                
                # Execute trade with dynamic size
                result = original_buy(product_id, adjusted_size, strategy)
                if result:
                    # Track position with enhanced risk manager
                    position = self.enhanced_risk_manager.create_position(
                        product_id=product_id,
                        entry_price=current_price,
                        size=adjusted_size / current_price,  # Convert back to base units
                        stop_loss=position_params['stop_loss'],
                        take_profit=position_params['take_profit'],
                        side='long',
                        strategy=strategy
                    )
                    
                    self.positions[product_id] = self.positions.get(product_id, 0) + adjusted_size
                    self.position_entry_times[product_id] = time.time()
                    self.trades_today += 1
                    
                    self.logger.info(f"BUY {product_id}: ${adjusted_size:.2f} @ {current_price:.4f} "
                                   f"[{volatility_regime} vol, SL={position_params['stop_loss']:.4f}, "
                                   f"TP={position_params['take_profit']:.4f}]")
                
                return result
                
            finally:
                with self.connection_lock:
                    self.active_connections -= 1
                    
        def wrapped_sell(product_id, size, strategy):
            with self.connection_lock:
                if self.active_connections >= self.max_connections:
                    self.logger.warning("Connection limit reached, skipping trade")
                    return None
                self.active_connections += 1
                
            try:
                # Get exit price before selling
                bid, ask = self.market_data_manager.get_price(product_id)
                exit_price = bid if bid > 0 else 0  # Sell at bid
                
                result = original_sell(product_id, size, strategy)
                if result:
                    # Calculate P&L before updating position
                    if product_id in self.position_targets:
                        entry_price = self.position_targets[product_id]['entry_price']
                        if entry_price > 0 and exit_price > 0:
                            # Calculate dollar P&L
                            pnl = (exit_price - entry_price) * (size / exit_price)
                            self._update_profit(pnl)
                            
                            # Log the trade result
                            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                            self.logger.info(f"SELL {product_id}: ${size:.2f} @ ${exit_price:.2f} "
                                           f"(P&L: ${pnl:.2f} / {pnl_pct:.2f}%) [{strategy}]")
                    
                    # Update position
                    self.positions[product_id] = max(0, self.positions.get(product_id, 0) - size)
                    
                    # Clean up if position closed
                    if self.positions[product_id] == 0:
                        self.position_entry_times.pop(product_id, None)
                        self.position_targets.pop(product_id, None)
                    
                    self.trades_today += 1
                    
                    # Log to UI
                    self.ui_integration.log_trade({
                        'side': 'SELL',
                        'product': product_id,
                        'size': size,
                        'price': exit_price,
                        'status': 'filled',
                        'strategy': strategy
                    })
                return result
            finally:
                with self.connection_lock:
                    self.active_connections -= 1
                    
        self.executor.market_buy = wrapped_buy
        self.executor.market_sell = wrapped_sell
        
    def _update_profit(self, pnl: float):
        """Update profit and handle withdrawals"""
        self.total_profit += pnl
        self.working_capital += pnl
        
        # Track daily loss
        if pnl < 0:
            self.daily_loss += abs(pnl)
        else:
            self.winning_trades += 1
            
        # Check for profit withdrawal
        available_profit = self.total_profit - self.withdrawn_profit
        if available_profit >= PROFIT_WITHDRAWAL_THRESHOLD:
            win_rate = self.winning_trades / max(1, self.trades_today)
            
            # Dynamic withdrawal based on performance
            if win_rate > 0.7:
                withdrawal_pct = 0.75
            elif win_rate > 0.6:
                withdrawal_pct = 0.5
            else:
                withdrawal_pct = 0.25
                
            withdrawal = available_profit * withdrawal_pct
            
            # Keep minimum working capital
            if self.working_capital - withdrawal >= MIN_WORKING_CAPITAL:
                self.working_capital -= withdrawal
                self.withdrawn_profit += withdrawal
                self.logger.info(f"💰 Profit withdrawn: ${withdrawal:.2f} (Total: ${self.withdrawn_profit:.2f})")
                
    async def _update_strategy_parameters(self):
        """AI-driven parameter updates - no modes!"""
        try:
            # Get current state
            stats = self.executor.get_daily_stats()
            market_vol = await self._get_market_volatility()
            win_rate = self.winning_trades / max(1, self.trades_today)
            
            # Calculate strategy weights based on volatility
            strategy_weights = volatility_manager.calculate_strategy_weights(
                market_vol,
                INITIAL_SCALPING_WEIGHT,
                INITIAL_BREAKOUT_WEIGHT
            )
            
            # AI decision
            prompt = f"""
            Current state:
            - P&L: ${stats.get('pnl', 0):.2f}
            - Win rate: {win_rate:.1%}
            - Market volatility: {market_vol:.1%}
            - Errors: {self.errors_today}
            - Hour: {datetime.now().hour}
            
            Return JSON with:
            - aggression: 0.0-1.0 (0=safe, 1=aggressive)
            - position_multiplier: 0.5-2.0
            - trades_per_hour: 1-20
            - scalping_weight: 0.0-1.0
            - breakout_weight: 0.0-1.0
            """
            
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a quant trading AI. Return only JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=150
                )
                
                params = json.loads(response.choices[0].message.content)
                
                # Apply parameters
                self.current_aggression = params.get('aggression', 0.5)
                self.position_size_multiplier = params.get('position_multiplier', 1.0)
                self.max_trades_per_hour = params.get('trades_per_hour', 10)
                
                # Override strategy weights if AI provides them
                if 'scalping_weight' in params and 'breakout_weight' in params:
                    strategy_weights['scalping'] = params['scalping_weight']
                    strategy_weights['breakout'] = params['breakout_weight']
                
                # Apply strategy weights to engine
                if hasattr(self, 'strategy_engine') and self.strategy_engine:
                    self._apply_strategy_weights(strategy_weights)
                
                # Update risk manager with dynamic limits
                self.risk_manager.max_position_size = (self.working_capital * MAX_TOTAL_EXPOSURE) / 5
                
                self.logger.info(f"Parameters updated: aggression={self.current_aggression:.2f}, "
                               f"scalping={strategy_weights['scalping']:.2f}, "
                               f"breakout={strategy_weights['breakout']:.2f}")
                
            except Exception as e:
                self.logger.error(f"AI update failed: {e}")
                # Safe defaults
                self.current_aggression = 0.3
                self.position_size_multiplier = 0.5
                self.max_trades_per_hour = 5
                
        except Exception as e:
            self.logger.error(f"Parameter update error: {e}")
            
    async def _get_market_volatility(self) -> float:
        """Calculate market volatility"""
        try:
            vols = []
            for pair in TRADING_PAIRS[:3]:  # Use top 3 pairs
                atr = risk_calculator.calculate_atr(pair)
                if atr:
                    # Get current price
                    candles = get_candles(pair, granularity=3600, limit=1)
                    if candles and candles[0]:
                        price = candles[0]['close']
                        volatility = atr / price
                        vols.append(volatility)
                        
            return np.mean(vols) if vols else 0.02
        except Exception as e:
            self.logger.error(f"Failed to calculate market volatility: {e}")
            return 0.02
            
    def _apply_strategy_weights(self, weights: Dict[str, float]):
        """Apply strategy weights to the trading engine"""
        try:
            # This would typically update the strategy engine's internal weights
            # For now, log the weights
            self.logger.info(f"Applying strategy weights: {weights}")
            
            # Store for reference
            self.current_strategy_weights = weights
            
            # Could update strategy engine parameters here if it supports dynamic weighting
            # self.strategy_engine.set_weights(weights)
            
        except Exception as e:
            self.logger.error(f"Failed to apply strategy weights: {e}")
            
    async def _check_position_ages(self):
        """Force close old positions"""
        current_time = time.time()
        max_age = 3600  # 1 hour
        
        for product_id, entry_time in list(self.position_entry_times.items()):
            if current_time - entry_time > max_age:
                size = self.positions.get(product_id, 0)
                if size > 0:
                    self.logger.warning(f"Force closing aged position: {product_id}")
                    self.executor.market_sell(product_id, size, "ForceExit")
                    
    async def _check_stop_loss_take_profit(self):
        """Check and execute stop loss / take profit orders"""
        for product_id, targets in list(self.position_targets.items()):
            if product_id not in self.positions or self.positions[product_id] <= 0:
                continue
                
            # Get current price
            bid, ask = self.market_data_manager.get_price(product_id)
            current_price = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
            
            if current_price <= 0:
                continue
                
            # Check stop loss
            if current_price <= targets['stop_loss']:
                self.logger.info(f"Stop loss triggered for {product_id}: ${current_price:.2f} <= ${targets['stop_loss']:.2f}")
                self.executor.market_sell(product_id, self.positions[product_id], "StopLoss")
                
            # Check take profit
            elif current_price >= targets['take_profit']:
                self.logger.info(f"Take profit triggered for {product_id}: ${current_price:.2f} >= ${targets['take_profit']:.2f}")
                self.executor.market_sell(product_id, self.positions[product_id], "TakeProfit")
                
    def _update_dynamic_risk_limits(self):
        """Update risk limits based on current performance"""
        # Update max daily loss based on current capital
        self.max_daily_loss = self.working_capital * MAX_DAILY_LOSS_PERCENT
        
        # Adjust position size multiplier based on recent performance
        if self.trades_today > 10:
            recent_win_rate = self.winning_trades / self.trades_today
            
            if recent_win_rate < 0.4:
                # Poor performance - reduce risk
                self.position_size_multiplier = max(0.5, self.position_size_multiplier * 0.9)
                self.max_trades_per_hour = max(3, self.max_trades_per_hour - 1)
                self.logger.info(f"Reducing risk: multiplier={self.position_size_multiplier:.2f}, max_trades={self.max_trades_per_hour}")
                
            elif recent_win_rate > 0.65:
                # Good performance - can increase risk slightly
                self.position_size_multiplier = min(2.0, self.position_size_multiplier * 1.1)
                self.max_trades_per_hour = min(20, self.max_trades_per_hour + 1)
                self.logger.info(f"Increasing risk: multiplier={self.position_size_multiplier:.2f}, max_trades={self.max_trades_per_hour}")
                
        # Update risk manager limits
        self.risk_manager.max_position_size = (self.working_capital * MAX_TOTAL_EXPOSURE) / 5  # Divide by max positions
        self.risk_manager.max_daily_trades = self.max_trades_per_hour * 24
                    
    def _update_ui(self):
        """Update terminal UI"""
        try:
            stats = self.executor.get_daily_stats()
            market_overview = self.market_data_manager.get_market_overview()
            
            # Prepare UI data
            ui_stats = {
                'mode': f"AGG={self.current_aggression:.1f}",
                'daily_pnl': stats.get('pnl', 0),
                'daily_capital_used': self.initial_capital - self.working_capital,
                'daily_capital_remaining': self.working_capital,
                'total_trades': self.trades_today,
                'win_rate': self.winning_trades / max(1, self.trades_today),
                'avg_slippage_bps': stats.get('avg_slippage_bps', 0),
                'withdrawn_profit': self.withdrawn_profit,
                'market_overview': market_overview
            }
            
            self.ui_integration.update_stats(ui_stats)
            self.ui_integration.update_positions(self.executor._inventory)
            
            # Health status
            health = {
                'executor_healthy': True,
                'strategy_healthy': (self.strategy_engine._thread and self.strategy_engine._thread.is_alive()) if self.strategy_engine else False,
                'market_data_healthy': len(market_overview) > 0 and any(data.get('price', 0) > 0 for data in market_overview.values()),
                'ai_healthy': True,
                'errors': self.errors_today,
                'uptime': (datetime.now(timezone.utc) - self.start_time).total_seconds(),
                'memory_mb': self._get_memory_usage()
            }
            self.ui_integration.update_health(health)
            
        except Exception as e:
            self.logger.error(f"UI update error: {e}")
            
    def _get_memory_usage(self) -> float:
        """Get memory usage"""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except:
            return 0
            
    async def run(self):
        """Main execution loop"""
        self.logger.info("Starting main trading loop...")
        
        last_param_update = time.time()
        last_position_check = time.time()
        last_risk_update = time.time()
        last_sentiment_update = time.time()
        
        # Start sentiment updates in background
        sentiment_task = asyncio.create_task(
            update_market_sentiment([coin for coin in TRADING_PAIRS])
        )
        
        while self._running and not self._shutdown.is_set():
            try:
                # Check daily loss limit
                if self.daily_loss >= self.max_daily_loss:
                    if not hasattr(self, '_loss_limit_logged'):
                        self.logger.warning(f"Daily loss limit reached: ${self.daily_loss:.2f} >= ${self.max_daily_loss:.2f}")
                        self._loss_limit_logged = True
                    # Stop trading but keep monitoring
                    await asyncio.sleep(5)
                    continue
                    
                # Update risk limits every minute
                if time.time() - last_risk_update > 60:
                    self._update_dynamic_risk_limits()
                    last_risk_update = time.time()
                
                # Update parameters every 5 minutes
                if time.time() - last_param_update > 300:
                    await self._update_strategy_parameters()
                    last_param_update = time.time()
                    
                # Check positions every 30 seconds
                if time.time() - last_position_check > 30:
                    await self._check_positions_enhanced()
                    last_position_check = time.time()
                    
                # Update sentiment-based strategy weights every 2 minutes
                if time.time() - last_sentiment_update > 120:
                    self._update_sentiment_weights()
                    last_sentiment_update = time.time()
                    
                # Update UI
                self._update_ui()
                
                # Update profit manager
                current_balance = self.working_capital + self.total_profit
                self.profit_manager.update_balance(current_balance, self.total_profit)
                
                # Rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                self.errors_today += 1
                self.logger.error(f"Main loop error: {e}")
                
                # Error backoff
                if self.errors_today > 10:
                    self.logger.warning("Too many errors, pausing 60s...")
                    await asyncio.sleep(60)
                    self.errors_today = 0
                    
        # Cancel sentiment task
        sentiment_task.cancel()
        
    async def _check_positions_enhanced(self):
        """Enhanced position checking with trailing stops and time-based exits"""
        try:
            # Get current market data
            market_data = {}
            for product_id in list(self.positions.keys()):
                bid, ask = self.market_data_manager.get_price(product_id)
                price = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
                
                # Get ATR for trailing stop
                candles = get_candles(product_id, granularity=3600, limit=20)
                if candles and len(candles) >= 14:
                    import pandas as pd
                    candle_df = pd.DataFrame(candles)
                    atr = self.enhanced_risk_manager.calculate_atr(candle_df)
                else:
                    atr = 0
                    
                market_data[product_id] = {
                    'price': price,
                    'atr': atr
                }
                
            # Check for exit signals
            exit_signals = self.enhanced_risk_manager.update_positions(market_data)
            
            # Execute exits
            for signal in exit_signals:
                product_id = signal['product_id']
                position = signal['position']
                exit_price = signal['exit_price']
                reason = signal['reason']
                
                if position.size > 0:
                    # Calculate P&L
                    pnl = (exit_price - position.entry_price) * position.size
                    
                    # Execute sell
                    self.executor.market_sell(
                        product_id, 
                        position.size * exit_price,  # USD value
                        f"{position.strategy}_{reason}"
                    )
                    
                    # Update tracking
                    self.enhanced_risk_manager.remove_position(product_id)
                    if product_id in self.positions:
                        del self.positions[product_id]
                    if product_id in self.position_entry_times:
                        del self.position_entry_times[product_id]
                        
                    # Update profit tracking
                    self.total_profit += pnl
                    if pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
                        
                    self.logger.info(f"Position exit {product_id}: {reason} @ {exit_price:.4f}, "
                                   f"P&L: ${pnl:.2f}")
                                   
        except Exception as e:
            self.logger.error(f"Enhanced position check error: {e}")
            
    def _update_sentiment_weights(self):
        """Update strategy weights based on market sentiment"""
        try:
            # Get sentiment for our trading pairs
            symbols = [pair.replace('-USD', '') for pair in TRADING_PAIRS]
            symbol_sentiments = self.sentiment_analyzer.aggregate_sentiment(symbols, hours=1.0)
            
            # Adjust strategy weights
            adjusted_weights = self.sentiment_analyzer.adjust_strategy_weights(
                self.strategy_weights,
                symbol_sentiments
            )
            
            # Apply adjusted weights
            self.strategy_weights = adjusted_weights
            
            # Log changes
            market_sentiment = self.sentiment_analyzer.get_market_sentiment()
            self.logger.info(f"Sentiment update - Market: {market_sentiment['overall']:.2f}, "
                           f"Weights adjusted for {len(symbol_sentiments)} symbols")
                           
        except Exception as e:
            self.logger.error(f"Sentiment weight update error: {e}")
            
    async def shutdown(self):
        """Graceful shutdown with comprehensive report"""
        self.logger.info("Shutting down...")
        self._running = False
        
        # Close all positions
        for product_id, size in list(self.positions.items()):
            if size > 0:
                self.logger.info(f"Closing position: {product_id}")
                self.executor.market_sell(product_id, size, "Shutdown")
                
        # Stop components
        if self.strategy_engine:
            self.strategy_engine.stop()
            
        # Stop WebSocket
        self.coinbase_client.stop_websocket()
            
        # Generate performance report
        self._generate_performance_report()
        
        # Final report
        runtime = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600
        avg_trades_per_hour = self.trades_today / max(1, runtime)
        
        print(f"""
{Fore.CYAN}{'='*60}
FINAL PERFORMANCE REPORT
{'='*60}{Style.RESET_ALL}
Runtime: {runtime:.2f} hours
Total Trades: {self.trades_today}
Trades/Hour: {avg_trades_per_hour:.1f}
Win Rate: {self.winning_trades / max(1, self.trades_today):.1%}
Total Profit: ${self.total_profit:.2f}
Withdrawn: ${self.withdrawn_profit:.2f}
Working Capital: ${self.working_capital:.2f}
Daily Loss: ${self.daily_loss:.2f}
Max Daily Loss: ${self.max_daily_loss:.2f}
{Fore.CYAN}{'='*60}{Style.RESET_ALL}
        """)
        
        # Stop UI
        self.terminal_ui.stop()
        
    def _generate_performance_report(self):
        """Generate detailed performance report"""
        try:
            report = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'runtime_hours': (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600,
                'total_trades': self.trades_today,
                'winning_trades': self.winning_trades,
                'win_rate': self.winning_trades / max(1, self.trades_today),
                'total_profit': self.total_profit,
                'withdrawn_profit': self.withdrawn_profit,
                'working_capital': self.working_capital,
                'daily_loss': self.daily_loss,
                'max_daily_loss': self.max_daily_loss,
                'errors_today': self.errors_today,
                'final_aggression': self.current_aggression,
                'final_position_multiplier': self.position_size_multiplier,
                'final_max_trades_per_hour': self.max_trades_per_hour,
                'trading_pairs': TRADING_PAIRS,
                'risk_percent': RISK_PERCENT,
                'max_total_exposure': MAX_TOTAL_EXPOSURE
            }
            
            # Get trade history from executor
            if hasattr(self.executor, 'get_trade_history'):
                report['trade_history'] = self.executor.get_trade_history()
                
            # Save report
            filename = f"logs/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            self.logger.info(f"Performance report saved to {filename}")
            
            # Also save to CSV for analysis
            if self.trades_today > 0:
                summary_data = {
                    'Date': datetime.now().strftime('%Y-%m-%d'),
                    'Runtime_Hours': report['runtime_hours'],
                    'Total_Trades': report['total_trades'],
                    'Win_Rate': report['win_rate'],
                    'Total_Profit': report['total_profit'],
                    'Withdrawn': report['withdrawn_profit'],
                    'Final_Capital': report['working_capital']
                }
                
                # Append to summary CSV
                import csv
                csv_file = 'logs/performance_summary.csv'
                file_exists = os.path.exists(csv_file)
                
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=summary_data.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(summary_data)
                    
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
        

async def main():
    """Entry point"""
    bot = ProductionBot()
    
    try:
        await bot.initialize()
        await bot.run()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Shutdown requested{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Fatal error: {e}{Style.RESET_ALL}")
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    # Ensure logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Check environment
    if not os.path.exists(".env"):
        print(f"{Fore.RED}ERROR: .env file not found!{Style.RESET_ALL}")
        exit(1)
        
    # Run
    asyncio.run(main()) 