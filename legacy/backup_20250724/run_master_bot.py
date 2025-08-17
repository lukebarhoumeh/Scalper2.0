#!/usr/bin/env python3
"""
run_master_bot.py - UNIFIED PRODUCTION HFT BOT
=====================================
The ONE bot to rule them all. Combines:
- Production-grade risk management
- AI-powered dynamic strategy selection
- Self-healing error recovery
- Auto-restart capabilities
- Real-time performance monitoring
- Unified terminal UI
"""

import sys
import os
import time
import json
import signal
import logging
import asyncio
import argparse
import threading
from datetime import datetime, timezone
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np

# Colorful terminal output
import colorama
from colorama import Fore, Style, Back
colorama.init()

# Import all necessary components
from strategy_engine_production import ProductionStrategyEngine
from trade_executor_production import ProductionTradeExecutor
from trade_logger import TradeLogger
from monitoring import TelegramNotifier
from enhanced_output_formatter import EnhancedFormatter as EnhancedOutputFormatter
from market_data import start_websocket_feed, USE_WS_FEED
from risk_manager import HFTRiskManager as RiskManager
from config_production_hft import *
from unified_terminal_ui import UnifiedTerminalUI, TerminalUIIntegration

# AI Modules for dynamic risk management
from ai_risk_manager import AIRiskManager
from market_intelligence import MarketIntelligence

class UnifiedMasterBot:
    """
    The ultimate trading bot that combines all features:
    - Dynamic AI-powered risk management
    - Self-healing capabilities
    - Unified monitoring
    - Production-grade execution
    """
    
    def __init__(self):
        # Setup logging FIRST
        self.setup_logger = self._setup_logging()
        self.formatter = EnhancedOutputFormatter()
        
        # Terminal UI
        self.terminal_ui = UnifiedTerminalUI()
        self.ui_integration = TerminalUIIntegration(self.terminal_ui)
        
        # Core components
        self.trade_logger = TradeLogger()
        
        # Create a proper logger for the executor (not TradeLogger)
        executor_logger = logging.getLogger("ProductionTradeExecutor")
        self.executor = ProductionTradeExecutor(executor_logger)
        
        # Give executor access to trade logger for CSV logging
        self.executor._trade_logger = self.trade_logger
        
        self.risk_manager = RiskManager()
        
        # Use setup_logger for this class, NOT trade_logger
        self.logger = self.setup_logger
        
        # AI components
        self.ai_risk_manager = AIRiskManager()
        self.market_intelligence = MarketIntelligence()
        
        # Strategy engine with AI integration
        self.strategy_engine = None
        self.notifier = None
        
        # Profit preservation system
        self.profit_manager = None
        
        # Performance tracking
        self.performance_history = []
        self.error_count = 0
        self.last_error_time = None
        self.consecutive_errors = 0
        
        # Self-healing state
        self.health_check_interval = 60  # seconds
        self.last_health_check = time.time()
        self.recovery_attempts = 0
        
        # Dynamic mode management
        self.current_mode = "conservative"  # conservative, balanced, aggressive
        self.mode_switch_threshold = {
            "profit_threshold": 50,    # Switch to aggressive if daily P&L > $50
            "loss_threshold": -30,     # Switch to conservative if daily P&L < -$30
            "error_threshold": 3       # Switch to conservative if errors > 3
        }
        
        # Shutdown handling
        self._shutdown = threading.Event()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup unified logging with proper formatting"""
        logger = logging.getLogger("UnifiedMasterBot")
        logger.setLevel(logging.INFO)
        
        # Console handler with color
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(
            f"logs/unified_bot_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console)
        logger.addHandler(file_handler)
        
        return logger
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"\n{Fore.YELLOW}Received signal {signum}. Shutting down gracefully...{Style.RESET_ALL}")
        self._shutdown.set()
    
    async def initialize(self):
        """Initialize all components with proper error handling"""
        try:
            # Display startup banner
            self.formatter.display_startup_banner()
            
            # Initialize WebSocket feed
            if USE_WS_FEED:
                self.logger.info(f"{Fore.CYAN}Starting WebSocket feed...{Style.RESET_ALL}")
                start_websocket_feed(TRADE_COINS)
            
            # Initialize notifications
            if ENABLE_NOTIFICATIONS and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                self.notifier = TelegramNotifier(
                    bot_token=TELEGRAM_BOT_TOKEN,
                    chat_id=TELEGRAM_CHAT_ID
                )
                await self.notifier.send_startup_message()
            else:
                self.logger.info(f"{Fore.YELLOW}Telegram notifications disabled{Style.RESET_ALL}")
            
            # Initialize AI components
            self.logger.info(f"{Fore.CYAN}Initializing AI risk management...{Style.RESET_ALL}")
            await self.ai_risk_manager.initialize()
            await self.market_intelligence.initialize()
            
            # Create strategy engine with dynamic configuration
            self._create_strategy_engine()
            
            # Initialize profit preservation system
            from unified_profit_manager import integrate_profit_manager
            self.profit_manager = integrate_profit_manager(self)
            
            self.logger.info(f"{Fore.GREEN}[OK] All systems initialized successfully{Style.RESET_ALL}")
            
        except Exception as e:
            self.logger.error(f"{Fore.RED}Initialization failed: {e}{Style.RESET_ALL}")
            raise
    
    def _create_strategy_engine(self):
        """Create strategy engine with current mode settings"""
        # Get AI-recommended configuration
        ai_config = self.ai_risk_manager.get_optimal_config(
            self.current_mode,
            self.performance_history
        )
        
        # Create strategy engine (only takes executor)
        self.strategy_engine = ProductionStrategyEngine(self.executor)
        
        # Configure risk limits based on mode
        self._apply_mode_settings(ai_config)
        
        # Hook into strategy engine for UI updates
        self._setup_strategy_hooks()
        
        # Start the strategy engine
        self.strategy_engine.start()
        self.logger.info(f"{Fore.GREEN}Strategy engine started{Style.RESET_ALL}")
    
    def _apply_mode_settings(self, config: Dict):
        """Apply mode-specific settings"""
        self.logger.info(f"{Fore.YELLOW}Applying {self.current_mode} mode settings{Style.RESET_ALL}")
        
        # Update risk limits
        if self.current_mode == "conservative":
            self.risk_manager.max_position_size = MIN_POSITION_SIZE * 1.5
            self.risk_manager.max_daily_trades = 10
        elif self.current_mode == "balanced":
            self.risk_manager.max_position_size = (MIN_POSITION_SIZE + MAX_POSITION_SIZE) / 2
            self.risk_manager.max_daily_trades = 20
        else:  # aggressive
            self.risk_manager.max_position_size = MAX_POSITION_SIZE
            self.risk_manager.max_daily_trades = 50
    
    def _setup_strategy_hooks(self):
        """Setup hooks to capture signals and trades for UI"""
        if self.strategy_engine:
            # Override strategy engine's signal method to capture for UI
            original_process = self.strategy_engine._process_signal if hasattr(self.strategy_engine, '_process_signal') else None
            
            def capture_signal(signal):
                # Log signal to UI
                self.ui_integration.log_signal({
                    'side': signal.get('action', 'UNKNOWN'),
                    'product': signal.get('coin', 'UNKNOWN') + '-USD',
                    'size': signal.get('size_usd', 0),
                    'strategy': signal.get('strategy', 'Unknown'),
                    'confidence': signal.get('confidence', 0) * 100
                })
                # Call original if exists
                if original_process:
                    return original_process(signal)
            
            if hasattr(self.strategy_engine, '_process_signal'):
                self.strategy_engine._process_signal = capture_signal
    
    async def run(self):
        """Main execution loop with self-healing capabilities"""
        self.logger.info(f"{Fore.GREEN}Starting unified master bot...{Style.RESET_ALL}")
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        
        # Main trading loop
        while not self._shutdown.is_set():
            try:
                # Health check
                if time.time() - self.last_health_check > self.health_check_interval:
                    await self._perform_health_check()
                
                # Check if mode switch is needed
                await self._check_mode_switch()
                
                # Run strategy engine
                if self.strategy_engine and self.strategy_engine.is_healthy():
                    # Let strategy engine run its own loop
                    await asyncio.sleep(1)
                else:
                    # Attempt recovery
                    await self._attempt_recovery()
                
                # Reset error counter on successful iteration
                self.consecutive_errors = 0
                
            except Exception as e:
                self.consecutive_errors += 1
                self.error_count += 1
                self.last_error_time = time.time()
                
                self.logger.error(f"{Fore.RED}Main loop error ({self.consecutive_errors}): {e}{Style.RESET_ALL}")
                
                # Self-healing: attempt recovery
                if self.consecutive_errors >= 3:
                    await self._attempt_recovery()
                
                # Prevent tight error loops
                await asyncio.sleep(5)
    
    async def _perform_health_check(self):
        """Comprehensive system health check"""
        self.last_health_check = time.time()
        
        health_status = {
            'timestamp': datetime.now(timezone.utc),
            'executor_healthy': self._check_executor_health(),
            'strategy_healthy': self._check_strategy_health(),
            'market_data_healthy': self._check_market_data_health(),
            'ai_systems_healthy': await self._check_ai_health(),
            'memory_usage': self._get_memory_usage(),
            'error_rate': self.error_count / max(1, time.time() - self.last_health_check)
        }
        
        # Log health status
        if all(v for k, v in health_status.items() if k.endswith('_healthy')):
            self.logger.info(f"{Fore.GREEN}[OK] Health check passed{Style.RESET_ALL}")
        else:
            self.logger.warning(f"{Fore.YELLOW}[WARNING] Health issues detected: {health_status}{Style.RESET_ALL}")
            
            # Notify if critical
            if self.notifier and not health_status['executor_healthy']:
                await self.notifier.send_alert("Critical: Executor unhealthy!")
    
    async def _check_mode_switch(self):
        """Check if trading mode should be switched based on performance"""
        stats = self.executor.get_daily_stats()
        daily_pnl = stats['daily_pnl']
        
        # Get AI recommendation
        ai_recommendation = await self.ai_risk_manager.recommend_mode(
            current_pnl=daily_pnl,
            error_count=self.consecutive_errors,
            market_conditions=await self.market_intelligence.get_market_conditions()
        )
        
        # Extract mode from recommendation (it's a dict)
        if isinstance(ai_recommendation, dict):
            recommended_mode = ai_recommendation.get('mode', 'balanced')
        else:
            recommended_mode = ai_recommendation
            
        new_mode = self.current_mode
        
        # Logic for mode switching
        if recommended_mode != self.current_mode:
            new_mode = recommended_mode
        elif daily_pnl > self.mode_switch_threshold['profit_threshold'] and self.current_mode != "aggressive":
            new_mode = "aggressive"
        elif daily_pnl < self.mode_switch_threshold['loss_threshold'] and self.current_mode != "conservative":
            new_mode = "conservative"
        elif self.consecutive_errors > self.mode_switch_threshold['error_threshold']:
            new_mode = "conservative"
        
        # Switch mode if needed
        if new_mode != self.current_mode:
            self.logger.info(f"{Fore.YELLOW}Switching from {self.current_mode} to {new_mode} mode{Style.RESET_ALL}")
            self.current_mode = new_mode
            
            # Recreate strategy engine with new settings
            if self.strategy_engine:
                self.strategy_engine.stop()
            self._create_strategy_engine()
            
            # Notify
            if self.notifier:
                await self.notifier.send_message(f"Mode switched to {new_mode} (P&L: ${daily_pnl:.2f})")
    
    async def _attempt_recovery(self):
        """Attempt to recover from errors"""
        self.recovery_attempts += 1
        self.logger.warning(f"{Fore.YELLOW}Attempting recovery (attempt #{self.recovery_attempts})...{Style.RESET_ALL}")
        
        try:
            # Stop current components
            if self.strategy_engine:
                self.strategy_engine.stop()
            
            # Wait for cleanup
            await asyncio.sleep(5)
            
            # Switch to conservative mode for safety
            self.current_mode = "conservative"
            
            # Reinitialize components
            await self.initialize()
            
            # Start fresh strategy engine
            self._create_strategy_engine()
            # No need to start here, _create_strategy_engine() now starts it
            
            self.logger.info(f"{Fore.GREEN}[OK] Recovery successful{Style.RESET_ALL}")
            self.consecutive_errors = 0
            
        except Exception as e:
            self.logger.error(f"{Fore.RED}Recovery failed: {e}{Style.RESET_ALL}")
            
            # If multiple recovery attempts fail, wait longer
            if self.recovery_attempts > 3:
                wait_time = min(300, 60 * self.recovery_attempts)  # Max 5 minutes
                self.logger.info(f"{Fore.YELLOW}Waiting {wait_time}s before next recovery attempt...{Style.RESET_ALL}")
                await asyncio.sleep(wait_time)
    
    def _monitor_loop(self):
        """Background monitoring loop for terminal UI"""
        # Start UI in separate thread
        ui_thread = threading.Thread(target=self.terminal_ui.run, daemon=True)
        ui_thread.start()
        
        while not self._shutdown.is_set():
            try:
                # Get current stats
                stats = self.executor.get_daily_stats()
                stats['mode'] = self.current_mode
                
                # Update UI with all data
                self.ui_integration.update_stats(stats)
                self.ui_integration.update_positions(stats.get('positions', {}))
                self.ui_integration.update_health({
                    'executor_healthy': self._check_executor_health(),
                    'strategy_healthy': self._check_strategy_health(),
                    'market_data_healthy': self._check_market_data_health(),
                    'ai_healthy': True,  # Simplified for now
                    'errors': self.consecutive_errors,
                    'uptime': time.time() - (self.last_error_time or time.time()),
                    'memory_mb': self._get_memory_usage()
                })
                
                # Update market data
                market_update = {}
                for coin in ['BTC', 'ETH', 'SOL']:
                    try:
                        from market_data import get_best_bid_ask
                        bid, ask = get_best_bid_ask(f"{coin}-USD")
                        market_update[f"{coin}-USD"] = {
                            'price': (bid + ask) / 2,
                            '24h_change': 0,  # Would need real data
                            'volume': 0
                        }
                    except:
                        pass
                self.ui_integration.update_market(market_update)
                
                # Sleep based on update frequency
                time.sleep(1)
                
            except Exception as e:
                self.logger.debug(f"Monitor error: {e}")
                time.sleep(5)
    
    def _check_executor_health(self) -> bool:
        """Check if executor is healthy"""
        try:
            stats = self.executor.get_daily_stats()
            return stats is not None
        except:
            return False
    
    def _check_strategy_health(self) -> bool:
        """Check if strategy engine is healthy"""
        return self.strategy_engine is not None and hasattr(self.strategy_engine, 'is_healthy') and self.strategy_engine.is_healthy()
    
    def _check_market_data_health(self) -> bool:
        """Check if market data feed is healthy"""
        try:
            from market_data import get_best_bid_ask
            bid, ask = get_best_bid_ask("BTC-USD")
            return bid > 0 and ask > 0
        except:
            return False
    
    async def _check_ai_health(self) -> bool:
        """Check if AI systems are healthy"""
        try:
            return await self.ai_risk_manager.health_check() and await self.market_intelligence.health_check()
        except:
            return False
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info(f"{Fore.YELLOW}Shutting down unified master bot...{Style.RESET_ALL}")
        
        # Stop strategy engine
        if self.strategy_engine:
            self.strategy_engine.stop()
        
        # Send shutdown notification
        if self.notifier:
            stats = self.executor.get_daily_stats()
            await self.notifier.send_shutdown_message(stats)
        
        # Save performance history
        self._save_performance_history()
        
        # Stop terminal UI
        self.terminal_ui.stop()
        
        self.logger.info(f"{Fore.GREEN}[OK] Shutdown complete{Style.RESET_ALL}")
    
    def _save_performance_history(self):
        """Save performance history to file"""
        try:
            import json
            filename = f"logs/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump({
                    'mode': self.current_mode,
                    'performance': list(self.performance_history),
                    'final_stats': self.executor.get_daily_stats()
                }, f, indent=2, default=str)
            self.logger.info(f"Performance history saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save performance history: {e}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Unified Master Trading Bot')
    parser.add_argument('--mode', choices=['conservative', 'balanced', 'aggressive'], 
                      default='balanced', help='Starting trading mode')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()
    
    # Create and run bot
    bot = UnifiedMasterBot()
    bot.current_mode = args.mode
    
    try:
        await bot.initialize()
        
        # Strategy engine is already started in _create_strategy_engine()
        
        # Run main loop
        await bot.run()
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Fatal error: {e}{Style.RESET_ALL}")
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Run the bot
    asyncio.run(main()) 