#!/usr/bin/env python3
"""
Unified Profit Manager for ScalperBot 2.0
Handles profit preservation, tracking, and dynamic capital scaling
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Any
import json
import os

class UnifiedProfitManager:
    """Manages profit preservation and capital scaling"""
    
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.logger = logging.getLogger("UnifiedProfitManager")
        
        # Profit tracking
        self.daily_profit = 0.0
        self.session_profit = 0.0
        self.peak_profit = 0.0
        self.drawdown_limit = 0.2  # 20% drawdown from peak
        
        # Capital scaling
        self.initial_capital = 1000.0
        self.current_capital = self.initial_capital
        self.scaling_enabled = True
        
        # Profit preservation thresholds
        self.profit_lock_threshold = 50.0  # Lock in profits after $50
        self.profit_lock_percentage = 0.5  # Lock 50% of profits
        
        # State tracking
        self.trades_today = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
    def update_profit(self, pnl: float):
        """Update profit tracking"""
        self.daily_profit += pnl
        self.session_profit += pnl
        
        # Update peak profit
        if self.session_profit > self.peak_profit:
            self.peak_profit = self.session_profit
            
        # Update trade counts
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        self.trades_today += 1
        
        # Check profit preservation
        self._check_profit_preservation()
        
    def _check_profit_preservation(self):
        """Check if we need to preserve profits"""
        # Check drawdown from peak
        if self.peak_profit > 0:
            current_drawdown = (self.peak_profit - self.session_profit) / self.peak_profit
            if current_drawdown > self.drawdown_limit:
                self.logger.warning(f"Drawdown limit reached: {current_drawdown:.1%}")
                self._trigger_profit_preservation()
                
        # Check profit lock threshold
        if self.session_profit >= self.profit_lock_threshold:
            locked_amount = self.session_profit * self.profit_lock_percentage
            self.logger.info(f"Profit threshold reached. Locking ${locked_amount:.2f}")
            
    def _trigger_profit_preservation(self):
        """Trigger profit preservation mode"""
        # Reduce position sizes
        if hasattr(self.bot, 'risk_manager'):
            self.bot.risk_manager.max_position_size *= 0.5
            
        # Switch to conservative mode
        if hasattr(self.bot, 'current_mode'):
            self.bot.current_mode = 'conservative'
            
        self.logger.info("Profit preservation mode activated")
        
    def scale_capital(self) -> float:
        """Dynamic capital scaling based on performance"""
        if not self.scaling_enabled:
            return self.current_capital
            
        # Calculate win rate
        total_trades = self.winning_trades + self.losing_trades
        if total_trades > 10:  # Need minimum trades
            win_rate = self.winning_trades / total_trades
            
            # Scale up if performing well
            if win_rate > 0.6 and self.daily_profit > 20:
                scaling_factor = 1.1  # 10% increase
            elif win_rate > 0.55:
                scaling_factor = 1.05  # 5% increase
            elif win_rate < 0.45:
                scaling_factor = 0.9  # 10% decrease
            else:
                scaling_factor = 1.0  # No change
                
            self.current_capital = min(
                self.current_capital * scaling_factor,
                self.initial_capital * 2  # Max 2x initial
            )
            
        return self.current_capital
        
    def get_stats(self) -> Dict[str, Any]:
        """Get current profit statistics"""
        total_trades = self.winning_trades + self.losing_trades
        win_rate = self.winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'daily_profit': self.daily_profit,
            'session_profit': self.session_profit,
            'peak_profit': self.peak_profit,
            'current_drawdown': (self.peak_profit - self.session_profit) / self.peak_profit if self.peak_profit > 0 else 0,
            'trades_today': self.trades_today,
            'win_rate': win_rate,
            'current_capital': self.current_capital,
            'profit_preservation_active': self.bot.current_mode == 'conservative' if hasattr(self.bot, 'current_mode') else False
        }
        
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_profit = 0.0
        self.trades_today = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.logger.info("Daily stats reset")


def integrate_profit_manager(bot_instance) -> UnifiedProfitManager:
    """Factory function to integrate profit manager with bot"""
    profit_manager = UnifiedProfitManager(bot_instance)
    
    # Hook into trade execution
    if hasattr(bot_instance, 'executor') and bot_instance.executor:
        # Store original methods
        original_buy = bot_instance.executor.market_buy
        original_sell = bot_instance.executor.market_sell
        
        def wrapped_buy(product_id, usd_notional, strategy):
            result = original_buy(product_id, usd_notional, strategy)
            # Note: We'll update profit after trade completion
            return result
            
        def wrapped_sell(product_id, usd_notional, strategy):
            result = original_sell(product_id, usd_notional, strategy)
            # Note: We'll update profit after trade completion
            return result
            
        bot_instance.executor.market_buy = wrapped_buy
        bot_instance.executor.market_sell = wrapped_sell
        
        # Also hook into daily stats to track P&L
        if hasattr(bot_instance.executor, 'get_daily_stats'):
            original_stats = bot_instance.executor.get_daily_stats
            
            def wrapped_stats():
                stats = original_stats()
                if 'pnl' in stats:
                    # Update profit manager with current P&L
                    current_pnl = stats['pnl']
                    if current_pnl != profit_manager.session_profit:
                        pnl_diff = current_pnl - profit_manager.session_profit
                        profit_manager.update_profit(pnl_diff)
                return stats
                
            bot_instance.executor.get_daily_stats = wrapped_stats
    
    return profit_manager 