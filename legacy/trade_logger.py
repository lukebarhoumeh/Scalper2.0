#!/usr/bin/env python3
"""
Trade Logger - CSV-based trade recording system
=============================================
Logs all trades with comprehensive details for analysis
"""

import csv
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import threading
import logging

logger = logging.getLogger(__name__)


class TradeLogger:
    """
    CSV-based trade logger with thread safety
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.lock = threading.Lock()
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate daily log filename
        today = datetime.now().strftime("%Y%m%d")
        self.trades_file = os.path.join(log_dir, f"trades_{today}.csv")
        self.signals_file = os.path.join(log_dir, f"signals_{today}.csv")
        
        # Initialize CSV files if they don't exist
        self._initialize_csv_files()
        
        logger.info(f"TradeLogger initialized: {self.trades_file}")
    
    def _initialize_csv_files(self):
        """Initialize CSV files with headers if they don't exist"""
        # Trade log headers
        trade_headers = [
            'timestamp', 'product_id', 'action', 'size', 'price', 'value',
            'reason', 'execution_type', 'pnl', 'cumulative_pnl',
            'position_size_after', 'capital_used', 'capital_remaining'
        ]
        
        # Signal log headers
        signal_headers = [
            'timestamp', 'product_id', 'action', 'confidence', 'reason',
            'executed', 'rejection_reason', 'spread_bps', 'rsi', 'sma_fast', 'sma_slow'
        ]
        
        with self.lock:
            # Initialize trades CSV
            if not os.path.exists(self.trades_file):
                with open(self.trades_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(trade_headers)
            
            # Initialize signals CSV
            if not os.path.exists(self.signals_file):
                with open(self.signals_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(signal_headers)
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """
        Log a completed trade to CSV
        
        Args:
            trade_data: Dictionary containing trade information
        """
        try:
            with self.lock:
                with open(self.trades_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    
                    row = [
                        trade_data.get('timestamp', datetime.now(timezone.utc).isoformat()),
                        trade_data.get('product_id', ''),
                        trade_data.get('action', ''),
                        trade_data.get('size', 0),
                        trade_data.get('price', 0),
                        trade_data.get('value', 0),
                        trade_data.get('reason', ''),
                        trade_data.get('execution_type', 'PAPER'),
                        trade_data.get('pnl', 0),
                        trade_data.get('cumulative_pnl', 0),
                        trade_data.get('position_size_after', 0),
                        trade_data.get('capital_used', 0),
                        trade_data.get('capital_remaining', 0)
                    ]
                    
                    writer.writerow(row)
                    
            logger.debug(f"Logged trade: {trade_data.get('action')} {trade_data.get('product_id')} "
                        f"${trade_data.get('value', 0):.2f}")
                        
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
    
    def log_signal(self, signal_data: Dict[str, Any]):
        """
        Log a trading signal (executed or rejected) to CSV
        
        Args:
            signal_data: Dictionary containing signal information
        """
        try:
            with self.lock:
                with open(self.signals_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    
                    row = [
                        signal_data.get('timestamp', datetime.now(timezone.utc).isoformat()),
                        signal_data.get('product_id', ''),
                        signal_data.get('action', ''),
                        signal_data.get('confidence', 0),
                        signal_data.get('reason', ''),
                        signal_data.get('executed', False),
                        signal_data.get('rejection_reason', ''),
                        signal_data.get('spread_bps', 0),
                        signal_data.get('rsi', 0),
                        signal_data.get('sma_fast', 0),
                        signal_data.get('sma_slow', 0)
                    ]
                    
                    writer.writerow(row)
                    
        except Exception as e:
            logger.error(f"Failed to log signal: {e}")
    
    def get_daily_stats(self) -> Dict[str, Any]:
        """
        Read today's trade log and calculate statistics
        
        Returns:
            Dictionary with daily trading statistics
        """
        stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'total_volume': 0.0,
            'win_rate': 0.0,
            'avg_pnl_per_trade': 0.0
        }
        
        try:
            if os.path.exists(self.trades_file):
                with open(self.trades_file, 'r') as f:
                    reader = csv.DictReader(f)
                    
                    for row in reader:
                        stats['total_trades'] += 1
                        
                        pnl = float(row.get('pnl', 0))
                        value = float(row.get('value', 0))
                        
                        stats['total_pnl'] += pnl
                        stats['total_volume'] += value
                        
                        if pnl > 0:
                            stats['winning_trades'] += 1
                
                # Calculate derived stats
                if stats['total_trades'] > 0:
                    stats['win_rate'] = stats['winning_trades'] / stats['total_trades']
                    stats['avg_pnl_per_trade'] = stats['total_pnl'] / stats['total_trades']
                    
        except Exception as e:
            logger.error(f"Failed to calculate daily stats: {e}")
        
        return stats
    
    def get_recent_trades(self, limit: int = 10) -> list:
        """
        Get the most recent trades from today's log
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            List of recent trade dictionaries
        """
        trades = []
        
        try:
            if os.path.exists(self.trades_file):
                with open(self.trades_file, 'r') as f:
                    reader = csv.DictReader(f)
                    all_trades = list(reader)
                    
                    # Return last 'limit' trades
                    trades = all_trades[-limit:] if len(all_trades) > limit else all_trades
                    
        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
        
        return trades


# Global logger instance
_trade_logger: Optional[TradeLogger] = None


def get_trade_logger() -> TradeLogger:
    """Get or create global trade logger instance"""
    global _trade_logger
    if _trade_logger is None:
        _trade_logger = TradeLogger()
    return _trade_logger


def log_trade(trade_data: Dict[str, Any]):
    """Convenience function to log a trade"""
    get_trade_logger().log_trade(trade_data)


def log_signal(signal_data: Dict[str, Any]):
    """Convenience function to log a signal"""
    get_trade_logger().log_signal(signal_data)
