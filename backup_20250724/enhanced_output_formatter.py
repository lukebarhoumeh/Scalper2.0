#!/usr/bin/env python3
"""
Enhanced Output Formatter for ScalperBot 2.0
Provides clean, professional terminal output with real-time updates
"""

import os
import sys
from datetime import datetime, timezone
from colorama import init, Fore, Back, Style
import logging

# Initialize colorama for Windows
init(autoreset=True)

class EnhancedFormatter:
    """Format bot output for professional appearance"""
    
    @staticmethod
    def clear_screen():
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def print_header():
        """Print professional header"""
        print(f"\n{Fore.CYAN}{'═' * 80}")
        print(f"{Fore.CYAN}║{' ' * 20}{Fore.WHITE}SCALPERBOT 2.0 - PRODUCTION HFT{' ' * 20}{Fore.CYAN}║")
        print(f"{Fore.CYAN}║{' ' * 22}{Fore.YELLOW}High-Frequency Trading System{' ' * 22}{Fore.CYAN}║")
        print(f"{Fore.CYAN}{'═' * 80}{Style.RESET_ALL}\n")
    
    @staticmethod
    def display_startup_banner():
        """Display startup banner for unified bot"""
        print(f"\n{Fore.CYAN}{'=' * 80}")
        print(f"{Fore.CYAN}||{' ' * 20}{Fore.WHITE}SCALPERBOT 2.0 - UNIFIED MASTER{' ' * 20}{Fore.CYAN}||")
        print(f"{Fore.CYAN}||{' ' * 22}{Fore.YELLOW}AI-Powered HFT Trading System{' ' * 22}{Fore.CYAN}||")
        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}[INFO] Starting unified master bot...{Style.RESET_ALL}")
        print(f"{Fore.GREEN}[INFO] Self-healing enabled{Style.RESET_ALL}")
        print(f"{Fore.GREEN}[INFO] AI risk management active{Style.RESET_ALL}\n")

    @staticmethod
    def print_config_summary(config):
        """Print configuration in a clean format"""
        print(f"{Fore.GREEN}> CONFIGURATION SUMMARY{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}• Daily Capital:{Fore.GREEN} ${config.get('DAILY_CAPITAL_LIMIT', 1000):,.2f}{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}• Trading Coins:{Fore.CYAN} {len(config.get('TRADE_COINS', []))} assets{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}• Base Trade Size:{Fore.YELLOW} ${config.get('BASE_TRADE_SIZE_USD', 100)}{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}• API Rate:{Fore.MAGENTA} {config.get('REST_RATE_LIMIT', 5)} req/s{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}• Mode:{Fore.RED} {'PAPER TRADING' if config.get('PAPER_TRADING', True) else 'LIVE TRADING'}{Style.RESET_ALL}\n")
    
    @staticmethod
    def print_market_status(market_data):
        """Print market status in a clean format"""
        print(f"{Fore.GREEN}▶ MARKET STATUS{Style.RESET_ALL}")
        for coin, data in market_data.items():
            trend = "↑" if data.get('trend') == 'up' else "↓"
            color = Fore.GREEN if data.get('trend') == 'up' else Fore.RED
            print(f"  {Fore.WHITE}{coin}:{color} ${data.get('price', 0):,.2f} {trend} {data.get('change_pct', 0):+.2f}%{Style.RESET_ALL}")
    
    @staticmethod
    def print_signal(signal):
        """Print trading signal in a clean format"""
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        action_color = Fore.GREEN if signal['action'] == 'BUY' else Fore.RED
        
        print(f"\n{Fore.YELLOW}⚡ SIGNAL [{timestamp}]{Style.RESET_ALL}")
        print(f"  {action_color}▸ {signal['action']} {signal['product_id']}{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}• Size: ${signal['size_usd']:.2f}")
        print(f"  {Fore.WHITE}• Strategy: {signal['strategy']}")
        print(f"  {Fore.WHITE}• Confidence: {signal['confidence']:.1%}")
        print(f"  {Fore.WHITE}• Risk Score: {signal.get('risk_score', 0):.2f}")
    
    @staticmethod
    def print_performance_dashboard(stats):
        """Print performance dashboard"""
        print(f"\n{Fore.GREEN}▶ PERFORMANCE DASHBOARD{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}• Trades Today: {stats.get('trades_today', 0)}")
        print(f"  {Fore.WHITE}• Win Rate: {stats.get('win_rate', 0):.1%}")
        print(f"  {Fore.WHITE}• P&L: {Fore.GREEN if stats.get('pnl', 0) >= 0 else Fore.RED}${stats.get('pnl', 0):+,.2f}{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}• Sharpe Ratio: {stats.get('sharpe', 0):.2f}")
        print(f"  {Fore.WHITE}• Active Positions: {stats.get('positions', 0)}")
    
    @staticmethod
    def print_status_line(message, status='info'):
        """Print status line with appropriate color"""
        colors = {
            'info': Fore.CYAN,
            'success': Fore.GREEN,
            'warning': Fore.YELLOW,
            'error': Fore.RED
        }
        color = colors.get(status, Fore.WHITE)
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"{Fore.BLACK}[{timestamp}] {color}{message}{Style.RESET_ALL}")
    
    @staticmethod
    def create_progress_bar(current, total, width=50):
        """Create a progress bar"""
        percentage = current / total if total > 0 else 0
        filled = int(width * percentage)
        bar = "█" * filled + "░" * (width - filled)
        return f"{bar} {percentage:.1%}"

class CleanLogger(logging.Handler):
    """Custom logging handler for clean output"""
    
    def __init__(self, formatter=None):
        super().__init__()
        self.formatter = formatter or EnhancedFormatter()
    
    def emit(self, record):
        """Emit log record with clean formatting"""
        # Filter out noisy logs
        if any(skip in record.getMessage() for skip in [
            "WebSocket message",
            "Volatility Tracker:",
            "Slow candle fetch",
            "Rate limited"
        ]):
            return
        
        # Map log levels to status types
        status_map = {
            logging.INFO: 'info',
            logging.WARNING: 'warning',
            logging.ERROR: 'error',
            logging.CRITICAL: 'error'
        }
        
        status = status_map.get(record.levelno, 'info')
        self.formatter.print_status_line(record.getMessage(), status)

def setup_clean_logging():
    """Setup clean logging for the bot"""
    # Remove existing handlers
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add our clean handler
    clean_handler = CleanLogger()
    logger.addHandler(clean_handler)
    logger.setLevel(logging.INFO)
    
    # Also setup for specific loggers
    for name in ['market_data', 'strategy_engine_production', 'trade_executor_production']:
        specific_logger = logging.getLogger(name)
        specific_logger.handlers.clear()
        specific_logger.addHandler(clean_handler)
        specific_logger.setLevel(logging.INFO)

# Test the formatter
if __name__ == "__main__":
    formatter = EnhancedFormatter()
    
    # Test header
    formatter.clear_screen()
    formatter.print_header()
    
    # Test config
    test_config = {
        'DAILY_CAPITAL_LIMIT': 1000,
        'TRADE_COINS': ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC'],
        'BASE_TRADE_SIZE_USD': 100,
        'REST_RATE_LIMIT': 5,
        'PAPER_TRADING': True
    }
    formatter.print_config_summary(test_config)
    
    # Test market status
    test_market = {
        'BTC-USD': {'price': 118256.30, 'trend': 'up', 'change_pct': 2.45},
        'ETH-USD': {'price': 3521.42, 'trend': 'down', 'change_pct': -1.23},
        'SOL-USD': {'price': 245.67, 'trend': 'up', 'change_pct': 5.67}
    }
    formatter.print_market_status(test_market)
    
    # Test signal
    test_signal = {
        'action': 'BUY',
        'product_id': 'BTC-USD',
        'size_usd': 150.00,
        'strategy': 'MultiIndicatorScalper',
        'confidence': 0.85,
        'risk_score': 0.3
    }
    formatter.print_signal(test_signal)
    
    # Test performance
    test_stats = {
        'trades_today': 42,
        'win_rate': 0.643,
        'pnl': 127.50,
        'sharpe': 2.1,
        'positions': 3
    }
    formatter.print_performance_dashboard(test_stats)
    
    # Test status lines
    formatter.print_status_line("WebSocket connected successfully", 'success')
    formatter.print_status_line("Low volatility detected on ETH-USD", 'warning')
    formatter.print_status_line("Strategy engine initialized", 'info') 