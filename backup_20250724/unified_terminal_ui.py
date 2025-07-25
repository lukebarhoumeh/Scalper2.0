"""
unified_terminal_ui.py - All-in-One Terminal Dashboard
====================================================
Professional terminal UI that displays:
- Real-time P&L and positions
- Trading signals and executions
- Market data and indicators
- System health and performance
"""

import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque
import numpy as np

# Terminal colors
from colorama import init, Fore, Back, Style
init()

class UnifiedTerminalUI:
    """
    Professional terminal UI for the unified trading bot
    """
    
    def __init__(self):
        self.running = True
        self.last_update = time.time()
        
        # Data storage
        self.stats = {}
        self.positions = {}
        self.recent_trades = deque(maxlen=10)
        self.recent_signals = deque(maxlen=5)
        self.market_data = {}
        self.health_status = {}
        self.signal_rejections = {}  # Track why signals are rejected
        
        # Display settings
        self.terminal_width = 120
        self.refresh_rate = 1.0  # seconds
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def update_data(self, **kwargs):
        """Update display data"""
        for key, value in kwargs.items():
            if key == 'stats':
                self.stats = value
                # Extract market overview if present
                if isinstance(value, dict) and 'market_overview' in value:
                    self.market_overview = value['market_overview']
            elif key == 'positions':
                self.positions = value
            elif key == 'trade':
                self.recent_trades.append(value)
            elif key == 'signal':
                self.recent_signals.append(value)
            elif key == 'market':
                self.market_data.update(value)
            elif key == 'health':
                self.health_status = value
            elif key == 'signal_rejections':
                self.signal_rejections = value
    
    def display(self):
        """Main display method"""
        self.clear_screen()
        
        # Header
        self._display_header()
        
        # Main sections
        self._display_account_summary()
        self._display_positions()
        self._display_recent_activity()
        self._display_market_overview()
        self._display_system_health()
        self._display_signal_health()
        
        # Footer
        self._display_footer()
    
    def _display_header(self):
        """Display header with title and time"""
        print(f"{Back.BLUE}{Fore.WHITE}{'═' * self.terminal_width}{Style.RESET_ALL}")
        title = "SCALPERBOT 2.0 - UNIFIED MASTER TRADING SYSTEM"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mode_raw = self.stats.get('mode', 'BALANCED')
        mode = str(mode_raw).upper() if mode_raw else 'BALANCED'
        
        print(f"{Back.BLUE}{Fore.WHITE}{title.center(self.terminal_width - 20)}{mode.rjust(20)}{Style.RESET_ALL}")
        print(f"{Back.BLUE}{Fore.WHITE}{timestamp.center(self.terminal_width)}{Style.RESET_ALL}")
        print(f"{Back.BLUE}{Fore.WHITE}{'═' * self.terminal_width}{Style.RESET_ALL}")
    
    def _display_account_summary(self):
        """Display account summary section"""
        print(f"\n{Fore.CYAN}📊 ACCOUNT SUMMARY{Style.RESET_ALL}")
        print("─" * 60)
        
        daily_pnl = self.stats.get('daily_pnl', 0)
        pnl_color = Fore.GREEN if daily_pnl >= 0 else Fore.RED
        
        # Create two-column layout
        left_col = [
            f"Daily P&L: {pnl_color}${daily_pnl:,.2f}{Style.RESET_ALL}",
            f"Capital Used: ${self.stats.get('daily_capital_used', 0):,.2f}",
            f"Capital Remaining: ${self.stats.get('daily_capital_remaining', 0):,.2f}",
        ]
        
        right_col = [
            f"Total Trades: {self.stats.get('total_trades', 0)}",
            f"Win Rate: {self.stats.get('win_rate', 0)*100:.1f}%",
            f"Avg Slippage: {self.stats.get('avg_slippage_bps', 0):.1f} bps",
        ]
        
        for left, right in zip(left_col, right_col):
            print(f"  {left:<40} {right}")
    
    def _display_positions(self):
        """Display current positions"""
        print(f"\n{Fore.CYAN}💼 CURRENT POSITIONS{Style.RESET_ALL}")
        print("─" * 60)
        
        if not self.positions:
            print(f"  {Fore.WHITE}No open positions{Style.RESET_ALL}")
        else:
            print(f"  {'Symbol':<10} {'Size':<15} {'Entry':<10} {'Current':<10} {'P&L':<12} {'%':<8}")
            print("  " + "─" * 58)
            
            for symbol, pos in self.positions.items():
                pnl = pos.get('unrealized_pnl', 0)
                pnl_pct = (pnl / (pos['base'] * pos['avg_entry'])) * 100 if pos['avg_entry'] > 0 else 0
                pnl_color = Fore.GREEN if pnl >= 0 else Fore.RED
                
                current_price = pos['usd'] / pos['base'] if pos['base'] != 0 else 0
                
                print(f"  {symbol:<10} ${pos['usd']:>13,.2f} ${pos['avg_entry']:>8,.2f} "
                      f"${current_price:>8,.2f} {pnl_color}${pnl:>10,.2f} "
                      f"{pnl_pct:>6.1f}%{Style.RESET_ALL}")
    
    def _display_recent_activity(self):
        """Display recent trades and signals"""
        print(f"\n{Fore.CYAN}⚡ RECENT ACTIVITY{Style.RESET_ALL}")
        print("─" * 60)
        
        # Recent signals
        if self.recent_signals:
            print(f"  {Fore.YELLOW}Latest Signals:{Style.RESET_ALL}")
            for signal in list(self.recent_signals)[-3:]:
                time_str = signal['timestamp'].strftime("%H:%M:%S")
                side_color = Fore.GREEN if signal['side'] == 'BUY' else Fore.RED
                print(f"    [{time_str}] {side_color}{signal['side']:<4}{Style.RESET_ALL} "
                      f"{signal['product']:<8} ${signal['size']:>8,.2f} "
                      f"({signal['strategy']}, {signal['confidence']:.0f}%)")
        
        # Recent trades
        if self.recent_trades:
            print(f"\n  {Fore.YELLOW}Latest Trades:{Style.RESET_ALL}")
            for trade in list(self.recent_trades)[-3:]:
                time_str = trade['timestamp'].strftime("%H:%M:%S")
                side_color = Fore.GREEN if trade['side'] == 'BUY' else Fore.RED
                status = "✓" if trade.get('status') == 'filled' else "✗"
                print(f"    [{time_str}] {status} {side_color}{trade['side']:<4}{Style.RESET_ALL} "
                      f"{trade['product']:<8} ${trade['size']:>8,.2f} @ ${trade['price']:,.2f}")
    
    def _display_market_overview(self):
        """Display market overview"""
        print(f"\n{Fore.CYAN}📈 MARKET OVERVIEW{Style.RESET_ALL}")
        print("─" * 60)
        
        # Display top movers or key markets
        markets = ['BTC-USD', 'ETH-USD', 'SOL-USD']
        print(f"  {'Market':<10} {'Price':<12} {'24h Change':<12} {'Volume':<15}")
        print("  " + "─" * 48)
        
        for market in markets:
            data = self.market_data.get(market, {})
            price = data.get('price', 0)
            change = data.get('24h_change', 0)
            volume = data.get('volume', 0)
            
            change_color = Fore.GREEN if change >= 0 else Fore.RED
            print(f"  {market:<10} ${price:>10,.2f} {change_color}{change:>10.1f}%{Style.RESET_ALL} "
                  f"${volume:>13,.0f}")
    
    def _display_system_health(self):
        """Display system health status"""
        print(f"\n{Fore.CYAN}🔧 SYSTEM HEALTH{Style.RESET_ALL}")
        print("─" * 60)
        
        health_items = [
            ('Executor', self.health_status.get('executor_healthy', False)),
            ('Strategy', self.health_status.get('strategy_healthy', False)),
            ('Market Data', self.health_status.get('market_data_healthy', False)),
            ('AI Systems', self.health_status.get('ai_healthy', False)),
        ]
        
        errors = self.health_status.get('errors', 0)
        uptime = self.health_status.get('uptime', 0)
        memory = self.health_status.get('memory_mb', 0)
        
        # Health indicators
        print("  ", end="")
        for name, status in health_items:
            icon = "✓" if status else "✗"
            color = Fore.GREEN if status else Fore.RED
            print(f"{color}{icon} {name}{Style.RESET_ALL}  ", end="")
        
        print(f"\n  Errors: {errors}  |  Uptime: {self._format_uptime(uptime)}  |  Memory: {memory:.0f}MB")
    
    def _display_signal_health(self):
        """Display why signals are being rejected"""
        if not self.signal_rejections:
            return
            
        print(f"\n{Fore.YELLOW}📊 SIGNAL HEALTH{Style.RESET_ALL}")
        print("─" * 60)
        
        # Show latest rejection reason for each product
        for product, rejections in self.signal_rejections.items():
            if rejections and len(rejections) > 0:
                latest = rejections[-1]
                print(f"  {Fore.RED}❌ {product}: {latest.get('reason', 'Unknown')}{Style.RESET_ALL}")
    
    def _display_footer(self):
        """Display footer with commands"""
        print(f"\n{Back.WHITE}{Fore.BLACK}{'─' * self.terminal_width}{Style.RESET_ALL}")
        commands = "[Q] Quit  [P] Pause  [M] Mode  [S] Stats  [H] Help"
        print(f"{Back.WHITE}{Fore.BLACK}{commands.center(self.terminal_width)}{Style.RESET_ALL}")
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def run(self):
        """Run the UI loop"""
        while self.running:
            try:
                if time.time() - self.last_update >= self.refresh_rate:
                    self.display()
                    self.last_update = time.time()
                time.sleep(0.1)
            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                print(f"{Fore.RED}UI Error: {e}{Style.RESET_ALL}")
                time.sleep(1)
    
    def stop(self):
        """Stop the UI"""
        self.running = False


class TerminalUIIntegration:
    """
    Integration class to connect UI with the trading bot
    """
    
    def __init__(self, ui: UnifiedTerminalUI):
        self.ui = ui
    
    def update_stats(self, stats: Dict):
        """Update trading statistics"""
        self.ui.update_data(stats=stats)
    
    def update_positions(self, positions: Dict):
        """Update position information"""
        self.ui.update_data(positions=positions)
    
    def log_trade(self, trade: Dict):
        """Log a new trade"""
        trade['timestamp'] = datetime.now()
        self.ui.update_data(trade=trade)
    
    def log_signal(self, signal: Dict):
        """Log a new signal"""
        signal['timestamp'] = datetime.now()
        self.ui.update_data(signal=signal)
    
    def update_market(self, market_data: Dict):
        """Update market data"""
        self.ui.update_data(market=market_data)
    
    def update_health(self, health: Dict):
        """Update system health"""
        self.ui.update_data(health=health)
    
    def update_signal_health(self, rejections: Dict):
        """Update signal rejection reasons"""
        self.ui.update_data(signal_rejections=rejections) 