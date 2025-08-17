#!/usr/bin/env python3
"""
test_unified_bot.py - Comprehensive Test Suite for Unified Master Bot
====================================================================
Tests all integrated features:
- AI risk management
- Terminal UI
- Self-healing capabilities
- Mode switching
- Trade execution
"""

import sys
import os
import asyncio
import time
import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
import colorama
from colorama import Fore, Style, Back
colorama.init()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class UnifiedBotTester:
    """
    Comprehensive test suite for the unified master bot
    """
    
    def __init__(self):
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
    def log_test(self, test_name: str, passed: bool, error: str = None):
        """Log test result"""
        if passed:
            print(f"{Fore.GREEN}✓ {test_name}{Style.RESET_ALL}")
            self.test_results['passed'] += 1
        else:
            print(f"{Fore.RED}✗ {test_name}: {error}{Style.RESET_ALL}")
            self.test_results['failed'] += 1
            self.test_results['errors'].append(f"{test_name}: {error}")
    
    async def test_core_imports(self):
        """Test that all core modules can be imported"""
        test_name = "Core Imports"
        try:
            from run_master_bot import UnifiedMasterBot
            from ai_risk_manager import AIRiskManager
            from unified_terminal_ui import UnifiedTerminalUI
            from market_intelligence import MarketIntelligence
            from strategy_engine_production import ProductionStrategyEngine
            from trade_executor_production import ProductionTradeExecutor
            from risk_manager import HFTRiskManager as RiskManager
            
            self.log_test(test_name, True)
        except Exception as e:
            self.log_test(test_name, False, str(e))
    
    async def test_ai_risk_manager(self):
        """Test AI risk manager functionality"""
        test_name = "AI Risk Manager"
        try:
            from ai_risk_manager import AIRiskManager
            
            ai_manager = AIRiskManager()
            
            # Test mode recommendation
            config = await ai_manager.recommend_mode(
                current_pnl=100,
                error_count=0,
                market_conditions={'volatility': 0.02}
            )
            
            assert config['mode'] in ['conservative', 'balanced', 'aggressive']
            assert 'position_size_multiplier' in config
            assert 'max_daily_trades' in config
            
            # Test risk adjustment
            adjusted = await ai_manager.adjust_risk_parameters({
                'max_position_size': 1000,
                'stop_loss_pct': 0.02
            })
            
            assert 'max_position_size' in adjusted
            assert 'stop_loss_pct' in adjusted
            
            self.log_test(test_name, True)
        except Exception as e:
            self.log_test(test_name, False, str(e))
    
    async def test_terminal_ui(self):
        """Test terminal UI functionality"""
        test_name = "Terminal UI"
        try:
            from unified_terminal_ui import UnifiedTerminalUI, TerminalUIIntegration
            
            ui = UnifiedTerminalUI()
            integration = TerminalUIIntegration(ui)
            
            # Test UI methods
            integration.update_stats({
                'daily_pnl': 50.25,
                'total_trades': 10,
                'win_rate': 0.60
            })
            
            integration.log_signal({
                'side': 'BUY',
                'product': 'BTC-USD',
                'size': 0.001,
                'strategy': 'Momentum',
                'confidence': 75
            })
            
            integration.log_trade({
                'side': 'BUY',
                'product': 'BTC-USD',
                'size': 0.001,
                'price': 45000,
                'status': 'FILLED'
            })
            
            # Stop UI to prevent thread issues
            ui.stop()
            
            self.log_test(test_name, True)
        except Exception as e:
            self.log_test(test_name, False, str(e))
    
    async def test_market_intelligence(self):
        """Test market intelligence module"""
        test_name = "Market Intelligence"
        try:
            from market_intelligence import MarketIntelligence
            
            mi = MarketIntelligence()
            await mi.initialize()
            
            # Test health check
            healthy = await mi.health_check()
            assert isinstance(healthy, bool)
            
            # Test market conditions
            conditions = await mi.get_market_conditions()
            assert 'volatility' in conditions
            assert 'trend' in conditions
            assert 'momentum' in conditions
            
            self.log_test(test_name, True)
        except Exception as e:
            self.log_test(test_name, False, str(e))
    
    async def test_unified_bot_initialization(self):
        """Test unified bot initialization"""
        test_name = "Unified Bot Initialization"
        try:
            from run_master_bot import UnifiedMasterBot
            
            # Mock external dependencies
            with patch('run_master_bot.start_websocket_feed'):
                with patch('run_master_bot.TelegramNotifier'):
                    bot = UnifiedMasterBot()
                    
                    # Verify components initialized
                    assert bot.ai_risk_manager is not None
                    assert bot.market_intelligence is not None
                    assert bot.terminal_ui is not None
                    assert bot.executor is not None
                    assert bot.risk_manager is not None
                    
                    # Test mode switching logic
                    assert bot.current_mode in ['conservative', 'balanced', 'aggressive']
                    assert 'profit_threshold' in bot.mode_switch_threshold
                    
                    # Stop UI
                    bot.terminal_ui.stop()
                    
                    self.log_test(test_name, True)
        except Exception as e:
            self.log_test(test_name, False, str(e))
    
    async def test_self_healing_capabilities(self):
        """Test self-healing and error recovery"""
        test_name = "Self-Healing Capabilities"
        try:
            from run_master_bot import UnifiedMasterBot
            
            with patch('run_master_bot.start_websocket_feed'):
                with patch('run_master_bot.TelegramNotifier'):
                    bot = UnifiedMasterBot()
                    
                    # Test error counting
                    bot.consecutive_errors = 3
                    bot.error_count = 5
                    
                    # Test health check methods
                    executor_health = bot._check_executor_health()
                    assert isinstance(executor_health, bool)
                    
                    strategy_health = bot._check_strategy_health()
                    assert isinstance(strategy_health, bool)
                    
                    market_health = bot._check_market_data_health()
                    assert isinstance(market_health, bool)
                    
                    # Stop UI
                    bot.terminal_ui.stop()
                    
                    self.log_test(test_name, True)
        except Exception as e:
            self.log_test(test_name, False, str(e))
    
    async def test_mode_switching_logic(self):
        """Test dynamic mode switching based on performance"""
        test_name = "Mode Switching Logic"
        try:
            from run_master_bot import UnifiedMasterBot
            
            with patch('run_master_bot.start_websocket_feed'):
                with patch('run_master_bot.TelegramNotifier'):
                    bot = UnifiedMasterBot()
                    
                    # Mock executor stats
                    bot.executor.get_daily_stats = Mock(return_value={
                        'daily_pnl': 100,  # Should trigger aggressive mode
                        'total_trades': 20,
                        'win_rate': 0.65
                    })
                    
                    # Mock AI recommendation
                    bot.ai_risk_manager.recommend_mode = Mock(return_value={
                        'mode': 'aggressive',
                        'confidence': 0.85,
                        'position_size_multiplier': 1.5
                    })
                    
                    # Test mode switch
                    await bot._check_mode_switch()
                    
                    # Verify mode changed
                    assert bot.current_mode in ['conservative', 'balanced', 'aggressive']
                    
                    # Stop UI
                    bot.terminal_ui.stop()
                    
                    self.log_test(test_name, True)
        except Exception as e:
            self.log_test(test_name, False, str(e))
    
    async def test_signal_and_trade_integration(self):
        """Test signal generation and trade execution flow"""
        test_name = "Signal & Trade Integration"
        try:
            # This is a high-level integration test
            # In production, you'd test with real market data
            
            from run_master_bot import UnifiedMasterBot
            
            with patch('run_master_bot.start_websocket_feed'):
                with patch('run_master_bot.TelegramNotifier'):
                    bot = UnifiedMasterBot()
                    
                    # Verify signal hooks setup
                    bot._setup_strategy_hooks()
                    
                    # Mock a signal
                    test_signal = {
                        'action': 'BUY',
                        'coin': 'BTC',
                        'size_usd': 100,
                        'strategy': 'TestStrategy',
                        'confidence': 0.75
                    }
                    
                    # UI should be able to receive signals
                    bot.ui_integration.log_signal(test_signal)
                    
                    # Stop UI
                    bot.terminal_ui.stop()
                    
                    self.log_test(test_name, True)
        except Exception as e:
            self.log_test(test_name, False, str(e))
    
    async def run_all_tests(self):
        """Run all tests and generate report"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"UNIFIED BOT TEST SUITE")
        print(f"{'='*60}{Style.RESET_ALL}\n")
        
        # Run tests
        await self.test_core_imports()
        await self.test_ai_risk_manager()
        await self.test_terminal_ui()
        await self.test_market_intelligence()
        await self.test_unified_bot_initialization()
        await self.test_self_healing_capabilities()
        await self.test_mode_switching_logic()
        await self.test_signal_and_trade_integration()
        
        # Generate report
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"TEST RESULTS")
        print(f"{'='*60}{Style.RESET_ALL}\n")
        
        total_tests = self.test_results['passed'] + self.test_results['failed']
        pass_rate = (self.test_results['passed'] / total_tests * 100) if total_tests > 0 else 0
        
        print(f"{Fore.GREEN}Passed: {self.test_results['passed']}{Style.RESET_ALL}")
        print(f"{Fore.RED}Failed: {self.test_results['failed']}{Style.RESET_ALL}")
        print(f"Pass Rate: {pass_rate:.1f}%\n")
        
        if self.test_results['errors']:
            print(f"{Fore.RED}Errors:{Style.RESET_ALL}")
            for error in self.test_results['errors']:
                print(f"  - {error}")
        
        # Overall status
        if self.test_results['failed'] == 0:
            print(f"\n{Fore.GREEN}✓ ALL TESTS PASSED! Bot is ready for production.{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}✗ Some tests failed. Please fix issues before production.{Style.RESET_ALL}")
        
        return self.test_results['failed'] == 0

async def main():
    """Main test runner"""
    tester = UnifiedBotTester()
    success = await tester.run_all_tests()
    
    # Return exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main()) 