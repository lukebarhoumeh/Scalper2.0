"""Configuration Optimizer for Maximum Profitability

This module provides optimized configurations for different market conditions
and trading styles, designed to maximize profit generation.
"""

from typing import Dict, Any
import os


class ConfigOptimizer:
    """Provides optimized trading configurations"""
    
    @staticmethod
    def get_aggressive_config() -> Dict[str, Any]:
        """Ultra-aggressive configuration for maximum trade generation"""
        return {
            # High frequency polling
            "POLL_INTERVAL_SEC": "0.1",
            "UI_REFRESH_HZ": "10",
            
            # Aggressive strategy parameters
            "RSI_BUY_THRESHOLD": "52",
            "RSI_SELL_THRESHOLD": "48",
            "SMA_FAST": "3",
            "SMA_SLOW": "7",
            "MAX_SPREAD_BPS": "800",
            
            # Tight exits for scalping
            "TAKE_PROFIT_BPS": "15",
            "STOP_LOSS_BPS": "25",
            "TRAILING_STOP_BPS": "12",
            
            # Increased risk tolerance
            "RISK_PERCENT": "0.025",
            "MAX_DAILY_LOSS_PERCENT": "0.10",
            "INVENTORY_CAP_USD": "5000.0",
            "MAX_POSITION_SIZE": "1000.0",
            
            # High activity target
            "ACTIVITY_TARGET_TPH": "200",
            
            # Enable all strategies
            "ENABLE_MEAN_REVERSION": "true",
            "ENABLE_BREAKOUT": "true",
            "ENABLE_GRID_OVERLAY": "false"
        }
    
    @staticmethod
    def get_balanced_config() -> Dict[str, Any]:
        """Balanced configuration for consistent profits"""
        return {
            "POLL_INTERVAL_SEC": "0.25",
            "UI_REFRESH_HZ": "8",
            
            "RSI_BUY_THRESHOLD": "45",
            "RSI_SELL_THRESHOLD": "55",
            "SMA_FAST": "5",
            "SMA_SLOW": "12",
            "MAX_SPREAD_BPS": "500",
            
            "TAKE_PROFIT_BPS": "25",
            "STOP_LOSS_BPS": "35",
            "TRAILING_STOP_BPS": "20",
            
            "RISK_PERCENT": "0.02",
            "MAX_DAILY_LOSS_PERCENT": "0.08",
            "INVENTORY_CAP_USD": "4000.0",
            "MAX_POSITION_SIZE": "500.0",
            
            "ACTIVITY_TARGET_TPH": "120",
            
            "ENABLE_MEAN_REVERSION": "true",
            "ENABLE_BREAKOUT": "true",
            "ENABLE_GRID_OVERLAY": "false"
        }
    
    @staticmethod
    def get_conservative_config() -> Dict[str, Any]:
        """Conservative configuration for steady growth"""
        return {
            "POLL_INTERVAL_SEC": "0.5",
            "UI_REFRESH_HZ": "5",
            
            "RSI_BUY_THRESHOLD": "35",
            "RSI_SELL_THRESHOLD": "65",
            "SMA_FAST": "8",
            "SMA_SLOW": "21",
            "MAX_SPREAD_BPS": "300",
            
            "TAKE_PROFIT_BPS": "35",
            "STOP_LOSS_BPS": "45",
            "TRAILING_STOP_BPS": "30",
            
            "RISK_PERCENT": "0.015",
            "MAX_DAILY_LOSS_PERCENT": "0.05",
            "INVENTORY_CAP_USD": "3000.0",
            "MAX_POSITION_SIZE": "300.0",
            
            "ACTIVITY_TARGET_TPH": "60",
            
            "ENABLE_MEAN_REVERSION": "true",
            "ENABLE_BREAKOUT": "false",
            "ENABLE_GRID_OVERLAY": "false"
        }
    
    @staticmethod
    def get_synthetic_optimized_config() -> Dict[str, Any]:
        """Optimized specifically for synthetic data testing"""
        return {
            "POLL_INTERVAL_SEC": "0.05",  # Very fast for synthetic
            "UI_REFRESH_HZ": "20",
            
            # Very loose parameters for synthetic random walk
            "RSI_BUY_THRESHOLD": "55",
            "RSI_SELL_THRESHOLD": "45",
            "SMA_FAST": "2",
            "SMA_SLOW": "5",
            "MAX_SPREAD_BPS": "1000",
            
            # Quick profits on synthetic
            "TAKE_PROFIT_BPS": "10",
            "STOP_LOSS_BPS": "20",
            "TRAILING_STOP_BPS": "8",
            
            # Higher risk for testing
            "RISK_PERCENT": "0.03",
            "MAX_DAILY_LOSS_PERCENT": "0.15",
            "INVENTORY_CAP_USD": "7000.0",
            "MAX_POSITION_SIZE": "1500.0",
            
            "ACTIVITY_TARGET_TPH": "300",
            
            "ENABLE_MEAN_REVERSION": "true",
            "ENABLE_BREAKOUT": "true",
            "ENABLE_GRID_OVERLAY": "false"
        }
    
    @staticmethod
    def apply_config(config_dict: Dict[str, Any]) -> None:
        """Apply configuration to environment variables"""
        for key, value in config_dict.items():
            os.environ[key] = str(value)
    
    @staticmethod
    def get_market_adaptive_config(volatility: float, trend_strength: float) -> Dict[str, Any]:
        """
        Get configuration adapted to current market conditions
        
        Args:
            volatility: 0-100 volatility percentile
            trend_strength: -1 to 1 (-1 = strong down, 0 = range, 1 = strong up)
        """
        config = ConfigOptimizer.get_balanced_config()
        
        # Adjust for volatility
        if volatility > 80:
            # High volatility - wider stops, reduced size
            config["STOP_LOSS_BPS"] = "50"
            config["TAKE_PROFIT_BPS"] = "40"
            config["RISK_PERCENT"] = "0.015"
            config["MAX_SPREAD_BPS"] = "600"
        elif volatility < 20:
            # Low volatility - tighter stops
            config["STOP_LOSS_BPS"] = "20"
            config["TAKE_PROFIT_BPS"] = "15"
            config["RISK_PERCENT"] = "0.025"
            config["MAX_SPREAD_BPS"] = "300"
        
        # Adjust for trend
        if abs(trend_strength) > 0.7:
            # Strong trend - momentum focus
            config["ENABLE_BREAKOUT"] = "true"
            config["RSI_BUY_THRESHOLD"] = "50"
            config["RSI_SELL_THRESHOLD"] = "50"
            config["SMA_FAST"] = "3"
            config["SMA_SLOW"] = "8"
        elif abs(trend_strength) < 0.3:
            # Range bound - mean reversion focus
            config["ENABLE_MEAN_REVERSION"] = "true"
            config["ENABLE_BREAKOUT"] = "false"
            config["RSI_BUY_THRESHOLD"] = "30"
            config["RSI_SELL_THRESHOLD"] = "70"
        
        return config
    
    @staticmethod
    def optimize_for_symbol(symbol: str, historical_performance: Dict) -> Dict[str, Any]:
        """
        Optimize configuration for a specific symbol based on its performance
        
        Args:
            symbol: Trading symbol
            historical_performance: Dict with win_rate, avg_profit, volatility
        """
        base_config = ConfigOptimizer.get_balanced_config()
        
        win_rate = historical_performance.get('win_rate', 0.5)
        avg_profit_bps = historical_performance.get('avg_profit_bps', 20)
        volatility = historical_performance.get('volatility', 1.0)
        
        # Adjust position sizing based on win rate
        if win_rate > 0.65:
            base_config["RISK_PERCENT"] = "0.025"
            base_config["MAX_POSITION_SIZE"] = "750.0"
        elif win_rate < 0.45:
            base_config["RISK_PERCENT"] = "0.015"
            base_config["MAX_POSITION_SIZE"] = "300.0"
        
        # Adjust exits based on average profit
        if avg_profit_bps > 30:
            # Let winners run
            base_config["TAKE_PROFIT_BPS"] = str(int(avg_profit_bps * 1.2))
            base_config["TRAILING_STOP_BPS"] = "25"
        elif avg_profit_bps < 15:
            # Quick exits
            base_config["TAKE_PROFIT_BPS"] = "20"
            base_config["TRAILING_STOP_BPS"] = "15"
        
        # Adjust for symbol-specific volatility
        if symbol == "BTC-USD":
            base_config["MAX_SPREAD_BPS"] = "300"
            base_config["SMA_FAST"] = "5"
        elif symbol == "ETH-USD":
            base_config["MAX_SPREAD_BPS"] = "400"
            base_config["SMA_FAST"] = "4"
        elif symbol == "SOL-USD":
            base_config["MAX_SPREAD_BPS"] = "500"
            base_config["SMA_FAST"] = "3"
            base_config["RISK_PERCENT"] = "0.025"  # More volatile
        
        return base_config


def create_optimal_env_file(mode: str = "aggressive") -> str:
    """
    Create a complete .env file with optimal settings
    
    Args:
        mode: "aggressive", "balanced", "conservative", or "synthetic"
    
    Returns:
        Complete .env file contents
    """
    # Get configuration based on mode
    if mode == "aggressive":
        config = ConfigOptimizer.get_aggressive_config()
    elif mode == "balanced":
        config = ConfigOptimizer.get_balanced_config()
    elif mode == "conservative":
        config = ConfigOptimizer.get_conservative_config()
    elif mode == "synthetic":
        config = ConfigOptimizer.get_synthetic_optimized_config()
    else:
        config = ConfigOptimizer.get_balanced_config()
    
    # Build complete .env file
    env_content = f"""# === OPTIMIZED CONFIGURATION: {mode.upper()} MODE ===
# Generated by ConfigOptimizer for maximum profitability

# === DATA SOURCE ===
DATA_SOURCE=synthetic
# DATA_SOURCE=coinbase_rest

PAPER_TRADING=true
TRADING_PAIRS=BTC-USD,ETH-USD,SOL-USD

# === ACCOUNT / STATE ===
STARTING_CAPITAL=10000.0
RESET_STATE_ON_START=true
STATE_PATH=bot_state.json
LOGS_DIR=logs
LOG_LEVEL=INFO

# === SAFETY RAILS ===
INVENTORY_CAP_USD={config.get('INVENTORY_CAP_USD', '4000.0')}
MAX_POSITION_SIZE={config.get('MAX_POSITION_SIZE', '500.0')}
ALT_EXPOSURE_USD_CAP=3000.0
RISK_PERCENT={config.get('RISK_PERCENT', '0.02')}
MAX_DAILY_LOSS_PERCENT={config.get('MAX_DAILY_LOSS_PERCENT', '0.08')}
MAX_ERRORS=50
MAX_REQUESTS_PER_SECOND=10

# === ENGINE SETTINGS ===
POLL_INTERVAL_SEC={config.get('POLL_INTERVAL_SEC', '0.25')}
UI_REFRESH_HZ={config.get('UI_REFRESH_HZ', '8')}
LEDGER_FLUSH_SEC=5

# === STRATEGY PARAMETERS ===
RSI_BUY_THRESHOLD={config.get('RSI_BUY_THRESHOLD', '45')}
RSI_SELL_THRESHOLD={config.get('RSI_SELL_THRESHOLD', '55')}
SMA_FAST={config.get('SMA_FAST', '5')}
SMA_SLOW={config.get('SMA_SLOW', '12')}
MAX_SPREAD_BPS={config.get('MAX_SPREAD_BPS', '500')}

# === EXIT PARAMETERS ===
TAKE_PROFIT_BPS={config.get('TAKE_PROFIT_BPS', '25')}
STOP_LOSS_BPS={config.get('STOP_LOSS_BPS', '35')}
TRAILING_STOP_BPS={config.get('TRAILING_STOP_BPS', '20')}

# === FEATURES ===
TUNING_ENABLED=true
ACTIVITY_TARGET_TPH={config.get('ACTIVITY_TARGET_TPH', '120')}
ENABLE_MEAN_REVERSION={config.get('ENABLE_MEAN_REVERSION', 'true')}
ENABLE_BREAKOUT={config.get('ENABLE_BREAKOUT', 'true')}
ENABLE_GRID_OVERLAY={config.get('ENABLE_GRID_OVERLAY', 'false')}

# === API KEYS ===
# COINBASE_API_KEY=your_key_here
# COINBASE_API_SECRET=your_secret_here
# OPENAI_API_KEY=your_openai_key_here
"""
    
    return env_content
