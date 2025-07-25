"""
config_production_hft.py – Elite Production HFT Configuration
Built by a Quant HFT Veteran for Maximum Profitability
"""

from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables
load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
# API CREDENTIALS
# ═══════════════════════════════════════════════════════════════════════════
COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Telegram notifications (optional)
ENABLE_NOTIFICATIONS = False  # Set to True when you want Telegram alerts
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ═══════════════════════════════════════════════════════════════════════════
# TRADING ENGINE SETTINGS
# ═══════════════════════════════════════════════════════════════════════════
POLL_INTERVAL = 30  # seconds between market data updates

# Position sizing
MIN_POSITION_SIZE = 50  # Minimum position size in USD
MAX_POSITION_SIZE = 500  # Maximum position size in USD

# ═══════════════════════════════════════════════════════════════════════════
# ELITE HFT TRADING UNIVERSE - Import from unified config
# ═══════════════════════════════════════════════════════════════════════════
from config_unified import TRADING_PAIRS

# Convert to coin symbols (remove -USD suffix)
TRADE_COINS = [pair.replace("-USD", "") for pair in TRADING_PAIRS]

# ═══════════════════════════════════════════════════════════════════════════
# POSITION SIZING & CAPITAL MANAGEMENT (Production Grade)
# ═══════════════════════════════════════════════════════════════════════════
# Daily capital allocation - as requested
MAX_DAILY_CAPITAL = 1000.0              # $1,000 daily trading capital

# Dynamic position sizing based on volatility
BASE_TRADE_SIZE_USD = 100.0             # Base size (will be adjusted by volatility)
MIN_TRADE_SIZE_USD = 50.0               # Minimum trade size
MAX_TRADE_SIZE_USD = 200.0              # Maximum single trade

# Portfolio limits
INVENTORY_CAP_USD = 500.0               # Max total exposure at any time
PER_COIN_POSITION_LIMIT = 150.0         # Max exposure per coin
MAX_COIN_ALLOCATION_PCT = 0.15          # Max 15% of capital per coin

# ═══════════════════════════════════════════════════════════════════════════
# API OPTIMIZATION - Balanced for reliability and speed
# ═══════════════════════════════════════════════════════════════════════════
POLL_INTERVAL_SEC = 30                  # Main loop interval (reduced as requested)
FAST_POLL_INTERVAL_SEC = 10             # For high volatility periods
REST_RATE_LIMIT_PER_S = 5               # Conservative to avoid rate limits
WS_RATE_LIMIT_PER_S = 20               # WebSocket can handle more

# Connection management
USE_WS_FEED = True                      # Primary data source
WS_HEARTBEAT_INTERVAL = 30              # Keep alive interval
REST_TIMEOUT_SEC = 5                    # API call timeout
MAX_RETRIES = 3                         # Retry failed requests

# ═══════════════════════════════════════════════════════════════════════════
# HFT STRATEGY PARAMETERS - Optimized from backtesting
# ═══════════════════════════════════════════════════════════════════════════
# Scalper settings (primary strategy)
SCALPER_CONFIG = {
    "sma_window": 20,                   # Fast SMA for trend
    "vol_thresh": 3.0,                  # Minimum volatility % to trade
    "spread_thresh": 0.25,              # Maximum spread % to enter
    "rsi_period": 14,                   # RSI lookback
    "rsi_oversold": 25,                 # Buy below this
    "rsi_overbought": 75,               # Sell above this
}

# Breakout settings (secondary strategy)
BREAKOUT_CONFIG = {
    "lookback": 48,                     # ~24 hours at 30min candles
    "atr_window": 14,                   # ATR period
    "atr_multiplier": 1.5,              # Breakout threshold
    "volume_multiplier": 1.2,           # Volume confirmation
}

# Market making settings (advanced strategy)
MARKET_MAKER_CONFIG = {
    "spread_target_bps": 20,            # Target 20 basis points
    "inventory_skew": 0.7,              # Bias orders based on inventory
    "order_refresh_sec": 60,            # Cancel/replace cycle
}

# Strategy weights (dynamic based on market conditions)
STRATEGY_WEIGHTS = {
    "scalper": 0.5,                     # Base weight
    "breakout": 0.3,                    # Base weight
    "market_maker": 0.2,                # Base weight
}

# ═══════════════════════════════════════════════════════════════════════════
# ADVANCED RISK MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════
# Position limits
RISK_LIMITS = {
    "max_correlated_exposure": 0.3,     # Max 30% in correlated assets
    "max_drawdown_pct": 10.0,           # Stop trading at 10% drawdown
    "max_daily_loss_usd": 200.0,        # 20% of daily capital
    "max_consecutive_losses": 5,         # Circuit breaker trigger
    "min_sharpe_ratio": 0.5,            # Minimum acceptable Sharpe
}

# Execution limits
EXECUTION_LIMITS = {
    "max_slippage_bps": 10,             # Cancel if slippage > 10bps
    "max_spread_bps": 30,               # Don't trade wide spreads
    "min_liquidity_usd": 10000,         # Minimum order book depth
    "max_market_impact_bps": 5,         # Expected price impact
}

# Time-based controls
TIME_CONTROLS = {
    "cooldown_sec": 120,                # Wait between trades per coin
    "news_blackout_min": 30,            # Pause after major news
    "low_volume_hours": [0, 1, 2, 3],   # UTC hours to reduce activity
}

# ═══════════════════════════════════════════════════════════════════════════
# VOLATILITY ADAPTIVE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
VOLATILITY_BANDS = {
    "low": {"min": 0.0, "max": 2.0, "size_mult": 0.5},
    "normal": {"min": 2.0, "max": 5.0, "size_mult": 1.0},
    "high": {"min": 5.0, "max": 10.0, "size_mult": 1.5},
    "extreme": {"min": 10.0, "max": 100.0, "size_mult": 0.3},
}

# ═══════════════════════════════════════════════════════════════════════════
# AI CONFIGURATION - Production optimized
# ═══════════════════════════════════════════════════════════════════════════
OPENAI_MODEL = "gpt-3.5-turbo"         # Fast and cost-effective
AI_TEMPERATURE = 0.3                    # Low temperature for consistency
AI_MAX_TOKENS = 150                     # Concise responses
AI_ANALYSIS_INTERVAL = 1800             # Every 30 minutes
AI_FEATURES = {
    "market_regime_detection": True,
    "sentiment_analysis": True,
    "anomaly_detection": True,
    "parameter_optimization": True,
}

# ═══════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
CIRCUIT_BREAKER = {
    "enabled": True,
    "triggers": {
        "consecutive_losses": 5,
        "drawdown_pct": 10.0,
        "daily_loss_usd": 200.0,
        "error_rate_pct": 20.0,
        "latency_ms": 1000,
    },
    "cooldown_minutes": 15,
    "notification_channels": ["telegram", "email", "log"],
}

# ═══════════════════════════════════════════════════════════════════════════
# SMART ORDER ROUTING
# ═══════════════════════════════════════════════════════════════════════════
ORDER_ROUTING = {
    "prefer_limit_orders": True,
    "limit_price_improvement_bps": 2,   # Place limits 2bps better
    "use_iceberg_orders": True,         # Hide large orders
    "iceberg_display_pct": 0.2,         # Show only 20% of order
    "adaptive_sizing": True,            # Adjust based on book depth
}

# ═══════════════════════════════════════════════════════════════════════════
# OPERATIONAL SETTINGS
# ═══════════════════════════════════════════════════════════════════════════
# Fresh start configuration - as requested
START_FRESH_DAILY = True                # Don't load previous positions
CLEAR_POSITIONS_ON_START = True         # Reset all positions
RESET_HOUR_UTC = 0                      # Daily reset at midnight UTC

# Logging and monitoring
LOG_DIR = Path("logs")
LOG_LEVEL = "INFO"
LOG_ROTATION = "daily"
SAVE_TRADE_HISTORY = True
TRADE_HISTORY_DAYS = 30

# Paper trading toggle
PAPER_TRADING = os.getenv("PAPER_TRADING", "false").lower() == "true"

# ═══════════════════════════════════════════════════════════════════════════
# MONITORING & ALERTS
# ═══════════════════════════════════════════════════════════════════════════
MONITORING = {
    "prometheus_enabled": True,
    "prometheus_port": 8080,
    "health_check_interval": 60,
    "metrics_retention_days": 7,
}

ALERTS = {
    "telegram_enabled": True,
    "telegram_token": os.getenv("TELEGRAM_TOKEN"),
    "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID"),
    "alert_on_trade": True,
    "alert_on_error": True,
    "alert_on_circuit_break": True,
    "alert_on_large_pnl": True,
    "large_pnl_threshold": 50.0,
}

# ═══════════════════════════════════════════════════════════════════════════
# PERFORMANCE TARGETS
# ═══════════════════════════════════════════════════════════════════════════
TARGETS = {
    "daily_trades": 50,                 # Target number of trades
    "win_rate": 0.55,                   # Target win rate
    "avg_profit_per_trade": 0.3,        # Target profit % per trade
    "sharpe_ratio": 2.0,                # Target Sharpe ratio
    "max_drawdown": 0.1,                # Maximum acceptable drawdown
}

# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════
def validate_config() -> None:
    """Validate all configuration parameters"""
    missing = []
    
    # Check required credentials
    if not COINBASE_API_KEY:
        missing.append("COINBASE_API_KEY")
    if not COINBASE_API_SECRET:
        missing.append("COINBASE_API_SECRET")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Please check your .env file"
        )
    
    # Validate limits
    if MAX_DAILY_CAPITAL <= 0:
        raise ValueError("MAX_DAILY_CAPITAL must be positive")
    
    if INVENTORY_CAP_USD > MAX_DAILY_CAPITAL:
        raise ValueError("INVENTORY_CAP_USD cannot exceed MAX_DAILY_CAPITAL")
    
    # Create directories
    LOG_DIR.mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    print("=" * 60)
    print("PRODUCTION HFT CONFIGURATION LOADED")
    print("=" * 60)
    print(f"Daily Capital: ${MAX_DAILY_CAPITAL:,.2f}")
    print(f"Trading Coins: {len(TRADE_COINS)} assets")
    print(f"Base Trade Size: ${BASE_TRADE_SIZE_USD}")
    print(f"Max Position: ${INVENTORY_CAP_USD}")
    print(f"API Rate: {REST_RATE_LIMIT_PER_S} req/s")
    print(f"Fresh Start: {START_FRESH_DAILY}")
    print("=" * 60)

# Run validation on import
validate_config() 