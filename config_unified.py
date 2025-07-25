#!/usr/bin/env python3
"""
Unified Configuration for ScalperBot Production
==============================================
Centralized configuration with dynamic risk management parameters
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─── API Credentials ─────────────────────────────────────────────────────────
CB_API_KEY = os.getenv("CB_API_KEY")
CB_API_SECRET = os.getenv("CB_API_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ─── Trading Universe ────────────────────────────────────────────────────────
# Expanded high-liquidity pairs
DEFAULT_TRADING_PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "DOGE-USD", 
                        "MATIC-USD", "ADA-USD", "LINK-USD", "XLM-USD", "XRP-USD"]
TRADING_PAIRS = os.getenv("TRADING_PAIRS", ",".join(DEFAULT_TRADING_PAIRS)).split(",")

# Clean up trading pairs (remove extra spaces)
TRADING_PAIRS = [pair.strip() for pair in TRADING_PAIRS if pair.strip()]

# ─── Position Sizing Parameters ──────────────────────────────────────────────
BASE_POSITION_SIZE = float(os.getenv("BASE_POSITION_SIZE", "75.0"))
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "200.0"))
INVENTORY_CAP_USD = float(os.getenv("INVENTORY_CAP_USD", "600.0"))
STARTING_CAPITAL = float(os.getenv("STARTING_CAPITAL", "1000.0"))
MAX_DAILY_CAPITAL = float(os.getenv("MAX_DAILY_CAPITAL", "1500.0"))

# ─── Risk Management Parameters ──────────────────────────────────────────────
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "0.01"))  # 1% risk per trade
MAX_TOTAL_EXPOSURE = float(os.getenv("MAX_TOTAL_EXPOSURE", "0.10"))  # 10% max exposure
MAX_DAILY_LOSS_PERCENT = float(os.getenv("MAX_DAILY_LOSS_PERCENT", "0.03"))  # 3% daily loss limit

# Stop loss and take profit
DEFAULT_RISK_REWARD_RATIO = float(os.getenv("RISK_REWARD_RATIO", "1.0"))  # 1:1 risk/reward
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))  # For dynamic stop loss calculation
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "2.0"))  # Stop loss = ATR * multiplier

# ─── Technical Indicators ────────────────────────────────────────────────────
RSI_BUY_THRESHOLD = float(os.getenv("RSI_BUY_THRESHOLD", "35"))
RSI_SELL_THRESHOLD = float(os.getenv("RSI_SELL_THRESHOLD", "65"))
SMA_FAST = int(os.getenv("SMA_FAST", "5"))
SMA_SLOW = int(os.getenv("SMA_SLOW", "20"))
MIN_PRICE_HISTORY = int(os.getenv("MIN_PRICE_HISTORY", "20"))

# ─── Spread Management ───────────────────────────────────────────────────────
MIN_SPREAD_BPS = float(os.getenv("MIN_SPREAD_BPS", "30"))  # 0.3% minimum spread
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", "150"))  # 1.5% max spread

# ─── API Rate Limiting ───────────────────────────────────────────────────────
MAX_REQUESTS_PER_SECOND = int(os.getenv("MAX_REQUESTS_PER_SECOND", "8"))
WS_RECONNECT_TIMEOUT = int(os.getenv("WS_RECONNECT_TIMEOUT", "30"))

# ─── Strategy Parameters ─────────────────────────────────────────────────────
# Dynamic strategy weights (will be adjusted based on market conditions)
INITIAL_SCALPING_WEIGHT = float(os.getenv("SCALPING_WEIGHT", "0.6"))
INITIAL_BREAKOUT_WEIGHT = float(os.getenv("BREAKOUT_WEIGHT", "0.4"))

# Market condition thresholds
VOLUME_MA_PERIOD = int(os.getenv("VOLUME_MA_PERIOD", "20"))
VOLUME_MULTIPLIER = float(os.getenv("VOLUME_MULTIPLIER", "1.2"))  # Volume must exceed MA * multiplier
MAX_SPREAD_PERCENT = float(os.getenv("MAX_SPREAD_PERCENT", "0.002"))  # 0.2% max spread

# ─── Circuit Breaker Parameters ──────────────────────────────────────────────
MAX_TRADES_PER_HOUR = int(os.getenv("MAX_TRADES_PER_HOUR", "60"))
MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "5"))

# ─── Execution Parameters ────────────────────────────────────────────────────
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() in ("1", "true", "yes")
POLL_INTERVAL_SEC = int(os.getenv("POLL_INTERVAL_SEC", "30"))
USE_WS_FEED = os.getenv("USE_WS_FEED", "true").lower() in ("1", "true", "yes")

# ─── Logging Configuration ───────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_ROTATION_SIZE = int(os.getenv("LOG_ROTATION_SIZE", "10485760"))  # 10MB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

# ─── Account Management ──────────────────────────────────────────────────────
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "1000.0"))
MIN_WORKING_CAPITAL = float(os.getenv("MIN_WORKING_CAPITAL", "500.0"))
PROFIT_WITHDRAWAL_THRESHOLD = float(os.getenv("PROFIT_WITHDRAWAL_THRESHOLD", "50.0"))
PROFIT_WITHDRAWAL_PERCENT = float(os.getenv("PROFIT_WITHDRAWAL_PERCENT", "0.50"))
MIN_PROFIT_FOR_WITHDRAWAL = float(os.getenv("MIN_PROFIT_FOR_WITHDRAWAL", "50.0"))

# ─── Helper Functions ────────────────────────────────────────────────────────
def get_config_dict() -> Dict[str, Any]:
    """Get all configuration as a dictionary"""
    return {
        "trading_pairs": TRADING_PAIRS,
        "base_position_size": BASE_POSITION_SIZE,
        "max_position_size": MAX_POSITION_SIZE,
        "inventory_cap_usd": INVENTORY_CAP_USD,
        "starting_capital": STARTING_CAPITAL,
        "max_daily_capital": MAX_DAILY_CAPITAL,
        "risk_percent": RISK_PERCENT,
        "max_total_exposure": MAX_TOTAL_EXPOSURE,
        "max_daily_loss_percent": MAX_DAILY_LOSS_PERCENT,
        "risk_reward_ratio": DEFAULT_RISK_REWARD_RATIO,
        "rsi_buy_threshold": RSI_BUY_THRESHOLD,
        "rsi_sell_threshold": RSI_SELL_THRESHOLD,
        "sma_fast": SMA_FAST,
        "sma_slow": SMA_SLOW,
        "min_spread_bps": MIN_SPREAD_BPS,
        "max_spread_bps": MAX_SPREAD_BPS,
        "max_requests_per_second": MAX_REQUESTS_PER_SECOND,
        "max_trades_per_hour": MAX_TRADES_PER_HOUR,
        "max_consecutive_losses": MAX_CONSECUTIVE_LOSSES,
        "paper_trading": PAPER_TRADING,
        "initial_capital": INITIAL_CAPITAL,
    }

def validate_config():
    """Validate configuration parameters"""
    errors = []
    
    if not CB_API_KEY:
        errors.append("CB_API_KEY not set")
    if not CB_API_SECRET:
        errors.append("CB_API_SECRET not set")
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY not set")
        
    if RISK_PERCENT <= 0 or RISK_PERCENT > 0.05:
        errors.append("RISK_PERCENT must be between 0 and 0.05 (5%)")
        
    if MAX_TOTAL_EXPOSURE <= 0 or MAX_TOTAL_EXPOSURE > 0.5:
        errors.append("MAX_TOTAL_EXPOSURE must be between 0 and 0.5 (50%)")
        
    if BASE_POSITION_SIZE <= 0 or BASE_POSITION_SIZE > 1000:
        errors.append("BASE_POSITION_SIZE must be between 0 and 1000")
        
    if MAX_POSITION_SIZE < BASE_POSITION_SIZE:
        errors.append("MAX_POSITION_SIZE must be >= BASE_POSITION_SIZE")
        
    if len(TRADING_PAIRS) == 0:
        errors.append("TRADING_PAIRS cannot be empty")
        
    return errors

def get_api_credentials():
    """Get API credentials with fallback"""
    return {
        "api_key": CB_API_KEY,
        "api_secret": CB_API_SECRET,
        "openai_key": OPENAI_API_KEY
    }