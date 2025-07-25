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
COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ─── Trading Universe ────────────────────────────────────────────────────────
# High-liquidity pairs only - can be overridden by env var
DEFAULT_TRADING_PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "DOGE-USD"]
TRADING_PAIRS = os.getenv("TRADING_PAIRS", ",".join(DEFAULT_TRADING_PAIRS)).split(",")

# ─── Risk Management Parameters ──────────────────────────────────────────────
# Position sizing
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "0.01"))  # 1% risk per trade
MAX_TOTAL_EXPOSURE = float(os.getenv("MAX_TOTAL_EXPOSURE", "0.10"))  # 10% max exposure
MAX_DAILY_LOSS_PERCENT = float(os.getenv("MAX_DAILY_LOSS_PERCENT", "0.03"))  # 3% daily loss limit

# Stop loss and take profit
DEFAULT_RISK_REWARD_RATIO = float(os.getenv("RISK_REWARD_RATIO", "1.0"))  # 1:1 risk/reward
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))  # For dynamic stop loss calculation
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "2.0"))  # Stop loss = ATR * multiplier

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

# ─── Execution Parameters ────────────────────────────────────────────────────
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() in ("1", "true", "yes")
POLL_INTERVAL_SEC = int(os.getenv("POLL_INTERVAL_SEC", "5"))
USE_WS_FEED = os.getenv("USE_WS_FEED", "true").lower() in ("1", "true", "yes")

# ─── Logging Configuration ───────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_ROTATION_SIZE = int(os.getenv("LOG_ROTATION_SIZE", "10485760"))  # 10MB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

# ─── Account Management ──────────────────────────────────────────────────────
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "1000.0"))
MIN_WORKING_CAPITAL = float(os.getenv("MIN_WORKING_CAPITAL", "500.0"))
PROFIT_WITHDRAWAL_THRESHOLD = float(os.getenv("PROFIT_WITHDRAWAL_THRESHOLD", "50.0"))

# ─── Helper Functions ────────────────────────────────────────────────────────
def get_config_dict() -> Dict[str, Any]:
    """Get all configuration as a dictionary"""
    return {
        "trading_pairs": TRADING_PAIRS,
        "risk_percent": RISK_PERCENT,
        "max_total_exposure": MAX_TOTAL_EXPOSURE,
        "max_daily_loss_percent": MAX_DAILY_LOSS_PERCENT,
        "risk_reward_ratio": DEFAULT_RISK_REWARD_RATIO,
        "atr_period": ATR_PERIOD,
        "atr_multiplier": ATR_MULTIPLIER,
        "max_requests_per_second": MAX_REQUESTS_PER_SECOND,
        "paper_trading": PAPER_TRADING,
        "initial_capital": INITIAL_CAPITAL,
    }

def validate_config():
    """Validate configuration parameters"""
    errors = []
    
    if not COINBASE_API_KEY:
        errors.append("COINBASE_API_KEY not set")
    if not COINBASE_API_SECRET:
        errors.append("COINBASE_API_SECRET not set")
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY not set")
        
    if RISK_PERCENT <= 0 or RISK_PERCENT > 0.05:
        errors.append("RISK_PERCENT must be between 0 and 0.05 (5%)")
        
    if MAX_TOTAL_EXPOSURE <= 0 or MAX_TOTAL_EXPOSURE > 0.5:
        errors.append("MAX_TOTAL_EXPOSURE must be between 0 and 0.5 (50%)")
        
    if not TRADING_PAIRS:
        errors.append("No trading pairs defined")
        
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")
        
# Validate on import
validate_config() 