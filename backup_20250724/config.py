"""
config_hft_optimized.py – Production-grade HFT configuration
Optimized based on 24-48 hour performance analysis
"""

from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─── API credentials (required) ──────────────────────────────────────────────
COINBASE_API_KEY    = os.getenv("COINBASE_API_KEY")
COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET")
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")

# ─── OPTIMIZED HFT Trading Universe ──────────────────────────────────────────
# Import from unified config
from config_unified import TRADING_PAIRS

# Convert to coin symbols (remove -USD suffix)
TRADE_COINS = [pair.replace("-USD", "") for pair in TRADING_PAIRS]

# ─── HFT-OPTIMIZED Engine Parameters ─────────────────────────────────────────
POLL_INTERVAL_SEC = 20              # Sweet spot: Not too aggressive on API
TRADE_SIZE_USD = 75.0               # Larger for meaningful profits
REST_RATE_LIMIT_PER_S = 8           # Safe but fast
USE_WS_FEED = True                  # WebSocket for real-time data
MAX_DAILY_CAPITAL = 1000.0          # $1,000 daily capital limit

# ─── AGGRESSIVE Risk Controls (But Safe) ─────────────────────────────────────
COOLDOWN_SEC = 180                  # 3 minutes - allows 20 trades/hour per coin
INVENTORY_CAP_USD = 600.0           # Total exposure limit (increased for more opportunities)
PER_COIN_POSITION_LIMIT = 200.0     # Max $200 per coin (increased from $100)
MAX_POSITION_USD = 150.0            # Max per trade (increased from $75)

# ─── HFT-TUNED Strategy Parameters ───────────────────────────────────────────
# Scalper settings - AGGRESSIVE for high frequency
SCALPER_SMA_WINDOW = 20             # Faster signals (was 30)
SCALPER_VOL_THRESH = 8.0            # Much lower - catch more opportunities
SCALPER_SPREAD_THRESH = 0.5         # Tighter - more signals

# Breakout settings - RESPONSIVE
BREAKOUT_LOOKBACK = 60              # Faster detection
BREAKOUT_ATR_WINDOW = 14            # Standard ATR
BREAKOUT_ATR_MULT = 1.8             # More sensitive

# Strategy weights - Favor scalping
SCALPER_WEIGHT = 0.7
BREAKOUT_WEIGHT = 0.3

# ─── Volatility Parameters ───────────────────────────────────────────────────
TARGET_VOL_PCT = 5.0                # Lower target for more trades
VOL_FLOOR_PCT = 1.0                 # Much lower floor

# ─── Circuit Breakers & Safety ───────────────────────────────────────────────
CIRCUIT_BREAKER_THRESHOLD = 10      # More tolerance
CIRCUIT_BREAKER_TIMEOUT = 300       # 5 minute timeout
MAX_DAILY_LOSS_USD = 200.0          # Daily stop loss
MAX_CONSECUTIVE_LOSSES = 5          # Stop after 5 losses in a row

# ─── AI Configuration (Cost-Optimized) ───────────────────────────────────────
OPENAI_MODEL = "gpt-3.5-turbo"      # Cheapest option
AI_ANALYSIS_INTERVAL = 3600         # Only analyze hourly
DISABLE_AI_TRADING = False          # Keep AI but minimize usage

# ─── Logging & Monitoring ────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() in ("1", "true", "yes")
DEBUG_MODE = False                  # Reduce log noise

# ─── Validation ──────────────────────────────────────────────────────────────
def validate_config() -> None:
    missing = [
        name for name, val in (
            ("COINBASE_API_KEY",    COINBASE_API_KEY),
            ("COINBASE_API_SECRET", COINBASE_API_SECRET),
            ("OPENAI_API_KEY",      OPENAI_API_KEY),
        ) if not val
    ]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

validate_config()

# ─── Performance Expectations ────────────────────────────────────────────────
"""
With these settings, expect:
- 50-100 trades per day
- 3-5 trades per coin per hour (during active periods)
- Average spread capture: 0.3-0.5%
- Win rate: 55-60%
- Daily volume: $3,000-7,500
"""

print("=" * 60)
print("HFT OPTIMIZED CONFIGURATION LOADED")
print("=" * 60)
print(f"Trading Coins: {', '.join(TRADE_COINS)}")
print(f"Poll Interval: {POLL_INTERVAL_SEC}s")
print(f"Trade Size: ${TRADE_SIZE_USD}")
print(f"Position Limits: ${PER_COIN_POSITION_LIMIT}/coin, ${INVENTORY_CAP_USD} total")
print(f"Strategy: {SCALPER_WEIGHT*100:.0f}% Scalper, {BREAKOUT_WEIGHT*100:.0f}% Breakout")
print(f"Thresholds: Vol={SCALPER_VOL_THRESH}%, Spread={SCALPER_SPREAD_THRESH}%")
print("=" * 60) 