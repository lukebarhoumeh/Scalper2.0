"""
config.py – centralised runtime parameters for ScalperBot 2.0
All values can be overridden via .env or regular environment variables.
"""

from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
#  .env  → os.environ
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

# ─── API credentials (required) ──────────────────────────────────────────────
COINBASE_API_KEY    = os.getenv("COINBASE_API_KEY")
COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET")
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")

# ─── Trading universe & sizing ───────────────────────────────────────────────
TRADE_COINS    = [
    c.strip().upper()
    for c in os.getenv("TRADE_COINS", "XRP,XLM,ADA,HBAR").split(",")
    if c.strip()
]
TRADE_SIZE_USD = float(os.getenv("TRADE_SIZE_USD", 10))

# ─── Engine behaviour ────────────────────────────────────────────────────────
POLL_INTERVAL_SEC     = int(os.getenv("POLL_INTERVAL_SEC", 15))       # tick spacing
REST_RATE_LIMIT_PER_S = int(os.getenv("REST_RATE_LIMIT_PER_S", 10))   # keep < 20/s
USE_WS_FEED           = os.getenv("USE_WS_FEED", "true").lower() in ("1", "true", "yes")

# ─── Phase 2 risk controls ───────────────────────────────────────────────────
COOLDOWN_SEC        = int(os.getenv("COOLDOWN_SEC", 60))
INVENTORY_CAP_USD   = float(os.getenv("INVENTORY_CAP_USD", 200))
TARGET_VOL_PCT      = float(os.getenv("TARGET_VOL_PCT", 10))   # annualised %
VOL_FLOOR_PCT       = float(os.getenv("VOL_FLOOR_PCT", 3))     # ignore signals below this

# ─── Local paths & mode ──────────────────────────────────────────────────────
LOG_DIR        = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(exist_ok=True)
PAPER_TRADING  = os.getenv("PAPER_TRADING", "true").lower() in ("1", "true", "yes")

# ─── Basic validation ────────────────────────────────────────────────────────
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
