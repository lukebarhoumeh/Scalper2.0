# adaptive_utils.py
import time
from collections import defaultdict
from typing import Dict

from config import (
    COOLDOWN_SEC,
    INVENTORY_CAP_USD,
    TARGET_VOL_PCT,
    VOL_FLOOR_PCT,
    TRADE_SIZE_USD,
)

class CooldownTracker:
    """
    Per‑product cool‑down timer.
    """
    def __init__(self, cooldown_sec: int = COOLDOWN_SEC):
        self._cd = cooldown_sec
        self._last_ts: Dict[str, float] = defaultdict(lambda: 0.0)

    def ready(self, product_id: str) -> bool:
        return (time.time() - self._last_ts[product_id]) >= self._cd

    def stamp(self, product_id: str) -> None:
        self._last_ts[product_id] = time.time()


def vol_scaled_size(realised_vol_pct: float) -> float:
    """
    Scale base TRADE_SIZE_USD so that dollar risk approximates TARGET_VOL_PCT.
    """
    vol = max(realised_vol_pct, VOL_FLOOR_PCT)
    factor = TARGET_VOL_PCT / vol
    return TRADE_SIZE_USD * factor


def inventory_allows(executor, product_id: str, side: str, usd_notional: float) -> bool:
    """
    FIXED: Enforce strict position limits to prevent directional bias.
    Check both long and short position limits per coin.
    """
    from config import PER_COIN_POSITION_LIMIT, INVENTORY_CAP_USD
    
    current_usd = executor.position_usd(product_id)
    
    # Use the stricter of the two limits
    position_limit = min(PER_COIN_POSITION_LIMIT, INVENTORY_CAP_USD)
    
    if side == "BUY":
        # Prevent excessive long positions
        new_position = current_usd + usd_notional
        if new_position > position_limit:
            return False
        # Also check if already short - allow buys to reduce short position
        return True
    else:  # SELL
        # Prevent excessive short positions
        new_position = current_usd - usd_notional
        if new_position < -position_limit:
            return False
        # Also check if already long - allow sells to reduce long position  
        return True
