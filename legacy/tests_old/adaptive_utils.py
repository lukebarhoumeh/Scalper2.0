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
    Enforce INVENTORY_CAP_USD. Skip BUY if long exposure would exceed cap.
    Force SELL if long exposure > cap.
    """
    current = executor.position_usd(product_id)
    if side == "BUY":
        return (current + usd_notional) <= INVENTORY_CAP_USD
    else:  # SELL
        return True  # always allow sell
