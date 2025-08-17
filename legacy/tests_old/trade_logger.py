# trade_logger.py
from __future__ import annotations
import csv
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, Any

from config import LOG_DIR

_HEADERS = [
    "order_id",
    "timestamp",
    "product_id",
    "side",
    "qty_base",
    "price",
    "notional_usd",
    "strategy",
    "running_pnl",
]

class TradeLogger:
    """
    CSV‑based trade logger. Designed to be append‑only & crash‑safe.
    """

    def __init__(self, log_dir: str = LOG_DIR):
        os.makedirs(log_dir, exist_ok=True)
        self._path = os.path.join(log_dir, "trades.csv")
        if not os.path.isfile(self._path):
            with open(self._path, "w", newline="") as f:
                csv.writer(f).writerow(_HEADERS)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────
    def log(
        self,
        product_id: str,
        side: str,
        qty_base: float,
        price: float,
        strategy: str,
        pnl_running: float,
        order_id: Optional[str] = None,
    ) -> None:
        order_id = order_id or str(uuid.uuid4())
        with open(self._path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    order_id,
                    datetime.now(timezone.utc).isoformat(),
                    product_id,
                    side.lower(),
                    f"{qty_base:.10f}",
                    f"{price:.8f}",
                    f"{qty_base*price:.2f}",
                    strategy,
                    f"{pnl_running:.2f}",
                ]
            )

    def __repr__(self) -> str:
        return f"TradeLogger(path={self._path})"
