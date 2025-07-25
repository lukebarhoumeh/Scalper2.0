# trade_logger.py
from __future__ import annotations
import csv
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, Any
from pathlib import Path

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
    Creates dated files for each run (e.g., 7/18Trades.csv).
    """

    def __init__(self):
        # Create dated filename - use hyphen instead of slash for Windows compatibility
        current_date = datetime.now().strftime("%m-%d")
        self._filename = f"{current_date}Trades.csv"
        self._filepath = str(LOG_DIR / self._filename)
        
        # Write header if new file
        if not Path(self._filepath).exists():
            with open(self._filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(_HEADERS)
            print(f"[INFO] Created new trade log: {self._filename}")

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
        with open(self._filepath, "a", newline="") as f:
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
        return f"TradeLogger(path={self._filepath})"
